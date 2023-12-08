#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
from torch.cuda import nvtx
from collections import OrderedDict
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from polygraphy.backend.common import bytes_from_path
from polygraphy import util
from polygraphy.backend.trt import ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import (
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine
)
from polygraphy.logger import G_LOGGER
import tensorrt as trt
from logging import error, warning
from tqdm import tqdm
import copy 

class Registry(dict):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def __str__(self):
        return self.name

    def choices(self):
        return list(self.keys())

    def register(self, name: str):
        def decorator(fn, fn_name=name):
            self[name] = fn
            return fn

        self[name] = decorator
        return self[name]


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
G_LOGGER.module_severity = G_LOGGER.ERROR

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}

class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}
        self.outputs = {}
        self.cuda_graph_instance = None  # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def reset(self, engine_path=None):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors
        self.engine_path = engine_path

        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}
        self.outputs = {}

    def refit_from_dict(self, refit_dict):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        assert len(all_weights)

        refit_wts_cnt = 0
        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if layer_name in refit_dict:
                print(f"[I] refit layer {layer_name}")
                refitter.set_weights(layer_name, weights_role, refit_dict[layer_name])
                refit_wts_cnt += 1

        if not refitter.refit_cuda_engine():
            print("Failed to refit!")
            exit(0)
        print(f"[I] Total refit {refit_wts_cnt} weights.")


    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = [Profile()]
        if input_profile:
            p = [Profile() for i in range(len(input_profile))]
            for _p, i_profile in zip(p, input_profile):
                for name, dims in i_profile.items():
                    assert len(dims) == 3
                    _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        network = network_from_onnx_path(
            onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
        )
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)

        builder = network[0]
        config = builder.create_builder_config()
        config.progress_monitor = TQDMProgressMonitor()

        config.set_flag(trt.BuilderFlag.FP16) if fp16 else None
        config.set_flag(trt.BuilderFlag.REFIT) if enable_refit else None

        cache = None
        try:
            with util.LockFile(timing_cache):
                timing_cache_data = util.load_file(
                    timing_cache, description="tactic timing cache"
                )
                cache = config.create_timing_cache(timing_cache_data)
        except FileNotFoundError:
            warning(
                "Timing cache file {} not found, falling back to empty timing cache.".format(
                    timing_cache
                )
            )
        if cache is not None:
            config.set_timing_cache(cache, ignore_mismatch=True)

        profiles = copy.deepcopy(p)
        for profile in profiles:
            # Last profile is used for set_calibration_profile.
            calib_profile = profile.fill_defaults(network[1]).to_trt(builder, network[1])
            config.add_optimization_profile(calib_profile)

        try:
            engine = engine_from_network(
                network,
                config,
                save_timing_cache=timing_cache,
            )
        except Exception as e:
            error(f"Failed to build engine: {e}")
            return 1
        try:
            save_engine(engine, path=self.engine_path)
        except Exception as e:
            error(f"Failed to save engine: {e}")
            return 1
        return 0

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if self.engine.binding_is_input(binding):
                self.inputs[binding] = idx
            else:
                self.outputs[binding] = idx

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        nvtx.range_push("allocate_buffers")

        for binding, idx in self.inputs.items():
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            self.context.set_binding_shape(idx, shape_dict[binding])

        for binding, idx in self.outputs.items():
            shape = self.context.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor

        nvtx.range_pop()

    # TODO Allow to parse buffffers for Output
    def infer(self, feed_dict, stream, use_cuda_graph=False):
        nvtx.range_push("set_tensors")

        # Inputs
        for name, buf in feed_dict.items():
            self.context.set_tensor_address(name, buf.data_ptr())

        # Outputs
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        nvtx.range_pop()
        nvtx.range_push("execute")
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")
        nvtx.range_pop()
        return self.tensors

    def __str__(self):
        out = ""
        for opt_profile in range(self.engine.num_optimization_profiles):
            for binding_idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(binding_idx)
                shape = self.engine.get_profile_shape(opt_profile, name)
                out += f"\t{name} = {shape}\n"
        return out
