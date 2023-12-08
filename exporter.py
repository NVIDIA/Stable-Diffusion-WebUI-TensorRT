import torch
import torch.nn.functional as F
import onnx
from logging import info, error
import time
import shutil
import onnx_graphsurgeon as gs
import numpy as np
from safetensors.numpy import save_file
from modules import shared
import gc

from utilities import Engine
from models_helper import UNetModelSplit
import os
from pathlib import Path
from optimum.onnx.utils import (
    _get_onnx_external_data_tensors,
    check_model_uses_external_data,
)
from datastructures import ProfileSettings


def apply_lora(model, lora_name, inputs):
    try:
        import sys

        sys.path.append("extensions-builtin/Lora")
        import importlib

        networks = importlib.import_module("networks")
        network = importlib.import_module("network")
        lora_net = importlib.import_module("extra_networks_lora")
    except Exception as e:
        error(e)
        error("LoRA not found. Please install LoRA extension first from ...")
    model.forward(*inputs)
    networks.load_networks([lora_name], [1.0], [1.0], [None])

    model.forward(*inputs)
    return model


def lora_from_pt(pytorch_model_sd: dict, onnx_path: str, delta: bool, prefix="unet"):
    def convert_int64(arr):
        if len(arr.shape) == 0:
            return np.array([np.int32(arr)])
        return arr

    def add_to_map(
        refit_dict, onnx_array, torch_array, delta, eps=1e-6
    ):  # Shoud be sufficient since fp16 only goes to 6e-5
        assert onnx_array.name not in refit_dict

        if torch_array.dtype == np.int64:
            torch_array = convert_int64(torch_array)

        if torch_array.shape != onnx_array.values.shape:
            torch_array = torch_array.transpose()
            assert torch_array.shape == onnx_array.values.shape

        if delta:
            assert torch_array.dtype == onnx_array.values.dtype
            torch_array = torch_array - onnx_array.values
        if torch_array.sum() < eps:
            return
        refit_dict[onnx_array.name] = torch_array

    print(f"Refitting TensorRT engine with LoRA weights")

    # Construct mapping from weight names in original ONNX model -> LoRA fused PyTorch model
    nodes = gs.import_onnx(onnx.load(onnx_path)).toposort().nodes

    refit_dict = {}
    lora_keywords = ["to_q", "to_k", "to_v", "to_out"]
    for node in nodes:
        for kw in lora_keywords:
            if kw in node.name and "MatMul" in node.name:
                for inp in node.inputs:
                    if inp.__class__ == gs.Constant:
                        onnx_node_name = node.name
                        pt_weight_name = onnx_node_name.replace(f"/{kw}/{kw}", f"/{kw}")
                        pt_weight_name = ".".join(
                            [prefix, *pt_weight_name.split("/")[2:]]
                        )
                        pt_weight_name = pt_weight_name.replace("MatMul", "weight")

                        # assert pt_weight_name in pytorch_model_sd
                        if pt_weight_name not in pytorch_model_sd:
                            print(
                                f"Warning: {pt_weight_name} not found in PyTorch model"
                            )
                        add_to_map(
                            refit_dict,
                            inp,
                            pytorch_model_sd[pt_weight_name].cpu().detach().numpy(),
                            delta=delta,
                        )

    return refit_dict


def export_lora(
    modelobj: UNetModelSplit, onnx_path: str, lora_name: str, profile: ProfileSettings
):
    inputs = modelobj.get_encoder_sample_input(
        profile.bs_opt * 2, profile.h_opt // 8, profile.w_opt // 8, profile.t_min
    )
    with torch.inference_mode(), torch.autocast("cuda"):
        modelobj.unet = apply_lora(modelobj.unet, lora_name, inputs)

    encoder_refit = lora_from_pt(
        modelobj.encoder.state_dict(),
        os.path.join(onnx_path, "encoder.onnx"),
        delta=True,
    )
    decoder_refit = lora_from_pt(
        modelobj.decoder.state_dict(),
        os.path.join(onnx_path, "decoder.onnx"),
        delta=True,
    )

    return encoder_refit, decoder_refit


def export_onnx(
    onnx_path: str,
    modelobj: UNetModelSplit = None,
    profile=None,
    opset=17,
    diable_optimizations=False,
    lora_path=None,
):
    swap_sdpa = hasattr(F, "scaled_dot_product_attention")
    old_sdpa = getattr(F, "scaled_dot_product_attention", None) if swap_sdpa else None
    if swap_sdpa:
        delattr(F, "scaled_dot_product_attention")

    info("Exporting to ONNX...")
    inputs = modelobj.get_encoder_sample_input(
        profile["sample"][1][0],
        profile["sample"][1][-2],
        profile["sample"][1][-1],
        profile["encoder_hidden_states"][1][1],
    )

    if not os.path.exists(os.path.join(onnx_path, "encoder.onnx")):
        _export_onnx(
            modelobj.encoder,
            inputs,
            Path(os.path.join(onnx_path, "encoder.onnx")),
            opset,
            modelobj.get_encoder_input_names(),
            modelobj.get_encoder_output_names(),
            modelobj.get_encoder_dynamic_axes(),
            modelobj.optimize if not diable_optimizations else None,
        )

    inputs = modelobj.get_decoder_sample_input(
        profile["sample"][1][0],
        profile["sample"][1][-2],
        profile["sample"][1][-1],
        profile["encoder_hidden_states"][1][1],
    )
    if not os.path.exists(os.path.join(onnx_path, "decoder.onnx")):
        _export_onnx(
            modelobj.decoder,
            inputs,
            Path(os.path.join(onnx_path, "decoder.onnx")),
            opset,
            modelobj.get_decoder_input_names(),
            modelobj.get_decoder_output_names(),
            modelobj.get_decoder_dynamic_axes(),
            modelobj.optimize if not diable_optimizations else None,
        )
    # CleanUp
    if swap_sdpa and old_sdpa:
        setattr(F, "scaled_dot_product_attention", old_sdpa)


def _export_onnx(
    model, inputs, path, opset, in_names, out_names, dyn_axes, optimizer=None
):
    os.makedirs(os.path.abspath("onnx_tmp"), exist_ok=True)
    tmp_path = os.path.abspath(os.path.join("onnx_tmp", "model.onnx"))
    try:
        info("Exporting to ONNX...")
        with torch.inference_mode(), torch.autocast("cuda"):
            torch.onnx.export(
                model,
                inputs,
                tmp_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dyn_axes,
            )
    except Exception as e:
        error("Exporting to ONNX failed. {}".format(e))
        return

    info("Optimize ONNX.")
    os.makedirs(path.parent, exist_ok=True)
    onnx_model = onnx.load(tmp_path, load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        info("ONNX model uses external data. Saving as external data.")
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(tmp_path, load_external_data=True)
        onnx.save(
            onnx_model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + "_data",
            size_threshold=1024,
        )
        shutil.rmtree(os.path.abspath("onnx_tmp"))

    if optimizer is not None:
        try:
            onnx_opt_graph = optimizer("encoder", onnx_model)
            onnx.save(onnx_opt_graph, path)
        except Exception as e:
            error("Optimizing ONNX failed. {}".format(e))
            return


def export_trt(trt_path, onnx_path, timing_cache, profile, use_fp16):
    engine = Engine(trt_path)

    # TODO Still approx. 2gb of VRAM unaccounted for...
    model = shared.sd_model.cpu()
    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_refit=True,
        enable_preview=True,
        timing_cache=timing_cache,
        input_profile=[profile],
        # hwCompatibility=hwCompatibility,
    )
    e = time.time()
    info(f"Time taken to build: {(e-s)}s")
    del engine

    shared.sd_model = model.cuda()
    return ret
