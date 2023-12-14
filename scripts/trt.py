import os
import numpy as np

import torch
from torch.cuda import nvtx
from modules import (
    script_callbacks,
    sd_unet,
    devices,
    scripts,
    sd_models,
)
import ui_trt
from utilities import Engine
from typing import List
from model_manager import TRT_MODEL_DIR, modelmanager
from polygraphy.logger import G_LOGGER
import gradio as gr
from scripts.lora import apply_loras
import re
from models_helper import get_unet_shape_dict, get_cnet_shape_dict
from datastructures import UNetEngineArgs, ModelType

G_LOGGER.module_severity = G_LOGGER.ERROR


class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str, filename: List[dict]):
        self.label = f"[TRT] {name}"
        self.model_name = name
        self.configs = filename

    def create_unet(self):
        return TrtUnet(self.model_name, self.configs)


GLOBAL_ARGS = UNetEngineArgs(0, 0, None, {})


class SplitUNetTRT:
    def __init__(self, configs: list):
        self.configs = configs
        self.engine_vram_req = 0
        self.profile_idx = GLOBAL_ARGS.idx
        self.device = devices.device
        self.loaded_config = self.configs[self.profile_idx]
        self.encoder = Engine(
            os.path.join(self.loaded_config["filepath"], "encoder.trt")
        )
        self.decoder = Engine(
            os.path.join(self.loaded_config["filepath"], "decoder.trt")
        )

        self.workspace_memory = None
        self.shape_hash = 0
        self.refitted_keys = set()

    def allocate_buffers(self, shape_dict):
        self.encoder.allocate_buffers(shape_dict)
        self.decoder.allocate_buffers(shape_dict)

        self.engine_vram_req = max(
            self.encoder.engine.device_memory_size,
            self.decoder.engine.device_memory_size,
        )

        self.workspace_memory = torch.empty(
            self.engine_vram_req, dtype=torch.int8, device=self.device
        )
        self.encoder.context.device_memory = self.workspace_memory.data_ptr()
        self.decoder.context.device_memory = self.workspace_memory.data_ptr()

    def forward(self, x, timesteps, context, y=None, hs=None, cuda_stream=None):
        if not self.profile_idx == GLOBAL_ARGS.idx:
            self.switch_engine()

        curr_hash = hash(x.shape + context.shape)
        if self.shape_hash != curr_hash:
            shape_dict = get_unet_shape_dict(x, context, y)
            self.allocate_buffers(shape_dict)
            self.shape_hash = curr_hash

        nvtx.range_push("forward")
        feed_dict = {
            "sample": x.float(),
            "timesteps": timesteps.float(),
            "encoder_hidden_states": context.float(),
        }
        if y is not None:
            feed_dict["y"] = y.float()

        out = self.encoder.infer(feed_dict, cuda_stream)

        # Add hidden states from controlnets
        if hs is not None:
            nvtx.range_push("hidden_states")
            for k, v in hs.items():
                assert out[k].dtype == v.dtype
                out[k] += v
            nvtx.range_pop()

        out.update({"encoder_hidden_states": context.float()})
        out = self.decoder.infer(out, cuda_stream)["latent"]

        nvtx.range_pop()
        return out

    def activate(self):
        self.shape_hash = 0

        self.encoder.load()
        self.decoder.load()

        self.encoder.activate(True)
        self.decoder.activate(True)

    def deactivate(self):
        del self.encoder
        del self.decoder
        del self.workspace_memory

    def apply_loras(self):
        if GLOBAL_ARGS.lora is None:
            enc, dec = {}, {}
        else:
            enc, dec = GLOBAL_ARGS.lora
        if not self.refitted_keys.issubset(set(enc.keys()) | set(dec.keys())):
            # Need to ensure that weights that have been modified before and are not present anymore are reset.
            self.refitted_keys = set()
            self.switch_engine()

        self.encoder.refit_from_dict(enc, is_fp16=True)
        self.decoder.refit_from_dict(dec, is_fp16=True)

        self.refitted_keys = set(enc.keys()) | set(dec.keys())

    def switch_engine(self):
        self.profile_idx = GLOBAL_ARGS.idx
        self.loaded_config = self.configs[self.profile_idx]
        self.encoder.reset(os.path.join(self.loaded_config["filepath"], "encoder.trt"))
        self.decoder.reset(os.path.join(self.loaded_config["filepath"], "decoder.trt"))
        self.activate()
        self.shape_hash = 0


class ControlNetsTRT:
    def __init__(self, configs: dict):
        self.configs = configs
        self.engine_vram_req = 0

        self.profile_idx = {}
        self.loaded_config = {}
        self.engines = {}
        self.hidden_states = {}

        for k, v in GLOBAL_ARGS.controlnets.items():
            self.profile_idx[k] = v.idx
            self.loaded_config[k] = self.configs[k][self.profile_idx[k]]
            self.engines[k] = Engine(
                os.path.join(self.loaded_config[k]["filepath"], "cnet.trt")
            )

        self.workspace_memory = None
        self.shape_hash = 0
        self.device = devices.device

    def allocate_buffers(self, shape_dict):
        for k, cnet in self.engines.items():
            cnet.allocate_buffers(shape_dict)  # TODO handle hs buffers internally
            self.engine_vram_req = max(
                self.engine_vram_req, cnet.engine.device_memory_size
            )

        self.workspace_memory = torch.empty(
            self.engine_vram_req, dtype=torch.int8, device=self.device
        )
        for k, cnet in self.engines.items():
            cnet.context.device_memory = self.workspace_memory.data_ptr()

    def forward(self, x, timesteps, context, y=None, cuda_stream=None):
        if y is not None:
            # FIXME
            gr.Error("ControlNet does not support SDXL yet")
            return
        # Ensure that currently loaded controlnets match the selected ones
        if self.engines.keys() != GLOBAL_ARGS.controlnets.keys():
            engines_to_delete = [
                k
                for k in self.engines.keys()
                if k not in GLOBAL_ARGS.controlnets.keys()
            ]
            engines_to_load = [
                k
                for k in GLOBAL_ARGS.controlnets.keys()
                if k not in self.engines.keys()
            ]

            for k in engines_to_delete:
                del self.engines[k]

            for k in engines_to_load:
                self.profile_idx[k] = GLOBAL_ARGS.controlnets[k].idx
                self.loaded_config[k] = self.configs[k][self.profile_idx[k]]
                self.engines[k] = Engine(
                    os.path.join(self.loaded_config[k]["filepath"], "cnet.trt")
                )
                self.engines[k].load()
                self.engines[k].activate(True)
                self.shape_hash = 0

        # Switch engines if necessary
        for k, cnet in self.engines.items():
            if not self.profile_idx[k] == GLOBAL_ARGS.controlnets[k].idx:
                self.switch_engine(k)

        # Allocate buffers if shape or models change
        curr_hash = hash(x.shape + context.shape)
        if self.shape_hash != curr_hash:
            shape_dict = get_cnet_shape_dict(x, context, y)
            self.allocate_buffers(shape_dict)
            self.shape_hash = curr_hash

        feed_dict = {  # TODO fix to float
            "sample": x.half(),
            "timesteps": timesteps[:1].half(),
            "encoder_hidden_states": context.half(),
            "controlnet_cond": GLOBAL_ARGS.controlnets[k].condition.half(),
            "conditioning_scale": torch.tensor(
                [GLOBAL_ARGS.controlnets[k].scale],
                device=devices.device,
                dtype=torch.float16,
            ),
        }
        out = None
        for cnet in self.engines.values():
            _out = cnet.infer(
                feed_dict, cuda_stream
            )  # Out tensors need to be handled globally to reduce memory usage
            if out is None:
                out = _out
            else:
                for k, v in _out.items():
                    out[k] += v.float()

        return out

    def activate(self):
        for k, v in self.engines.items():
            v.load()
            v.activate(True)
            self.engine_vram_req = max(
                self.engine_vram_req, v.engine.device_memory_size
            )

    def deactivate(self):
        for k, v in self.engines.items():
            del v

    def switch_engine(self, k):
        self.profile_idx[k] = GLOBAL_ARGS.controlnets[k].idx
        self.loaded_config[k] = self.configs[k][self.profile_idx[k]]
        self.engines[k].reset(
            os.path.join(self.loaded_config[k]["filepath"], "cnet.trt")
        )
        self.engines[k].load()
        self.engines[k].activate(True)
        self.shape_hash = 0


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, model_name: str, configs: List[dict], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stream = None
        self.model_name = model_name
        self.unet = SplitUNetTRT(configs)
        self.controlnets = ControlNetsTRT(
            modelmanager.get_available_models(ModelType.CONTROLNET)
        )

    def forward(self, x, timesteps, context, *args, **kwargs):
        y = kwargs.get("y", None)
        hs = None
        cuda_stream = torch.cuda.current_stream().cuda_stream
        if GLOBAL_ARGS.controlnets:
            hs = self.controlnets.forward(
                x, timesteps, context, y=y, cuda_stream=cuda_stream
            )
        out = self.unet.forward(
            x, timesteps, context, y=y, hs=hs, cuda_stream=cuda_stream
        )

        return out

    def activate(self):
        self.unet.activate()
        self.controlnets.activate()

    def deactivate(self):
        self.unet.deactivate()
        self.controlnets.deactivate()

    def apply_loras(self):
        self.unet.apply_loras()


class TensorRTScript(scripts.Script):
    def __init__(self) -> None:
        self.loaded_model = None
        self.lora_hash = ""
        self.update_lora = False

    def title(self):
        return "TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def setup(self, p, *args):
        return super().setup(p, *args)

    def before_process(self, p, *args):  # 1
        # Check divisibilty
        if p.width % 64 or p.height % 64:
            gr.Error("Target resolution must be divisible by 64 in both dimensions.")

        if p.enable_hr:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            if hr_w % 64 or hr_h % 64:
                gr.Error(
                    "HIRES Fix resolution must be divisible by 64 in both dimensions. Please change the upscale factor or disable HIRES Fix."
                )

    def get_profile_idx(self, p, model_name):
        best_hr = None
        hr_scale = p.hr_scale if p.enable_hr else 1
        (
            valid_models,
            distances,
            idx,
        ) = modelmanager.get_valid_models(
            model_name, p.width, p.height, p.batch_size, 77, ModelType.UNET
        )  # TODO: max_embedding, just ignore?
        if len(valid_models) == 0:
            gr.Error(
                f"""No valid profile found for ({model_name}) LOWRES. Please go to the TensorRT tab and generate an engine with the necessary profile. 
                If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."""
            )
        best = idx[np.argmin(distances)]

        if hr_scale != 1:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            valid_models_hr, distances_hr, idx_hr = modelmanager.get_valid_models(
                model_name, hr_w, hr_h, p.batch_size, 77, ModelType.UNET
            )  # TODO: max_embedding
            if len(valid_models_hr) == 0:
                gr.Error(
                    f"""No valid profile found for ({model_name}) HIRES. Please go to the TensorRT tab and generate an engine with the necessary profile. 
                    If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."""
                )
            merged_idx = [i for i, id in enumerate(idx) if id in idx_hr]
            if len(merged_idx) == 0:
                gr.Warning(
                    "No model available for both ({}) LOWRES ({}x{}) and HIRES ({}x{}). This will slow-down inference.".format(
                        model_name, p.width, p.height, hr_w, hr_h
                    )
                )
                best_hr = idx_hr[np.argmin(distances_hr)]
            else:
                _distances = [distances[i] for i in merged_idx]
                best_hr = idx_hr[merged_idx[np.argmin(_distances)]]
                best = best_hr

        return best, best_hr

    def process(self, p, *args):
        # before unet_init
        sd_unet_option = sd_unet.get_unet_option()
        if sd_unet_option is None:
            return

        if not sd_unet_option.model_name == p.sd_model_name:
            gr.Error(
                """Selected torch model ({}) does not match the selected TensorRT U-Net ({}). 
                Please ensure that both models are the same or select Automatic from the SD UNet dropdown.""".format(
                    p.sd_model_name, sd_unet_option.model_name
                )
            )
        GLOBAL_ARGS.idx, GLOBAL_ARGS.hr_idx = self.get_profile_idx(p, p.sd_model_name)

        self.get_loras(p)

        controlnets = p.__dict__.get("controlnet", [])
        GLOBAL_ARGS.controlnets = {}
        for controlnet in controlnets:
            controlnet.idx, controlnet.hr_idx = self.get_profile_idx(p, controlnet.name)
            GLOBAL_ARGS.controlnets[controlnet.name] = controlnet

    def get_loras(self, p):
        lora_pathes = []
        lora_scales = []

        # get lora from prompt
        _prompt = p.prompt
        extra_networks = re.findall("\<(.*?)\>", _prompt)
        loras = [net for net in extra_networks if net.startswith("lora")]

        # Avoid that extra networks will be loaded
        for lora in loras:
            _prompt = _prompt.replace(f"<{lora}>", "")
        p.prompt = _prompt

        # check if lora config has changes
        if self.lora_hash != "".join(loras):
            self.lora_hash = "".join(loras)
            self.update_lora = True
            if self.lora_hash == "":
                GLOBAL_ARGS.lora = None
                return
        else:
            return

        # Get pathes
        print("Apllying LoRAs: " + str(loras))
        for lora in loras:
            lora_name, lora_scale = lora.split(":")[1:]
            lora_scales.append(float(lora_scale))
            if lora_name not in modelmanager.get_available_models(ModelType.LORA):
                gr.Error(
                    f"Please export the LoRA checkpoint {lora_name} first from the TensorRT LoRA tab"
                )
            lora_pathes.append(
                os.path.join(
                    TRT_MODEL_DIR,
                    modelmanager.get_available_models(ModelType.LORA)[lora_name][0][
                        "filepath"
                    ],
                )
            )

        # Merge lora refit dicts
        base_name, base_path = modelmanager.get_onnx_path(p.sd_model_name)
        refit_dict_encoder = apply_loras(base_path, lora_pathes, lora_scales, "encoder")
        refit_dict_decoder = apply_loras(base_path, lora_pathes, lora_scales, "decoder")

        GLOBAL_ARGS.lora = (refit_dict_encoder, refit_dict_decoder)

    def process_batch(self, p, *args, **kwargs):
        # Called for each batch count
        return super().process_batch(p, *args, **kwargs)

    def before_hr(self, p, *args):
        GLOBAL_ARGS.idx = GLOBAL_ARGS.hr_idx

        for cnet in GLOBAL_ARGS.controlnets.values():
            cnet.idx = cnet.hr_idx

        return super().before_hr(p, *args)  # 4 (Only when HR starts.....)

    def after_extra_networks_activate(self, p, *args, **kwargs):
        if self.update_lora:
            self.update_lora = False
            # Not the fastest, but safest option. Larger bottlenecks to solve first!
            # Other two options: Overengingeer, Refit whole model
            sd_unet.current_unet.apply_loras()


def list_unets(l):
    model = modelmanager.get_available_models(ModelType.UNET)
    for k, v in model.items():
        l.append(TrtUnetOption(k, v))


script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
