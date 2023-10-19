import os
import numpy as np

import ldm.modules.diffusionmodules.openaimodel

import torch
from torch.cuda import nvtx
from modules import script_callbacks, sd_unet, devices

import ui_trt
from utilities import Engine
from typing import List
from model_manager import TRT_MODEL_DIR, modelmanager
from modules import sd_models, shared


class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str, filename: List[dict]):
        self.label = f"[TRT] {name}"
        self.model_name = name
        self.configs = filename

    def create_unet(self):
        lora_path = None
        if self.configs[0]["config"].lora:
            lora_path = os.path.join(TRT_MODEL_DIR, self.configs[0]["filepath"])
            self.model_name = self.configs[0]["base_model"]
            self.configs = modelmanager.available_models()[self.model_name]
        validate_sd_version(self.model_name, exact=True)
        return TrtUnet(self.model_name, self.configs, lora_path)


def validate_sd_version(model_name, exact=False):
    loaded_model = shared.sd_model.sd_checkpoint_info.model_name
    if exact:
        if not loaded_model == model_name:
            raise ValueError(
                f"Selected torch model ({loaded_model}) does not match the selected TensorRT U-Net ({model_name}). Please ensure that both models are the same."
            )
    else:
        if shared.sd_model.is_sdxl:
            if not "xl" in model_name:
                raise ValueError(
                    f"Selected torch model ({loaded_model}) does not match the selected TensorRT U-Net ({model_name}). Please ensure that both models are the same."
                )
        loaded_version = 1 if shared.sd_model.is_sd1 else 2
        if f"v{loaded_version}" not in model_name:
            raise ValueError(
                f"Selected torch model ({loaded_model}) does not match the selected TensorRT U-Net ({model_name}). Please ensure that both models are the same."
            )


class TrtUnet(sd_unet.SdUnet):
    def __init__(
        self, model_name: str, configs: List[dict], lora_path, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.configs = configs
        self.stream = None
        self.model_name = model_name
        self.lora_path = lora_path
        self.engine_vram_req = 0

        self.loaded_config = self.configs[0]
        self.shape_hash = 0
        self.engine = Engine(
            os.path.join(TRT_MODEL_DIR, self.loaded_config["filepath"])
        )

    def forward(self, x, timesteps, context, *args, **kwargs):
        nvtx.range_push("forward")
        feed_dict = {
            "sample": x.float(),
            "timesteps": timesteps.float(),
            "encoder_hidden_states": context.float(),
        }
        if "y" in kwargs:
            feed_dict["y"] = kwargs["y"].float()

        # Need to check compatibility on the fly
        if self.shape_hash != hash(x.shape):
            nvtx.range_push("switch_engine")
            if x.shape[-1] % 8 or x.shape[-2] % 8:
                raise ValueError(
                    "Input shape must be divisible by 64 in both dimensions."
                )
            self.switch_engine(feed_dict)
            self.shape_hash = hash(x.shape)
            nvtx.range_pop()

        tmp = torch.empty(
            self.engine_vram_req, dtype=torch.uint8, device=devices.device
        )
        self.engine.context.device_memory = tmp.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)["latent"]

        nvtx.range_pop()
        return out

    def switch_engine(self, feed_dict):
        valid_models, distances = modelmanager.get_valid_models(
            self.model_name, feed_dict
        )
        if len(valid_models) == 0:
            raise ValueError(
                "No valid profile found. Please go to the TensorRT tab and generate an engine with the necessary profile. If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."
            )

        best = valid_models[np.argmin(distances)]
        if best["filepath"] == self.loaded_config["filepath"]:
            return
        self.deactivate()
        self.engine = Engine(os.path.join(TRT_MODEL_DIR, best["filepath"]))
        self.activate()
        self.loaded_config = best

    def activate(self):
        self.engine.load()
        print(self.engine)
        self.engine_vram_req = self.engine.engine.device_memory_size
        self.engine.activate(True)

        if self.lora_path is not None:
            self.engine.refit_from_dump(self.lora_path)

    def deactivate(self):
        self.shape_hash = 0
        del self.engine


def list_unets(l):
    model = modelmanager.available_models()
    for k, v in model.items():
        label = "{} ({})".format(k, v[0]["base_model"]) if v[0]["config"].lora else k
        l.append(TrtUnetOption(label, v))

script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
