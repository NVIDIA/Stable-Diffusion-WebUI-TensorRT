import os
import numpy as np

import torch
from torch.cuda import nvtx
from modules import script_callbacks, sd_unet, devices, scripts

import ui_trt
from utilities import Engine
from typing import List
from model_manager import TRT_MODEL_DIR, modelmanager
from polygraphy.logger import G_LOGGER
import gradio as gr

G_LOGGER.module_severity = G_LOGGER.ERROR


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
        return TrtUnet(self.model_name, self.configs, lora_path)


# This is ugly. Is there a better way to parse this as kwargs to the SD Unet?
GLOBAL_KWARGS = {"profile_idx": None, "profile_hr_idx": None, "model_name": ""}


class TrtUnet(sd_unet.SdUnet):
    def __init__(
        self, model_name: str, configs: List[dict], lora_path, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not model_name == GLOBAL_KWARGS["model_name"]:
            raise ValueError(
                """Selected torch model ({}) does not match the selected TensorRT U-Net ({}). 
                Please ensure that both models are the same or select Automatic from the SD UNet dropdown.""".format(
                    GLOBAL_KWARGS["model_name"], model_name
                )
            )
        self.configs = configs
        self.stream = None
        self.model_name = model_name
        self.lora_path = lora_path
        self.engine_vram_req = 0

        self.profile_idx = GLOBAL_KWARGS["profile_idx"]
        self.loaded_config = self.configs[self.profile_idx]
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

        if not self.profile_idx == GLOBAL_KWARGS["profile_idx"]:
            self.switch_engine()

        tmp = torch.empty(
            self.engine_vram_req, dtype=torch.uint8, device=devices.device
        )
        self.engine.context.device_memory = tmp.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)["latent"]

        nvtx.range_pop()
        return out

    def switch_engine(self):
        self.profile_idx = GLOBAL_KWARGS["profile_idx"]
        self.loaded_config = self.configs[self.profile_idx]
        self.deactivate()
        self.engine = Engine(
            os.path.join(TRT_MODEL_DIR, self.loaded_config["filepath"])
        )
        self.activate()

    def activate(self):
        self.engine.load()
        print(f"\nLoaded Profile: {self.profile_idx}")
        print(self.engine)
        self.engine_vram_req = self.engine.engine.device_memory_size
        self.engine.activate(True)

        if self.lora_path is not None:
            self.engine.refit_from_dump(self.lora_path)

    def deactivate(self):
        self.shape_hash = 0
        del self.engine


class TensorRTScript(scripts.Script):
    def __init__(self) -> None:
        self.loaded_model = None
        pass

    def title(self):
        return "TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def setup(self, p, *args):
        return super().setup(p, *args)

    def before_process(self, p, *args):  # 1
        # Check divisibilty
        if p.width % 64 or p.height % 64:
            raise ValueError(
                "Target resolution must be divisible by 64 in both dimensions."
            )

        if p.enable_hr:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            if hr_w % 64 or hr_h % 64:
                raise ValueError(
                    "HIRES Fix resolution must be divisible by 64 in both dimensions. Please change the upscale factor or disable HIRES Fix."
                )

        # lora p.prompt == '<lora:BarbieCore:1>

    def process(self, p, *args):  # 2
        # before unet_init
        hr_scale = p.hr_scale if p.enable_hr else 1
        (
            valid_models,
            distances,
            idx,
        ) = modelmanager.get_valid_models(
            p.sd_model_name, p.width, p.height, p.batch_size, 77
        )  # TODO: max_embedding
        if len(valid_models) == 0:
            raise ValueError(
                """No valid profile found for LOWRES. Please go to the TensorRT tab and generate an engine with the necessary profile. 
                If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."""
            )
        best = idx[np.argmin(distances)]

        if hr_scale != 1:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            valid_models_hr, distances_hr, idx_hr = modelmanager.get_valid_models(
                p.sd_model_name, hr_w, hr_h, p.batch_size, 77
            )  # TODO: max_embedding
            if len(valid_models) == 0:
                raise ValueError(
                    "No valid profile found for HIRES. Please go to the TensorRT tab and generate an engine with the necessary profile. If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."
                )
            merged_idx = [i for i, id in enumerate(idx) if id in idx_hr]
            if len(merged_idx) == 0:
                gr.Warning(
                    "No model available for both LOWRES ({}x{}) and HIRES ({}x{}). This will slow-down inference.".format(
                        p.width, p.height, hr_w, hr_h
                    )
                )
                best_hr = idx_hr[np.argmin(distances_hr)]
            else:
                _distances = [distances[i] for i in merged_idx]
                best_hr = idx_hr[merged_idx[np.argmin(_distances)]]
                best = best_hr
            GLOBAL_KWARGS["profile_hr_idx"] = best_hr
        GLOBAL_KWARGS["profile_idx"] = best
        GLOBAL_KWARGS["model_name"] = p.sd_model_name

    def process_batch(self, p, *args, **kwargs):
        return super().process_batch(p, *args, **kwargs)

    def before_hr(self, p, *args):
        GLOBAL_KWARGS["profile_idx"] = GLOBAL_KWARGS["profile_hr_idx"]
        return super().before_hr(p, *args)  # 4 (Only when HR starts.....)

    def after_extra_networks_activate(self, p, *args, **kwargs):
        # if self.lora_path is not None:
        #    self.engine.refit_from_dump(self.lora_path)

        # Called after UNet activate
        # p.extra_network_data
        # Contains dict of modules.extra_networks.ExtraNetworkParams
        return super().after_extra_networks_activate(p, *args, **kwargs)  # 3


def list_unets(l):
    model = modelmanager.available_models()
    for k, v in model.items():
        label = "{} ({})".format(k, v[0]["base_model"]) if v[0]["config"].lora else k
        l.append(TrtUnetOption(label, v))


script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
