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

G_LOGGER.module_severity = G_LOGGER.ERROR


class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str, filename: List[dict]):
        self.label = f"[TRT] {name}"
        self.model_name = name
        self.configs = filename

    def create_unet(self):
        return TrtUnet(self.model_name, self.configs)


# This is ugly. Is there a better way to parse this as kwargs to the SD Unet?
GLOBAL_KWARGS = {
    "profile_idx": None,
    "profile_hr_idx": None,
    "refit_dict": None,
}


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, model_name: str, configs: List[dict], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configs = configs
        self.stream = None
        self.model_name = model_name
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

        if GLOBAL_KWARGS["refit_dict"] is not None:
            nvtx.range_push("refit")
            self.engine.refit_from_dict(GLOBAL_KWARGS["refit_dict"])
            nvtx.range_pop()

    def deactivate(self):
        self.shape_hash = 0
        del self.engine


class TensorRTScript(scripts.Script):
    def __init__(self) -> None:
        self.loaded_model = None
        self.lora_hash = ""
        self.update_lora = False
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
            gr.Error("Target resolution must be divisible by 64 in both dimensions.")

        if p.enable_hr:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            if hr_w % 64 or hr_h % 64:
                gr.Error(
                    "HIRES Fix resolution must be divisible by 64 in both dimensions. Please change the upscale factor or disable HIRES Fix."
                )

    def process(self, p, *args):  # 2
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

        hr_scale = p.hr_scale if p.enable_hr else 1
        (
            valid_models,
            distances,
            idx,
        ) = modelmanager.get_valid_models(
            p.sd_model_name, p.width, p.height, p.batch_size, 77
        )  # TODO: max_embedding, just irnore?
        if len(valid_models) == 0:
            gr.Error(
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
                gr.Error(
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

        self.get_loras(p)

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
                GLOBAL_KWARGS["refit_dict"] = None
                return
        else:
            return

        # Get pathes
        print("Apllying LoRAs: " + str(loras))
        for lora in loras:
            lora_name, lora_scale = lora.split(":")[1:]
            lora_scales.append(float(lora_scale))
            if lora_name not in modelmanager.available_models():
                gr.Error(
                    f"Please export the LoRA checkpoint {lora_name} first from the TensorRT LoRA tab"
                )
            lora_pathes.append(
                os.path.join(
                    TRT_MODEL_DIR,
                    modelmanager.available_models()[lora_name][0]["filepath"],
                )
            )

        # Merge lora refit dicts
        model_hash = sd_models.checkpoint_aliases.get(
            modelmanager.available_models()[lora_name][0]["base_model"]
        ).hash
        base_name, base_path = modelmanager.get_onnx_path(p.sd_model_name, model_hash)
        refit_dict = apply_loras(base_path, lora_pathes, lora_scales)

        GLOBAL_KWARGS["refit_dict"] = refit_dict

    def process_batch(self, p, *args, **kwargs):
        # Called for each batch count
        return super().process_batch(p, *args, **kwargs)

    def before_hr(self, p, *args):
        GLOBAL_KWARGS["profile_idx"] = GLOBAL_KWARGS["profile_hr_idx"]
        return super().before_hr(p, *args)  # 4 (Only when HR starts.....)

    def after_extra_networks_activate(self, p, *args, **kwargs):
        if self.update_lora:
            self.update_lora = False
            # Not the fastest, but safest option. Larger bottlenecks to solve first!
            # Other two options: Overengingeer, Refit whole model
            sd_unet.current_unet.switch_engine()


def list_unets(l):
    model = modelmanager.available_models()
    for k, v in model.items():
        if v[0]["config"].lora:
            continue
        l.append(TrtUnetOption(k, v))


script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
