import json
from json import JSONEncoder

import os
from logging import info, warning
from dataclasses import dataclass
import torch
from exporter import get_cc
from modules import paths_internal

ONNX_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-onnx")
if not os.path.exists(ONNX_MODEL_DIR):
    os.makedirs(ONNX_MODEL_DIR)
TRT_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-trt")
if not os.path.exists(TRT_MODEL_DIR):
    os.makedirs(TRT_MODEL_DIR)
LORA_MODEL_DIR = os.path.join(paths_internal.models_path, "Lora")
NVIDIA_CACHE_URL = ""

MODEL_FILE = os.path.join(TRT_MODEL_DIR, "model.json")

cc_major, cc_minor = get_cc()


class ModelManager:
    def __init__(self, model_file=MODEL_FILE) -> None:
        self.all_models = {}
        self.model_file = model_file
        self.cc = "cc{}{}".format(cc_major, cc_minor)
        if not os.path.exists(model_file):
            warning("Model file does not exist. Creating new one.")
        else:
            self.all_models = self.read_json()

        self.update()

    @staticmethod
    def get_onnx_path(model_name, model_hash):
        onnx_filename = "_".join([model_name, model_hash]) + ".onnx"
        onnx_path = os.path.join(ONNX_MODEL_DIR, onnx_filename)
        return onnx_filename, onnx_path

    def get_trt_path(self, model_name, model_hash, profile, static_shape):
        profile_hash = []
        n_profiles = 1 if static_shape else 3
        for k, v in profile.items():
            dim_hash = []
            for i in range(n_profiles):
                dim_hash.append("x".join([str(x) for x in v[i]]))
            profile_hash.append(k + "=" + "+".join(dim_hash))

        profile_hash = "-".join(profile_hash)
        trt_filename = (
            "_".join([model_name, model_hash, self.cc, profile_hash]) + ".trt"
        )
        trt_path = os.path.join(TRT_MODEL_DIR, trt_filename)

        return trt_filename, trt_path

    def update(self):
        trt_engines = [
            trt_file
            for trt_file in os.listdir(TRT_MODEL_DIR)
            if trt_file.endswith(".trt")
        ]

        tmp_all_models = self.all_models.copy()
        for cc, base_models in tmp_all_models.items():
            for base_model, models in base_models.items():
                tmp_config_list = {}
                for model_config in models:
                    if model_config["filepath"] not in trt_engines:
                        info(
                            "Model config outdated. {} was not found".format(model_config["filepath"])
                        )
                        continue
                    tmp_config_list[model_config["filepath"]] = model_config
                
                tmp_config_list = list(tmp_config_list.values()) 
                if len(tmp_config_list) == 0:
                    self.all_models[cc].pop(base_model)
                else:
                    self.all_models[cc][base_model] = models

        self.write_json()


    def __del__(self):
        self.update()

    def add_entry(
        self,
        model_name,
        model_hash,
        profile,
        static_shapes,
        fp32,
        inpaint,
        refit,
        vram,
        unet_hidden_dim,
        lora,
    ):
        config = ModelConfig(
            profile, static_shapes, fp32, inpaint, refit, lora, vram, unet_hidden_dim
        )
        trt_name, trt_path = self.get_trt_path(
            model_name, model_hash, profile, static_shapes
        )

        base_model_name = f"{model_name}"  # _{model_hash}
        if self.cc not in self.all_models:
            self.all_models[self.cc] = {}

        if base_model_name not in self.all_models[self.cc]:
            self.all_models[self.cc][base_model_name] = []
        self.all_models[self.cc][base_model_name].append(
            {
                "filepath": trt_name,
                "config": config,
            }
        )

        self.write_json()

    def add_lora_entry(
        self, base_model, lora_name, trt_lora_path, fp32, inpaint, vram, unet_hidden_dim
    ):
        config = ModelConfig(
            [[], [], []], False, fp32, inpaint, True, True, vram, unet_hidden_dim
        )

        self.all_models[self.cc][lora_name] = [
            {
                "filepath": trt_lora_path,
                "base_model": base_model,
                "config": config,
            }
        ]

        self.write_json()

    def write_json(self):
        with open(self.model_file, "w") as f:
            json.dump(self.all_models, f, indent=4, cls=ModelConfigEncoder)

    def read_json(self, encode_config=True):
        with open(self.model_file, "r") as f:
            out = json.load(f)

        if not encode_config:
            return out

        for cc, models in out.items():
            for base_model, configs in models.items():
                for i in range(len(configs)):
                    out[cc][base_model][i]["config"] = ModelConfig(
                        **configs[i]["config"]
                    )
        return out

    def available_models(self):
        available = self.all_models.get(self.cc, {})
        return available

    def get_timing_cache(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache = os.path.join(
            current_dir,
            "timing_caches",
            "timing_cache_{}_{}.cache".format(
                "win" if os.name == "nt" else "linux", self.cc
            ),
        )

        return cache

    def get_valid_models(self, base_model: str, feed_dict: dict):
        valid_models = []
        distances = []
        models = self.available_models()
        for model in models[base_model]:
            valid, distance = model["config"].is_compatible(feed_dict)
            if valid:
                valid_models.append(model)
                distances.append(distance)

        return valid_models, distances


@dataclass
class ModelConfig:
    profile: dict
    static_shapes: bool
    fp32: bool
    inpaint: bool
    refit: bool
    lora: bool
    vram: int
    unet_hidden_dim: int = 4

    def is_compatible(self, feed_dict: dict):
        distance = 0
        for k, v in feed_dict.items():
            _min, _opt, _max = self.profile[k]
            v_tensor = torch.Tensor(list(v.shape))
            r_min = torch.Tensor(_max) - v_tensor
            r_opt = (torch.Tensor(_opt) - v_tensor).abs()
            r_max = v_tensor - torch.Tensor(_min)
            if torch.any(r_min < 0) or torch.any(r_max < 0):
                return (False, distance)
            distance += r_opt.sum() + 0.5 * (r_max.sum() + 0.5 * r_min.sum())
        return (True, distance)


class ModelConfigEncoder(JSONEncoder):
    def default(self, o: ModelConfig):
        return o.__dict__


modelmanager = ModelManager()
