import json
from json import JSONEncoder

import os
from logging import info, warning
from dataclasses import dataclass
import torch
from modules import paths_internal
from copy import copy
from datastructures import ModelType

ONNX_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-onnx")
if not os.path.exists(ONNX_MODEL_DIR):
    os.makedirs(ONNX_MODEL_DIR)

TRT_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-trt")
if not os.path.exists(TRT_MODEL_DIR):
    os.makedirs(TRT_MODEL_DIR)

CNET_MODEL_PATH = os.path.join(paths_internal.models_path, "ControlNet")
if not os.path.exists(CNET_MODEL_PATH):
    os.makedirs(CNET_MODEL_PATH)


def get_cc():
    cc_major = torch.cuda.get_device_properties(0).major
    cc_minor = torch.cuda.get_device_properties(0).minor
    return cc_major, cc_minor


cc_major, cc_minor = get_cc()


class ModelManager:
    def __init__(self) -> None:
        self.all_models = {}
        self.model_files = []
        self.cc = "cc{}{}".format(cc_major, cc_minor)

        for cc in os.listdir(TRT_MODEL_DIR):
            cc_dir = os.path.join(TRT_MODEL_DIR, cc)
            if not os.path.isdir(cc_dir):
                continue
            for model_type in os.listdir(cc_dir):
                arch_dir = os.path.join(cc_dir, model_type)
                for model_name in os.listdir(arch_dir):
                    model_dir = os.path.join(arch_dir, model_name)
                    if not os.path.isdir(model_dir):
                        continue
                    for profile_hash in os.listdir(model_dir):
                        profile_dir = os.path.join(model_dir, profile_hash)
                        if not os.path.isdir(profile_dir):
                            continue
                        model_file = os.path.join(profile_dir, "model.json")
                        if not os.path.exists(model_file):
                            warning(
                                "Model file does not exist at {}... This might cause issues.".format(
                                    profile_dir
                                )
                            )
                            continue
                        self.model_files.append(model_file)
                        config = self.read_json(model_file)
                        self.all_models.setdefault(cc, {}).setdefault(
                            model_type, {}
                        ).setdefault(model_name, []).append(
                            {"config": config, "filepath": profile_dir}
                        )

    @staticmethod
    def get_onnx_path(model_name: str):
        onnx_filename = model_name + ".onnx"
        onnx_path = os.path.join(ONNX_MODEL_DIR, onnx_filename)
        return onnx_filename, onnx_path

    def get_trt_path(
        self, model_name: str, profile: dict, static_shape: bool, model_type: ModelType
    ):
        hash_dims = ["sample", "encoder_hidden_states"]
        profile_hash = []
        n_profiles = 1 if static_shape else 3
        dim_hash = ""
        for k, v in profile.items():
            if k not in hash_dims:
                continue
            for i in range(n_profiles):
                dim_hash += "".join([str(x) for x in v[i]])

        if model_type == ModelType.LORA:
            dim_hash = "RefitDict"
        else:
            dim_hash = str(hex(int(dim_hash))[2:])

        profile_hash = "id" + dim_hash
        trt_path = os.path.join(
            TRT_MODEL_DIR, self.cc, str(model_type), model_name, profile_hash
        )

        return profile_hash, trt_path

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

    def add_entry(
        self,
        model_name: str,
        profile: dict,
        static_shapes: bool,
        fp32: bool,
        refit: bool,
        vram: int,
        model_type: ModelType,
    ):
        config = ModelConfig(profile, static_shapes, fp32, refit, model_type, vram)
        trt_name, trt_path = self.get_trt_path(
            model_name, profile, static_shapes, model_type
        )

        self.write_json(os.path.join(trt_path, "model.json"), config)
        model_type = str(model_type).lower()
        if self.cc not in self.all_models:
            self.all_models[self.cc] = {}

        if model_type not in self.all_models[self.cc]:
            self.all_models[self.cc][model_type] = {}

        if model_name not in self.all_models[self.cc][model_type]:
            self.all_models[self.cc][model_type][model_name] = []
        self.all_models[self.cc][model_type][model_name].append(
            {
                "filepath": trt_path,
                "config": config,
            }
        )

    def get_weights_map_path(self, model_name: str, model_type: ModelType):
        return os.path.join(
            TRT_MODEL_DIR, self.cc, str(model_type), model_name, "weights_map.json"
        )

    def get_available_models(self, model_type: ModelType):
        available = self.all_models.get(self.cc, {})
        if model_type == ModelType.UNDEFINED:
            out = {}
            for mtype in ModelType:
                out.update(available.get(str(mtype), {}))
            return out
        else:
            return available.get(str(model_type), {})

    def get_valid_models_from_dict(
        self, base_model: str, feed_dict: dict, model_type: ModelType
    ):
        valid_models = []
        distances = []
        idx = []
        models = self.get_available_models(model_type)
        for i, model in enumerate(models[base_model]):
            valid, distance = model["config"].is_compatible_from_dict(feed_dict)
            if valid:
                valid_models.append(model)
                distances.append(distance)
                idx.append(i)

        return valid_models, distances, idx

    def get_valid_models(
        self,
        base_model: str,
        width: int,
        height: int,
        batch_size: int,
        max_embedding: int,
        model_type: ModelType,
    ):
        valid_models = []
        distances = []
        idx = []
        models = self.get_available_models(model_type)
        for i, model in enumerate(models[base_model]):
            valid, distance = model["config"].is_compatible(
                width, height, batch_size, max_embedding
            )
            if valid:
                valid_models.append(model)
                distances.append(distance)
                idx.append(i)

        return valid_models, distances, idx

    def write_json(self, model_file, config):
        with open(model_file, "w") as f:
            json.dump(config, f, indent=4, cls=ModelConfigEncoder)

    def read_json(self, model_file, encode_config=True):
        with open(model_file, "r") as f:
            out = json.load(f)

        if not encode_config:
            return out

        return ModelConfig(**out)


@dataclass
class ModelConfig:
    profile: dict
    static_shapes: bool
    fp32: bool
    refit: bool
    model_type: ModelType
    vram: int

    def __dict__(self):
        return {
            "profile": self.profile,
            "static_shapes": self.static_shapes,
            "fp32": self.fp32,
            "refit": self.refit,
            "model_type": str(self.model_type),
            "vram": self.vram,
        }

    def is_compatible_from_dict(self, feed_dict: dict):
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

    def is_compatible(
        self, width: int, height: int, batch_size: int, max_embedding: int
    ):
        distance = 0
        sample = self.profile["sample"]
        embedding = self.profile["encoder_hidden_states"]

        batch_size *= 2
        width = width // 8
        height = height // 8

        _min, _opt, _max = sample
        if _min[0] > batch_size or _max[0] < batch_size:
            return (False, distance)
        if _min[2] > height or _max[2] < height:
            return (False, distance)
        if _min[3] > width or _max[3] < width:
            return (False, distance)

        _min_em, _opt_em, _max_em = embedding
        if _min_em[1] > max_embedding or _max_em[1] < max_embedding:
            return (False, distance)

        distance = (
            abs(_opt[0] - batch_size)
            + abs(_opt[2] - height)
            + abs(_opt[3] - width)
            + 0.5 * (abs(_max[2] - height) + abs(_max[3] - width))
        )

        return (True, distance)


class ModelConfigEncoder(JSONEncoder):
    def default(self, o: ModelConfig):
        return o.__dict__()


modelmanager = ModelManager()
