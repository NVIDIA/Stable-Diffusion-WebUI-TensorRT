import os
from typing import List
import numpy as np
from safetensors.numpy import load_file
import onnx_graphsurgeon as gs
import onnx


def merge_loras(loras: List[str], scales: List[str], model_name: str):
    refit_dict = {}
    for lora, scale in zip(loras, scales):
        lora_dict = load_file(os.path.join(lora, model_name + ".refit"))
        for k, v in lora_dict.items():
            if k in refit_dict:
                refit_dict[k] += scale * v
            else:
                refit_dict[k] = scale * v
    return refit_dict


def apply_loras(
    base_path: str, loras: List[str], scales: List[str], model_name: str
) -> dict:
    refit_dict = merge_loras(loras, scales, model_name)
    base = gs.import_onnx(
        onnx.load(os.path.join(base_path, model_name + ".onnx"))
    ).toposort().nodes

    def convert_int64(arr):
        if len(arr.shape) == 0:
            return np.array([np.int32(arr)])
        return arr

    lora_keywords = ["to_q", "to_k", "to_v", "to_out"]
    for node in base:
        for kw in lora_keywords:
            if kw in node.name and "MatMul" in node.name:
                for inp in node.inputs:
                    if inp.__class__ == gs.Constant:
                        if inp.name not in refit_dict:
                            continue
                        if inp.values.dtype == np.int64:
                            inp.values = convert_int64(inp.values)
                        assert inp.values.shape == refit_dict[inp.name].shape
                        assert inp.values.dtype == refit_dict[inp.name].dtype
                        refit_dict[inp.name] += inp.values

    return refit_dict
