import os
from typing import List
import cupy
from safetensors.numpy import load_file
import onnx_graphsurgeon as gs
import onnx


def merge_loras(loras: List[str], scales: List[str]):
    refit_dict = {}
    for lora, scale in zip(loras, scales):
        lora_dict = load_file(lora)
        for k, v in lora_dict.items():
            if k in refit_dict:
                refit_dict[k] += scale * cupy.array(v)
            else:
                refit_dict[k] = scale * cupy.array(v)
    return refit_dict


def apply_loras(base_path: str, loras: List[str], scales: List[str]) -> dict:
    refit_dict = merge_loras(loras, scales)
    base = gs.import_onnx(onnx.load(base_path)).toposort()

    def add_to_map(refit_dict, name, value):
        if name in refit_dict:
            refit_dict[name] += value

    for n in base.nodes:
        if n.op == "Constant":
            name = n.outputs[0].name
            add_to_map(refit_dict, name, n.outputs[0].values)

        # Handle scale and bias weights
        elif n.op == "Conv":
            if n.inputs[1].__class__ == gs.Constant:
                name = n.name + "_TRTKERNEL"
                add_to_map(refit_dict, name, n.inputs[1].values)

            if n.inputs[2].__class__ == gs.Constant:
                name = n.name + "_TRTBIAS"
                add_to_map(refit_dict, name, n.inputs[2].values)

        # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
        else:
            for inp in n.inputs:
                name = inp.name
                if inp.__class__ == gs.Constant:
                    add_to_map(refit_dict, name, inp.values)

    return refit_dict
