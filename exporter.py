import torch
import torch.nn.functional as F
import onnx
from logging import info, error
import time
import shutil
import numpy as np
from modules import shared
from collections import OrderedDict
from onnx import numpy_helper
from utilities import Engine
from models_helper import UNetModelSplit
import os
from pathlib import Path
from optimum.onnx.utils import (
    _get_onnx_external_data_tensors,
    check_model_uses_external_data,
)
from datastructures import ProfileSettings
import json


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


def get_refit_weights(
    state_dict, onnx_opt_path, weight_name_mapping, weight_shape_mapping
):
    refit_weights = OrderedDict()
    onnx_opt_dir = os.path.dirname(onnx_opt_path)
    onnx_opt_model = onnx.load(onnx_opt_path)
    # Create initializer data hashes
    initializer_hash_mapping = {}
    onnx_data_mapping = {}
    for initializer in onnx_opt_model.graph.initializer:
        initializer_data = numpy_helper.to_array(
            initializer, base_dir=onnx_opt_dir
        ).astype(np.float16)
        initializer_hash = hash(initializer_data.data.tobytes())
        initializer_hash_mapping[initializer.name] = initializer_hash
        onnx_data_mapping[initializer.name] = initializer_data

    for torch_name, initializer_name in weight_name_mapping.items():
        initializer_hash = initializer_hash_mapping[initializer_name]
        wt = state_dict[torch_name]

        # get shape transform info
        initializer_shape, is_transpose = weight_shape_mapping[torch_name]
        if is_transpose:
            wt = torch.transpose(wt, 0, 1)
        else:
            wt = torch.reshape(wt, initializer_shape)

        # include weight if hashes differ
        wt_hash = hash(wt.cpu().detach().numpy().astype(np.float16).data.tobytes())
        if initializer_hash != wt_hash:
            delta = wt - torch.tensor(onnx_data_mapping[initializer_name]).to(wt.device)
            refit_weights[initializer_name] = delta.contiguous()

    return refit_weights


def export_lora(
    modelobj: UNetModelSplit,
    onnx_path: str,
    weights_map_path: str,
    lora_name: str,
    profile: ProfileSettings,
):
    inputs = modelobj.get_encoder_sample_input(
        profile.bs_opt * 2, profile.h_opt // 8, profile.w_opt // 8, profile.t_min
    )
    with open(weights_map_path, "r") as fp_wts:
        print(f"[I] Loading weights map: {weights_map_path} ")
        [weights_name_mapping, weights_shape_mapping] = json.load(fp_wts)

    with torch.inference_mode(), torch.autocast("cuda"):
        modelobj.unet = apply_lora(
            modelobj.unet, os.path.splitext(lora_name)[0], inputs
        )

        encoder_refit = get_refit_weights(
            modelobj.unet.state_dict(),
            os.path.join(onnx_path, "encoder.onnx"),
            weights_name_mapping["encoder"],
            weights_shape_mapping["encoder"],
        )
        decoder_refit = get_refit_weights(
            modelobj.unet.state_dict(),
            os.path.join(onnx_path, "decoder.onnx"),
            weights_name_mapping["decoder"],
            weights_shape_mapping["decoder"],
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
    tmp_dir = os.path.abspath("onnx_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "model.onnx")
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

    if optimizer is not None:
        try:
            onnx_opt_graph = optimizer("unet", onnx_model)
            onnx.save(onnx_opt_graph, path)
        except Exception as e:
            error("Optimizing ONNX failed. {}".format(e))
            return

    if not model_uses_external_data and optimizer is None:
        shutil.move(tmp_path, str(path))

    shutil.rmtree(tmp_dir)


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
