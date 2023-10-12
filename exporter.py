import torch
import torch.nn.functional as F
import onnx
from logging import info, error
import time
import shutil

from modules import sd_hijack, sd_unet, shared

from utilities import Engine
import os


def get_cc():
    cc_major = torch.cuda.get_device_properties(0).major
    cc_minor = torch.cuda.get_device_properties(0).minor
    return cc_major, cc_minor


def apply_lora(model, lora_path, inputs):
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
    lora_name = os.path.splitext(os.path.basename(lora_path))[0]
    networks.load_networks(
        [lora_name], [1.0], [1.0], [None]
    )  # todo: UI for parameters, multiple loras -> Struct of Arrays?

    model.forward(*inputs)
    return model


def export_onnx(
    onnx_path,
    modelobj=None,
    profile=None,
    opset=17,
    diable_optimizations=False,
    lora_path=None,
):
    swap_sdpa = hasattr(F, "scaled_dot_product_attention")
    old_sdpa = getattr(F, "scaled_dot_product_attention", None) if swap_sdpa else None
    if swap_sdpa:
        delattr(F, "scaled_dot_product_attention")

    def disable_checkpoint(self):
        if getattr(self, "use_checkpoint", False) == True:
            self.use_checkpoint = False
        if getattr(self, "checkpoint", False) == True:
            self.checkpoint = False

    shared.sd_model.model.diffusion_model.apply(disable_checkpoint)
    is_xl = shared.sd_model.is_sdxl

    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.apply_optimizations("None")

    os.makedirs("onnx_tmp", exist_ok=True)
    tmp_path = os.path.abspath(os.path.join("onnx_tmp", "tmp.onnx"))

    try:
        info("Exporting to ONNX...")
        with torch.inference_mode(), torch.autocast("cuda"):
            inputs = modelobj.get_sample_input(
                profile["sample"][1][0] // 2,
                profile["sample"][1][-2] * 8,
                profile["sample"][1][-1] * 8,
            )
            model = shared.sd_model.model.diffusion_model

            if lora_path:
                model = apply_lora(model, lora_path, inputs)

            torch.onnx.export(
                model,
                inputs,
                tmp_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=modelobj.get_input_names(),
                output_names=modelobj.get_output_names(),
                dynamic_axes=modelobj.get_dynamic_axes(),
            )

        info("Optimize ONNX.")

        onnx_graph = onnx.load(tmp_path)
        if diable_optimizations:
            onnx_opt_graph = onnx_graph
        else:
            onnx_opt_graph = modelobj.optimize(onnx_graph)

        if onnx_opt_graph.ByteSize() > 2147483648 or is_xl:
            onnx.save_model(
                onnx_opt_graph,
                onnx_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )
        else:
            try:
                onnx.save(onnx_opt_graph, onnx_path)
            except Exception as e:
                error(e)
                error("ONNX file too large. Saving as external data.")
                onnx.save_model(
                    onnx_opt_graph,
                    onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False,
                )
        info("ONNX export complete.")
        del onnx_opt_graph
    except Exception as e:
        error(e)
        exit()

    # CleanUp
    if swap_sdpa and old_sdpa:
        setattr(F, "scaled_dot_product_attention", old_sdpa)
    shutil.rmtree(os.path.abspath("onnx_tmp"))
    del model


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

    shared.sd_model = model.cuda()
    return ret
