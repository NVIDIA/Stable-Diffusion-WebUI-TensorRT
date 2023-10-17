import os

from modules import sd_models, shared
import gradio as gr

from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import cmd_opts
from modules.ui_components import FormRow

from exporter import export_onnx, export_trt
from utilities import PIPELINE_TYPE, Engine
from models import make_OAIUNetXL, make_OAIUNet
import logging
import gc
import torch
from model_manager import modelmanager, cc_major, TRT_MODEL_DIR
from time import sleep
from collections import defaultdict
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

logging.basicConfig(level=logging.INFO)


def get_version_from_model(sd_model):
    if sd_model.is_sd1:
        return "1.5"
    if sd_model.is_sd2:
        return "2.1"
    if sd_model.is_sdxl:
        return "xl-1.0"


class LogLevel:
    Debug = 0
    Info = 1
    Warning = 2
    Error = 3


def log_md(logging_history, message, prefix="**[INFO]:**"):
    logging_history += f"{prefix} {message} \n"
    return logging_history


def export_unet_to_trt(
    batch_min,
    batch_opt,
    batch_max,
    height_min,
    height_opt,
    height_max,
    width_min,
    width_opt,
    width_max,
    token_count_min,
    token_count_opt,
    token_count_max,
    force_export,
    static_shapes,
    preset,
    controlnet=None,
):
    logging_history = ""

    if preset == "Default":
        (
            batch_min,
            batch_opt,
            batch_max,
            height_min,
            height_opt,
            height_max,
            width_min,
            width_opt,
            width_max,
            token_count_min,
            token_count_opt,
            token_count_max,
        ) = export_default_unet_to_trt()
    is_inpaint = False
    use_fp32 = False
    if cc_major < 7:
        use_fp32 = True
        logging_history = log_md(
            logging_history, "FP16 has been disabled because your GPU does not support it."
        )
        yield logging_history

    unet_hidden_dim = shared.sd_model.model.diffusion_model.in_channels
    if unet_hidden_dim == 9:
        is_inpaint = True

    model_hash = shared.sd_model.sd_checkpoint_info.hash
    model_name = shared.sd_model.sd_checkpoint_info.model_name
    onnx_filename, onnx_path = modelmanager.get_onnx_path(model_name, model_hash)

    logging_history = log_md(
        logging_history, f"Exporting {model_name} to TensorRT", prefix="###"
    )
    yield logging_history

    timing_cache = modelmanager.get_timing_cache()

    version = get_version_from_model(shared.sd_model)

    pipeline = PIPELINE_TYPE.TXT2IMG
    if is_inpaint:
        pipeline = PIPELINE_TYPE.INPAINT
    controlnet = None

    min_textlen = (token_count_min // 75) * 77
    opt_textlen = (token_count_opt // 75) * 77
    max_textlen = (token_count_max // 75) * 77
    if static_shapes:
        min_textlen = max_textlen = opt_textlen

    if shared.sd_model.is_sdxl:
        pipeline = PIPELINE_TYPE.SD_XL_BASE
        modelobj = make_OAIUNetXL(
            version, pipeline, "cuda", False, batch_max, opt_textlen, max_textlen
        )
        diable_optimizations = True
    else:
        modelobj = make_OAIUNet(
            version,
            pipeline,
            "cuda",
            False,
            batch_max,
            opt_textlen,
            max_textlen,
            controlnet,
        )
        diable_optimizations = False

    profile = modelobj.get_input_profile(
        batch_min,
        batch_opt,
        batch_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        static_shapes,
    )
    print(profile)

    if not os.path.exists(onnx_path):
        logging_history = log_md(logging_history, "No ONNX file found. Exporting ONNX…")
        yield logging_history
        export_onnx(
            onnx_path,
            modelobj,
            profile=profile,
            diable_optimizations=diable_optimizations,
        )
        logging_history = log_md(logging_history, "Exported to ONNX.")
        yield logging_history

    trt_engine_filename, trt_path = modelmanager.get_trt_path(
        model_name, model_hash, profile, static_shapes
    )

    if not os.path.exists(trt_path) or force_export:
        logging_history = log_md(
            logging_history,
            "Building TensorRT engine... This can take a while, please check the progress in the terminal.",
        )
        yield logging_history
        gc.collect()
        torch.cuda.empty_cache()
        ret = export_trt(
            trt_path,
            onnx_path,
            timing_cache,
            profile=profile,
            use_fp16=not use_fp32,
        )
        if ret:
            yield logging_history + "\n --- \n ## Export Failed due to unknown reason. See shell for more information. \n"
            return
        logging_history = log_md(
            logging_history, "TensorRT engines has been saved to disk."
        )
        yield logging_history
        modelmanager.add_entry(
            model_name,
            model_hash,
            profile,
            static_shapes,
            fp32=use_fp32,
            inpaint=is_inpaint,
            refit=True,
            vram=0,
            unet_hidden_dim=unet_hidden_dim,
            lora=False,
        )
    else:
        logging_history = log_md(
            logging_history,
            "TensorRT engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed.",
        )
        yield logging_history

    yield logging_history + "\n --- \n ## Exported Successfully \n"


def export_lora_to_trt(lora_name, force_export):
    logging_history = ""
    is_inpaint = False
    use_fp32 = False
    if cc_major < 7:
        use_fp32 = True
        logging_history = log_md(
            logging_history, "FP16 has been disabled because your GPU does not support it."
        )
        yield logging_history
    unet_hidden_dim = shared.sd_model.model.diffusion_model.in_channels
    if unet_hidden_dim == 9:
        is_inpaint = True

    model_hash = shared.sd_model.sd_checkpoint_info.hash
    model_name = shared.sd_model.sd_checkpoint_info.model_name
    base_name = f"{model_name}"  # _{model_hash}

    available_lora_models = get_lora_checkpoints()
    lora_name = lora_name.split(" ")[0]
    lora_model = available_lora_models[lora_name]

    onnx_base_filename, onnx_base_path = modelmanager.get_onnx_path(
        model_name, model_hash
    )
    onnx_lora_filename, onnx_lora_path = modelmanager.get_onnx_path(
        lora_name, base_name
    )

    version = get_version_from_model(shared.sd_model)

    pipeline = PIPELINE_TYPE.TXT2IMG
    if is_inpaint:
        pipeline = PIPELINE_TYPE.INPAINT

    if shared.sd_model.is_sdxl:
        pipeline = PIPELINE_TYPE.SD_XL_BASE
        modelobj = make_OAIUNetXL(version, pipeline, "cuda", False, 1, 77, 77)
        diable_optimizations = True
    else:
        modelobj = make_OAIUNet(
            version,
            pipeline,
            "cuda",
            False,
            1,
            77,
            77,
            None,
        )
        diable_optimizations = False

    if not os.path.exists(onnx_lora_path):
        logging_history = log_md(logging_history, "No ONNX file found. Exporting ONNX…")
        yield logging_history
        export_onnx(
            onnx_lora_path,
            modelobj,
            profile=modelobj.get_input_profile(
                1, 1, 1, 512, 512, 512, 512, 512, 512, True
            ),
            diable_optimizations=diable_optimizations,
            lora_path=lora_model["filename"],
        )
        logging_history = log_md(logging_history, "Exported to ONNX.")
        yield logging_history

    trt_lora_name = onnx_lora_filename.replace(".onnx", ".trt")
    trt_lora_path = os.path.join(TRT_MODEL_DIR, trt_lora_name)

    available_trt_unet = modelmanager.available_models()
    if len(available_trt_unet[base_name]) == 0:
        logging_history = log_md(logging_history, "Please export the base model first.")
        yield logging_history
    trt_base_path = os.path.join(
        TRT_MODEL_DIR, available_trt_unet[base_name][0]["filepath"]
    )

    if not os.path.exists(onnx_base_path):
        raise ValueError("Please export the base model first.")

    if not os.path.exists(trt_lora_path) or force_export:
        logging_history = log_md(
            logging_history, "No TensorRT engine found. Building..."
        )
        yield logging_history
        engine = Engine(trt_base_path)
        engine.load()
        engine.refit(onnx_base_path, onnx_lora_path, dump_refit_path=trt_lora_path)
        logging_history = log_md(logging_history, "Built TensorRT engine.")
        yield logging_history

        modelmanager.add_lora_entry(
            base_name,
            lora_name,
            trt_lora_name,
            use_fp32,
            is_inpaint,
            0,
            unet_hidden_dim,
        )
    yield logging_history + "\n --- \n ## Exported Successfully \n"


def export_default_unet_to_trt():
    is_xl = shared.sd_model.is_sdxl

    batch_min = 1
    batch_opt = 1
    batch_max = 4
    height_min = 768 if is_xl else 512
    height_opt = 1024 if is_xl else 512
    height_max = 1024 if is_xl else 768
    width_min = 768 if is_xl else 512
    width_opt = 1024 if is_xl else 512
    width_max = 1024 if is_xl else 768
    token_count_min = 75
    token_count_opt = 75
    token_count_max = 150

    return (
        batch_min,
        batch_opt,
        batch_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        token_count_min,
        token_count_opt,
        token_count_max,
    )


profile_presets = {
    "512x512 | Batch Size 1 (Static)": (
        1,
        1,
        1,
        512,
        512,
        512,
        512,
        512,
        512,
        75,
        75,
        75,
    ),
    "768x768 | Batch Size 1 (Static)": (
        1,
        1,
        1,
        768,
        768,
        768,
        768,
        768,
        768,
        75,
        75,
        75,
    ),
    "1024x1024 | Batch Size 1 (Static)": (
        1,
        1,
        1,
        1024,
        1024,
        1024,
        1024,
        1024,
        1024,
        75,
        75,
        75,
    ),
    "256x256 - 512x512 | Batch Size 1-4 (Dynamic)": (
        1,
        1,
        4,
        256,
        512,
        512,
        256,
        512,
        512,
        75,
        75,
        150,
    ),
    "512x512 - 768x768 | Batch Size 1-4 (Dynamic)": (
        1,
        1,
        4,
        512,
        512,
        768,
        512,
        512,
        768,
        75,
        75,
        150,
    ),
    "768x768 - 1024x1024 | Batch Size 1-4 (Dynamic)": (
        1,
        1,
        4,
        768,
        1024,
        1024,
        768,
        1024,
        1024,
        75,
        75,
        150,
    ),
}


def get_settings_from_version(version):
    static = False
    if version == "Default":
        return *list(profile_presets.values())[-2], static
    if "Static" in version:
        static = True
    return *profile_presets[version], static


def diable_export(version):
    if version == "Default":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)

def disable_lora_export(lora):
    if lora is None:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def diable_visibility(hide):
    num_outputs = 8
    out = [gr.update(visible=not hide) for _ in range(num_outputs)]
    return out


def engine_profile_card():
    def get_md_table(
        h_min,
        h_opt,
        h_max,
        w_min,
        w_opt,
        w_max,
        b_min,
        b_opt,
        b_max,
        t_min,
        t_opt,
        t_max,
    ):
        md_table = (
            "|             	|   Min   	|   Opt   	|   Max   	| \n"
            "|-------------	|:-------:	|:-------:	|:-------:	| \n"
            "| Height      	| {h_min} 	| {h_opt} 	| {h_max} 	| \n"
            "| Width       	| {w_min} 	| {w_opt} 	| {w_max} 	| \n"
            "| Batch Size  	| {b_min} 	| {b_opt} 	| {b_max} 	| \n"
            "| Text-length 	| {t_min} 	| {t_opt} 	| {t_max} 	| \n"
        )
        return md_table.format(
            h_min=h_min,
            h_opt=h_opt,
            h_max=h_max,
            w_min=w_min,
            w_opt=w_opt,
            w_max=w_max,
            b_min=b_min,
            b_opt=b_opt,
            b_max=b_max,
            t_min=t_min,
            t_opt=t_opt,
            t_max=t_max,
        )

    available_models = modelmanager.available_models()

    model_md = defaultdict(list)
    loras_md = {}
    for base_model, models in available_models.items():
        for i, m in enumerate(models):
            if m["config"].lora:
                loras_md[base_model] = m.get("base_model", None)
                continue

            s_min, s_opt, s_max = m["config"].profile.get(
                "sample", [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            )
            t_min, t_opt, t_max = m["config"].profile.get(
                "encoder_hidden_states", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            )
            profile_table = get_md_table(
                s_min[2] * 8,
                s_opt[2] * 8,
                s_max[2] * 8,
                s_min[3] * 8,
                s_opt[3] * 8,
                s_max[3] * 8,
                max(s_min[0] // 2, 1),
                max(s_opt[0] // 2, 1),
                max(s_max[0] // 2, 1),
                (t_min[1] // 77) * 75,
                (t_opt[1] // 77) * 75,
                (t_max[1] // 77) * 75,
            )

            model_md[base_model].append(profile_table)

    for lora, base_model in loras_md.items():
        model_md[f"{lora} (*{base_model}*)"] = model_md[base_model]

    return model_md


def get_version_from_filename(name):
    if "v1-" in name:
        return "1.5"
    elif "v2-" in name:
        return "2.1"
    elif "xl" in name:
        return "xl-1.0"
    else:
        return "Unknown"


def get_lora_checkpoints():
    available_lora_models = {}
    canditates = list(
        shared.walk_files(
            shared.cmd_opts.lora_dir,
            allowed_extensions=[".pt", ".ckpt", ".safetensors"],
        )
    )
    for filename in canditates:
        name = os.path.splitext(os.path.basename(filename))[0]
        try:
            metadata = sd_models.read_metadata_from_safetensors(filename)
            version = get_version_from_filename(metadata.get("ss_sd_model_name"))
        except (AssertionError, TypeError):
            version = "Unknown"
        available_lora_models[name] = {
            "filename": filename,
            "version": version,
        }
    return available_lora_models


def get_valid_lora_checkpoints():
    available_lora_models = get_lora_checkpoints()
    return [
        f"{k} ({v['version']})"
        for k, v in available_lora_models.items()
        if v["version"] == get_version_from_model(shared.sd_model)
        or v["version"] == "Unknown"
    ]


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="trt_tabs"):
                    with gr.Tab(label="TensorRT Exporter"):
                        gr.Markdown(
                            value="# TensorRT Exporter",
                        )

                        default_version = list(profile_presets.keys())[-2]
                        default_vals = list(profile_presets.values())[-2]
                        version = gr.Dropdown(
                            label="Preset",
                            choices=list(profile_presets.keys()) + ["Default"],
                            elem_id="sd_version",
                            default="Default",
                            value="Default",
                        )

                        with gr.Accordion("Advanced Settings", open=False, visible=False) as advanced_settings:
                            with FormRow(
                                elem_classes="checkboxes-row", variant="compact"
                            ):
                                static_shapes = gr.Checkbox(
                                    label="Use static shapes.",
                                    value=False,
                                    elem_id="trt_static_shapes",
                                )

                            with gr.Column(elem_id="trt_max_batch"):
                                trt_min_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Min batch-size",
                                    value=default_vals[0],
                                    elem_id="trt_min_batch",
                                )

                                trt_opt_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Optimal batch-size",
                                    value=default_vals[1],
                                    elem_id="trt_opt_batch",
                                )
                                trt_max_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Max batch-size",
                                    value=default_vals[2],
                                    elem_id="trt_max_batch",
                                )

                            with gr.Column(elem_id="trt_height"):
                                trt_height_min = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Min height",
                                    value=default_vals[3],
                                    elem_id="trt_min_height",
                                )
                                trt_height_opt = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Optimal height",
                                    value=default_vals[4],
                                    elem_id="trt_opt_height",
                                )
                                trt_height_max = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Max height",
                                    value=default_vals[5],
                                    elem_id="trt_max_height",
                                )

                            with gr.Column(elem_id="trt_width"):
                                trt_width_min = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Min width",
                                    value=default_vals[6],
                                    elem_id="trt_min_width",
                                )
                                trt_width_opt = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Optimal width",
                                    value=default_vals[7],
                                    elem_id="trt_opt_width",
                                )
                                trt_width_max = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Max width",
                                    value=default_vals[8],
                                    elem_id="trt_max_width",
                                )

                            with gr.Column(elem_id="trt_token_count"):
                                trt_token_count_min = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Min prompt token count",
                                    value=default_vals[9],
                                    elem_id="trt_opt_token_count_min",
                                )
                                trt_token_count_opt = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Optimal prompt token count",
                                    value=default_vals[10],
                                    elem_id="trt_opt_token_count_opt",
                                )
                                trt_token_count_max = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Max prompt token count",
                                    value=default_vals[11],
                                    elem_id="trt_opt_token_count_max",
                                )

                            with FormRow(
                                elem_classes="checkboxes-row", variant="compact"
                            ):
                                force_rebuild = gr.Checkbox(
                                    label="Force Rebuild.",
                                    value=False,
                                    elem_id="trt_force_rebuild",
                                )

                        button_export_unet = gr.Button(
                            value="Export Engine",
                            variant="primary",
                            elem_id="trt_export_unet",
                            visible=False,
                        )

                        button_export_default_unet = gr.Button(
                            value="Export Default Engine",
                            variant="primary",
                            elem_id="trt_export_default_unet",
                            visible=True,
                        )

                        version.change(
                            get_settings_from_version,
                            version,
                            [
                                trt_min_batch,
                                trt_opt_batch,
                                trt_max_batch,
                                trt_height_min,
                                trt_height_opt,
                                trt_height_max,
                                trt_width_min,
                                trt_width_opt,
                                trt_width_max,
                                trt_token_count_min,
                                trt_token_count_opt,
                                trt_token_count_max,
                                static_shapes,
                            ],
                        )
                        version.change(
                            diable_export,
                            version,
                            [button_export_unet, button_export_default_unet, advanced_settings],
                        )

                        static_shapes.change(
                            diable_visibility,
                            static_shapes,
                            [
                                trt_min_batch,
                                trt_max_batch,
                                trt_height_min,
                                trt_height_max,
                                trt_width_min,
                                trt_width_max,
                                trt_token_count_min,
                                trt_token_count_max,
                            ],
                        )

                    with gr.Tab(label="TensorRT LoRA"):
                        gr.Markdown("# Apply LoRA checkpoint to TensorRT model")
                        lora_refresh_button = gr.Button(
                            value="Refresh",
                            variant="primary",
                            elem_id="trt_lora_refresh",
                        )

                        trt_lora_dropdown = gr.Dropdown(
                            choices=get_valid_lora_checkpoints(),
                            elem_id="lora_model",
                            label="LoRA Model",
                            default=None,
                        )

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            trt_lora_force_rebuild = gr.Checkbox(
                                label="Force Rebuild.",
                                value=False,
                                elem_id="trt_lora_force_rebuild",
                            )

                        button_export_lora_unet = gr.Button(
                            value="Convert to TensorRT",
                            variant="primary",
                            elem_id="trt_lora_export_unet",
                            visible=False,
                        )

                        lora_refresh_button.click(
                            get_valid_lora_checkpoints,
                            None,
                            trt_lora_dropdown,
                        )
                        trt_lora_dropdown.change(
                            disable_lora_export, trt_lora_dropdown, button_export_lora_unet
                        )

            with gr.Column(variant="panel"):
                with open(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "info.md"),
                    "r",
                    encoding='utf-8',
                ) as f:
                    trt_info = gr.Markdown(elem_id="trt_info", value=f.read())

        with gr.Row(equal_height=False):
            with gr.Accordion("Output", open=True):
                trt_result = gr.Markdown(elem_id="trt_result", value="")

        with gr.Column(variant="panel"):
            with gr.Row(equal_height=True, variant="compact"):
                button_refresh_profiles = ToolButton(value=refresh_symbol, elem_id="trt_refresh_profiles", visible=True)
                profile_header_md = gr.Markdown(
                    value=f"## Available TensorRT Engine Profiles"
                )
            engines_md = engine_profile_card()
            for model, profiles in engines_md.items():
                with gr.Row(equal_height=False):
                    row_name = model + " ({} Profiles)".format(len(profiles))
                    with gr.Accordion(row_name, open=False):
                        out_string = ""
                        for i, profile in enumerate(profiles):
                            out_string += f"#### Profile {i} \n"
                            out_string += profile
                            out_string += "\n\n"
                        gr.Markdown(elem_id=f"trt_{model}_{i}", value=out_string)

        button_export_unet.click(
            export_unet_to_trt,
            inputs=[
                trt_min_batch,
                trt_opt_batch,
                trt_max_batch,
                trt_height_min,
                trt_height_opt,
                trt_height_max,
                trt_width_min,
                trt_width_opt,
                trt_width_max,
                trt_token_count_min,
                trt_token_count_opt,
                trt_token_count_max,
                force_rebuild,
                static_shapes,
                version,
            ],
            outputs=[trt_result],
        )

        button_export_default_unet.click(
            export_unet_to_trt,
            inputs=[
                trt_min_batch,
                trt_opt_batch,
                trt_max_batch,
                trt_height_min,
                trt_height_opt,
                trt_height_max,
                trt_width_min,
                trt_width_opt,
                trt_width_max,
                trt_token_count_min,
                trt_token_count_opt,
                trt_token_count_max,
                force_rebuild,
                static_shapes,
                version,
            ],
            outputs=[trt_result],
        )

        button_export_lora_unet.click(
            export_lora_to_trt,
            inputs=[trt_lora_dropdown, trt_lora_force_rebuild],
            outputs=[trt_result],
        )

        
        # TODO Dynamically update available profiles. Not possible with gradio?!
        button_refresh_profiles.click(
                fn=shared.state.request_restart,
                _js='restart_reload',
                inputs=[],
                outputs=[],
        )

    return [(trt_interface, "TensorRT", "tensorrt")]
