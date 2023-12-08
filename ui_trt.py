import os

import gradio as gr
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import cmd_opts, sd_model
from modules.ui_components import FormRow
from modules import sd_hijack, sd_unet, sd_models
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

from exporter import export_onnx, export_trt, export_lora
from utilities import Engine
from safetensors.numpy import save_file

from models_helper import UNetModelSplit, CNetModel
import logging
import gc
import torch
from model_manager import modelmanager, cc_major
from collections import defaultdict
import json

from datastructures import ProfilePrests, ProfileSettings, SDVersion, ModelType

profile_presets = ProfilePrests()
available_cnets = CNetModel.list_cnet_models()

logging.basicConfig(level=logging.INFO)


# TODO get info from model config
def get_context_dim():
    if sd_model.is_sd1:
        return 768
    elif sd_model.is_sd2:
        return 1024
    elif sd_model.is_sdxl:
        return 2048


def is_fp32():
    use_fp32 = False
    if cc_major < 7:
        use_fp32 = True
        print("FP16 has been disabled because your GPU does not support it.")
    return use_fp32


def export_cnet_to_trt(
    cnet_name,
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
):
    model_id = "lllyasviel/sd-controlnet-" + cnet_name
    model_name = model_id.replace("/", "_")

    profile_settings = ProfileSettings(
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
    if preset == "Default":
        profile_settings = profile_presets.get_default(is_xl=False)
    use_fp32 = is_fp32()

    print(f"Exporting {model_name} to TensorRT using - {profile_settings}")
    profile_settings.token_to_dim(static_shapes)

    onnx_filename, onnx_path = modelmanager.get_onnx_path(model_name)
    timing_cache = modelmanager.get_timing_cache()

    modelobj = CNetModel(model_id)
    profile = modelobj.get_input_profile(profile_settings)

    if not os.path.exists(onnx_path):
        # TODO use _export_onnx
        torch.onnx.export(
            modelobj.model,
            args=modelobj.get_sample_input(
                1, 512, 512, device="cuda", dtype=torch.float16
            ),
            f=onnx_path,
            opset_version=17,
            input_names=modelobj.get_input_names(),
            output_names=modelobj.get_output_names(),
            dynamic_axes=modelobj.get_dynamic_axes(),
            do_constant_folding=True,
            verbose=False,
            export_params=True,
        )
        print("Exported to ONNX.")

    refit = False
    trt_engine_filename, trt_path = modelmanager.get_trt_path(
        model_name, profile, static_shapes, ModelType.CONTROLNET
    )

    del modelobj
    gc.collect()
    torch.cuda.empty_cache()

    if os.path.exists(trt_path) and not force_export:
        print(
            "TensorRT engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed."
        )
        return "## Exported Successfully \n"

    trt_path = os.path.join(trt_path, "cnet.trt")
    engine = Engine(trt_path)
    engine.build(
        onnx_path,
        (not use_fp32),
        enable_refit=refit,
        timing_cache=timing_cache,
        input_profile=[profile],
        enable_preview=True,
    )
    modelmanager.add_entry(
        model_name,
        profile,
        static_shapes,
        fp32=use_fp32,
        refit=refit,
        model_type=ModelType.CONTROLNET,
        vram=0,
    )

    return "## Exported Successfully \n"


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
):
    is_xl = sd_model.is_sdxl
    model_name = sd_model.sd_checkpoint_info.model_name

    profile_settings = ProfileSettings(
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
    if preset == "Default":
        profile_settings = profile_presets.get_default(is_xl=is_xl)
    use_fp32 = is_fp32()

    print(f"Exporting {model_name} to TensorRT using - {profile_settings}")
    profile_settings.token_to_dim(static_shapes)

    onnx_filename, onnx_path = modelmanager.get_onnx_path(model_name)
    timing_cache = modelmanager.get_timing_cache()

    diable_optimizations = is_xl
    embedding_dim = get_context_dim()

    def disable_checkpoint(self):
        if getattr(self, "use_checkpoint", False) == True:
            self.use_checkpoint = False
        if getattr(self, "checkpoint", False) == True:
            self.checkpoint = False

    sd_model.model.diffusion_model.apply(disable_checkpoint)
    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.apply_optimizations("None")

    modelobj = UNetModelSplit(
        sd_model.model.diffusion_model,
        embedding_dim,
        text_minlen=profile_settings.t_min,
        is_xl=is_xl,
    )

    profile_encoder = modelobj.get_encoder_input_profile(
        profile_settings,
    )
    profile_decoder = modelobj.get_decoder_input_profile(
        profile_settings,
    )
    print(profile_encoder)
    export_onnx(
        onnx_path,
        modelobj,
        profile=profile_encoder,
        diable_optimizations=diable_optimizations,
    )

    trt_engine_filename, trt_path = modelmanager.get_trt_path(
        model_name, profile_encoder, static_shapes, ModelType.UNET
    )
    if not os.path.exists(trt_path):
        os.makedirs(trt_path, exist_ok=True)

    print(
        "Building TensorRT engine... This can take a while, please check the progress in the terminal."
    )
    gr.Info(
        "Building TensorRT engine... This can take a while, please check the progress in the terminal."
    )
    gc.collect()
    torch.cuda.empty_cache()

    _trt_path = os.path.join(trt_path, "encoder.trt")
    _onnx_path = os.path.join(onnx_path, "encoder.onnx")
    if not os.path.exists(_trt_path) or force_export:
        ret = export_trt(
            _trt_path,
            _onnx_path,
            timing_cache,
            profile=profile_encoder,
            use_fp16=not use_fp32,
        )
        if ret:
            return "## Export Failed due to unknown reason. See shell for more information. \n"

    _trt_path = os.path.join(trt_path, "decoder.trt")
    _onnx_path = os.path.join(onnx_path, "decoder.onnx")
    if not os.path.exists(_trt_path) or force_export:
        ret = export_trt(
            _trt_path,
            _onnx_path,
            timing_cache,
            profile=profile_decoder,
            use_fp16=not use_fp32,
        )
        if ret:
            return "## Export Failed due to unknown reason. See shell for more information. \n"

    print("TensorRT engines has been saved to disk.")
    modelmanager.add_entry(
        model_name,
        profile_encoder,
        static_shapes,
        fp32=use_fp32,
        refit=True,
        model_type=ModelType.UNET,
        vram=0,
    )
    print(
        "TensorRT engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed."
    )

    return "## Exported Successfully \n"


def export_lora_to_trt(lora_name, force_export):
    is_xl = sd_model.is_sdxl

    available_lora_models = get_lora_checkpoints()
    lora_name = lora_name.split(" ")[0]
    lora_model = available_lora_models.get(lora_name, None)
    if lora_model is None:
        return f"## No LoRA model found for {lora_name}"

    version = lora_model.get("version", SDVersion.Unknown)
    if version == SDVersion.Unknown:
        print(
            "LoRA SD version couldm't be determined. Please ensure the correct SD Checkpoint is selected."
        )

    model_name = sd_model.sd_checkpoint_info.model_name
    if not version.match(sd_model):
        print(
            f"""LoRA SD version ({version}) does not match the current SD version ({model_name}). 
            Please ensure the correct SD Checkpoint is selected."""
        )

    profile_settings = profile_presets.get_default(is_xl=False)
    print(f"Exporting {lora_name} to TensorRT using - {profile_settings}")
    profile_settings.token_to_dim(True)

    onnx_base_filename, onnx_base_path = modelmanager.get_onnx_path(model_name)
    if not os.path.exists(onnx_base_path):
        return f"## Please export the base model ({model_name}) first."

    embedding_dim = get_context_dim()

    def disable_checkpoint(self):
        if getattr(self, "use_checkpoint", False) == True:
            self.use_checkpoint = False
        if getattr(self, "checkpoint", False) == True:
            self.checkpoint = False

    sd_model.model.diffusion_model.apply(disable_checkpoint)
    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.apply_optimizations("None")

    modelobj = UNetModelSplit(
        sd_model.model.diffusion_model,
        embedding_dim,
        text_minlen=profile_settings.t_min,
        is_xl=is_xl,
    )
    # TODO ensure that checkpoint is LoRA! LÖyCoris needs to be postponed see kohya module?!

    hash, lora_trt_path = modelmanager.get_trt_path(lora_name, {}, {}, ModelType.LORA)
    enc_dict, dec_dict = export_lora(
        modelobj, onnx_base_path, lora_model["filename"], profile_settings
    )

    os.makedirs(lora_trt_path, exist_ok=True)

    if (
        os.path.exists(os.path.join(lora_trt_path, "encoder.refit"))
        and os.path.exists(os.path.join(lora_trt_path, "decoder.refit"))
        and not force_export
    ):
        print(
            "TensorRT engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed."
        )
        return "## Exported Successfully \n"
    save_file(enc_dict, os.path.join(lora_trt_path, "encoder.refit"))
    save_file(dec_dict, os.path.join(lora_trt_path, "decoder.refit"))

    modelmanager.add_entry(
        lora_name,
        {},
        True,
        fp32=True,
        refit=True,
        model_type=ModelType.LORA,
        vram=0,
    )

    return "## Exported Successfully \n"


def get_lora_checkpoints():
    available_lora_models = {}
    allowed_extensions = ["pt", "ckpt", "safetensors"]
    candidates = [
        p
        for p in os.listdir(cmd_opts.lora_dir)
        if p.split(".")[-1] in allowed_extensions
    ]

    for filename in candidates:
        metadata = {}
        name, ext = os.path.splitext(filename)
        config_file = os.path.join(cmd_opts.lora_dir, name + ".json")

        if ext == ".safetensors":
            metadata = sd_models.read_metadata_from_safetensors(
                os.path.join(cmd_opts.lora_dir, filename)
            )
        else:
            print(
                """LoRA {} is not a safetensor. This might cause issues when exporting to TensorRT.
                   Please ensure that the correct base model is selected when exporting.""".format(
                    name
                )
            )

        base_model = metadata.get("ss_sd_model_name", "Unknown")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
            version = SDVersion.from_str(config["sd version"])

        else:
            version = SDVersion.Unknown
            print(
                "No config file found for {}. You can generate it in the LoRA tab.".format(
                    name
                )
            )

        available_lora_models[name] = {
            "filename": filename,
            "version": version,
            "base_model": base_model,
        }
    return available_lora_models


def get_valid_lora_checkpoints():
    available_lora_models = get_lora_checkpoints()
    return [f"{k} ({v['version']})" for k, v in available_lora_models.items()]


def diable_export(version):
    if version == "Default":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )


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
    ):  # TODO add engine path since we are using hasehs now
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

    available_models = modelmanager.get_available_models(ModelType.UNDEFINED)

    model_md = defaultdict(list)
    loras_md = {}
    for base_model, models in available_models.items():
        for i, m in enumerate(models):
            if m["config"].model_type == ModelType.LORA:
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


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="trt_tabs"):
                    with gr.Tab(label="TensorRT Exporter"):
                        gr.Markdown(
                            value="# TensorRT Exporter",
                        )

                        default_vals = profile_presets.get_default(is_xl=False)
                        version = gr.Dropdown(
                            label="Preset",
                            choices=profile_presets.get_choices(),
                            elem_id="sd_version",
                            default="Default",
                            value="Default",
                        )

                        with gr.Accordion(
                            "Advanced Settings", open=False, visible=False
                        ) as advanced_settings:
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
                                    value=default_vals.bs_min,
                                    elem_id="trt_min_batch",
                                )

                                trt_opt_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Optimal batch-size",
                                    value=default_vals.bs_opt,
                                    elem_id="trt_opt_batch",
                                )
                                trt_max_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Max batch-size",
                                    value=default_vals.bs_min,
                                    elem_id="trt_max_batch",
                                )

                            with gr.Column(elem_id="trt_height"):
                                trt_height_min = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Min height",
                                    value=default_vals.h_min,
                                    elem_id="trt_min_height",
                                )
                                trt_height_opt = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Optimal height",
                                    value=default_vals.h_opt,
                                    elem_id="trt_opt_height",
                                )
                                trt_height_max = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Max height",
                                    value=default_vals.h_max,
                                    elem_id="trt_max_height",
                                )

                            with gr.Column(elem_id="trt_width"):
                                trt_width_min = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Min width",
                                    value=default_vals.w_min,
                                    elem_id="trt_min_width",
                                )
                                trt_width_opt = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Optimal width",
                                    value=default_vals.w_opt,
                                    elem_id="trt_opt_width",
                                )
                                trt_width_max = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Max width",
                                    value=default_vals.w_max,
                                    elem_id="trt_max_width",
                                )

                            with gr.Column(elem_id="trt_token_count"):
                                trt_token_count_min = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Min prompt token count",
                                    value=default_vals.t_min,
                                    elem_id="trt_opt_token_count_min",
                                )
                                trt_token_count_opt = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Optimal prompt token count",
                                    value=default_vals.t_opt,
                                    elem_id="trt_opt_token_count_opt",
                                )
                                trt_token_count_max = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Max prompt token count",
                                    value=default_vals.t_max,
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
                            profile_presets.get_settings_from_version,
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
                            [
                                button_export_unet,
                                button_export_default_unet,
                                advanced_settings,
                            ],
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
                            disable_lora_export,
                            trt_lora_dropdown,
                            button_export_lora_unet,
                        )

                    with gr.Tab(label="TensorRT ControlNet"):
                        gr.Markdown(
                            value="# TensorRT ControlNet",
                        )
                        cnet = gr.Dropdown(
                            choices=available_cnets,
                            label="ControlNet Model",
                            elem_id="cnet_model",
                        )

                        default_vals = profile_presets.get_default(is_xl=False)
                        version_cnet = gr.Dropdown(
                            label="Preset",
                            choices=profile_presets.get_choices(),
                            elem_id="sd_version",
                            default="Default",
                            value="Default",
                        )

                        with gr.Accordion(
                            "Advanced Settings", open=False, visible=False
                        ) as advanced_settings_cnet:
                            with FormRow(
                                elem_classes="checkboxes-row", variant="compact"
                            ):
                                static_shapes_cnet = gr.Checkbox(
                                    label="Use static shapes.",
                                    value=False,
                                    elem_id="trt_static_shapes",
                                )

                            with gr.Column(elem_id="trt_max_batch"):
                                trt_min_batch_cnet = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Min batch-size",
                                    value=default_vals.bs_min,
                                    elem_id="trt_min_batch",
                                )

                                trt_opt_batch_cnet = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Optimal batch-size",
                                    value=default_vals.bs_opt,
                                    elem_id="trt_opt_batch",
                                )
                                trt_max_batch_cnet = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Max batch-size",
                                    value=default_vals.bs_max,
                                    elem_id="trt_max_batch",
                                )

                            with gr.Column(elem_id="trt_height"):
                                trt_height_min_cnet = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Min height",
                                    value=default_vals.h_min,
                                    elem_id="trt_min_height",
                                )
                                trt_height_opt_cnet = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Optimal height",
                                    value=default_vals.h_opt,
                                    elem_id="trt_opt_height",
                                )
                                trt_height_max_cnet = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Max height",
                                    value=default_vals.h_max,
                                    elem_id="trt_max_height",
                                )

                            with gr.Column(elem_id="trt_width"):
                                trt_width_min_cnet = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Min width",
                                    value=default_vals.w_min,
                                    elem_id="trt_min_width",
                                )
                                trt_width_opt_cnet = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Optimal width",
                                    value=default_vals.w_opt,
                                    elem_id="trt_opt_width",
                                )
                                trt_width_max_cnet = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Max width",
                                    value=default_vals.w_max,
                                    elem_id="trt_max_width",
                                )

                            with gr.Column(elem_id="trt_token_count"):
                                trt_token_count_min_cnet = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Min prompt token count",
                                    value=default_vals.t_min,
                                    elem_id="trt_opt_token_count_min",
                                )
                                trt_token_count_opt_cnet = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Optimal prompt token count",
                                    value=default_vals.t_opt,
                                    elem_id="trt_opt_token_count_opt",
                                )
                                trt_token_count_max_cnet = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Max prompt token count",
                                    value=default_vals.t_max,
                                    elem_id="trt_opt_token_count_max",
                                )

                            with FormRow(
                                elem_classes="checkboxes-row", variant="compact"
                            ):
                                force_rebuild_cnet = gr.Checkbox(
                                    label="Force Rebuild.",
                                    value=False,
                                    elem_id="trt_force_rebuild",
                                )

                        button_export_cnet = gr.Button(
                            value="Export Engine",
                            variant="primary",
                            elem_id="trt_export_cnet",
                            visible=False,
                        )

                        button_export_default_cnet = gr.Button(
                            value="Export Default Engine",
                            variant="primary",
                            elem_id="trt_export_default_cnet",
                            visible=True,
                        )

                        version_cnet.change(
                            profile_presets.get_settings_from_version,
                            version_cnet,
                            [
                                trt_min_batch_cnet,
                                trt_opt_batch_cnet,
                                trt_max_batch_cnet,
                                trt_height_min_cnet,
                                trt_height_opt_cnet,
                                trt_height_max_cnet,
                                trt_width_min_cnet,
                                trt_width_opt_cnet,
                                trt_width_max_cnet,
                                trt_token_count_min_cnet,
                                trt_token_count_opt_cnet,
                                trt_token_count_max_cnet,
                                static_shapes_cnet,
                            ],
                        )
                        version_cnet.change(
                            diable_export,
                            version_cnet,
                            [
                                button_export_cnet,
                                button_export_default_cnet,
                                advanced_settings_cnet,
                            ],
                        )

                        static_shapes_cnet.change(
                            diable_visibility,
                            static_shapes_cnet,
                            [
                                trt_min_batch_cnet,
                                trt_max_batch_cnet,
                                trt_height_min_cnet,
                                trt_height_max_cnet,
                                trt_width_min_cnet,
                                trt_width_max_cnet,
                                trt_token_count_min_cnet,
                                trt_token_count_max_cnet,
                            ],
                        )

            with gr.Column(variant="panel"):
                with open(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "info.md"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    trt_info = gr.Markdown(elem_id="trt_info", value=f.read())

        with gr.Row(equal_height=False):
            with gr.Accordion("Output", open=True):
                trt_result = gr.Markdown(elem_id="trt_result", value="")

        def get_trt_profiles_markdown():
            profiles_md_string = ""
            for model, profiles in engine_profile_card().items():
                profiles_md_string += f"<details><summary>{model} ({len(profiles)} Profiles)</summary>\n\n"
                for i, profile in enumerate(profiles):
                    profiles_md_string += f"#### Profile {i} \n{profile}\n\n"
                profiles_md_string += "</details>\n"
            profiles_md_string += "</details>\n"
            return profiles_md_string

        with gr.Column(variant="panel"):
            with gr.Row(equal_height=True, variant="compact"):
                button_refresh_profiles = ToolButton(
                    value=refresh_symbol, elem_id="trt_refresh_profiles", visible=True
                )
                profile_header_md = gr.Markdown(
                    value=f"## Available TensorRT Engine Profiles"
                )
            with gr.Row(equal_height=True):
                trt_profiles_markdown = gr.Markdown(
                    elem_id=f"trt_profiles_markdown", value=get_trt_profiles_markdown()
                )

        button_refresh_profiles.click(
            lambda: gr.Markdown.update(value=get_trt_profiles_markdown()),
            outputs=[trt_profiles_markdown],
        )

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

        button_export_cnet.click(
            export_cnet_to_trt,
            inputs=[
                cnet,
                trt_min_batch_cnet,
                trt_opt_batch_cnet,
                trt_max_batch_cnet,
                trt_height_min_cnet,
                trt_height_opt_cnet,
                trt_height_max_cnet,
                trt_width_min_cnet,
                trt_width_opt_cnet,
                trt_width_max_cnet,
                trt_token_count_min_cnet,
                trt_token_count_opt_cnet,
                trt_token_count_max_cnet,
                force_rebuild_cnet,
                static_shapes_cnet,
                version_cnet,
            ],
            outputs=[trt_result],
        )

        button_export_default_cnet.click(
            export_cnet_to_trt,
            inputs=[
                cnet,
                trt_min_batch_cnet,
                trt_opt_batch_cnet,
                trt_max_batch_cnet,
                trt_height_min_cnet,
                trt_height_opt_cnet,
                trt_height_max_cnet,
                trt_width_min_cnet,
                trt_width_opt_cnet,
                trt_width_max_cnet,
                trt_token_count_min_cnet,
                trt_token_count_opt_cnet,
                trt_token_count_max_cnet,
                force_rebuild_cnet,
                static_shapes_cnet,
                version_cnet,
            ],
            outputs=[trt_result],
        )

    return [(trt_interface, "TensorRT", "tensorrt")]
