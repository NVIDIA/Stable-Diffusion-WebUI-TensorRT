import gradio as gr
from model_manager import controlnetmanager

from image_processor import PREPROCESSOR, resize, preprocess_controlnet_images
from PIL import Image
from datastructures import ControlModelType, ResizeOption
from dataclasses import dataclass

import torch
from diffusers import ControlNetModel

cnet_list = controlnetmanager.list_controlnet("sd15")
preprocessor_list = PREPROCESSOR.choices()
resize_option_list = ResizeOption.list_resize_options()


@dataclass
class ControlNetTrtBase:
    image: Image = None
    preprocessor: str = None
    resize_option: str = ResizeOption.CROP
    model_id: str = None
    hf_id: str = None
    model: str = None
    weight: float = 1.0
    enable: bool = True
    is_single_image: bool = True
    control_type: ControlModelType = ControlModelType.CONTROLNET
    controlnet_cond: torch.Tensor = None

    def __eq__(self, other):
        if not isinstance(other, ControlNetTrtBase):
            return False

        return vars(self) == vars(other)

    def load_model(self):
        if self.hf_id is None:
            self.hf_id = controlnetmanager.get_hf_id(self.model_id)
        self.model = (
            ControlNetModel.from_pretrained(self.hf_id, torch_dtype=torch.float16)
            .to(device="cuda")
            .eval()
            .half()
        )

    def get_cond(self, batch_size: int, width: int, height: int):
        img = resize(self.image, width, height, self.resize_option)
        condition = PREPROCESSOR[self.preprocessor](img)
        condition = preprocess_controlnet_images(batch_size, [condition])[0].half()
        self.controlnet_cond = condition


class ControlNetTrtUi(object):
    def __init__(self, controlnet_trt: ControlNetTrtBase):
        self.controlnet_trt = controlnet_trt

    def __repr__(self) -> str:
        return self.controlnet_trt.__repr__()

    def get_cnet_trt(
        self, image, preprocessor, resize_option, model, weight
    ) -> ControlNetTrtBase:
        return ControlNetTrtBase(
            image=image,
            preprocessor=preprocessor,
            resize_option=resize_option,
            model_id=model,
            weight=weight,
        )

    def run_preprocessor(self, image, preprocessor, resize_option):
        if image is None:
            gr.Error("No image uploaded")
            return
        image = resize(image, 512, 512, resize_option)

        if preprocessor is None:
            raise ValueError("Preprocessor not selected")
        return PREPROCESSOR[preprocessor](image)

    def render(self, is_img2img: bool, tabname: str, elem_id: str):
        with gr.Group():
            with gr.Tabs():
                with gr.Tab(label="Single Image"):
                    with gr.Column(elem_classes=["cnet-trt-column-layout"]):
                        with gr.Row(elem_classes=["cnet-trt-row-layout"]):
                            self.image = gr.Image(
                                source="upload",
                                mirror_webcam=False,
                                type="pil",
                                elem_id=f"{elem_id}_{tabname}_input_image",
                                value=self.controlnet_trt.image,
                                width=256,
                                height=256,
                            )

                            self.preview_image = gr.Image(
                                source="upload",
                                mirror_webcam=False,
                                type="pil",
                                elem_id=f"{elem_id}_{tabname}_preview_image",
                                width=256,
                                height=256,
                            )

                        with gr.Row(elem_classes=["cnet-trt-row-layout"]):
                            self.preprocessor = gr.Dropdown(
                                label="ControlNet Preprocessor",
                                choices=preprocessor_list,
                                value=self.controlnet_trt.preprocessor,
                                elem_id=f"{elem_id}_{tabname}_model",
                                scale=3,
                            )

                            self.preview_btn = gr.Button(
                                value="Preview",
                                elem_id=f"{elem_id}_{tabname}_preview_btn",
                                elem_classes=["cnet-trt-preview-button"],
                                scale=2,
                            )

                            self.model = gr.Dropdown(
                                label="ControlNet Model",
                                choices=cnet_list,
                                value=self.controlnet_trt.model,
                                elem_id=f"{elem_id}_{tabname}_model",
                                scale=5,
                            )

                        with gr.Row(elem_classes=["cnet-trt-row-layout"]):
                            self.resize_option = gr.Radio(
                                label="Resize Option",
                                choices=resize_option_list,
                                value=self.controlnet_trt.resize_option,
                                elem_id=f"{elem_id}_{tabname}_resize_option",
                            )

                        with gr.Row(elem_classes=["cnet-trt-row-layout"]):
                            self.weight = gr.Slider(
                                label="ControlNet Scale",
                                minimum=0,
                                maximum=2,
                                step=0.05,
                                value=self.controlnet_trt.weight,
                                elem_id=f"{elem_id}_{tabname}_scale",
                            )

        self.preview_btn.click(
            fn=self.run_preprocessor,
            inputs=[self.image, self.preprocessor, self.resize_option],
            outputs=self.preview_image,
        )

    def wire_up_component_states(self):
        cnet_ui_state = gr.State(self.controlnet_trt)
        cnet_trt_unit = [
            self.image,
            self.preprocessor,
            self.resize_option,
            self.model,
            self.weight,
        ]

        for trigger in cnet_trt_unit:
            event_subscribers = []
            if hasattr(trigger, "edit"):
                event_subscribers.append(trigger.edit)
            elif hasattr(trigger, "click"):
                event_subscribers.append(trigger.click)
            elif isinstance(trigger, gr.Slider) and hasattr(trigger, "release"):
                event_subscribers.append(trigger.release)
            elif hasattr(trigger, "change"):
                event_subscribers.append(trigger.change)
            if hasattr(trigger, "clear"):
                event_subscribers.append(trigger.clear)

            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=self.get_cnet_trt, inputs=cnet_trt_unit, outputs=cnet_ui_state
                )

        return cnet_ui_state
