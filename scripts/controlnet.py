from modules import scripts
import gradio as gr
from model_manager import modelmanager
from image_processor import PREPROCESSOR, crop_resize, preprocess_controlnet_images
from PIL import Image
from modules.ui_components import InputAccordion
from datastructures import ControlNetParams, ModelType

cnet_list = modelmanager.get_available_models(ModelType.CONTROLNET)
preprocessor_list = PREPROCESSOR.choices()


class ControlNetTensorRTScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.image: Image = None
        self.model: str = None
        self.preprocessor: str = None
        self.scale: float = 1.0
        self.resize_option: str = None
        self.enable_controlnet: bool = False

    def title(self):
        return "ControlNet TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def setup(self, p, *args):
        return super().setup(p, *args)

    def before_process(self, p, *args):
        # Only run ControlNet if the accordion is open
        if not self.enable_controlnet:
            return
        crop = fill = False
        if self.resize_option == "Crop":
            crop = True
        elif self.resize_option == "Fill":
            fill = True

        assert self.image is not None
        assert self.model is not None
        assert self.preprocessor is not None

        img = crop_resize(self.image, p.width, p.height, crop, fill)
        condition = PREPROCESSOR[self.preprocessor](img)
        condition = preprocess_controlnet_images(p.batch_size, [condition])[0]

        p.controlnet = [ControlNetParams(self.model, self.scale, condition)]

    def run_preprocessor(self, image, preprocessor, resize_option):
        crop = fill = False
        if resize_option == "Crop":
            crop = True
        elif resize_option == "Fill":
            fill = True

        if image is None:
            gr.Error("No image uploaded")
            return
        image = crop_resize(image, 512, 512, crop, fill)

        if preprocessor is None:
            raise ValueError("Preprocessor not selected")
        return PREPROCESSOR[preprocessor](image)

    def update_data(self, image, preprocessor, resize_option, model, weight):
        self.image = image
        self.preprocessor = preprocessor
        self.resize_option = resize_option
        self.model = model
        self.scale = weight

    def ui(self, is_img2img):
        elem_id_tabname = ("img2img" if is_img2img else "txt2img") + "_controlnet"
        with gr.Group(elem_id=elem_id_tabname):
            with InputAccordion(
                False, label="ControlNet TensorRT", elem_id="txt2img_controlnet"
            ) as enable_controlnet:
                with gr.Column():
                    with gr.Tabs():
                        with gr.Tab(label="Single Image") as self.upload_tab:
                            with gr.Row(equal_height=True):
                                with gr.Column():
                                    image = gr.Image(
                                        source="upload",
                                        mirror_webcam=False,
                                        type="pil",
                                        elem_id=f"{elem_id_tabname}_input_image",
                                        elem_classes=["cnet-image"],
                                        width=256,
                                        height=256,
                                    )

                                with gr.Column():
                                    preview = gr.Image(
                                        source="upload",
                                        mirror_webcam=False,
                                        type="pil",
                                        elem_id=f"{elem_id_tabname}_preview_image",
                                        elem_classes=["cnet-preview-image"],
                                        width=256,
                                        height=256,
                                    )

                            with gr.Row(equal_height=True):
                                model = gr.Dropdown(
                                    label="ControlNet Model",
                                    choices=cnet_list,
                                    value=None,
                                    elem_id=f"{elem_id_tabname}_model",
                                )

                                preprocessor = gr.Dropdown(
                                    label="ControlNet Preprocessor",
                                    choices=preprocessor_list,
                                    value=None,
                                    elem_id=f"{elem_id_tabname}_model",
                                )

                            with gr.Row(equal_height=False):
                                resize_option = gr.Radio(
                                    label="Resize Option",
                                    choices=["Resize", "Crop", "Fill"],
                                    value="Crop",
                                    elem_id=f"{elem_id_tabname}_resize_option",
                                )

                                preview_btn = gr.Button(
                                    value="Preview", elem_id=f"{elem_id_tabname}_preview_btn", scale=0.3
                                )
                            
                            with gr.Row(equal_height=False):
                                weight = gr.Slider(
                                    label="ControlNet Scale",
                                    minimum=0,
                                    maximum=2,
                                    step=0.05,
                                    value=1,
                                    elem_id=f"{elem_id_tabname}_scale",
                                )

        preview_btn.click(
            self.run_preprocessor,
            inputs=[image, preprocessor, resize_option],
            outputs=[preview],
        )
        triggers = [
            image,
            preprocessor,
            resize_option,
            model,
            weight,
        ]
        for trigger in triggers:
            trigger.change(
                self.update_data,
                inputs=[image, preprocessor, resize_option, model, weight],
                outputs=[],
            )

        def activate(a):
            self.enable_controlnet = a

        enable_controlnet.change(activate, inputs=[enable_controlnet])
