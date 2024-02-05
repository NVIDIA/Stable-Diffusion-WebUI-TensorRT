from modules import scripts
import gradio as gr
from image_processor import PREPROCESSOR, crop_resize, preprocess_controlnet_images
from modules import scripts, script_callbacks, shared
from modules.ui_components import InputAccordion
from datastructures import ControlNetParams
from scripts.controlnet_trt_unit import ControlNetTrtBase, ControlNetTrtUi

import logging

logger = logging.getLogger('ControlNetTrt')

class ControlNetTensorRTScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.enable_controlnet: bool = False

    def title(self):
        return "ControlNet TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def setup(self, p, *args):
        return super().setup(p, *args)

    def before_process(self, p, *args):
        logger.debug(f"TRT ControlNets: {args}")

        # Only run ControlNet if the accordion is open
        if not self.enable_controlnet:
            return

        controlnets_list = []
        for cnet in args:
            logger.debug(f"TRT CNet: {cnet}")
            if cnet.image is None:
                continue
            crop = fill = False
            if cnet.resize_option == "Crop":
                crop = True
            elif cnet.resize_option == "Fill":
                fill = True

            assert cnet.image is not None
            assert cnet.model is not None
            assert cnet.preprocessor is not None

            img = crop_resize(cnet.image, p.width, p.height, crop, fill)
            condition = PREPROCESSOR[cnet.preprocessor](img)
            condition = preprocess_controlnet_images(p.batch_size, [condition])[0]
            controlnets_list.append(ControlNetParams(cnet.model, cnet.weight, condition))

        p.controlnet = controlnets_list if len(controlnets_list) > 0 else None

    def ui_groups(self):
        cnets = []
        for _ in range(shared.opts.data.get("controlnet_trt_unit_count", 3)):
            cnets.append((ControlNetTrtUi(ControlNetTrtBase())))
        return cnets

    def ui(self, is_img2img):
        elem_id = ("img2img" if is_img2img else "txt2img") + "_controlnet_trt"
        controlnet_trt_units = ()
        # Create 3 tabs and loop over them
        with gr.Group(elem_id=elem_id, visible=not is_img2img):
            with InputAccordion(
                False, label="ControlNet TensorRT", elem_id=f"{elem_id}_accordion"
            ) as self.enable_controlnet:
                with gr.Tabs():
                    i = 0
                    for group in self.ui_groups():
                        i = i + 1
                        with gr.Tab(label=f"ControlNet-TRT-{i}"):
                            group.render(is_img2img, f"tab_{i}", elem_id)
                            state = group.wire_up_component_states()
                            controlnet_trt_units += (state,)

        def activate(a):
            self.enable_controlnet = a

        self.enable_controlnet.change(activate, inputs=[self.enable_controlnet])

        return controlnet_trt_units

def on_ui_settings():
    section = ('control_net_trt', "ControlNet TRT")
    shared.opts.add_option("controlnet_trt_unit_count", shared.OptionInfo(
        3, "Multi-ControlNet TRT: ControlNet TRT unit number (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))

script_callbacks.on_ui_settings(on_ui_settings)
