# TensorRT Extension

Use this extension to generate optimized engines and enable the best performance on NVIDIA RTX GPUs with TensorRT. Please follow the instructions below to set everything up.

## Set Up

1. Click on the "Generate Default Engines" button. This step can take 2-10 min depending on your GPU. You can generate engines for other combinations. 
2. Go to Settings → User Interface → Quick Settings List, add sd_unet. Apply these settings, then reload the UI.
3. Back in the main UI, select the TRT model from the sd_unet dropdown menu at the top of the page.
4. You can now start generating images accelerated by TRT. If you need to create more Engines, go to the TensorRT tab.

Happy prompting!

## More Information

TensorRT uses optimized engines for specific resolutions and batch sizes. You can generate as many optimized engines as desired. Types:

- The "Export Default Engines" selection adds support for resolutions between 512x512 and 768x768 for Stable Diffusion 1.5 and 768x768 to 1024x1024 for SDXL with batch sizes 1 to 4.
- Static engines support a single specific output resolution and batch size.
- Dynamic engines support a range of resolutions and batch sizes, at a small cost in performance. Wider ranges will use more VRAM. 

---

Each preset can be adjusted with the "Advanced Settings" option.

For more information, please visit the TensorRT Extension GitHub page [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui-tensorrt).
