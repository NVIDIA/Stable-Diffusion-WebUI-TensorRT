# TensorRT Extension for Stable Diffusion 

This extension enables the best performance on NVIDIA RTX GPUs for Stable Diffusion with TensorRT. 

You need to install the extension and generate optimized engines before using the extension. Please follow the instructions below to set everything up. 

Supports Stable Diffusion 1.5 and 2.1. Native SDXL support coming in a future release. Please use the [dev branch](https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/dev) if you would like to use it today. Note that the Dev branch is not intended for production work and may break other things that you are currently using.

## Installation

Example instructions for Automatic1111:

1. Start the webui.bat
2. Select the Extensions tab and click on Install from URL
3. Copy the link to this repository and paste it into URL for extension's git repository
4. Click Install

## How to use

1. Click on the “Generate Default Engines” button. This step takes 2-10 minutes depending on your GPU. You can generate engines for other combinations. 
2. Go to Settings → User Interface → Quick Settings List, add sd_unet. Apply these settings, then reload the UI.
3. Back in the main UI, select the TRT model from the sd_unet dropdown menu at the top of the page. 
4. You can now start generating images accelerated by TRT. If you need to create more Engines, go to the TensorRT tab. 

Happy prompting!

## More Information

TensorRT uses optimized engines for specific resolutions and batch sizes. You can generate as many optimized engines as desired. Types:

- The "Export Default Engines” selection adds support for resolutions between 512x512 and 768x768 for Stable Diffusion 1.5 and 768x768 to 1024x1024 for SDXL with batch sizes 1 to 4.
- Static engines support a single specific output resolution and batch size. 
- Dynamic engines support a range of resolutions and batch sizes, at a small cost in performance. Wider ranges will use more VRAM. 

Each preset can be adjusted with the “Advanced Settings” option. More detailed instructions can be found [here](https://nvidia.custhelp.com/app/answers/detail/a_id/5487/~/tensorrt-extension-for-stable-diffusion-web-ui).

### Common Issues/Limitations

**HIRES FIX:** If using the hires.fix option in Automatic1111 you must build engines that match both the starting and ending resolutions. For instance, if initial size is `512 x 512` and hires.fix upscales to `1024 x 1024`, you must either generate two engines, one at 512 and one at 1024, or generate a single dynamic engine that covers the whole range.
Having two seperate engines will heavily impact performance at the moment. Stay tuned for updates.

**Resolution:** When generating images the resolution needs to be a multiple of 64. This applies to hires.fix as well, requiring the low and high-res to be divisible by 64.

**Failing CMD arguments:**

- `medvram` and `lowvram` Have caused issues when compiling the engine and running it.
- `api` Has caused the `model.json` to not be updated. Resulting in SD Unets not appearing after compilation.

**Failing installation or TensorRT tab not appearing in UI:** This is most likely due to a failed install. To resolve this manually use this [guide](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT/issues/27#issuecomment-1767570566).

## Requirements

**Driver**:

- Linux: >= 450.80.02
- Windows: >=452.39

We always recommend keeping the driver up-to-date for system wide performance improvments.