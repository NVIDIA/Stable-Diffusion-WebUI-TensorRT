import launch
from importlib_metadata import version
from modules import shared

def install():
    if launch.is_installed("tensorrt"):
        if not version("tensorrt") == "9.0.1.post11.dev4":
            launch.run(["python","-m","pip","uninstall","-y","tensorrt"], "removing old version of tensorrt")
        
    
    if not launch.is_installed("tensorrt"):
        print("TensorRT is not installed! Installing...")
        launch.run_pip("install nvidia-cudnn-cu11==8.9.4.25", "nvidia-cudnn-cu11")
        launch.run_pip("install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4", "tensorrt", live=True)
        launch.run(["python","-m","pip","uninstall","-y","nvidia-cudnn-cu11"], "removing nvidia-cudnn-cu11")
        
    if launch.is_installed("nvidia-cudnn-cu11"):
        if version("nvidia-cudnn-cu11") == "8.9.4.25":
            launch.run(["python","-m","pip","uninstall","-y","nvidia-cudnn-cu11"], "removing nvidia-cudnn-cu11")

    # Polygraphy	
    if not launch.is_installed("polygraphy"):
        print("Polygraphy is not installed! Installing...")
        launch.run_pip("install polygraphy --extra-index-url https://pypi.ngc.nvidia.com", "polygraphy", live=True)
    
    # ONNX GS
    if not launch.is_installed("onnx-graphsurgeon"):
        print("GS is not installed! Installing...")
        launch.run_pip("install protobuf==3.20.2", "protobuf", live=True)
        launch.run_pip('install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com', "onnx-graphsurgeon", live=True)

    if shared.opts is None:
        print("UI Config not initialized")
        return 
    
    if "sd_unet" not in shared.opts["quicksettings_list"]:
        shared.opts["quicksettings_list"].append("sd_unet")
        shared.opts.save(shared.config_filename)
 
install()