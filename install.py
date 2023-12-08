import launch
import sys
from importlib_metadata import version

python = sys.executable


def install():
    import torch

    if not torch.cuda.is_available():
        print(
            "Torch CUDA is not available! Please install Torch with CUDA and try again."
        )
        return

    if launch.is_installed("tensorrt"):
        if not version("tensorrt") == "9.0.1.post11.dev4":
            print("Removing old TensorRT package and try reinstalling...")
            launch.run(
                f'"{python}" -m pip uninstall -y tensorrt',
                "removing old version of tensorrt",
            )

    if not launch.is_installed("tensorrt"):
        launch.run_pip(
            "install --pre --extra-index-url https://pypi.nvidia.com  --no-cache-dir --no-deps tensorrt==9.0.1.post11.dev4",
            "tensorrt",
            live=True,
        )

    # Polygraphy
    if not launch.is_installed("polygraphy"):
        print("Polygraphy is not installed! Installing...")
        launch.run_pip(
            "install polygraphy --extra-index-url https://pypi.ngc.nvidia.com",
            "polygraphy",
            live=True,
        )

    # ONNX GS
    if not launch.is_installed("onnx_graphsurgeon"):
        print("GS is not installed! Installing...")
        launch.run_pip("install protobuf==3.20.2", "protobuf", live=True)
        launch.run_pip(
            "install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com",
            "onnx-graphsurgeon",
            live=True,
        )

    ###########################################################
    ############# ControlNet ##################################

    # Diffusers
    if not launch.is_installed("diffusers"):
        print("Diffusers is not installed! Installing...")
        launch.run_pip(
            "install diffusers",
            "diffusers",
            live=True,
        )
    if not launch.is_installed("transformers"):
        print("Transformers is not installed! Installing...")
        launch.run_pip(
            "install transformers",
            "transformers",
            live=True,
        )
    if not launch.is_installed("accelerate"):
        print("accelerate is not installed! Installing...")
        launch.run_pip(
            "install accelerate",
            "accelerate",
            live=True,
        )
    if not launch.is_installed("optimum"):
        print("Diffusers is not installed! Installing...")
        launch.run_pip(
            "install optimum",
            "optimum",
            live=True,
        )

    # OpenCV
    if not launch.is_installed("cv2"):
        print("OpenCV is not installed! Installing...")
        launch.run_pip(
            "install opencv-contrib-python",
            "opencv-contrib-python",
            live=True,
        )

    # ControlNet AUX
    if not launch.is_installed("cv2"):
        print("ControlNetAux is not installed! Installing...")
        launch.run_pip(
            "install controlnet-aux",
            "controlnet-aux",
            live=True,
        )


install()
