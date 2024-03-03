import launch
import pkg_resources

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def install_package(package_name, version_spec=None, uninstall_first=False, extra_index_url=None, no_cache_dir=False):
    package_install_cmd = f"{package_name}{'==' + version_spec if version_spec else ''}"
    if extra_index_url:
        package_install_cmd += f" --extra-index-url {extra_index_url}"
    if no_cache_dir:
        package_install_cmd += " --no-cache-dir"
    
    if uninstall_first and launch.is_installed(package_name):
        launch.run(["python", "-m", "pip", "uninstall", "-y", package_name], f"removing {package_name}")
    
    launch.run_pip(f"install {package_install_cmd}", package_name, live=True)


def install():
    # TensorRT installation or upgrade
    tensorrt_version = get_installed_version("tensorrt")
    if not tensorrt_version or tensorrt_version != "9.3.0.post12.dev1":
        # nvidia-cudnn-cu11 installation
        if launch.is_installed("nvidia-cudnn-cu12") and get_installed_version("nvidia-cudnn-cu11") != "8.9.7.29":
            install_package("nvidia-cudnn-cu12", "8.9.6.50", uninstall_first=True, no_cache_dir=True)
        install_package("tensorrt", "9.3.0.post12.dev1", uninstall_first=True, extra_index_url="https://pypi.nvidia.com", no_cache_dir=True)


    # Polygraphy installation
    if not launch.is_installed("polygraphy"):
        install_package("polygraphy", extra_index_url="https://pypi.ngc.nvidia.com", no_cache_dir=True)

    # ONNX Graph Surgeon installation
    if not launch.is_installed("onnx_graphsurgeon"):
        install_package("protobuf", "3.20.3", no_cache_dir=True)
        install_package("onnx-graphsurgeon", extra_index_url="https://pypi.ngc.nvidia.com", no_cache_dir=True)

    # Optimum installation
    if not launch.is_installed("optimum"):
        install_package("optimum", no_cache_dir=True)

install()
