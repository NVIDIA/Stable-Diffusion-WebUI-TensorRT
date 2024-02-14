import numpy as np
import torch
import cv2
from controlnet_aux import OpenposeDetector, HEDdetector, MLSDdetector
from PIL import Image
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from utilities import Registry
from datastructures import ResizeOption

PREPROCESSOR = Registry("preprocessor")


@PREPROCESSOR.register("SEG")
def seg(image: Image):
    image_processor = AutoImageProcessor.from_pretrained(
        "openmmlab/upernet-convnext-small"
    )
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-small"
    )

    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)

    image = Image.fromarray(color_seg)
    return image


@PREPROCESSOR.register("Scribble")
def scribble(image: Image):
    return hed(image, scribble=True)


@PREPROCESSOR.register("OpenPose")
def openpose(
    image: Image,
):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    image = openpose(image)
    return image


@PREPROCESSOR.register("Normal")
def normal(image: Image):
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")

    image = depth_estimator(image)["predicted_depth"][0]

    image = image.numpy()

    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)

    bg_threhold = 0.4

    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0

    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0

    z = np.ones_like(x) * np.pi * 2.0

    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image


@PREPROCESSOR.register("MLSD")
def mlsd(image: Image):
    mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    image = mlsd(image)
    return image


@PREPROCESSOR.register("HED")
def hed(image: Image, scribble=False):
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    image = hed(image, scribble=scribble)
    return image


@PREPROCESSOR.register("Depth")
def depth(image: Image):
    depth_estimator = pipeline("depth-estimation")

    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return image


@PREPROCESSOR.register("Canny")
def canny(image: Image, low_threshold: int = 100, high_threshold: int = 200):
    image = np.array(image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return image


def resize(image: Image, width: int, height: int, mode: ResizeOption):
    crop = fill = False
    if mode == ResizeOption.CROP:
        crop = True
    elif mode == ResizeOption.FILL:
        fill = True
    elif mode == ResizeOption.RESIZE:
        pass

    return crop_resize(image, width, height, crop, fill)


def crop_resize(image: Image, width: int, height: int, crop: bool, fill: bool):
    if image.size == (width, height):
        return image

    ar_img = image.size[0] / image.size[1]
    ar_out = width / height
    out_size = (width, height)

    if ar_img > ar_out:
        dim = 1
    elif ar_img < ar_out:
        dim = 0
    else:
        crop = fill = False
        return image.resize(out_size, Image.BILINEAR)

    scale_factor = out_size[dim] / image.size[dim]
    image = image.resize(
        (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor)),
        Image.BILINEAR,
    )

    if crop:
        crop_size = min(image.size[not dim], out_size[not dim])
        pad = (image.size[not dim] - crop_size) // 2

        left = right = pad if dim else 0
        top = bottom = pad if not dim else 0

        image = image.crop((left, top, image.size[0] - right, image.size[1] - bottom))

    elif fill:
        fill_size = int(max(image.size[not dim], out_size[not dim]) / ar_out)
        pad = (fill_size - image.size[dim]) // 2

        left = right = pad if not dim else 0
        top = bottom = pad if dim else 0

        out_fill_size = (
            (image.size[0], fill_size) if dim else (fill_size, image.size[1])
        )
        result = Image.new(image.mode, out_fill_size, "black")
        result.paste(image, (left, top))
        image = result

    image = image.resize(out_size, Image.BILINEAR)
    return image


def preprocess_controlnet_images(batch_size: int, images: Image = None, device="cuda"):
    if images is None:
        return None
    images = [
        (np.array(i.convert("RGB")).astype(np.float32) / 255.0)[..., None]
        .transpose(3, 2, 0, 1)
        .repeat(batch_size, axis=0)
        for i in images
    ]
    # do_classifier_free_guidance
    images = [torch.cat([torch.from_numpy(i).to(device).float()] * 2) for i in images]
    images = torch.cat([image[None, ...] for image in images], dim=0)
    return images


palette = np.asarray(
    [
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]
)
