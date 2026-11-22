import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def stretch_and_normalize(image):
    """Stretch a uint8 image to 224 x 224 and apply ImageNet normalization."""
    image = F.resize(image, [224, 224], antialias=True)
    image = F.convert_image_dtype(image, torch.float32)
    return F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def resize_and_pad(image):
    """Fit a channel-first image inside a black 224 × 224 canvas."""
    size = 224
    height, width = image.shape[-2:]
    longest = max(height, width)
    new_height = max(1, round(size * height / longest))
    new_width = max(1, round(size * width / longest))
    resized = F.resize(
        image, [new_height, new_width],
        interpolation=InterpolationMode.BILINEAR, antialias=True,
    )

    horizontal = size - new_width
    vertical = size - new_height
    left, top = horizontal // 2, vertical // 2
    return F.pad(resized, [left, top, horizontal - left, vertical - top], fill=0)


def transform_without_augmentation():
    return resize_and_pad


def transform_with_augmentation():
    pass
