from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


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
