from torchvision import transforms
from typing import Callable, Any


def transform_if(predicate: bool, f: Callable[[Any], Any]) -> transforms.Lambda:
    return transforms.Lambda(lambda x: f(x) if predicate else x)


def tensor_normalize() -> transforms.Normalize:
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


def image_normalize() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        tensor_normalize(),
    ])


def resize_square(size: int) -> transforms.Compose:
    if size is None:
        return None

    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size)
    ])


def preproces(to_size: int = None, augmentation=None):
    return transforms.Compose([
        transform_if(augmentation is not None, augmentation),
        transform_if(to_size is not None, resize_square(to_size)),
        image_normalize()
    ])
