"""Image transform utilities for diffusion pretraining."""

from __future__ import annotations

from torchvision import transforms


def build_image_transforms(
    resolution: int,
    center_crop: bool = False,
    random_flip: bool = True,
    color_jitter: float = 0.0,
):
    """Build augmentation pipeline suitable for diffusion pretraining.

    Images are resized slightly larger, then randomly cropped to preserve diversity.
    Output is normalized to [-1, 1].
    """

    resize_size = int(resolution * 1.12)
    transform_ops = [transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR)]

    if center_crop:
        transform_ops.append(transforms.CenterCrop(resolution))
    else:
        transform_ops.append(transforms.RandomCrop(resolution))

    if random_flip:
        transform_ops.append(transforms.RandomHorizontalFlip(p=0.5))

    if color_jitter > 0:
        transform_ops.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(0.5, color_jitter / 2),
            )
        )

    transform_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(transform_ops)
