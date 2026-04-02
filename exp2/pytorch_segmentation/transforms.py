from __future__ import annotations

import random
from typing import Optional

from PIL import Image, ImageOps
import torchvision.transforms.functional as F

from exp2.pytorch_segmentation.config import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD


class RandomScaleCropFlip:
    def __init__(
        self,
        crop_size: int,
        min_scale: float,
        max_scale: float,
        ignore_label: int,
    ) -> None:
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ignore_label = ignore_label

    def __call__(self, image: Image.Image, mask: Image.Image):
        scale = random.uniform(self.min_scale, self.max_scale)
        new_width = max(1, int(image.width * scale))
        new_height = max(1, int(image.height * scale))
        image = image.resize((new_width, new_height), resample=Image.BILINEAR)
        mask = mask.resize((new_width, new_height), resample=Image.NEAREST)

        pad_width = max(self.crop_size - new_width, 0)
        pad_height = max(self.crop_size - new_height, 0)
        if pad_width or pad_height:
            image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=0)
            mask = ImageOps.expand(
                mask,
                border=(0, 0, pad_width, pad_height),
                fill=self.ignore_label,
            )

        left = random.randint(0, image.width - self.crop_size)
        top = random.randint(0, image.height - self.crop_size)
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        mask = mask.crop((left, top, left + self.crop_size, top + self.crop_size))

        if random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        image = F.to_tensor(image)
        image = F.normalize(image, mean=DEFAULT_IMAGE_MEAN, std=DEFAULT_IMAGE_STD)
        mask = F.pil_to_tensor(mask).long().squeeze(0)
        return image, mask


class EvalTransform:
    def __init__(self, long_size: Optional[int] = None) -> None:
        self.long_size = long_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        if self.long_size is not None:
            scale = self.long_size / max(image.height, image.width)
            new_width = max(1, int(round(image.width * scale)))
            new_height = max(1, int(round(image.height * scale)))
            image = image.resize((new_width, new_height), resample=Image.BILINEAR)

        image = F.to_tensor(image)
        image = F.normalize(image, mean=DEFAULT_IMAGE_MEAN, std=DEFAULT_IMAGE_STD)
        mask = F.pil_to_tensor(mask).long().squeeze(0)
        return image, mask


class PredictTransform:
    def __init__(self, long_size: Optional[int] = None) -> None:
        self.long_size = long_size

    def __call__(self, image: Image.Image):
        image = image.convert("RGB")
        original_size = image.size
        if self.long_size is not None:
            scale = self.long_size / max(image.height, image.width)
            new_width = max(1, int(round(image.width * scale)))
            new_height = max(1, int(round(image.height * scale)))
            image = image.resize((new_width, new_height), resample=Image.BILINEAR)
        tensor = F.to_tensor(image)
        tensor = F.normalize(tensor, mean=DEFAULT_IMAGE_MEAN, std=DEFAULT_IMAGE_STD)
        return tensor, original_size
