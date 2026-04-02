from __future__ import annotations

from typing import Optional

from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


def _resolve_segmentation_weights(name: str):
    if name == "none":
        return None
    if name == "voc":
        return DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    raise ValueError(f"Unsupported segmentation weights: {name}")


def _resolve_backbone_weights(name: str):
    if name == "none":
        return None
    if name == "imagenet":
        return getattr(ResNet50_Weights, "IMAGENET1K_V2", ResNet50_Weights.IMAGENET1K_V1)
    raise ValueError(f"Unsupported backbone weights: {name}")


def build_deeplabv3_resnet50(
    num_classes: int = 21,
    weights: str = "none",
    backbone_weights: str = "imagenet",
) -> nn.Module:
    segmentation_weights = _resolve_segmentation_weights(weights)

    if segmentation_weights is not None and num_classes != 21:
        raise ValueError("VOC pretrained segmentation weights only support num_classes=21.")

    model = deeplabv3_resnet50(
        weights=segmentation_weights,
        weights_backbone=None if segmentation_weights is not None else _resolve_backbone_weights(backbone_weights),
        num_classes=num_classes,
    )
    return model


def freeze_batch_norm(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            for parameter in child.parameters():
                parameter.requires_grad = False
