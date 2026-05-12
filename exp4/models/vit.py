from __future__ import annotations

from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_vit_model(
    num_classes: int = 10,
    pretrained: bool = True,
    train_mode: str = "full",
) -> nn.Module:
    """Build torchvision ViT-B/16 and replace the classification head.

    Args:
        num_classes: Number of target classes.
        pretrained: Load ImageNet pretrained weights when True.
        train_mode: "full" fine-tunes all parameters; "head_only" trains only
            the final classification head.
    """

    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    if train_mode == "head_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.heads.parameters():
            parameter.requires_grad = True
    elif train_mode != "full":
        raise ValueError("train_mode must be 'full' or 'head_only'")

    return model

