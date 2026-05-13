from __future__ import annotations

import warnings

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, embed_dim: int = 192) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(images)
        return tokens.flatten(2).permute(2, 0, 1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attention_output, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.dropout(attention_output)
        return tokens + self.mlp(self.norm2(tokens))


class LightweightVisionTransformer(nn.Module):
    """Small ViT fallback for torchvision versions without official ViT."""

    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embedding.num_patches
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(num_patches + 1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(images)
        batch_size = tokens.size(1)
        class_tokens = self.class_token.expand(-1, batch_size, -1)
        tokens = torch.cat((class_tokens, tokens), dim=0)
        tokens = self.dropout(tokens + self.position_embedding)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[0])


def _set_train_mode(model: nn.Module, train_mode: str, head: nn.Module) -> None:
    if train_mode == "head_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
    elif train_mode != "full":
        raise ValueError("train_mode must be 'full' or 'head_only'")


def _replace_torchvision_head(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model.heads
    if hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model.head
    raise AttributeError("Unsupported torchvision ViT head structure")


def _build_torchvision_vit(num_classes: int, pretrained: bool) -> tuple[nn.Module, nn.Module] | None:
    try:
        from torchvision.models import vit_b_16
    except ImportError:
        return None

    try:
        from torchvision.models import ViT_B_16_Weights
    except ImportError:
        ViT_B_16_Weights = None

    if ViT_B_16_Weights is not None:
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
    else:
        try:
            model = vit_b_16(pretrained=pretrained)
        except TypeError:
            model = vit_b_16()
            if pretrained:
                warnings.warn(
                    "This torchvision version exposes vit_b_16 but cannot load "
                    "pretrained weights through this API.",
                    RuntimeWarning,
                )

    head = _replace_torchvision_head(model, num_classes)
    return model, head


def build_vit_model(
    num_classes: int = 10,
    pretrained: bool = True,
    train_mode: str = "full",
) -> nn.Module:
    """Build torchvision ViT-B/16 and replace the classification head.

    Args:
        num_classes: Number of target classes.
        pretrained: Load ImageNet pretrained weights when the backend supports it.
        train_mode: "full" fine-tunes all parameters; "head_only" trains only
            the final classification head.
    """

    built = _build_torchvision_vit(num_classes=num_classes, pretrained=pretrained)
    if built is not None:
        model, head = built
        _set_train_mode(model, train_mode=train_mode, head=head)
        return model

    if pretrained:
        warnings.warn(
            "torchvision in this environment does not provide ViT-B/16. "
            "Falling back to a lightweight ViT initialized from scratch.",
            RuntimeWarning,
        )
    effective_train_mode = "full" if train_mode == "head_only" else train_mode
    model = LightweightVisionTransformer(num_classes=num_classes)
    _set_train_mode(model, train_mode=effective_train_mode, head=model.head)

    return model
