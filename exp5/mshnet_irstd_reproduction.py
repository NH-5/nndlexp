#!/usr/bin/env python
# coding: utf-8

# # MSHNet 论文复现实验
#
# 复现论文：Qiankun Liu 等，**Infrared Small Target Detection with Scale and Location Sensitivity**, CVPR 2024。
#
# 本 notebook 按论文与官方实现整理为一个自包含版本：
#
# - 复现 MSHNet：U-Net 编码解码器 + 4 个多尺度预测头 + 最终融合头。
# - 复现 SLS loss：scale-sensitive IoU 项 + location-sensitive 中心点惩罚项。
# - 提供 IRSTD-1k / NUDT-SIRST 风格的数据读取、训练、验证、指标和可视化。
# - 如果本地没有真实红外小目标数据，会自动使用合成小目标数据跑通 smoke test，方便检查代码是否可执行。

# ## 1. 环境与依赖
#
# 项目 `pyproject.toml` 已声明 `torch`、`torchvision`、`opencv-python`、`matplotlib`、`numpy`、`tqdm` 等依赖。真实训练建议使用 GPU；论文设置为输入 `256 x 256`、batch size 4、AdaGrad、学习率 0.05、训练 400 epochs。

# In[ ]:


import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mshnet_mpl_cache"))
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adagrad
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import cv2
except Exception:
    cv2 = None

PROJECT_ROOT = Path.cwd()
EXP_ROOT = PROJECT_ROOT / "exp5" if (PROJECT_ROOT / "exp5").exists() else PROJECT_ROOT
OUTPUT_ROOT = EXP_ROOT / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Experiment root: {EXP_ROOT}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# ## 2. 实验配置
#
# 真实数据按整理脚本输出的结构组织：
#
# ```text
# exp5/data/IRSTD-1k/
#   images/
#   masks/
#   trainval.txt
#   test.txt
#
# exp5/data/NUDT-SIRST/
#   images/
#   masks/
#   trainval.txt
#   test.txt
# ```
#
# `dataset_runs` 控制训练时要依次跑哪些数据集。默认会从头训练两个独立模型：先跑 `IRSTD-1k`，再跑 `NUDT-SIRST`。

# In[ ]:


@dataclass
class ExperimentConfig:
    seed: int = 42
    dataset_dir: Path = EXP_ROOT / "data" / "IRSTD-1k"
    dataset_runs: Tuple[Tuple[str, Path], ...] = (
        ("IRSTD-1k", EXP_ROOT / "data" / "IRSTD-1k"),
        ("NUDT-SIRST", EXP_ROOT / "data" / "NUDT-SIRST"),
    )
    output_root: Path = OUTPUT_ROOT
    image_size: int = 256
    crop_size: int = 256
    batch_size: int = 4
    num_workers: int = 0
    epochs: int = 400
    full_paper_epochs: int = 400
    warmup_epochs: int = 5
    lr: float = 0.05
    threshold: float = 0.5
    synthetic_train_size: int = 16
    synthetic_val_size: int = 4
    use_synthetic_if_missing: bool = True
    run_smoke_train: bool = True
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    save_checkpoints: bool = True
    save_experiment_logs: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = ExperimentConfig()
print(asdict(CFG))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(CFG.seed)


# ## 3. 数据集读取与合成 smoke-test 数据
#
# 论文使用 IRSTD-1k 与 NUDT-SIRST。这里的数据集类兼容官方 `images/ masks/ trainval.txt test.txt` 结构，并复现随机翻转、随机尺度裁剪、Gaussian blur、验证集 resize 到 `256 x 256` 的流程。没有真实数据时，合成数据会生成暗背景 + 微弱小亮点 + 二值 mask，用于验证训练链路。

# In[ ]:


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask.convert("L"), dtype=np.float32)
    arr = (arr > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def resize_pair(image: Image.Image, mask: Image.Image, size: int) -> Tuple[Image.Image, Image.Image]:
    return image.resize((size, size), Image.BILINEAR), mask.resize((size, size), Image.NEAREST)


class IRSTDDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, image_size: int = 256, crop_size: int = 256):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        self.crop_size = crop_size
        split_file = "trainval.txt" if split == "train" else "test.txt"
        self.list_path = self.dataset_dir / split_file
        self.image_dir = self.dataset_dir / "images"
        self.mask_dir = self.dataset_dir / "masks"
        if not self.list_path.exists():
            raise FileNotFoundError(f"Missing split file: {self.list_path}")
        with self.list_path.open("r", encoding="utf-8") as f:
            self.names = [line.strip() for line in f if line.strip()]
        if not self.names:
            raise ValueError(f"No samples listed in {self.list_path}")

    def __len__(self) -> int:
        return len(self.names)

    def _resolve_png(self, folder: Path, name: str) -> Path:
        p = Path(name)
        candidates = [folder / p.name]
        if p.suffix == "":
            candidates.append(folder / f"{p.name}.png")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Cannot find {name} in {folder}")

    def _train_transform(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        long_size = random.randint(int(self.image_size * 0.5), int(self.image_size * 2.0))
        w, h = image.size
        if h > w:
            oh = long_size
            ow = max(1, int(w * long_size / h + 0.5))
            short_size = ow
        else:
            ow = long_size
            oh = max(1, int(h * long_size / w + 0.5))
            short_size = oh
        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < self.crop_size:
            pad_w = max(0, self.crop_size - ow)
            pad_h = max(0, self.crop_size - oh)
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)

        w, h = image.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]
        image = Image.open(self._resolve_png(self.image_dir, name)).convert("RGB")
        mask = Image.open(self._resolve_png(self.mask_dir, name)).convert("L")
        if self.split == "train":
            image, mask = self._train_transform(image, mask)
        else:
            image, mask = resize_pair(image, mask, self.image_size)
        return image_to_tensor(image), mask_to_tensor(mask)


class SyntheticInfraredSmallTargetDataset(Dataset):
    def __init__(self, length: int, image_size: int = 256, seed: int = 42):
        self.length = length
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + idx)
        h = w = self.image_size
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        background = rng.normal(0.20, 0.035, size=(h, w)).astype(np.float32)
        background += np.linspace(0.0, 0.08, w, dtype=np.float32)[None, :]
        background += rng.normal(0.0, 0.012, size=(h, w)).astype(np.float32)
        mask = np.zeros((h, w), dtype=np.float32)

        target_count = int(rng.integers(1, 4))
        for _ in range(target_count):
            cx = float(rng.integers(16, w - 16))
            cy = float(rng.integers(16, h - 16))
            radius = float(rng.uniform(1.5, 4.5))
            amp = float(rng.uniform(0.45, 0.85))
            dist2 = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
            spot = amp * np.exp(-dist2 / (2.0 * radius ** 2))
            background += spot.astype(np.float32)
            mask[dist2 <= (radius * 1.4) ** 2] = 1.0

        image = np.clip(background, 0.0, 1.0)
        image_rgb = np.repeat(image[..., None], 3, axis=2)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return image_tensor, mask_tensor


def real_dataset_available(dataset_dir: Path) -> bool:
    dataset_dir = Path(dataset_dir)
    return (
        (dataset_dir / "images").exists()
        and (dataset_dir / "masks").exists()
        and (dataset_dir / "trainval.txt").exists()
        and (dataset_dir / "test.txt").exists()
    )


def build_loaders(cfg: ExperimentConfig) -> Tuple[DataLoader, DataLoader, str]:
    if real_dataset_available(cfg.dataset_dir):
        train_set = IRSTDDataset(cfg.dataset_dir, "train", cfg.image_size, cfg.crop_size)
        val_set = IRSTDDataset(cfg.dataset_dir, "val", cfg.image_size, cfg.crop_size)
        data_source = f"real dataset: {cfg.dataset_dir}"
    elif cfg.use_synthetic_if_missing:
        train_set = SyntheticInfraredSmallTargetDataset(cfg.synthetic_train_size, cfg.image_size, cfg.seed)
        val_set = SyntheticInfraredSmallTargetDataset(cfg.synthetic_val_size, cfg.image_size, cfg.seed + 10000)
        data_source = "synthetic smoke-test dataset"
    else:
        raise FileNotFoundError(f"Dataset not found: {cfg.dataset_dir}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, data_source

train_loader, val_loader, data_source = build_loaders(CFG)
print(f"Using {data_source}")
print(f"Train batches: {len(train_loader)}, val samples: {len(val_loader.dataset)}")


# ## 4. 可视化一个 batch
#
# 先检查图像和 mask 是否对齐。真实 IRSTD 图像通常是低对比灰度图，这里经过 ImageNet 均值方差归一化，显示时会反归一化。

# In[ ]:


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    x = tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return x


def show_batch(images: torch.Tensor, masks: torch.Tensor, max_items: int = 4) -> None:
    n = min(max_items, images.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(7, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n):
        axes[i, 0].imshow(denormalize_image(images[i]))
        axes[i, 0].set_title("image")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i, 0].cpu(), cmap="gray")
        axes[i, 1].set_title("mask")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()

sample_images, sample_masks = next(iter(train_loader))
show_batch(sample_images, sample_masks)


# ## 5. MSHNet 模型
#
# 论文的 MSHNet 思路是：普通 U-Net 解码器产生 4 个尺度的特征图，每个尺度接一个 `3 x 3` 预测头；低分辨率预测上采样后与高分辨率预测拼接，再通过最终 `3 x 3` 卷积得到最终预测。
#
# 这里保留 logits 形式，不在模型内部做 sigmoid，便于使用 `BCEWithLogits` 风格和数值稳定的 loss。训练时返回所有辅助尺度预测，推理时也可以只取 `final`。

# In[ ]:


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MSHNet(nn.Module):
    def __init__(self, in_channels: int = 3, channels: Sequence[int] = (16, 32, 64, 128, 256)):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.pool = nn.MaxPool2d(2, 2)

        self.enc0 = DoubleConv(in_channels, c0)
        self.enc1 = DoubleConv(c0, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.middle = DoubleConv(c3, c4)

        self.dec3 = DoubleConv(c3 + c4, c3)
        self.dec2 = DoubleConv(c2 + c3, c2)
        self.dec1 = DoubleConv(c1 + c2, c1)
        self.dec0 = DoubleConv(c0 + c1, c0)

        self.head0 = nn.Conv2d(c0, 1, kernel_size=3, padding=1)
        self.head1 = nn.Conv2d(c1, 1, kernel_size=3, padding=1)
        self.head2 = nn.Conv2d(c2, 1, kernel_size=3, padding=1)
        self.head3 = nn.Conv2d(c3, 1, kernel_size=3, padding=1)
        self.final_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def _up_like(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        mid = self.middle(self.pool(e3))

        d3 = self.dec3(torch.cat([e3, self._up_like(mid, e3)], dim=1))
        d2 = self.dec2(torch.cat([e2, self._up_like(d3, e2)], dim=1))
        d1 = self.dec1(torch.cat([e1, self._up_like(d2, e1)], dim=1))
        d0 = self.dec0(torch.cat([e0, self._up_like(d1, e0)], dim=1))

        p4 = self.head0(d0)  # H x W
        p3 = self.head1(d1)  # H/2 x W/2
        p2 = self.head2(d2)  # H/4 x W/4
        p1 = self.head3(d3)  # H/8 x W/8

        final = self.final_head(torch.cat([
            p4,
            self._up_like(p3, p4),
            self._up_like(p2, p4),
            self._up_like(p1, p4),
        ], dim=1))
        return {"aux": [p4, p3, p2, p1], "final": final}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = MSHNet().to(CFG.device)
with torch.no_grad():
    outputs = model(sample_images[:1].to(CFG.device))
print(f"Trainable parameters: {count_parameters(model):,}")
print("Final shape:", tuple(outputs["final"].shape))
print("Aux shapes:", [tuple(x.shape) for x in outputs["aux"]])


# ## 6. SLS Loss
#
# 论文定义：
#
# - `L_S = 1 - w * IoU`，其中 `w = (min(|Ap|, |Agt|) + Var(|Ap|, |Agt|)) / (max(|Ap|, |Agt|) + Var(|Ap|, |Agt|))`。
# - `L_L = (1 - min(d_p, d_gt) / max(d_p, d_gt)) + 4 / pi^2 * (theta_p - theta_gt)^2`。
# - `L_SLS = L_S + L_L`。
#
# 在 soft segmentation 训练中，`Ap` 用 sigmoid 概率图近似，中心点用概率加权坐标计算。多尺度监督时，`p4, p3, p2, p1` 分别对应原尺度、1/2、1/4、1/8；mask 通过 max pooling 下采样以保留小目标。

# In[ ]:


class SLSLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, warmup_epochs: int = 5, use_location: bool = True):
        super().__init__()
        self.eps = eps
        self.warmup_epochs = warmup_epochs
        self.use_location = use_location

    def _scale_sensitive_iou_loss(self, probs: torch.Tensor, target: torch.Tensor, epoch: int) -> torch.Tensor:
        dims = (1, 2, 3)
        intersection = (probs * target).sum(dim=dims)
        pred_area = probs.sum(dim=dims)
        target_area = target.sum(dim=dims)
        union = pred_area + target_area - intersection
        iou = (intersection + self.eps) / (union + self.eps)

        if epoch <= self.warmup_epochs:
            return 1.0 - iou.mean()

        mean_area = 0.5 * (pred_area + target_area)
        variance = 0.5 * ((pred_area - mean_area) ** 2 + (target_area - mean_area) ** 2)
        weight = (torch.min(pred_area, target_area) + variance + self.eps) / (
            torch.max(pred_area, target_area) + variance + self.eps
        )
        return 1.0 - (weight * iou).mean()

    def _centers(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = probs.shape
        y_coords = torch.linspace(0, 1, h, device=probs.device, dtype=probs.dtype)
        x_coords = torch.linspace(0, 1, w, device=probs.device, dtype=probs.dtype)
        y, x = torch.meshgrid(y_coords, x_coords)
        mass = probs.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        center_x = (probs[:, 0] * x).sum(dim=(1, 2)) / mass
        center_y = (probs[:, 0] * y).sum(dim=(1, 2)) / mass
        return center_x, center_y

    def _location_loss(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_area = probs.sum(dim=(1, 2, 3))
        target_area = target.sum(dim=(1, 2, 3))
        valid = target_area > self.eps
        if valid.sum() == 0:
            return probs.sum() * 0.0

        px, py = self._centers(probs)
        gx, gy = self._centers(target)
        px, py, gx, gy = px[valid], py[valid], gx[valid], gy[valid]

        pred_dist = torch.sqrt(px ** 2 + py ** 2 + self.eps)
        gt_dist = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)
        length_loss = 1.0 - torch.min(pred_dist, gt_dist) / torch.max(pred_dist, gt_dist).clamp_min(self.eps)

        pred_theta = torch.atan2(py, px.clamp_min(self.eps))
        gt_theta = torch.atan2(gy, gx.clamp_min(self.eps))
        angle_loss = (4.0 / math.pi ** 2) * (pred_theta - gt_theta) ** 2
        return (length_loss + angle_loss).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        scale_loss = self._scale_sensitive_iou_loss(probs, target, epoch)
        if self.use_location and epoch > self.warmup_epochs:
            return scale_loss + self._location_loss(probs, target)
        return scale_loss


class MultiScaleSLSLoss(nn.Module):
    def __init__(self, warmup_epochs: int = 5):
        super().__init__()
        self.sls = SLSLoss(warmup_epochs=warmup_epochs)

    def forward(self, outputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]], target: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        losses = [self.sls(outputs["final"], target, epoch)]
        current_target = target
        for idx, logits in enumerate(outputs["aux"]):
            if idx == 0:
                current_target = target
            else:
                current_target = F.max_pool2d(current_target, kernel_size=2, stride=2)
            losses.append(self.sls(logits, current_target, epoch))
        return torch.stack(losses).mean()

criterion = MultiScaleSLSLoss(warmup_epochs=CFG.warmup_epochs)
loss_value = criterion(outputs, sample_masks[:1].to(CFG.device), epoch=CFG.warmup_epochs + 1)
print(f"SLS loss smoke value: {float(loss_value.detach().cpu()):.4f}")


# ## 7. 指标：IoU、Pd、Fa
#
# 论文采用像素级 IoU、目标级检测概率 `Pd` 和虚警率 `Fa`。`Pd` 的判断规则沿用常见 IRSTD 做法：预测连通域中心与真实连通域中心距离小于 3 像素则认为该目标被检出。

# In[ ]:


def binary_components(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if cv2 is not None:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        centers = centroids[1:].astype(np.float32)[:, ::-1]  # y, x
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        return centers, areas

    visited = np.zeros_like(mask, dtype=bool)
    centers: List[Tuple[float, float]] = []
    areas: List[int] = []
    h, w = mask.shape
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y0 in range(h):
        for x0 in range(w):
            if mask[y0, x0] == 0 or visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            coords = []
            while stack:
                y, x = stack.pop()
                coords.append((y, x))
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            arr = np.asarray(coords, dtype=np.float32)
            centers.append(tuple(arr.mean(axis=0)))
            areas.append(len(coords))
    return np.asarray(centers, dtype=np.float32), np.asarray(areas, dtype=np.float32)


class IRSTDMetrics:
    def __init__(self, threshold: float = 0.5, match_distance: float = 3.0):
        self.threshold = threshold
        self.match_distance = match_distance
        self.reset()

    def reset(self) -> None:
        self.intersection = 0.0
        self.union = 0.0
        self.false_pixels = 0.0
        self.total_pixels = 0.0
        self.detected_targets = 0.0
        self.total_targets = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = target.detach().cpu().numpy()
        pred_bin = probs[:, 0] >= self.threshold
        label_bin = labels[:, 0] > 0.5

        self.intersection += np.logical_and(pred_bin, label_bin).sum()
        self.union += np.logical_or(pred_bin, label_bin).sum()
        self.false_pixels += np.logical_and(pred_bin, np.logical_not(label_bin)).sum()
        self.total_pixels += np.prod(label_bin.shape)

        for pred_mask, gt_mask in zip(pred_bin, label_bin):
            pred_centers, pred_areas = binary_components(pred_mask)
            gt_centers, _ = binary_components(gt_mask)
            self.total_targets += len(gt_centers)
            used_pred = set()
            for gt_center in gt_centers:
                if len(pred_centers) == 0:
                    continue
                distances = np.linalg.norm(pred_centers - gt_center[None, :], axis=1)
                order = np.argsort(distances)
                for pred_idx in order:
                    if pred_idx not in used_pred and distances[pred_idx] < self.match_distance:
                        used_pred.add(int(pred_idx))
                        self.detected_targets += 1
                        break

    def compute(self) -> Dict[str, float]:
        return {
            "IoU": self.intersection / max(self.union, 1.0),
            "Pd": self.detected_targets / max(self.total_targets, 1.0),
            "Fa": self.false_pixels / max(self.total_pixels, 1.0),
        }


# ## 8. 训练与验证循环
#
# 默认会按 `CFG.dataset_runs` 依次训练 `IRSTD-1k` 和 `NUDT-SIRST`，每个数据集都重新初始化一个 MSHNet，checkpoint 和实验记录分别保存到 `exp5/outputs/<dataset>/`。每个 epoch 会写入 `history.csv` / `history.json`，并在 `exp5/outputs/summary.csv` 汇总两个数据集。要临时快速检查，可以把 `CFG.epochs` 改小、设置 `CFG.max_train_batches = 2`、`CFG.max_eval_batches = 5`、并关闭 `CFG.save_checkpoints`。论文报告的设置是 `batch_size=4`、`lr=0.05`、`AdaGrad`、`400 epochs`。

# In[ ]:


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    max_batches: Optional[int] = None,
    dataset_name: str = "dataset",
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    iterator = tqdm(loader, desc=f"{dataset_name} train epoch {epoch}", leave=False)
    for batch_idx, (images, masks) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, masks, epoch=epoch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        iterator.set_postfix(loss=total_loss / max(total_items, 1))
        if max_batches is not None and batch_idx >= max_batches:
            break
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
    dataset_name: str = "dataset",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    metrics = IRSTDMetrics(threshold=threshold)
    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc=f"{dataset_name} evaluate", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images)
        metrics.update(outputs["final"], masks)
        if max_batches is not None and batch_idx >= max_batches:
            break
    return metrics.compute()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def save_checkpoint(model: nn.Module, cfg: ExperimentConfig, dataset_name: str, metrics: Dict[str, float], epoch: int) -> Path:
    output_dir = cfg.output_root / safe_name(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"mshnet_epoch{epoch:03d}_iou{metrics['IoU']:.4f}.pt"
    torch.save({
        "dataset": dataset_name,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "metrics": metrics,
        "config": asdict(cfg),
    }, ckpt_path)
    return ckpt_path


def to_serializable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_history_csv(path: Path, history: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "epoch", "loss", "IoU", "Pd", "Fa", "Fa_x1e6"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            csv_row = {key: row.get(key, "") for key in fieldnames}
            csv_row["Fa_x1e6"] = float(row.get("Fa", 0.0)) * 1_000_000
            writer.writerow(csv_row)


def best_history_row(history: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not history:
        return None
    return max(history, key=lambda row: float(row.get("IoU", 0.0)))


def save_experiment_records(
    cfg: ExperimentConfig,
    dataset_name: str,
    history: List[Dict[str, float]],
    best_path: Optional[Path],
    data_source: str,
) -> None:
    output_dir = cfg.output_root / safe_name(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    write_json(output_dir / "config.json", {
        "dataset": dataset_name,
        "data_source": data_source,
        "created_or_updated_at": now,
        "config": asdict(cfg),
    })
    write_history_csv(output_dir / "history.csv", history)
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "latest_metrics.json", {
        "dataset": dataset_name,
        "data_source": data_source,
        "updated_at": now,
        "epochs_recorded": len(history),
        "latest": history[-1] if history else None,
        "best": best_history_row(history),
        "best_checkpoint": best_path,
    })


def save_all_results_summary(training_results: Dict[str, Dict[str, object]], cfg: ExperimentConfig) -> None:
    rows: List[Dict[str, object]] = []
    for dataset_name, result in training_results.items():
        dataset_history = result["history"]
        if not dataset_history:
            continue
        latest = dataset_history[-1]
        best = best_history_row(dataset_history)
        rows.append({
            "dataset": dataset_name,
            "epochs_recorded": len(dataset_history),
            "latest_epoch": latest["epoch"],
            "latest_loss": latest["loss"],
            "latest_IoU": latest["IoU"],
            "latest_Pd": latest["Pd"],
            "latest_Fa": latest["Fa"],
            "latest_Fa_x1e6": latest["Fa"] * 1_000_000,
            "best_epoch": best["epoch"] if best else "",
            "best_IoU": best["IoU"] if best else "",
            "best_Pd": best["Pd"] if best else "",
            "best_Fa": best["Fa"] if best else "",
            "best_Fa_x1e6": best["Fa"] * 1_000_000 if best else "",
            "best_checkpoint": result.get("best_path"),
            "data_source": result.get("data_source", ""),
        })

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = cfg.output_root / "summary.csv"
    fieldnames = [
        "dataset", "epochs_recorded", "latest_epoch", "latest_loss", "latest_IoU", "latest_Pd",
        "latest_Fa", "latest_Fa_x1e6", "best_epoch", "best_IoU", "best_Pd", "best_Fa",
        "best_Fa_x1e6", "best_checkpoint", "data_source",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_serializable(row.get(key, "")) for key in fieldnames})
    write_json(cfg.output_root / "summary.json", rows)
    print(f"Experiment summary saved to {summary_csv}")


def run_training_for_dataset(dataset_name: str, dataset_dir: Path, base_cfg: ExperimentConfig) -> Dict[str, object]:
    print(f"\n===== Training {dataset_name} =====")
    cfg = replace(base_cfg, dataset_dir=Path(dataset_dir))
    seed_everything(cfg.seed)
    train_loader, val_loader, data_source = build_loaders(cfg)
    print(f"Using {data_source}")
    print(f"Train batches: {len(train_loader)}, val samples: {len(val_loader.dataset)}")

    model = MSHNet().to(cfg.device)
    optimizer = Adagrad(model.parameters(), lr=cfg.lr)
    criterion = MultiScaleSLSLoss(warmup_epochs=cfg.warmup_epochs)

    history: List[Dict[str, float]] = []
    best_iou = -1.0
    best_path = None

    if cfg.run_smoke_train:
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                epoch=epoch,
                max_batches=cfg.max_train_batches,
                dataset_name=dataset_name,
            )
            metrics = evaluate(
                model,
                val_loader,
                cfg.device,
                threshold=cfg.threshold,
                dataset_name=dataset_name,
                max_batches=cfg.max_eval_batches,
            )
            row = {"dataset": dataset_name, "epoch": epoch, "loss": train_loss, **metrics}
            history.append(row)
            print(row)
            if metrics["IoU"] > best_iou:
                best_iou = metrics["IoU"]
                if cfg.save_checkpoints:
                    best_path = save_checkpoint(model, cfg, dataset_name, metrics, epoch)
            if cfg.save_experiment_logs:
                save_experiment_records(cfg, dataset_name, history, best_path, data_source)
        if cfg.save_experiment_logs:
            save_experiment_records(cfg, dataset_name, history, best_path, data_source)
        if cfg.save_checkpoints:
            print(f"Best checkpoint for {dataset_name}: {best_path}")
        else:
            print(f"Checkpoint saving is disabled for {dataset_name}.")
    else:
        print(f"CFG.run_smoke_train is False; skip training {dataset_name}.")
        if cfg.save_experiment_logs:
            save_experiment_records(cfg, dataset_name, history, best_path, data_source)

    model = model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "cfg": cfg,
        "history": history,
        "best_path": best_path,
        "model": model,
        "val_loader": val_loader,
        "data_source": data_source,
    }


training_results: Dict[str, Dict[str, object]] = {}

for dataset_name, dataset_dir in CFG.dataset_runs:
    training_results[dataset_name] = run_training_for_dataset(dataset_name, dataset_dir, CFG)

history: List[Dict[str, float]] = [
    row
    for result in training_results.values()
    for row in result["history"]
]

if CFG.save_experiment_logs:
    save_all_results_summary(training_results, CFG)

if training_results:
    first_result = next(iter(training_results.values()))
    model = first_result["model"]
    val_loader = first_result["val_loader"]


# ## 9. 结果曲线与预测可视化
#
# 下面会分别展示每个数据集的训练曲线和预测效果。真实训练时重点看 IoU、Pd、Fa；论文表格中的 `Fa` 单位常写成 `x10^-6`，这里也输出 `Fa_x1e6`。完整实验记录同时保存在 `exp5/outputs/<dataset>/history.csv` 和 `exp5/outputs/summary.csv`。

# In[ ]:


def plot_history(history: List[Dict[str, float]], title: str = "training") -> None:
    if not history:
        print(f"No history to plot for {title}.")
        return
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    keys = ["loss", "IoU", "Pd", "Fa"]
    for ax, key in zip(axes, keys):
        values = [r[key] for r in history]
        ax.plot(epochs, values, marker="o")
        ax.set_title(f"{title} {key}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_predictions(model: nn.Module, loader: DataLoader, device: str, threshold: float = 0.5, max_items: int = 4, title: str = "prediction") -> None:
    model = model.to(device)
    model.eval()
    images, masks = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(images)["final"]).cpu()
    n = min(max_items, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(title)
    for i in range(n):
        axes[i, 0].imshow(denormalize_image(images[i].cpu()))
        axes[i, 0].set_title("image")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i, 0].cpu(), cmap="gray")
        axes[i, 1].set_title("ground truth")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(probs[i, 0], cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title("probability")
        axes[i, 2].axis("off")
        axes[i, 3].imshow(probs[i, 0] >= threshold, cmap="gray")
        axes[i, 3].set_title("prediction")
        axes[i, 3].axis("off")
    plt.tight_layout()
    plt.show()
    model.to("cpu")


for dataset_name, result in training_results.items():
    dataset_history = result["history"]
    plot_history(dataset_history, title=dataset_name)
    if dataset_history:
        last = dataset_history[-1]
        print({**last, "Fa_x1e6": last["Fa"] * 1_000_000})
    show_predictions(
        result["model"],
        result["val_loader"],
        CFG.device,
        threshold=result["cfg"].threshold,
        title=f"{dataset_name} predictions",
    )


# ## 10. 真实数据复现实验建议
#
# 用真实 IRSTD-1k 和 NUDT-SIRST 复现论文结果时，建议按下面步骤执行：
#
# 1. 运行 `download_prepare_irstd.py`，确认 `exp5/data/IRSTD-1k` 和 `exp5/data/NUDT-SIRST` 都包含 `images/ masks/ trainval.txt test.txt`。
# 2. 保持 `CFG.dataset_runs` 同时包含两个数据集；如只想跑其中一个，就删掉另一个 tuple。
# 3. 设置 `CFG.epochs = CFG.full_paper_epochs`，`CFG.max_train_batches = None`，`CFG.max_eval_batches = None`，`CFG.save_checkpoints = True`。
# 4. 使用 GPU 运行完整训练；checkpoint 会分别保存到 `exp5/outputs/IRSTD-1k/` 和 `exp5/outputs/NUDT-SIRST/`。
# 5. 若要和论文表 1 对齐，IRSTD-1k 期望量级约为 IoU 67%、Pd 94%、Fa 15e-6；NUDT-SIRST 期望量级约为 IoU 81%、Pd 98%、Fa 12e-6。实际结果会受数据划分、实现细节、随机种子和训练轮数影响。
