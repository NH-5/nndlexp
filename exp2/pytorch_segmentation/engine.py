from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from exp2.pytorch_segmentation.metrics import SegmentationMetric
from exp2.pytorch_segmentation.utils import AverageMeter


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 20,
    logger: Optional[logging.Logger] = None,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    amp_enabled = scaler is not None and device.type == "cuda"
    logger = logger or logging.getLogger(__name__)

    for step, (images, masks, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)

        if step % log_interval == 0 or step == len(loader):
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:02d} | Step {step:04d}/{len(loader):04d} "
                f"| lr={lr:.6f} | loss={loss_meter.average:.4f}"
            )

    return {"loss": loss_meter.average}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> dict[str, float | list[float]]:
    model.eval()
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)
    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")

        outputs = model(images)["out"]
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(
                outputs,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = criterion(outputs, masks)
        predictions = outputs.argmax(dim=1)
        metric.update(predictions, masks)
        loss_meter.update(loss.item(), images.size(0))

    scores = metric.compute()
    scores["loss"] = loss_meter.average
    return scores
