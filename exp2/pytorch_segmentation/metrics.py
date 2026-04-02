from __future__ import annotations

import torch


class SegmentationMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.view(-1).cpu()
        target = target.view(-1).cpu()

        valid = target != self.ignore_index
        prediction = prediction[valid]
        target = target[valid]
        if prediction.numel() == 0:
            return

        indices = self.num_classes * target + prediction
        bins = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += bins.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float | list[float]]:
        hist = self.confusion_matrix
        diagonal = torch.diag(hist)
        union = hist.sum(dim=1) + hist.sum(dim=0) - diagonal

        pixel_acc = (diagonal.sum() / hist.sum()).item() if hist.sum() > 0 else 0.0
        per_class_iou = torch.where(union > 0, diagonal / union, torch.zeros_like(union))
        mean_iou = per_class_iou.mean().item()

        return {
            "pixel_accuracy": pixel_acc,
            "mean_iou": mean_iou,
            "per_class_iou": per_class_iou.tolist(),
        }
