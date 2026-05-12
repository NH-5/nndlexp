from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def macro_scores_from_report(report: dict) -> dict[str, float]:
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    return {
        "macro_precision": float(macro.get("precision", 0.0)),
        "macro_recall": float(macro.get("recall", 0.0)),
        "macro_f1": float(macro.get("f1-score", 0.0)),
        "weighted_precision": float(weighted.get("precision", 0.0)),
        "weighted_recall": float(weighted.get("recall", 0.0)),
        "weighted_f1": float(weighted.get("f1-score", 0.0)),
    }

