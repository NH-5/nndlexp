from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "nndl_exp4_matplotlib_cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp4.utils.data import CIFAR10_MEAN, CIFAR10_STD


def _prepare_output(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_loss_curve(history: list[dict], output_path: str | Path, title: str) -> None:
    output_path = _prepare_output(output_path)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(epochs, train_loss, marker="o", label="Train loss")
    plt.plot(epochs, val_loss, marker="s", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_curve(history: list[dict], output_path: str | Path, title: str) -> None:
    output_path = _prepare_output(output_path)
    epochs = [row["epoch"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(epochs, train_acc, marker="o", label="Train accuracy")
    plt.plot(epochs, val_acc, marker="s", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray | list[list[int]],
    class_names: tuple[str, ...] | list[str],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = _prepare_output(output_path)
    matrix = np.asarray(matrix)

    plt.figure(figsize=(8, 7), dpi=150)
    image = plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            plt.text(col, row, str(value), ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _unnormalize(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    mean = np.asarray(CIFAR10_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(CIFAR10_STD, dtype=np.float32).reshape(3, 1, 1)
    array = (array * std) + mean
    array = np.clip(array, 0.0, 1.0)
    return np.transpose(array, (1, 2, 0))


def plot_prediction_samples(
    samples: list[dict],
    class_names: tuple[str, ...] | list[str],
    output_path: str | Path,
    title: str,
    max_cols: int = 4,
) -> None:
    output_path = _prepare_output(output_path)
    if not samples:
        plt.figure(figsize=(6, 2), dpi=150)
        plt.text(0.5, 0.5, "No samples", ha="center", va="center")
        plt.axis("off")
        plt.savefig(output_path)
        plt.close()
        return

    cols = min(max_cols, len(samples))
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.2), dpi=150)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for axis in axes.ravel():
        axis.axis("off")

    for axis, sample in zip(axes.ravel(), samples):
        image = _unnormalize(sample["image"])
        target = class_names[sample["target"]]
        prediction = class_names[sample["prediction"]]
        confidence = sample.get("confidence", 0.0)
        color = "#137333" if sample["target"] == sample["prediction"] else "#b3261e"
        axis.imshow(image)
        axis.set_title(
            f"T: {target}\nP: {prediction} ({confidence:.2f})",
            color=color,
            fontsize=9,
        )
        axis.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_model_comparison(results: dict[str, dict], output_path: str | Path) -> None:
    output_path = _prepare_output(output_path)
    model_names = list(results.keys())
    accuracy = [results[name]["accuracy"] for name in model_names]
    macro_f1 = [results[name]["macro_f1"] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.36

    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.bar(x - width / 2, accuracy, width, label="Accuracy")
    plt.bar(x + width / 2, macro_f1, width, label="Macro F1")
    plt.xticks(x, [name.upper() for name in model_names])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("CNN vs ViT on CIFAR-10")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
