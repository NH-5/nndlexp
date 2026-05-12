from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, float]]
    best_checkpoint: Path
    last_checkpoint: Path
    history_csv: Path
    best_val_accuracy: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def save_json(data: dict, output_path: str | Path) -> None:
    def make_json_ready(value):
        if isinstance(value, dict):
            return {key: make_json_ready(item) for key, item in value.items()}
        if isinstance(value, list):
            return [make_json_ready(item) for item in value]
        if isinstance(value, tuple):
            return [make_json_ready(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(make_json_ready(data), file, indent=2, ensure_ascii=False)


def _write_history_csv(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
        "epoch_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _make_summary_writer(log_dir: Path, model_name: str, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError):
        return None
    return SummaryWriter(log_dir=log_dir / "tensorboard" / model_name)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    epoch: int,
    training: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    phase = "train" if training else "val"
    progress = tqdm(loader, desc=f"{model_name} {phase} epoch {epoch}", leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        if training:
            if optimizer is None:
                raise ValueError("optimizer is required for training")
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += batch_size
        progress.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_name: str,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | Path,
    log_dir: str | Path,
    criterion: nn.Module | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_tensorboard: bool = True,
) -> TrainingResult:
    """Train a classifier, log history, and save best/last checkpoints."""

    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    criterion = criterion or nn.CrossEntropyLoss()
    model = model.to(device)

    best_checkpoint = checkpoint_dir / f"{model_name}_best.pth"
    last_checkpoint = checkpoint_dir / f"{model_name}_last.pth"
    history_csv = log_dir / f"{model_name}_history.csv"
    writer = _make_summary_writer(log_dir=log_dir, model_name=model_name, enabled=use_tensorboard)

    best_val_accuracy = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            model_name=model_name,
            epoch=epoch,
            training=True,
            optimizer=optimizer,
        )
        val_stats = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            model_name=model_name,
            epoch=epoch,
            training=False,
        )
        if scheduler is not None:
            scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_acc": float(train_stats["accuracy"]),
            "val_loss": float(val_stats["loss"]),
            "val_acc": float(val_stats["accuracy"]),
            "lr": float(lr),
            "epoch_seconds": float(epoch_seconds),
        }
        history.append(row)
        _write_history_csv(history, history_csv)

        is_best = row["val_acc"] > best_val_accuracy
        if is_best:
            best_val_accuracy = row["val_acc"]
        checkpoint = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "history": history,
        }
        torch.save(checkpoint, last_checkpoint)
        if is_best:
            torch.save(checkpoint, best_checkpoint)

        if writer is not None:
            writer.add_scalar("loss/train", row["train_loss"], epoch)
            writer.add_scalar("loss/val", row["val_loss"], epoch)
            writer.add_scalar("accuracy/train", row["train_acc"], epoch)
            writer.add_scalar("accuracy/val", row["val_acc"], epoch)
            writer.add_scalar("lr", row["lr"], epoch)

        print(
            f"{model_name} epoch {epoch:02d}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f}"
        )

    if writer is not None:
        writer.close()

    save_json(
        {
            "model_name": model_name,
            "epochs": epochs,
            "best_val_accuracy": best_val_accuracy,
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "history_csv": str(history_csv),
        },
        log_dir / f"{model_name}_train_summary.json",
    )

    return TrainingResult(
        history=history,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        history_csv=history_csv,
        best_val_accuracy=best_val_accuracy,
    )
