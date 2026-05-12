from __future__ import annotations

import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from exp4.utils.metrics import macro_scores_from_report
from exp4.utils.plot import plot_confusion_matrix, plot_prediction_samples


def _load_torch_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=map_location, weights_only=False)


def load_model_weights(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = _load_torch_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_confusion_matrix_csv(
    matrix: np.ndarray,
    class_names: tuple[str, ...] | list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])


def _maybe_store_sample(
    reservoir: list[dict],
    candidate: dict,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(candidate)
        return
    replacement = rng.randint(0, seen - 1)
    if replacement < limit:
        reservoir[replacement] = candidate


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: tuple[str, ...] | list[str],
    model_name: str,
    checkpoint_path: str | Path | None = None,
    figures_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    criterion: nn.Module | None = None,
    num_samples: int = 12,
    num_wrong_samples: int = 12,
    seed: int = 42,
) -> dict:
    """Evaluate a classifier and save metrics, confusion matrix, and samples."""

    if checkpoint_path is not None:
        load_model_weights(model, checkpoint_path=checkpoint_path, device=device)
    model.to(device)
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    rng = random.Random(seed)

    all_targets: list[int] = []
    all_predictions: list[int] = []
    total_loss = 0.0
    total = 0
    correct = 0
    samples: list[dict] = []
    wrong_samples: list[dict] = []
    seen = 0

    for images, targets in tqdm(data_loader, desc=f"{model_name} test", leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        loss = criterion(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == targets).sum().item()
        total += batch_size

        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

        images_cpu = images.cpu()
        targets_cpu = targets.cpu()
        predictions_cpu = predictions.cpu()
        confidences_cpu = confidences.cpu()
        for index in range(batch_size):
            seen += 1
            candidate = {
                "image": images_cpu[index],
                "target": int(targets_cpu[index]),
                "prediction": int(predictions_cpu[index]),
                "confidence": float(confidences_cpu[index]),
            }
            _maybe_store_sample(samples, candidate, seen, num_samples, rng)
            if candidate["target"] != candidate["prediction"] and len(wrong_samples) < num_wrong_samples:
                wrong_samples.append(candidate)

    labels = list(range(len(class_names)))
    report_dict = classification_report(
        all_targets,
        all_predictions,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        all_targets,
        all_predictions,
        labels=labels,
        target_names=list(class_names),
        zero_division=0,
    )
    matrix = confusion_matrix(all_targets, all_predictions, labels=labels)
    accuracy = correct / max(total, 1)
    summary = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "test_loss": total_loss / max(total, 1),
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        **macro_scores_from_report(report_dict),
    }

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        plot_confusion_matrix(
            matrix,
            class_names=class_names,
            output_path=figures_dir / f"{model_name}_confusion_matrix.png",
            title=f"{model_name.upper()} confusion matrix",
        )
        plot_prediction_samples(
            samples,
            class_names=class_names,
            output_path=figures_dir / f"{model_name}_prediction_samples.png",
            title=f"{model_name.upper()} prediction samples",
        )
        if wrong_samples:
            plot_prediction_samples(
                wrong_samples,
                class_names=class_names,
                output_path=figures_dir / f"{model_name}_wrong_samples.png",
                title=f"{model_name.upper()} misclassified samples",
            )

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"{model_name}_classification_report.txt").open("w", encoding="utf-8") as file:
            file.write(report_text)
            file.write("\n")
        with (log_dir / f"{model_name}_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(_json_ready(summary), file, indent=2, ensure_ascii=False)
        _write_confusion_matrix_csv(
            matrix=matrix,
            class_names=class_names,
            output_path=log_dir / f"{model_name}_confusion_matrix.csv",
        )

    print(f"{model_name} test accuracy: {accuracy:.4f}")
    print(report_text)
    return summary

