from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp4.evaluate import evaluate_model
from exp4.models import SimpleCNN, build_vit_model
from exp4.train import resolve_device, save_json, seed_everything, train_model
from exp4.utils.data import build_cifar10_dataloaders
from exp4.utils.metrics import count_parameters
from exp4.utils.plot import plot_accuracy_curve, plot_loss_curve, plot_model_comparison


RUN_CONFIG = {
    "seed": 42,
    "device": "auto",
    "data_root": PROJECT_ROOT / "exp4" / "data",
    "output_root": PROJECT_ROOT / "exp4" / "outputs",
    "image_size": 224,
    "val_fraction": 0.1,
    "num_workers": 2,
    "download": True,
    "download_timeout_seconds": 60,
    "use_tensorboard": True,
    # Keep these as None for the full CIFAR-10 experiment. Set small integers
    # here for local smoke tests without changing the no-CLI workflow.
    "train_subset": None,
    "val_subset": None,
    "test_subset": None,
}

MODEL_CONFIGS = {
    "cnn": {
        "epochs": 5,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "vit": {
        "epochs": 3,
        "batch_size": 8,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "pretrained": True,
        "train_mode": "head_only",  # "head_only" or "full"
    },
}


def _create_experiment_dirs(output_root: Path) -> dict[str, Path]:
    experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_root / experiment_name
    paths = {
        "experiment": experiment_dir,
        "figures": experiment_dir / "figures",
        "checkpoints": experiment_dir / "checkpoints",
        "logs": experiment_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_experiment.txt").write_text(str(experiment_dir.resolve()), encoding="utf-8")
    return paths


def _build_model(model_name: str) -> nn.Module:
    if model_name == "cnn":
        return SimpleCNN(num_classes=10)
    if model_name == "vit":
        config = MODEL_CONFIGS["vit"]
        return build_vit_model(
            num_classes=10,
            pretrained=config["pretrained"],
            train_mode=config["train_mode"],
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _write_comparison_csv(results: dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "parameters",
                "trainable_parameters",
                "best_val_accuracy",
            ],
        )
        writer.writeheader()
        for model_name, result in results.items():
            row = {"model": model_name, **result}
            writer.writerow(row)


def main() -> None:
    seed_everything(RUN_CONFIG["seed"])
    device = resolve_device(RUN_CONFIG["device"])
    paths = _create_experiment_dirs(Path(RUN_CONFIG["output_root"]))
    save_json(
        {
            "run_config": RUN_CONFIG,
            "model_configs": MODEL_CONFIGS,
            "device": str(device),
            "experiment_dir": str(paths["experiment"]),
        },
        paths["logs"] / "run_config.json",
    )
    print(f"Device: {device}")
    print(f"Experiment directory: {paths['experiment']}", flush=True)

    results: dict[str, dict] = {}

    for model_name in ("cnn", "vit"):
        model_config = MODEL_CONFIGS[model_name]
        print(
            f"Loading CIFAR-10 for {model_name} "
            f"from {RUN_CONFIG['data_root']} (download={RUN_CONFIG['download']})",
            flush=True,
        )
        loaders = build_cifar10_dataloaders(
            data_root=RUN_CONFIG["data_root"],
            batch_size=model_config["batch_size"],
            num_workers=RUN_CONFIG["num_workers"],
            image_size=RUN_CONFIG["image_size"],
            val_fraction=RUN_CONFIG["val_fraction"],
            seed=RUN_CONFIG["seed"],
            download=RUN_CONFIG["download"],
            pin_memory=device.type == "cuda",
            train_subset=RUN_CONFIG["train_subset"],
            val_subset=RUN_CONFIG["val_subset"],
            test_subset=RUN_CONFIG["test_subset"],
            download_timeout=RUN_CONFIG["download_timeout_seconds"],
        )
        print(
            f"{model_name} data | train={loaders.train_size} "
            f"val={loaders.val_size} test={loaders.test_size}"
        )

        model = _build_model(model_name)
        total_parameters = count_parameters(model)
        trainable_parameters = count_parameters(model, trainable_only=True)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=model_config["lr"],
            weight_decay=model_config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, model_config["epochs"]),
        )

        training = train_model(
            model=model,
            train_loader=loaders.train_loader,
            val_loader=loaders.val_loader,
            device=device,
            model_name=model_name,
            epochs=model_config["epochs"],
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=paths["checkpoints"],
            log_dir=paths["logs"],
            criterion=nn.CrossEntropyLoss(),
            use_tensorboard=RUN_CONFIG["use_tensorboard"],
        )
        plot_loss_curve(
            training.history,
            paths["figures"] / f"{model_name}_loss_curve.png",
            title=f"{model_name.upper()} loss curve",
        )
        plot_accuracy_curve(
            training.history,
            paths["figures"] / f"{model_name}_acc_curve.png",
            title=f"{model_name.upper()} accuracy curve",
        )

        evaluation = evaluate_model(
            model=model,
            data_loader=loaders.test_loader,
            device=device,
            class_names=loaders.class_names,
            model_name=model_name,
            checkpoint_path=training.best_checkpoint,
            figures_dir=paths["figures"],
            log_dir=paths["logs"],
            criterion=nn.CrossEntropyLoss(),
            seed=RUN_CONFIG["seed"],
        )
        results[model_name] = {
            "accuracy": evaluation["accuracy"],
            "macro_precision": evaluation["macro_precision"],
            "macro_recall": evaluation["macro_recall"],
            "macro_f1": evaluation["macro_f1"],
            "parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "best_val_accuracy": training.best_val_accuracy,
        }

    save_json(results, paths["logs"] / "comparison_summary.json")
    _write_comparison_csv(results, paths["logs"] / "comparison_summary.csv")
    plot_model_comparison(results, paths["figures"] / "model_comparison.png")

    print("\nSummary")
    for model_name, result in results.items():
        print(
            f"{model_name.upper()} | acc={result['accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f} params={result['parameters']:,}"
        )
    print(f"Outputs saved to: {paths['experiment']}")


if __name__ == "__main__":
    main()
