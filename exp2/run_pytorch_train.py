from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp2.pytorch_segmentation.train import run_training


CONFIG = {
    "data_root": PROJECT_ROOT / "exp2" / "VOC2012",
    "output_dir": PROJECT_ROOT / "exp2" / "outputs" / "deeplabv3_resnet50",
    "train_split": "train",
    "val_split": "val",
    "epochs": 10,
    "batch_size": 4,
    "num_workers": 2,
    "crop_size": 513,
    "min_scale": 0.5,
    "max_scale": 2.0,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_classes": 21,
    "ignore_label": 255,
    "seed": 42,
    "weights": "none",
    "backbone_weights": "imagenet",
    "resume": None,
    "freeze_bn": False,
    "device": "auto",
    "eval_every": 1,
    "eval_long_size": None,
}


if __name__ == "__main__":
    run_training(Namespace(**CONFIG))
