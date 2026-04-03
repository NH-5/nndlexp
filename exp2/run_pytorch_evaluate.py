from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp2.pytorch_segmentation.evaluate import run_evaluation


CONFIG = {
    "data_root": PROJECT_ROOT / "exp2" / "VOC2012",
    "output_root": PROJECT_ROOT / "exp2" / "outputs",
    "experiment_name": "latest",
    "split": "val",
    "checkpoint": None,
    "weights": "none",
    "backbone_weights": "imagenet",
    "num_classes": 21,
    "ignore_label": 255,
    "num_workers": 2,
    "device": "auto",
    "long_size": None,
}


if __name__ == "__main__":
    metrics = run_evaluation(Namespace(**CONFIG))
    print(f"loss: {metrics['loss']:.4f}")
    print(f"pixel_accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"mean_iou: {metrics['mean_iou']:.4f}")
