from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp2.pytorch_segmentation.predict import run_prediction


CONFIG = {
    "input": PROJECT_ROOT / "exp2" / "VOC2012" / "JPEGImages" / "2007_000032.jpg",
    "output_dir": PROJECT_ROOT / "exp2" / "outputs" / "predictions",
    "checkpoint": PROJECT_ROOT / "exp2" / "outputs" / "deeplabv3_resnet50" / "best.pth",
    "weights": "none",
    "backbone_weights": "imagenet",
    "num_classes": 21,
    "device": "auto",
    "long_size": 513,
}


if __name__ == "__main__":
    run_prediction(Namespace(**CONFIG))
