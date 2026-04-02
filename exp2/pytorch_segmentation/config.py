from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_COLORMAP = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
]

DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class ExperimentConfig:
    data_root: Path = Path(__file__).resolve().parents[1] / "VOC2012"
    downloads_dir: Path = Path(__file__).resolve().parents[1] / "downloads"
    outputs_dir: Path = Path(__file__).resolve().parents[1] / "outputs"
    num_classes: int = 21
    ignore_label: int = 255
    crop_size: int = 513
    min_scale: float = 0.5
    max_scale: float = 2.0
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 42
