from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from exp2.pytorch_segmentation.config import ExperimentConfig, VOC_CLASSES
from exp2.pytorch_segmentation.dataset import VOCSegmentationDataset
from exp2.pytorch_segmentation.engine import evaluate
from exp2.pytorch_segmentation.model import build_deeplabv3_resnet50
from exp2.pytorch_segmentation.transforms import EvalTransform
from exp2.pytorch_segmentation.utils import get_device


def build_parser() -> argparse.ArgumentParser:
    config = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Evaluate DeepLabV3 on VOC2012.")
    parser.add_argument("--data-root", type=Path, default=config.data_root)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--weights", choices=["none", "voc"], default="none")
    parser.add_argument("--backbone-weights", choices=["none", "imagenet"], default="imagenet")
    parser.add_argument("--num-classes", type=int, default=config.num_classes)
    parser.add_argument("--ignore-label", type=int, default=config.ignore_label)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--long-size", type=int, default=None)
    return parser


def run_evaluation(args: argparse.Namespace) -> dict[str, float | list[float]]:
    device = get_device(args.device)

    dataset = VOCSegmentationDataset(
        data_root=args.data_root,
        split=args.split,
        transform=EvalTransform(long_size=args.long_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_deeplabv3_resnet50(
        num_classes=args.num_classes,
        weights=args.weights,
        backbone_weights=args.backbone_weights,
    ).to(device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")

    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_label,
    )
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_evaluation(args)
    print(f"loss: {metrics['loss']:.4f}")
    print(f"pixel_accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"mean_iou: {metrics['mean_iou']:.4f}")
    print("per_class_iou:")
    for class_name, class_iou in zip(VOC_CLASSES, metrics["per_class_iou"]):
        print(f"  {class_name:12s} {class_iou:.4f}")


if __name__ == "__main__":
    main()
