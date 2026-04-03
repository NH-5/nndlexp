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
from exp2.pytorch_segmentation.utils import (
    build_run_stamp,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_json,
    setup_logger,
)


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
    parser.add_argument("--output-dir", type=Path, default=config.outputs_dir / "evaluation")
    return parser


def run_evaluation(args: argparse.Namespace) -> dict[str, float | list[float]]:
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    run_stamp = build_run_stamp()
    logger = setup_logger("segmentation.evaluate", output_dir / "logs" / f"evaluate_{run_stamp}.log")
    logger.info("Evaluation started")
    logger.info("Run stamp: %s", run_stamp)
    logger.info("Arguments: %s", vars(args))

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
    logger.info("Evaluation samples: %d", len(dataset))
    logger.info("Device: %s", device)

    model = build_deeplabv3_resnet50(
        num_classes=args.num_classes,
        weights=args.weights,
        backbone_weights=args.backbone_weights,
    ).to(device)

    if args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_label,
    )
    metrics_payload = {
        "run_stamp": run_stamp,
        "checkpoint": args.checkpoint,
        "weights": args.weights,
        "backbone_weights": args.backbone_weights,
        "split": args.split,
        "device": str(device),
        "metrics": metrics,
    }
    save_json(metrics_payload, output_dir / f"evaluation_metrics_{run_stamp}.json")
    logger.info("loss: %.4f", metrics["loss"])
    logger.info("pixel_accuracy: %.4f", metrics["pixel_accuracy"])
    logger.info("mean_iou: %.4f", metrics["mean_iou"])
    for class_name, class_iou in zip(VOC_CLASSES, metrics["per_class_iou"]):
        logger.info("class=%s iou=%.4f", class_name, class_iou)
    logger.info("Saved evaluation metrics: %s", output_dir / f"evaluation_metrics_{run_stamp}.json")
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
