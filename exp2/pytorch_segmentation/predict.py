from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from exp2.pytorch_segmentation.config import ExperimentConfig, VOC_COLORMAP
from exp2.pytorch_segmentation.model import build_deeplabv3_resnet50
from exp2.pytorch_segmentation.transforms import PredictTransform
from exp2.pytorch_segmentation.utils import (
    build_run_stamp,
    ensure_dir,
    get_device,
    save_json,
    setup_logger,
)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    palette = []
    for color in VOC_COLORMAP:
        palette.extend(color)
    palette.extend([0] * (768 - len(palette)))

    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    image.putpalette(palette)
    return image


def blend_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.6) -> Image.Image:
    color_mask = colorize_mask(mask).convert("RGB")
    return Image.blend(image.convert("RGB"), color_mask, alpha=alpha)


def build_parser() -> argparse.ArgumentParser:
    config = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Run inference with DeepLabV3.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=config.outputs_dir / "predictions")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--weights", choices=["none", "voc"], default="none")
    parser.add_argument("--backbone-weights", choices=["none", "imagenet"], default="imagenet")
    parser.add_argument("--num-classes", type=int, default=config.num_classes)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--long-size", type=int, default=513)
    return parser


def iter_input_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([item for item in path.iterdir() if item.suffix.lower() in exts])


def run_prediction(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    run_stamp = build_run_stamp()
    logger = setup_logger("segmentation.predict", output_dir / "logs" / f"predict_{run_stamp}.log")
    logger.info("Prediction started")
    logger.info("Run stamp: %s", run_stamp)
    logger.info("Arguments: %s", vars(args))

    model = build_deeplabv3_resnet50(
        num_classes=args.num_classes,
        weights=args.weights,
        backbone_weights=args.backbone_weights,
    ).to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    model.eval()

    transform = PredictTransform(long_size=args.long_size)
    prediction_records = []
    for image_path in iter_input_images(args.input):
        image = Image.open(image_path).convert("RGB")
        image_tensor, original_size = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)["out"]
            output = F.interpolate(
                output,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False,
            )
            prediction = output.argmax(dim=1).squeeze(0).cpu().numpy()

        mask = colorize_mask(prediction)
        overlay = blend_overlay(image, prediction)
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        mask.save(mask_path)
        overlay.save(overlay_path)
        prediction_records.append(
            {
                "input_image": image_path,
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "original_size": {"width": original_size[0], "height": original_size[1]},
            }
        )
        logger.info("Saved prediction outputs for %s", image_path.name)

    manifest_path = output_dir / f"prediction_manifest_{run_stamp}.json"
    save_json(
        {
            "run_stamp": run_stamp,
            "checkpoint": args.checkpoint,
            "weights": args.weights,
            "backbone_weights": args.backbone_weights,
            "device": str(device),
            "items": prediction_records,
        },
        manifest_path,
    )
    logger.info("Saved prediction manifest: %s", manifest_path)


def main() -> None:
    args = build_parser().parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
