from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from exp2.pytorch_segmentation.config import ExperimentConfig
from exp2.pytorch_segmentation.dataset import VOCSegmentationDataset
from exp2.pytorch_segmentation.engine import evaluate, train_one_epoch
from exp2.pytorch_segmentation.model import build_deeplabv3_resnet50, freeze_batch_norm
from exp2.pytorch_segmentation.transforms import EvalTransform, RandomScaleCropFlip
from exp2.pytorch_segmentation.utils import ensure_dir, get_device, save_checkpoint, seed_everything


def build_parser() -> argparse.ArgumentParser:
    config = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Train DeepLabV3 on VOC2012 with PyTorch.")
    parser.add_argument("--data-root", type=Path, default=config.data_root)
    parser.add_argument("--output-dir", type=Path, default=config.outputs_dir / "deeplabv3_resnet50")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--crop-size", type=int, default=config.crop_size)
    parser.add_argument("--min-scale", type=float, default=config.min_scale)
    parser.add_argument("--max-scale", type=float, default=config.max_scale)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--momentum", type=float, default=config.momentum)
    parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
    parser.add_argument("--num-classes", type=int, default=config.num_classes)
    parser.add_argument("--ignore-label", type=int, default=config.ignore_label)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--weights", choices=["none", "voc"], default="none")
    parser.add_argument("--backbone-weights", choices=["none", "imagenet"], default="imagenet")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-long-size", type=int, default=None)
    return parser


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = VOCSegmentationDataset(
        data_root=args.data_root,
        split=args.train_split,
        transform=RandomScaleCropFlip(
            crop_size=args.crop_size,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            ignore_label=args.ignore_label,
        ),
    )
    val_dataset = VOCSegmentationDataset(
        data_root=args.data_root,
        split=args.val_split,
        transform=EvalTransform(long_size=args.eval_long_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
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
    if args.freeze_bn:
        freeze_batch_norm(model)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    optimizer = SGD(
        params=(parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: max((1 - step / total_steps) ** 0.9, 0.0),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 1
    best_miou = -1.0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", best_miou)
        print(f"Resumed from checkpoint: {args.resume}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            scaler=scaler,
        )
        print(f"Epoch {epoch:02d} training loss: {train_stats['loss']:.4f}")

        if epoch % args.eval_every != 0:
            continue

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_label,
        )
        print(
            f"Epoch {epoch:02d} validation | loss={val_stats['loss']:.4f} "
            f"| pixel_acc={val_stats['pixel_accuracy']:.4f} "
            f"| mIoU={val_stats['mean_iou']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "args": vars(args),
        }
        save_checkpoint(checkpoint, output_dir / "last.pth")

        if val_stats["mean_iou"] > best_miou:
            best_miou = float(val_stats["mean_iou"])
            checkpoint["best_miou"] = best_miou
            save_checkpoint(checkpoint, output_dir / "best.pth")
            print(f"Saved best checkpoint to {output_dir / 'best.pth'}")


def main() -> None:
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
