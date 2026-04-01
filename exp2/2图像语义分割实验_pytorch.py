import random
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50


VOC_CLASSES = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}

VOC_COLORS = [
    "aliceblue",
    "grey",
    "red",
    "green",
    "darkorange",
    "lime",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "darkorange",
    "tan",
    "darkviolet",
    "cornflowerblue",
    "yellow",
    "crimson",
    "darkcyan",
]


RUN_CONFIG = {
    "command": "train",
    "data_root": "./VOC2012",
    "crop_size": 513,
    "batch_size": 4,
    "num_classes": 21,
    "ignore_label": 255,
    "workers": 2,
    "seed": 1,
    "device": "auto",
    "use_amp": True,
    "image_mean": [103.53, 116.28, 123.675],
    "image_std": [57.375, 57.120, 58.395],
    "epochs": 3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "min_scale": 0.5,
    "max_scale": 2.0,
    "output": "./exp2/model_pytorch.pth",
    "checkpoint": None,
    "init_mode": "scratch",
    "flip": True,
    "scales": [1.0],
    "num_images": 3,
    "save_dir": None,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def use_cuda_amp(args, device: torch.device) -> bool:
    return bool(getattr(args, "use_amp", True) and device.type == "cuda")


def dataloader_kwargs(device: torch.device, workers: int):
    kwargs = {"num_workers": workers}
    if device.type == "cuda":
        kwargs["pin_memory"] = True
    return kwargs


def build_args_from_config(config):
    normalized = dict(config)
    for key in {"data_root", "output", "checkpoint", "save_dir"}:
        if normalized.get(key) is not None:
            normalized[key] = Path(normalized[key])
    return type("Config", (), normalized)()


class VOCSegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        image_mean,
        image_std,
        crop_size: int,
        ignore_label: int = 255,
        training: bool = False,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ):
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.training = training
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

        split_file = self.data_root / "ImageSets" / "Segmentation" / f"{split}.txt"
        with split_file.open("r", encoding="utf-8") as f:
            image_ids = [line.strip() for line in f if line.strip()]

        self.samples = [
            (
                self.data_root / "JPEGImages" / f"{image_id}.jpg",
                self.data_root / "SegmentationClassGray" / f"{image_id}.png",
            )
            for image_id in image_ids
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"读取失败: {image_path} / {mask_path}")

        if self.training:
            image, mask = self._train_transform(image, mask)
        else:
            image, mask = self._eval_transform(image, mask)

        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        return image, mask

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image = (image - self.image_mean) / self.image_std
        return image.transpose(2, 0, 1)

    def _train_transform(self, image: np.ndarray, mask: np.ndarray):
        scale = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pad_h = max(self.crop_size - new_h, 0)
        pad_w = max(self.crop_size - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(
                mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label
            )

        offset_h = np.random.randint(0, image.shape[0] - self.crop_size + 1)
        offset_w = np.random.randint(0, image.shape[1] - self.crop_size + 1)
        image = image[offset_h : offset_h + self.crop_size, offset_w : offset_w + self.crop_size]
        mask = mask[offset_h : offset_h + self.crop_size, offset_w : offset_w + self.crop_size]

        if np.random.rand() > 0.5:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
        return self._normalize(image), mask

    def _eval_transform(self, image: np.ndarray, mask: np.ndarray):
        image = resize_long(image, self.crop_size)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        pad_h = max(self.crop_size - image.shape[0], 0)
        pad_w = max(self.crop_size - image.shape[1], 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(
                mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label
            )
        return self._normalize(image), mask


def create_gray_masks(data_root: Path, overwrite: bool = False):
    color_dir = data_root / "SegmentationClass"
    gray_dir = data_root / "SegmentationClassGray"
    gray_dir.mkdir(parents=True, exist_ok=True)

    for color_mask in sorted(color_dir.glob("*.png")):
        target = gray_dir / color_mask.name
        if target.exists() and not overwrite:
            continue
        with Image.open(color_mask) as mask_image:
            Image.fromarray(np.array(mask_image)).save(target)

    print(f"gray mask ready: {gray_dir}")


def build_model(num_classes: int, init_mode: str = "scratch") -> nn.Module:
    if init_mode == "torchvision":
        model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
            num_classes=21,
        )
        if num_classes != 21:
            classifier_in = model.classifier[-1].in_channels
            model.classifier[-1] = nn.Conv2d(classifier_in, num_classes, kernel_size=1)
            if model.aux_classifier is not None:
                aux_in = model.aux_classifier[-1].in_channels
                model.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, kernel_size=1)
        return model

    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
    return model


def resize_long(image: np.ndarray, long_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h > w:
        new_h = long_size
        new_w = int(long_size * w / h)
    else:
        new_w = long_size
        new_h = int(long_size * h / w)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def preprocess_for_inference(image, image_mean, image_std, crop_size):
    resized = resize_long(image, crop_size)
    resize_h, resize_w = resized.shape[:2]
    normalized = (resized.astype(np.float32) - np.array(image_mean)) / np.array(image_std)
    pad_h = max(crop_size - resize_h, 0)
    pad_w = max(crop_size - resize_w, 0)
    if pad_h > 0 or pad_w > 0:
        normalized = cv2.copyMakeBorder(normalized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    chw = normalized.transpose(2, 0, 1)
    return chw, resize_h, resize_w


@torch.inference_mode()
def predict_probabilities(args, model, images):
    device = torch.device(args.device)
    batch = []
    resize_hw = []
    for image in images:
        tensor, resize_h, resize_w = preprocess_for_inference(
            image, args.image_mean, args.image_std, args.crop_size
        )
        batch.append(tensor)
        resize_hw.append((resize_h, resize_w))

    batch_tensor = torch.from_numpy(np.stack(batch)).float().to(device)
    logits = model(batch_tensor)["out"]
    probs = torch.softmax(logits, dim=1)

    if args.flip:
        flipped = torch.flip(batch_tensor, dims=[3])
        flip_logits = model(flipped)["out"]
        probs = probs + torch.flip(torch.softmax(flip_logits, dim=1), dims=[3])

    results = []
    for idx, image in enumerate(images):
        prob = probs[idx, :, : resize_hw[idx][0], : resize_hw[idx][1]].permute(1, 2, 0).cpu().numpy()
        prob = cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        results.append(prob)
    return results


@torch.inference_mode()
def predict_mask_multi_scale(args, model, images):
    base_crop_size = args.crop_size
    all_probs = None
    for scale in args.scales:
        scaled_crop = int((base_crop_size - 1) * scale) + 1
        image_mean = args.image_mean
        image_std = args.image_std
        device = args.device
        batch = []
        resize_hw = []
        for image in images:
            tensor, resize_h, resize_w = preprocess_for_inference(image, image_mean, image_std, scaled_crop)
            batch.append(tensor)
            resize_hw.append((resize_h, resize_w))

        batch_tensor = torch.from_numpy(np.stack(batch)).float().to(device)
        logits = model(batch_tensor)["out"]
        probs_tensor = torch.softmax(logits, dim=1)
        if args.flip:
            flipped = torch.flip(batch_tensor, dims=[3])
            flip_logits = model(flipped)["out"]
            probs_tensor = probs_tensor + torch.flip(torch.softmax(flip_logits, dim=1), dims=[3])

        probs = []
        for idx, image in enumerate(images):
            prob = probs_tensor[idx, :, : resize_hw[idx][0], : resize_hw[idx][1]].permute(1, 2, 0).cpu().numpy()
            prob = cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            probs.append(prob)
        if all_probs is None:
            all_probs = probs
        else:
            for idx in range(len(probs)):
                all_probs[idx] += probs[idx]
    return [prob.argmax(axis=2).astype(np.uint8) for prob in all_probs]


def fast_hist(label_true, label_pred, num_classes, ignore_label):
    valid = (label_true >= 0) & (label_true < num_classes) & (label_true != ignore_label)
    hist = np.bincount(
        num_classes * label_true[valid].astype(np.int64) + label_pred[valid].astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def evaluate(args):
    create_gray_masks(args.data_root)
    device = resolve_device(args.device)
    print(f"using device: {device}")

    model = build_model(args.num_classes, init_mode="scratch").to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    val_list = args.data_root / "ImageSets" / "Segmentation" / "val.txt"
    with val_list.open("r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    hist = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    batch_images = []
    batch_masks = []
    processed = 0
    for image_id in image_ids:
        image_path = args.data_root / "JPEGImages" / f"{image_id}.jpg"
        mask_path = args.data_root / "SegmentationClassGray" / f"{image_id}.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"读取失败: {image_path} / {mask_path}")

        batch_images.append(image)
        batch_masks.append(mask)
        if len(batch_images) < args.batch_size:
            continue

        preds = predict_mask_multi_scale(args, model, batch_images)
        for target, pred in zip(batch_masks, preds):
            hist += fast_hist(target.flatten(), pred.flatten(), args.num_classes, args.ignore_label)
            processed += 1
            if processed % 100 == 0:
                print(f"processed {processed} images")
        batch_images = []
        batch_masks = []

    if batch_images:
        preds = predict_mask_multi_scale(args, model, batch_images)
        for target, pred in zip(batch_masks, preds):
            hist += fast_hist(target.flatten(), pred.flatten(), args.num_classes, args.ignore_label)
            processed += 1

    iou = np.diag(hist) / np.maximum(hist.sum(1) + hist.sum(0) - np.diag(hist), 1e-10)
    print("mean IoU", np.nanmean(iou))


def train(args):
    set_seed(args.seed)
    create_gray_masks(args.data_root)
    device = resolve_device(args.device)
    print(f"using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = VOCSegmentationDataset(
        data_root=args.data_root,
        split="train",
        image_mean=args.image_mean,
        image_std=args.image_std,
        crop_size=args.crop_size,
        ignore_label=args.ignore_label,
        training=True,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs(device, args.workers),
    )

    model = build_model(args.num_classes, init_mode=args.init_mode).to(device)
    if args.checkpoint and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=False)
    elif args.checkpoint:
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp(args, device))

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for step, (images, masks) in enumerate(loader, start=1):
            non_blocking = device.type == "cuda"
            images = images.to(device, non_blocking=non_blocking)
            masks = masks.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_cuda_amp(args, device)):
                logits = model(images)["out"]
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            if step % 10 == 0:
                avg_loss = running_loss / step
                lr = scheduler.get_last_lr()[0]
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss {avg_loss:.4f} lr {lr:.6f}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "num_classes": args.num_classes,
            },
            args.output,
        )
        print(f"saved checkpoint to {args.output}")


def visualize(args):
    create_gray_masks(args.data_root)
    device = resolve_device(args.device)
    print(f"using device: {device}")
    model = build_model(args.num_classes, init_mode="scratch").to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    val_list = args.data_root / "ImageSets" / "Segmentation" / "val.txt"
    with val_list.open("r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    selected = image_ids[: args.num_images]
    cmap = mcolors.ListedColormap(VOC_COLORS)
    norm = mcolors.BoundaryNorm(list(range(args.num_classes + 1)), cmap.N)

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for image_id in selected:
        image_path = args.data_root / "JPEGImages" / f"{image_id}.jpg"
        mask_path = args.data_root / "SegmentationClassGray" / f"{image_id}.png"

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        pred_mask = predict_mask_multi_scale(args, model, [bgr])[0]

        gt_mask = gt_mask.copy()
        gt_mask[gt_mask == args.ignore_label] = 0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title("Image")

        plt.subplot(1, 3, 2)
        plt.imshow(rgb)
        plt.imshow(pred_mask, alpha=0.8, interpolation="none", cmap=cmap, norm=norm)
        plt.axis("off")
        plt.title("Prediction")

        plt.subplot(1, 3, 3)
        plt.imshow(rgb)
        plt.imshow(gt_mask, alpha=0.8, interpolation="none", cmap=cmap, norm=norm)
        plt.axis("off")
        plt.title("Ground Truth")

        pred_classes = [VOC_CLASSES[idx] for idx in np.unique(pred_mask)]
        gt_classes = [VOC_CLASSES[idx] for idx in np.unique(gt_mask)]
        print(f"{image_id} prediction classes: {pred_classes}")
        print(f"{image_id} ground truth classes: {gt_classes}")

        if args.save_dir is not None:
            save_path = args.save_dir / f"{image_id}_vis.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"saved visualization to {save_path}")
            plt.close()
        else:
            plt.show()


def main():
    args = build_args_from_config(RUN_CONFIG)
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "visualize":
        visualize(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
