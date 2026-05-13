from __future__ import annotations

import socket
from contextlib import contextmanager
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_FOLDER = "cifar-10-batches-py"
CIFAR10_REQUIRED_FILES = (
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
    "batches.meta",
)


@dataclass(frozen=True)
class Cifar10Loaders:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: tuple[str, ...]
    train_size: int
    val_size: int
    test_size: int


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return train_transform, eval_transform


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def _limit_indices(indices: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return indices
    return indices[: max(0, min(limit, len(indices)))]


def _has_cifar10_files(data_root: Path) -> bool:
    data_dir = data_root / CIFAR10_FOLDER
    return all((data_dir / file_name).exists() for file_name in CIFAR10_REQUIRED_FILES)


@contextmanager
def _temporary_socket_timeout(timeout_seconds: int | None):
    if timeout_seconds is None:
        yield
        return

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous_timeout)


def _dataset_error_message(data_root: Path, timeout_seconds: int | None) -> str:
    timeout_text = "without a socket timeout" if timeout_seconds is None else f"with {timeout_seconds}s socket timeout"
    return (
        f"Failed to prepare CIFAR-10 under {data_root} {timeout_text}. "
        "If the server cannot reach the official CIFAR-10 download host, "
        "manually put the extracted cifar-10-batches-py directory under ./data "
        "or set RUN_CONFIG['download'] = False after the data is already present."
    )


def build_cifar10_dataloaders(
    data_root: str | Path = "./data",
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    download: bool = True,
    pin_memory: bool = False,
    train_subset: int | None = None,
    val_subset: int | None = None,
    test_subset: int | None = None,
    download_timeout: int | None = 60,
) -> Cifar10Loaders:
    """Create deterministic train/validation/test loaders for CIFAR-10."""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    data_root = Path(data_root)
    train_transform, eval_transform = build_transforms(image_size=image_size)

    data_ready = _has_cifar10_files(data_root)
    if download and not data_ready:
        print(
            f"CIFAR-10 was not found in {data_root}. "
            f"torchvision will download it now; timeout={download_timeout}s.",
            flush=True,
        )
    elif not download and not data_ready:
        raise FileNotFoundError(
            f"CIFAR-10 was not found in {data_root}. "
            "Expected ./data/cifar-10-batches-py with data_batch_1 ... test_batch."
        )

    try:
        with _temporary_socket_timeout(download_timeout if download and not data_ready else None):
            train_dataset = datasets.CIFAR10(
                root=data_root,
                train=True,
                transform=train_transform,
                download=download,
            )
            val_dataset = datasets.CIFAR10(
                root=data_root,
                train=True,
                transform=eval_transform,
                download=False,
            )
            test_dataset = datasets.CIFAR10(
                root=data_root,
                train=False,
                transform=eval_transform,
                download=download,
            )
    except Exception as exc:
        raise RuntimeError(_dataset_error_message(data_root, download_timeout)) from exc

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    val_size = int(len(indices) * val_fraction)
    val_indices = _limit_indices(indices[:val_size], val_subset)
    train_indices = _limit_indices(indices[val_size:], train_subset)
    test_indices = _limit_indices(list(range(len(test_dataset))), test_subset)

    train_subset_dataset = Subset(train_dataset, train_indices)
    val_subset_dataset = Subset(val_dataset, val_indices)
    test_subset_dataset = Subset(test_dataset, test_indices)

    loader_generator = torch.Generator().manual_seed(seed)
    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_subset_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_subset_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )

    return Cifar10Loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=CIFAR10_CLASSES,
        train_size=len(train_subset_dataset),
        val_size=len(val_subset_dataset),
        test_size=len(test_subset_dataset),
    )
