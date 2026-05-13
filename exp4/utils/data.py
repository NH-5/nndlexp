from __future__ import annotations

import hashlib
import random
import tarfile
import time
import urllib.request
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
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
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


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_cifar10_archive(data_root: Path, timeout_seconds: int | None) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / CIFAR10_ARCHIVE
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")

    if archive_path.exists():
        if _file_md5(archive_path) == CIFAR10_ARCHIVE_MD5:
            print(f"Found CIFAR-10 archive: {archive_path}", flush=True)
            return archive_path
        print(f"Existing CIFAR-10 archive has wrong MD5 and will be replaced: {archive_path}", flush=True)
        archive_path.unlink()

    if partial_path.exists():
        partial_path.unlink()

    print(f"Downloading CIFAR-10 from {CIFAR10_URL}", flush=True)
    request = urllib.request.Request(CIFAR10_URL, headers={"User-Agent": "Mozilla/5.0"})
    start_time = time.monotonic()
    last_report = start_time
    downloaded = 0
    total_bytes = None

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                total_bytes = int(content_length)

            with partial_path.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break

                    file.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    should_report = now - last_report >= 2.0
                    if total_bytes is not None and downloaded >= total_bytes:
                        should_report = True

                    if should_report:
                        elapsed = max(now - start_time, 1e-6)
                        speed = downloaded / elapsed
                        if total_bytes is None:
                            print(
                                f"Downloading CIFAR-10: {_format_size(downloaded)} "
                                f"at {_format_size(int(speed))}/s",
                                flush=True,
                            )
                        else:
                            percent = downloaded / total_bytes * 100
                            print(
                                f"Downloading CIFAR-10: {_format_size(downloaded)} / "
                                f"{_format_size(total_bytes)} ({percent:.1f}%) "
                                f"at {_format_size(int(speed))}/s",
                                flush=True,
                            )
                        last_report = now
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    partial_path.replace(archive_path)
    if _file_md5(archive_path) != CIFAR10_ARCHIVE_MD5:
        archive_path.unlink()
        raise RuntimeError("Downloaded CIFAR-10 archive failed MD5 verification.")

    print(f"CIFAR-10 archive downloaded: {archive_path}", flush=True)
    return archive_path


def _extract_cifar10_archive(archive_path: Path, data_root: Path) -> None:
    print(f"Extracting CIFAR-10 archive to {data_root}", flush=True)
    data_root_resolved = data_root.resolve()

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target_path = (data_root / member.name).resolve()
            try:
                target_path.relative_to(data_root_resolved)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe path in CIFAR-10 archive: {member.name}") from exc
        archive.extractall(data_root)

    if not _has_cifar10_files(data_root):
        raise RuntimeError(f"CIFAR-10 archive was extracted but required files were not found in {data_root}.")
    print(f"CIFAR-10 is ready: {data_root / CIFAR10_FOLDER}", flush=True)


def _ensure_cifar10_data(data_root: Path, download: bool, timeout_seconds: int | None) -> None:
    if _has_cifar10_files(data_root):
        print(f"CIFAR-10 found locally: {data_root / CIFAR10_FOLDER}", flush=True)
        return

    if not download:
        raise FileNotFoundError(
            f"CIFAR-10 was not found in {data_root}. "
            "Expected ./data/cifar-10-batches-py with data_batch_1 ... test_batch."
        )

    print(
        f"CIFAR-10 was not found in {data_root}. "
        f"Downloading with visible progress; socket timeout={timeout_seconds}s.",
        flush=True,
    )
    archive_path = _download_cifar10_archive(data_root, timeout_seconds)
    _extract_cifar10_archive(archive_path, data_root)


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

    try:
        _ensure_cifar10_data(data_root, download=download, timeout_seconds=download_timeout)
        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=train_transform,
            download=False,
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
            download=False,
        )
    except FileNotFoundError:
        raise
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
