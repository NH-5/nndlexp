from __future__ import annotations

import hashlib
import tarfile
import time
import urllib.request
from pathlib import Path


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


def has_cifar10_files(data_root: str | Path) -> bool:
    data_dir = Path(data_root) / CIFAR10_FOLDER
    return all((data_dir / file_name).exists() for file_name in CIFAR10_REQUIRED_FILES)


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_cifar10_archive(archive_path: str | Path) -> bool:
    archive_path = Path(archive_path)
    if not archive_path.exists():
        return False
    return file_md5(archive_path) == CIFAR10_ARCHIVE_MD5


def download_cifar10_archive(data_root: str | Path, timeout_seconds: int | None = 60) -> Path:
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / CIFAR10_ARCHIVE
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")

    if archive_path.exists():
        if verify_cifar10_archive(archive_path):
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
                                f"Downloading CIFAR-10: {format_size(downloaded)} "
                                f"at {format_size(int(speed))}/s",
                                flush=True,
                            )
                        else:
                            percent = downloaded / total_bytes * 100
                            print(
                                f"Downloading CIFAR-10: {format_size(downloaded)} / "
                                f"{format_size(total_bytes)} ({percent:.1f}%) "
                                f"at {format_size(int(speed))}/s",
                                flush=True,
                            )
                        last_report = now
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    partial_path.replace(archive_path)
    if not verify_cifar10_archive(archive_path):
        archive_path.unlink()
        raise RuntimeError("Downloaded CIFAR-10 archive failed MD5 verification.")

    print(f"CIFAR-10 archive downloaded: {archive_path}", flush=True)
    return archive_path


def extract_cifar10_archive(archive_path: str | Path, data_root: str | Path) -> None:
    archive_path = Path(archive_path)
    data_root = Path(data_root)
    print(f"Extracting CIFAR-10 archive to {data_root}", flush=True)
    data_root.mkdir(parents=True, exist_ok=True)
    data_root_resolved = data_root.resolve()

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target_path = (data_root / member.name).resolve()
            try:
                target_path.relative_to(data_root_resolved)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe path in CIFAR-10 archive: {member.name}") from exc
        archive.extractall(data_root)

    if not has_cifar10_files(data_root):
        raise RuntimeError(f"CIFAR-10 archive was extracted but required files were not found in {data_root}.")
    print(f"CIFAR-10 is ready: {data_root / CIFAR10_FOLDER}", flush=True)


def ensure_cifar10_data(
    data_root: str | Path,
    download: bool = True,
    timeout_seconds: int | None = 60,
    archive_path: str | Path | None = None,
) -> None:
    data_root = Path(data_root)
    if has_cifar10_files(data_root):
        print(f"CIFAR-10 found locally: {data_root / CIFAR10_FOLDER}", flush=True)
        return

    if archive_path is not None:
        archive_path = Path(archive_path)
    else:
        archive_path = data_root / CIFAR10_ARCHIVE

    if archive_path.exists():
        if not verify_cifar10_archive(archive_path):
            actual_md5 = file_md5(archive_path)
            raise RuntimeError(
                f"CIFAR-10 archive MD5 mismatch: {archive_path}. "
                f"Expected {CIFAR10_ARCHIVE_MD5}, got {actual_md5}."
            )
        print(f"Using existing CIFAR-10 archive: {archive_path}", flush=True)
        extract_cifar10_archive(archive_path, data_root)
        return

    if not download:
        raise FileNotFoundError(
            f"CIFAR-10 was not found in {data_root}. "
            f"Put {CIFAR10_ARCHIVE} under {data_root} and run exp4/download_cifar10.py, "
            "or extract cifar-10-batches-py there manually."
        )

    print(
        f"CIFAR-10 was not found in {data_root}. "
        f"Downloading with visible progress; socket timeout={timeout_seconds}s.",
        flush=True,
    )
    archive_path = download_cifar10_archive(data_root, timeout_seconds=timeout_seconds)
    extract_cifar10_archive(archive_path, data_root)


def dataset_error_message(data_root: str | Path, timeout_seconds: int | None) -> str:
    timeout_text = "without a socket timeout" if timeout_seconds is None else f"with {timeout_seconds}s socket timeout"
    return (
        f"Failed to prepare CIFAR-10 under {data_root} {timeout_text}. "
        f"If the server cannot reach the official CIFAR-10 download host, put {CIFAR10_ARCHIVE} "
        "under ./data and run `python exp4/download_cifar10.py`, or manually put the extracted "
        "cifar-10-batches-py directory under ./data."
    )
