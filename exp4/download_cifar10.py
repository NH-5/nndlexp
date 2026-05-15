from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp4.utils.cifar10_download import (
    CIFAR10_ARCHIVE,
    CIFAR10_ARCHIVE_MD5,
    CIFAR10_URL,
    ensure_cifar10_data,
    file_md5,
    has_cifar10_files,
    verify_cifar10_archive,
)


DOWNLOAD_CONFIG = {
    "data_root": PROJECT_ROOT / "data",
    # Set this to False on the server after you manually upload
    # data/cifar-10-python.tar.gz.
    "download_if_missing": False,
    "timeout_seconds": 60,
}


def main() -> None:
    data_root = Path(DOWNLOAD_CONFIG["data_root"])
    archive_path = data_root / CIFAR10_ARCHIVE

    print(f"CIFAR-10 URL: {CIFAR10_URL}")
    print(f"Expected MD5: {CIFAR10_ARCHIVE_MD5}")
    print(f"Data root: {data_root}")

    if has_cifar10_files(data_root):
        print(f"CIFAR-10 is already extracted under: {data_root / 'cifar-10-batches-py'}")
        return

    if archive_path.exists():
        actual_md5 = file_md5(archive_path)
        print(f"Found archive: {archive_path}")
        print(f"Actual MD5: {actual_md5}")
        if verify_cifar10_archive(archive_path):
            ensure_cifar10_data(
                data_root=data_root,
                download=False,
                timeout_seconds=DOWNLOAD_CONFIG["timeout_seconds"],
                archive_path=archive_path,
            )
            return

        raise RuntimeError(
            f"Archive MD5 mismatch. Delete {archive_path} and download it again from {CIFAR10_URL}."
        )

    if not DOWNLOAD_CONFIG["download_if_missing"]:
        print(
            "Archive not found. Download it manually on a machine with good network, then upload it here:\n"
            f"  {archive_path}\n\n"
            "After upload, run:\n"
            "  python exp4/download_cifar10.py\n\n"
            "Expected file:\n"
            f"  {CIFAR10_ARCHIVE}\n"
        )
        return

    ensure_cifar10_data(
        data_root=data_root,
        download=True,
        timeout_seconds=DOWNLOAD_CONFIG["timeout_seconds"],
    )


if __name__ == "__main__":
    main()
