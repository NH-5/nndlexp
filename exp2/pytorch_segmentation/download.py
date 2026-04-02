from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from exp2.pytorch_segmentation.config import ExperimentConfig
from exp2.pytorch_segmentation.utils import ensure_dir


VOC2012_URL = "https://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"


def download_with_progress(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)

    def _report(blocks: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(blocks * block_size, total_size)
        ratio = downloaded / total_size
        print(f"\rDownloading: {ratio:6.2%}", end="")

    urllib.request.urlretrieve(url, destination, _report)
    print()
    return destination


def extract_tar(archive_path: Path, target_dir: Path) -> Path:
    ensure_dir(target_dir)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(target_dir)
    return target_dir / "VOCdevkit" / "VOC2012"


def prepare_voc2012(
    data_root: Path,
    downloads_dir: Path,
    archive_url: str = VOC2012_URL,
    force_extract: bool = False,
) -> Path:
    if (data_root / "JPEGImages").exists() and not force_extract:
        print(f"Dataset is ready: {data_root}")
        return data_root

    archive_path = downloads_dir / Path(archive_url).name
    if not archive_path.exists():
        print(f"Archive not found locally, downloading from {archive_url}")
        download_with_progress(archive_url, archive_path)
    else:
        print(f"Using local archive: {archive_path}")

    extracted_root = extract_tar(archive_path, data_root.parent)
    if extracted_root != data_root and extracted_root.exists() and not data_root.exists():
        shutil.move(str(extracted_root), str(data_root))
    print(f"Dataset prepared at: {data_root}")
    return data_root


def parse_args() -> argparse.Namespace:
    config = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Download and extract VOC2012.")
    parser.add_argument("--data-root", type=Path, default=config.data_root)
    parser.add_argument("--downloads-dir", type=Path, default=config.downloads_dir)
    parser.add_argument("--url", default=VOC2012_URL)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def run_download(args: argparse.Namespace) -> Path:
    return prepare_voc2012(
        data_root=args.data_root,
        downloads_dir=args.downloads_dir,
        archive_url=args.url,
        force_extract=args.force_extract,
    )


def main() -> None:
    args = parse_args()
    run_download(args)


if __name__ == "__main__":
    main()
