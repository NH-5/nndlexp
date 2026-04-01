import shutil
import tarfile
import urllib.request
from urllib.error import ContentTooShortError, URLError
from pathlib import Path

import numpy as np
from PIL import Image


RUN_CONFIG = {
    "data_root": "./VOC2012",
    "download_dir": "./downloads",
    "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    "download_voc": True,
    "prepare_gray_masks": True,
    "force_download": False,
    "overwrite_gray_masks": False,
    "download_retries": 3,
}


def download_with_progress(url: str, target_path: Path) -> None:
    def reporthook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = downloaded * 100 / total_size
        print(f"\rdownloading {target_path.name}: {percent:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
    print()


def download_with_retry(url: str, target_path: Path, retries: int) -> None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if target_path.exists():
                target_path.unlink()
            print(f"download attempt {attempt}/{retries}")
            download_with_progress(url, target_path)
            return
        except (ContentTooShortError, URLError, EOFError) as exc:
            last_error = exc
            if target_path.exists():
                target_path.unlink()
            print(f"\ndownload failed on attempt {attempt}: {exc}")
    raise RuntimeError(f"下载失败，已重试 {retries} 次") from last_error


def download_voc2012(data_root: Path, download_dir: Path, url: str, force: bool, retries: int) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / "VOCtrainval_11-May-2012.tar"
    extract_dir = download_dir / "VOCdevkit"
    source_root = extract_dir / "VOC2012"

    if data_root.exists() and not force:
        print(f"{data_root} already exists, skip download")
        return

    if archive_path.exists() and force:
        archive_path.unlink()
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    if data_root.exists() and force:
        shutil.rmtree(data_root)

    if not archive_path.exists():
        print(f"download VOC2012 from {url}")
        download_with_retry(url, archive_path, retries)

    print(f"extracting {archive_path}")
    try:
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(path=download_dir)
    except tarfile.TarError as exc:
        if archive_path.exists():
            archive_path.unlink()
        raise RuntimeError("压缩包损坏，已删除本地 tar 文件，请重新运行脚本") from exc

    if not source_root.exists():
        raise FileNotFoundError(f"解压后未找到 {source_root}")

    if data_root.exists():
        shutil.rmtree(data_root)
    shutil.copytree(source_root, data_root)
    print(f"VOC2012 ready at {data_root}")


def create_gray_masks(data_root: Path, overwrite: bool) -> None:
    color_dir = data_root / "SegmentationClass"
    gray_dir = data_root / "SegmentationClassGray"
    gray_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for color_mask in sorted(color_dir.glob("*.png")):
        target = gray_dir / color_mask.name
        if target.exists() and not overwrite:
            continue
        with Image.open(color_mask) as mask_image:
            Image.fromarray(np.array(mask_image)).save(target)
        converted += 1

    print(f"gray mask ready: {gray_dir}, converted {converted} files")


def main():
    data_root = Path(RUN_CONFIG["data_root"])
    download_dir = Path(RUN_CONFIG["download_dir"])

    if RUN_CONFIG["download_voc"]:
        download_voc2012(
            data_root=data_root,
            download_dir=download_dir,
            url=RUN_CONFIG["url"],
            force=RUN_CONFIG["force_download"],
            retries=RUN_CONFIG["download_retries"],
        )

    if RUN_CONFIG["prepare_gray_masks"]:
        create_gray_masks(
            data_root=data_root,
            overwrite=RUN_CONFIG["overwrite_gray_masks"],
        )


if __name__ == "__main__":
    main()
