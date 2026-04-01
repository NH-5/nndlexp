import json
import logging
import shutil
import tarfile
import urllib.request
from datetime import datetime
from urllib.error import ContentTooShortError, URLError
from pathlib import Path

import numpy as np
from PIL import Image


RUN_CONFIG = {
    "data_root": "./VOC2012",
    "download_dir": "./downloads",
    "log_dir": "./exp2/logs",
    "log_name": None,
    "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    "download_voc": True,
    "prepare_gray_masks": True,
    "force_download": False,
    "overwrite_gray_masks": False,
    "download_retries": 3,
}


def setup_logger(log_dir: Path, log_name: str | None) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = log_name or f"prepare_data_{timestamp}"
    log_path = log_dir / f"{file_stem}.log"

    logger = logging.getLogger(f"exp2_prepare_{file_stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("log file: %s", log_path)
    return logger


def config_to_dict(config: dict) -> dict:
    normalized = {}
    for key, value in config.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
    return normalized


def download_with_progress(url: str, target_path: Path) -> None:
    def reporthook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = downloaded * 100 / total_size
        print(f"\rdownloading {target_path.name}: {percent:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
    print()


def download_with_retry(url: str, target_path: Path, retries: int, logger: logging.Logger) -> None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if target_path.exists():
                target_path.unlink()
            logger.info("download attempt %d/%d", attempt, retries)
            download_with_progress(url, target_path)
            return
        except (ContentTooShortError, URLError, EOFError) as exc:
            last_error = exc
            if target_path.exists():
                target_path.unlink()
            logger.warning("download failed on attempt %d: %s", attempt, exc)
    raise RuntimeError(f"下载失败，已重试 {retries} 次") from last_error


def download_voc2012(
    data_root: Path,
    download_dir: Path,
    url: str,
    force: bool,
    retries: int,
    logger: logging.Logger,
) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / "VOCtrainval_11-May-2012.tar"
    extract_dir = download_dir / "VOCdevkit"
    source_root = extract_dir / "VOC2012"

    if data_root.exists() and not force:
        logger.info("%s already exists, skip download", data_root)
        return

    if archive_path.exists() and force:
        archive_path.unlink()
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    if data_root.exists() and force:
        shutil.rmtree(data_root)

    if not archive_path.exists():
        logger.info("download VOC2012 from %s", url)
        download_with_retry(url, archive_path, retries, logger)

    logger.info("extracting %s", archive_path)
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
    logger.info("VOC2012 ready at %s", data_root)


def create_gray_masks(data_root: Path, overwrite: bool, logger: logging.Logger) -> None:
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

    logger.info("gray mask ready: %s, converted %d files", gray_dir, converted)


def main():
    data_root = Path(RUN_CONFIG["data_root"])
    download_dir = Path(RUN_CONFIG["download_dir"])
    log_dir = Path(RUN_CONFIG["log_dir"])
    logger = setup_logger(log_dir, RUN_CONFIG["log_name"])
    logger.info("start data preparation")
    logger.info("config: %s", json.dumps(config_to_dict(RUN_CONFIG), ensure_ascii=False, indent=2))

    if RUN_CONFIG["download_voc"]:
        download_voc2012(
            data_root=data_root,
            download_dir=download_dir,
            url=RUN_CONFIG["url"],
            force=RUN_CONFIG["force_download"],
            retries=RUN_CONFIG["download_retries"],
            logger=logger,
        )

    if RUN_CONFIG["prepare_gray_masks"]:
        create_gray_masks(
            data_root=data_root,
            overwrite=RUN_CONFIG["overwrite_gray_masks"],
            logger=logger,
        )

    logger.info("data preparation finished")


if __name__ == "__main__":
    main()
