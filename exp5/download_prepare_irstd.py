#!/usr/bin/env python3
"""Download and prepare IRSTD datasets for the exp5 MSHNet notebook.

The notebook expects this normalized structure:

    exp5/data/<dataset>/
      images/
      masks/
      trainval.txt
      test.txt

This script can download the public Google Drive archives used by common
IRSTD repositories, or normalize a zip/tar/source directory that you already
downloaded manually.
"""

from __future__ import annotations

import argparse
import html
import os
import random
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "data"
DEFAULT_DOWNLOAD_DIR = SCRIPT_DIR / "downloads"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_DIR_NAMES = {"image", "images", "img", "imgs", "jpegimages"}
MASK_DIR_NAMES = {
    "annotation",
    "annotations",
    "gt",
    "groundtruth",
    "label",
    "labels",
    "mask",
    "masks",
}


@dataclass(frozen=True)
class DatasetSource:
    name: str
    drive_file_id: str
    archive_name: str
    manual_url: str
    fallback_train_fraction: float


DATASET_SOURCES = {
    "IRSTD-1k": DatasetSource(
        name="IRSTD-1k",
        drive_file_id="1JoGDGF96v4CncKZprDnoIor0k1opaLZa",
        archive_name="IRSTD-1k.zip",
        manual_url="https://drive.google.com/file/d/1JoGDGF96v4CncKZprDnoIor0k1opaLZa/view?usp=sharing",
        fallback_train_fraction=0.8,
    ),
    "NUDT-SIRST": DatasetSource(
        name="NUDT-SIRST",
        drive_file_id="1LscYoPnqtE32qxv5v_dB4iOF4dW3bxL2",
        archive_name="BasicIRSTD_dataset_pack.zip",
        manual_url="https://drive.google.com/file/d/1LscYoPnqtE32qxv5v_dB4iOF4dW3bxL2/view?usp=sharing",
        fallback_train_fraction=0.5,
    ),
}


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonical_dataset_name(value: str) -> str:
    key = normalize_key(value)
    for name in DATASET_SOURCES:
        if key == normalize_key(name):
            return name
    raise ValueError(f"Unsupported dataset {value!r}. Choose from: {', '.join(DATASET_SOURCES)}")


def format_bytes(size: Optional[int]) -> str:
    if size is None:
        return "unknown size"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def drive_download_url(file_id: str, confirm: Optional[str] = None) -> str:
    query = {"export": "download", "id": file_id}
    if confirm:
        query["confirm"] = confirm
    return "https://drive.google.com/uc?" + urllib.parse.urlencode(query)


def get_drive_confirm_token(text: str, cookies: CookieJar) -> Optional[str]:
    for cookie in cookies:
        if cookie.name.startswith("download_warning"):
            return cookie.value

    patterns = [
        r"confirm=([0-9A-Za-z_-]+)",
        r'name="confirm"\s+value="([^"]+)"',
        r"confirm=([^&\"']+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return html.unescape(match.group(1))
    return None


def response_is_attachment(response: urllib.response.addinfourl) -> bool:
    content_disposition = response.headers.get("Content-Disposition", "")
    content_type = response.headers.get("Content-Type", "")
    return "attachment" in content_disposition.lower() or "application/" in content_type.lower()


def stream_response_to_file(response: urllib.response.addinfourl, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_header = response.headers.get("Content-Length")
    total = int(total_header) if total_header and total_header.isdigit() else None
    downloaded = 0
    last_printed = -1

    with output_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = int(downloaded * 100 / total)
                if percent // 5 != last_printed // 5:
                    print(f"  {percent:3d}% ({format_bytes(downloaded)} / {format_bytes(total)})")
                    last_printed = percent
    print(f"Downloaded {output_path} ({format_bytes(downloaded)})")


def download_google_drive_file(file_id: str, output_path: Path, force: bool = False) -> Path:
    if output_path.exists() and not force:
        print(f"Archive already exists: {output_path}")
        return output_path

    cookies = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    first_url = drive_download_url(file_id)
    try:
        response = opener.open(first_url, timeout=60)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Google Drive: {exc}") from exc

    if not response_is_attachment(response):
        page = response.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
        token = get_drive_confirm_token(page, cookies)
        if not token:
            raise RuntimeError(
                "Google Drive did not return a downloadable file. "
                "Open the manual URL in a browser, download the archive, then pass --archive."
            )
        response = opener.open(drive_download_url(file_id, token), timeout=60)

    if not response_is_attachment(response):
        raise RuntimeError(
            "Google Drive still returned an HTML page instead of an archive. "
            "The file may require login or may have exceeded quota; use --archive with a manually downloaded file."
        )

    stream_response_to_file(response, output_path)
    return output_path


def ensure_safe_member(base_dir: Path, member_name: str) -> Path:
    destination = (base_dir / member_name).resolve()
    base = base_dir.resolve()
    if not str(destination).startswith(str(base) + os.sep) and destination != base:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    return destination


def extract_archive(archive_path: Path, extract_dir: Path, force: bool = False) -> Path:
    archive_path = archive_path.resolve()
    extract_dir = extract_dir.resolve()
    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        print(f"Using existing extracted directory: {extract_dir}")
        return extract_dir
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive_path} -> {extract_dir}")
    suffixes = "".join(archive_path.suffixes).lower()
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                ensure_safe_member(extract_dir, info.filename)
            zf.extractall(extract_dir)
    elif tarfile.is_tarfile(archive_path) or suffixes.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(archive_path) as tf:
            for member in tf.getmembers():
                ensure_safe_member(extract_dir, member.name)
            tf.extractall(extract_dir)
    else:
        raise RuntimeError(f"Unsupported archive type: {archive_path}")
    return extract_dir


def is_archive_path(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return path.is_file() and (
        zipfile.is_zipfile(path)
        or tarfile.is_tarfile(path)
        or suffixes.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"))
    )


def find_nested_archive(root: Path, dataset: str) -> Optional[Path]:
    dataset_key = normalize_key(dataset)
    archives = [path for path in root.rglob("*") if is_archive_path(path)]
    if not archives:
        return None

    def score(path: Path) -> int:
        key = normalize_key(path.name)
        value = 0
        if dataset_key in key:
            value += 100
        if "dataset" in key:
            value += 10
        return value

    archives.sort(key=score, reverse=True)
    return archives[0]


def find_named_child_dirs(root: Path, names: set[str]) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and directory_name_matches(path.name, names):
            matches.append(path)
    return matches


def directory_name_matches(name: str, names: set[str]) -> bool:
    key = normalize_key(name)
    if key in names:
        return True
    if names is IMAGE_DIR_NAMES:
        return "image" in key or "img" in key
    if names is MASK_DIR_NAMES:
        return "mask" in key or "label" in key or "annotation" in key or "gt" == key
    return False


def directory_score(path: Path, dataset: str) -> int:
    dataset_key = normalize_key(dataset)
    path_key = normalize_key(str(path))
    score = 0
    if dataset_key in path_key:
        score += 50
    if any((path / name).exists() for name in ("images", "image", "imgs", "img")):
        score += 20
    if any((path / name).exists() for name in ("masks", "mask", "labels", "label", "annotations")):
        score += 20
    if (path / "img_idx").exists():
        score += 10
    return score


def find_dataset_root(raw_root: Path, dataset: str) -> Path:
    candidates = [raw_root]
    candidates.extend(path for path in raw_root.rglob("*") if path.is_dir())
    candidates.sort(key=lambda p: directory_score(p, dataset), reverse=True)

    for candidate in candidates:
        try:
            find_image_mask_dirs(candidate)
            return candidate
        except RuntimeError:
            continue
    raise RuntimeError(f"Cannot find images/masks folders under {raw_root}")


def find_image_mask_dirs(dataset_root: Path) -> tuple[Path, Path]:
    image_dirs = find_named_child_dirs(dataset_root, IMAGE_DIR_NAMES)
    mask_dirs = find_named_child_dirs(dataset_root, MASK_DIR_NAMES)

    def has_images(path: Path) -> bool:
        return any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir())

    image_dirs = [path for path in image_dirs if has_images(path)]
    mask_dirs = [path for path in mask_dirs if has_images(path)]
    if not image_dirs or not mask_dirs:
        raise RuntimeError(f"Missing image or mask directory below {dataset_root}")

    image_dirs.sort(key=lambda p: (0 if normalize_key(p.name) in {"image", "images", "img", "imgs"} else 1, len(p.parts)))
    mask_dirs.sort(key=lambda p: (0 if normalize_key(p.name) in {"mask", "masks"} else 1, len(p.parts)))
    return image_dirs[0], mask_dirs[0]


def list_files_by_stem(folder: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.setdefault(path.stem, path)
    return files


def split_file_score(path: Path, dataset: str, kind: str) -> int:
    name = normalize_key(path.name)
    parent = normalize_key(str(path.parent))
    dataset_key = normalize_key(dataset)
    score = 0
    if dataset_key in name or dataset_key in parent:
        score += 10
    if "imgidx" in parent:
        score += 5
    if kind == "train":
        if "trainval" in name:
            score += 30
        elif name.startswith("train"):
            score += 25
        if "test" in name:
            score -= 100
    else:
        if name.startswith("test") or "test" in name:
            score += 30
        if "train" in name:
            score -= 100
    return score


def discover_split_file(dataset_root: Path, dataset: str, kind: str) -> Optional[Path]:
    candidates = [path for path in dataset_root.rglob("*.txt") if path.is_file()]
    scored = [(split_file_score(path, dataset, kind), path) for path in candidates]
    scored = [(score, path) for score, path in scored if score > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -len(item[1].parts)), reverse=True)
    return scored[0][1]


def read_split_stems(split_path: Path) -> list[str]:
    stems: list[str] = []
    with split_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            token = stripped.split()[0]
            stems.append(Path(token).stem)
    return stems


def fallback_split(stems: list[str], train_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
    train_stems = sorted(shuffled[:train_count])
    test_stems = sorted(shuffled[train_count:])
    return train_stems, test_stems


def resolve_splits(dataset_root: Path, dataset: str, stems: list[str], train_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    train_file = discover_split_file(dataset_root, dataset, "train")
    test_file = discover_split_file(dataset_root, dataset, "test")
    available = set(stems)

    if train_file and test_file:
        train = [stem for stem in read_split_stems(train_file) if stem in available]
        test = [stem for stem in read_split_stems(test_file) if stem in available]
        if train and test:
            print(f"Using split files: {train_file} and {test_file}")
            return sorted(dict.fromkeys(train)), sorted(dict.fromkeys(test))
        print("Split files were found but did not match image/mask stems; falling back to deterministic split.")

    train, test = fallback_split(stems, train_fraction, seed)
    print(f"No usable split files found. Created deterministic split: {len(train)} train, {len(test)} test.")
    return train, test


def copy_pair(src_image: Path, src_mask: Path, output_name: str, output_images: Path, output_masks: Path) -> None:
    shutil.copy2(src_image, output_images / output_name)
    shutil.copy2(src_mask, output_masks / output_name)


def write_split(path: Path, names: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")


def prepare_from_source(
    dataset: str,
    source_root: Path,
    output_root: Path,
    force: bool = False,
    seed: int = 42,
) -> Path:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    output_dir = output_root / dataset

    dataset_root = find_dataset_root(source_root, dataset)
    image_dir, mask_dir = find_image_mask_dirs(dataset_root)
    print(f"Dataset root: {dataset_root}")
    print(f"Images: {image_dir}")
    print(f"Masks: {mask_dir}")

    image_files = list_files_by_stem(image_dir)
    mask_files = list_files_by_stem(mask_dir)
    stems = sorted(set(image_files) & set(mask_files))
    if not stems:
        raise RuntimeError(f"No image/mask pairs found in {image_dir} and {mask_dir}")

    source = DATASET_SOURCES[dataset]
    train_stems, test_stems = resolve_splits(
        dataset_root,
        dataset,
        stems,
        train_fraction=source.fallback_train_fraction,
        seed=seed,
    )
    selected_stems = sorted(set(train_stems) | set(test_stems))

    if output_dir.exists():
        if not force:
            raise RuntimeError(f"Output directory already exists: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)

    output_images = output_dir / "images"
    output_masks = output_dir / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    output_names: dict[str, str] = {}
    for stem in selected_stems:
        output_name = image_files[stem].name
        output_names[stem] = output_name
        copy_pair(image_files[stem], mask_files[stem], output_name, output_images, output_masks)

    write_split(output_dir / "trainval.txt", [output_names[stem] for stem in train_stems if stem in output_names])
    write_split(output_dir / "test.txt", [output_names[stem] for stem in test_stems if stem in output_names])

    print(f"Prepared {dataset}:")
    print(f"  output: {output_dir}")
    print(f"  pairs: {len(selected_stems)}")
    print(f"  train: {len(train_stems)}")
    print(f"  test: {len(test_stems)}")
    return output_dir


def prepare_dataset(args: argparse.Namespace, dataset: str) -> Path:
    dataset = canonical_dataset_name(dataset)
    output_root = Path(args.output_root)
    download_dir = Path(args.download_dir)

    if args.source_dir:
        return prepare_from_source(dataset, Path(args.source_dir), output_root, force=args.force, seed=args.seed)

    if args.archive:
        archive_path = Path(args.archive)
    else:
        source = DATASET_SOURCES[dataset]
        archive_path = download_dir / source.archive_name
        print(f"Downloading {dataset} from Google Drive.")
        print(f"Manual fallback URL: {source.manual_url}")
        download_google_drive_file(source.drive_file_id, archive_path, force=args.force_download)

    extract_root = download_dir / "extracted" / dataset
    source_root = extract_archive(archive_path, extract_root, force=args.force_extract)
    try:
        return prepare_from_source(dataset, source_root, output_root, force=args.force, seed=args.seed)
    except RuntimeError as exc:
        nested_archive = find_nested_archive(source_root, dataset)
        if nested_archive is None:
            raise exc
        print(f"Found nested archive for {dataset}: {nested_archive}")
        nested_extract_root = download_dir / "extracted" / f"{dataset}-nested"
        nested_source_root = extract_archive(nested_archive, nested_extract_root, force=True)
        return prepare_from_source(dataset, nested_source_root, output_root, force=args.force, seed=args.seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="IRSTD-1k",
        help="Dataset to prepare: IRSTD-1k, NUDT-SIRST, or all. Default: IRSTD-1k",
    )
    parser.add_argument("--archive", type=Path, help="Use an already downloaded zip/tar archive.")
    parser.add_argument("--source-dir", type=Path, help="Use an already extracted dataset directory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for normalized datasets.")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR, help="Where archives and extracted files are kept.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing normalized output directory.")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if the archive already exists.")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract even if the extracted directory already exists.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used only when split files are missing.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.archive and args.source_dir:
        parser.error("--archive and --source-dir are mutually exclusive.")
    if args.dataset.lower() == "all" and (args.archive or args.source_dir):
        parser.error("--dataset all is only supported with automatic downloads.")

    datasets = list(DATASET_SOURCES) if args.dataset.lower() == "all" else [canonical_dataset_name(args.dataset)]
    try:
        for dataset in datasets:
            prepare_dataset(args, dataset)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
