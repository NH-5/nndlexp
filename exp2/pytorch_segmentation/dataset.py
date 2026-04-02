from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class VOCSegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        split_file = self.data_root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.sample_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.images_dir = self.data_root / "JPEGImages"
        self.masks_dir = self.data_root / "SegmentationClass"

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        image_path = self.images_dir / f"{sample_id}.jpg"
        mask_path = self.masks_dir / f"{sample_id}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask, sample_id
