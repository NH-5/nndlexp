from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp2.pytorch_segmentation.download import run_download


CONFIG = {
    "data_root": PROJECT_ROOT / "exp2" / "VOC2012",
    "downloads_dir": PROJECT_ROOT / "exp2" / "downloads",
    "url": "https://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    "force_extract": False,
}


if __name__ == "__main__":
    run_download(Namespace(**CONFIG))
