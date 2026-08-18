from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
MINI_DATA = ROOT / "data" / "names_mini.txt"
FULL_DATA = ROOT / "data" / "names.txt"


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--full", action="store_true", help="使用完整 names.txt 和正式训练步数")
    parser.add_argument("--seed", type=int, default=2147483647)
    return parser.parse_args()


def data_path(full: bool) -> Path:
    path = FULL_DATA if full else MINI_DATA
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}；完整模式请先运行 python download_names.py")
    return path


def prepare_artifacts() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS

