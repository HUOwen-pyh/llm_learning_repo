"""下载 Karpathy makemore 的官方 names.txt。"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
TARGET = Path(__file__).resolve().parent / "data" / "names.txt"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=30) as response:
        payload = response.read()
    TARGET.write_bytes(payload)
    line_count = len(TARGET.read_text(encoding="utf-8").splitlines())
    print(f"已下载 {line_count} 个名字到 {TARGET}")


if __name__ == "__main__":
    main()

