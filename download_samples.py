"""Завантажує тестові відео в data/samples/ (у git вони не потрапляють — див. .gitignore).

Це демо-відео бібліотеки supervision (MIT). Файли перевіряються за MD5, тож повторний
запуск нічого не качає вдруге.

Запуск:
    python download_samples.py
    python download_samples.py --gt-videos     # ще й відео з лінією для ручного підрахунку
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from supervision.assets import VideoAssets, download_assets

DATA = Path(__file__).with_name("data")
SAMPLES = DATA / "samples"
GT_VIDEOS = DATA / "gt_videos"
CLIPS = [VideoAssets.PEOPLE_WALKING, VideoAssets.MARKET_SQUARE, VideoAssets.SUBWAY]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-videos", action="store_true",
                    help="згенерувати data/gt_videos/*_line.mp4 через draw_line.py")
    args = ap.parse_args()

    SAMPLES.mkdir(parents=True, exist_ok=True)
    for asset in CLIPS:
        path = download_assets(asset, SAMPLES)
        print(f"  {path}")

    if args.gt_videos:
        GT_VIDEOS.mkdir(parents=True, exist_ok=True)
        for asset in CLIPS[:2]:  # кліпи, для яких ведемо ручний підрахунок
            src = SAMPLES / asset.value
            dst = GT_VIDEOS / f"{src.stem}_line.mp4"
            if dst.exists():
                print(f"  {dst} (вже є)")
                continue
            # 4K-відео зменшуємо вдвічі, щоб його було зручно переглядати
            scale = ["--scale", "0.5"] if "market" in src.name else []
            subprocess.run([sys.executable, str(Path(__file__).with_name("draw_line.py")),
                            str(src), "-o", str(dst), *scale], check=True)


if __name__ == "__main__":
    main()
