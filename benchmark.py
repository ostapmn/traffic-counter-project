"""Проганяє всі кліпи з clips.csv усіма трьома варіантами й рахує похибку.

    bytetrack   основний варіант
    sort        baseline (b)
    none        baseline (a), без трекінгу

Підсумки дописуються в results/runs.csv, події — у results/events/, розмічені відео
(з --video) — в out/. Наприкінці запускається eval_count.py проти gt/gt.csv.

Запуск:
    python benchmark.py
    python benchmark.py --clips people-walking.mp4 --trackers bytetrack --video
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from clips import all_clips
from track import append_result, track_video

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", help="імена файлів із clips.csv; за замовчуванням усі наявні")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "sort", "none"],
                    choices=["bytetrack", "sort", "none"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--video", action="store_true", help="записати розмічені відео в out/")
    ap.add_argument("--results", default=str(ROOT / "results" / "runs.csv"))
    args = ap.parse_args()

    clips = [r["clip"] for r in all_clips() if not args.clips or r["clip"] in args.clips]

    for clip in clips:
        src = ROOT / "data" / "samples" / clip
        if not src.exists():
            print(f"пропускаю {clip}: немає {src} (python download_samples.py)")
            continue
        for tracker in args.trackers:
            print(f"\n=== {clip} · {tracker}")
            stem = Path(clip).stem
            stats = track_video(
                str(src),
                output=str(ROOT / "out" / f"{stem}_{tracker}.mp4") if args.video else None,
                tracker=tracker,
                device=args.device,
                events_csv=str(ROOT / "results" / "events" / f"{stem}_{tracker}.csv") if tracker != "none" else None,
            )
            append_result(args.results, stats)
            counts = {k: v for k, v in stats.items() if k.startswith(("person", "cyclist"))}
            print(f"    {stats['fps']} FPS @ {stats['imgsz']} на {stats['device']} | {counts}")

    print("\n=== оцінка проти ручного підрахунку")
    subprocess.run([sys.executable, str(ROOT / "eval_count.py"), "--results", args.results])


if __name__ == "__main__":
    main()
