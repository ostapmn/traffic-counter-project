"""Налаштування кліпів (лінія підрахунку + розмір входу) з clips.csv — спільні для всіх скриптів."""

from __future__ import annotations

import csv
from pathlib import Path

CLIPS_CSV = Path(__file__).with_name("clips.csv")


def all_clips() -> list[dict]:
    if not CLIPS_CSV.exists():
        return []
    with open(CLIPS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clip_config(source: str) -> dict | None:
    """Рядок clips.csv для цього відео (за іменем файлу), або None."""
    return next((r for r in all_clips() if r["clip"] == Path(source).name), None)


def clip_line(row: dict) -> tuple[int, int, int, int]:
    return tuple(int(row[k]) for k in ("x1", "y1", "x2", "y2"))
