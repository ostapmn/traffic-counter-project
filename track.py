"""Детекція + трекінг + підрахунок перетинів лінії на відео.

Ліцензія: AGPL-3.0 (див. LICENSE).

Цей файл використовує Ultralytics YOLO, яка поширюється під AGPL-3.0,
тому й сам зобов'язаний бути під AGPL-3.0. Якщо потрібна дозвільна
ліцензія — доведеться замінити детектор (наприклад NanoDet, Apache-2.0,
крок 18 плану) або купити комерційну ліцензію в Ultralytics.

Сторонні компоненти:
    ultralytics   AGPL-3.0   ← визначає ліцензію всього проєкту
    supervision   MIT
    opencv        Apache-2.0
    torch         BSD-3-Clause
    scipy         BSD-3-Clause

Дані (не входять до репозиторію):
    ваги yolo11n.pt      AGPL-3.0
    тестові відео        ліцензії їхніх власників (python download_samples.py)
    MOT17 / MOT20        CC BY-NC-SA 4.0 — лише інференс, без донавчання

Трекери (--tracker):
    bytetrack   основний варіант (ByteTrack з ultralytics, дефолтні гіперпараметри)
    sort        baseline (b): SORT, sort_tracker.py
    none        baseline (a): без трекінгу, лише прихід детекцій у смугу біля лінії

Запуск:
    python track.py data/samples/people-walking.mp4
    python track.py data/samples/market-square.mp4 -o out/market.mp4
    python track.py data/samples/people-walking.mp4 --tracker sort --line 0 540 1920 540

Лінія й розмір входу, якщо не задані явно, беруться з clips.csv за іменем файлу.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from clips import clip_config, clip_line
from counter import CounterConfig, Line, LineCounter, NaiveZoneCounter
from sort_tracker import Sort

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.5
THICK = 1
PERSON, BICYCLE = 0, 1  # id класів COCO
RESULT_FIELDS = ["time", "clip", "tracker", "imgsz", "device", "frames", "fps",
                 "person_A", "person_B", "cyclist_A", "cyclist_B", "person_total", "cyclist_total"]


def pick_device() -> str:
    """cuda на NVIDIA, mps на Apple Silicon, інакше cpu (зокрема Raspberry Pi)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def color_for(track_id: int) -> tuple[int, int, int]:
    """Стабільний колір на кожен ID — ID switch одразу видно по стрибку кольору."""
    return (track_id * 37 % 255, track_id * 97 % 255, track_id * 57 % 255)


def draw_box(frame, box, label, color) -> None:
    """Рамка + підпис, що не вилазить за краї кадру."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), base = cv2.getTextSize(label, FONT, SCALE, THICK)
    # по горизонталі: не вилізти за правий край
    tx = min(x1, w - tw - 2)
    # по вертикалі: над рамкою, а якщо місця немає — під верхньою межею
    ty = y1 - base - 2 if y1 - th - base - 2 >= 0 else y1 + th + base + 2

    cv2.rectangle(frame, (tx, ty - th - base), (tx + tw, ty + base), color, -1)
    cv2.putText(frame, label, (tx, ty), FONT, SCALE, (255, 255, 255), THICK)


def riders(persons: np.ndarray, bikes: np.ndarray, min_cover: float = 0.3) -> np.ndarray:
    """Які з людей сидять на велосипеді.

    COCO дає людину й велосипед окремими рамками, тож без цього велосипедист рахувався б
    двічі. Людина — велосипедист, якщо велосипед перекритий її рамкою хоча б на min_cover
    своєї площі, а центр велосипеда під нею по горизонталі. Обмеження: людина, що веде
    велосипед поруч, теж може потрапити сюди.
    """
    out = np.zeros(len(persons), bool)
    if len(persons) == 0 or len(bikes) == 0:
        return out
    for i, p in enumerate(persons):
        ix = np.clip(np.minimum(p[2], bikes[:, 2]) - np.maximum(p[0], bikes[:, 0]), 0, None)
        iy = np.clip(np.minimum(p[3], bikes[:, 3]) - np.maximum(p[1], bikes[:, 1]), 0, None)
        bike_area = (bikes[:, 2] - bikes[:, 0]) * (bikes[:, 3] - bikes[:, 1])
        cx = (bikes[:, 0] + bikes[:, 2]) / 2
        out[i] = bool(np.any((ix * iy / np.maximum(bike_area, 1e-9) >= min_cover)
                             & (cx >= p[0]) & (cx <= p[2])))
    return out


def track_video(
    source: str,
    output: str | None = None,
    weights: str = "yolo11n.pt",
    tracker: str = "bytetrack",
    conf: float = 0.25,
    imgsz: int | None = None,
    device: str | None = None,
    line: tuple[int, int, int, int] | None = None,
    counter_cfg: CounterConfig | None = None,
    events_csv: str | None = None,
    zone_px: float = 40.0,
) -> dict:
    """Прогнати відео через детектор + трекер і порахувати перетини лінії.

    line: (x1, y1, x2, y2) у пікселях; за замовчуванням — із clips.csv, інакше горизонталь
    посередині кадру. Напрямок A: рух із боку side<0 у бік side>0, B — навпаки.
    Рахуються лише треки людей; людина на велосипеді йде в окремий клас cyclist.
    Повертає статистику прогону.
    """
    device = device or pick_device()
    clip = clip_config(source)
    imgsz = imgsz or (int(clip["imgsz"]) if clip else 640)
    model = YOLO(weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"не відкривається: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError(f"не пишеться: {output}")

    if line is None and clip:
        line = clip_line(clip)
    a, b = (line[:2], line[2:]) if line else ((0, h // 2), (w, h // 2))
    gate = Line(tuple(map(float, a)), tuple(map(float, b)))

    counter = NaiveZoneCounter(gate, zone_px) if tracker == "none" else LineCounter(gate, counter_cfg or CounterConfig())
    sort = Sort() if tracker == "sort" else None
    rider_votes: dict[int, list[int]] = {}  # track_id -> [кадрів на велосипеді, усього кадрів]
    events = []

    seen_ids: set[int] = set()
    frames = 0
    started = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if tracker == "bytetrack":
            # persist=True — трекер зберігає стан між кадрами. Без нього трекінгу немає.
            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                classes=[PERSON, BICYCLE], conf=conf, imgsz=imgsz,
                device=device, verbose=False,
            )[0]
        else:
            result = model.predict(
                frame, classes=[PERSON, BICYCLE], conf=conf, imgsz=imgsz,
                device=device, verbose=False,
            )[0]

        bx = result.boxes
        xyxy = bx.xyxy.cpu().numpy()
        cls = bx.cls.cpu().numpy().astype(int)
        confs = bx.conf.cpu().numpy()
        persons, bikes = xyxy[cls == PERSON], xyxy[cls == BICYCLE]
        on_bike = riders(persons, bikes)

        if tracker == "none":
            counter.update({"person": list(persons[~on_bike]), "cyclist": list(persons[on_bike])})
            for box, rider in zip(persons, on_bike):
                draw_box(frame, box, "cyclist" if rider else "person", (0, 160, 255))
        else:
            if tracker == "bytetrack":
                # .id буває None (на відміну від .conf, яка просто порожня) — перевірка обов'язкова
                ids = bx.id.cpu().numpy().astype(int) if bx.id is not None else np.zeros(0, int)
                mask = cls[: len(ids)] == PERSON
                tracked = [(xyxy[k], int(ids[k]), confs[k]) for k in np.flatnonzero(mask)]
                track_bikes = on_bike if len(ids) else np.zeros(0, bool)
            else:
                out = sort.update(persons)
                tracked = [(row[:4], int(row[4]), None) for row in out]
                # SORT повертає згладжені рамки — велосипед шукаємо для кожної заново
                track_bikes = riders(out[:, :4], bikes)

            for (box, tid, p), rider in zip(tracked, track_bikes):
                seen_ids.add(tid)
                votes = rider_votes.setdefault(tid, [0, 0])
                votes[0] += int(rider)
                votes[1] += 1
                label = "cyclist" if votes[0] * 2 >= votes[1] else "person"

                ev = counter.update(frames, tid, box, label)
                if ev:
                    events.append(ev)

                text = f"#{tid} {label}" + (f" {p:.2f}" if p is not None else "")
                draw_box(frame, box, text, color_for(tid))
            counter.prune(frames)

        if writer:
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 0, 255), 3)
            if tracker == "none":
                txt = "  ".join(f"{k}: {v['total']}" for k, v in counter.counts.items()) or "0"
            else:
                txt = "  ".join(f"{k} A:{v['A']} B:{v['B']}" for k, v in counter.counts.items()) or "A: 0  B: 0"
            cv2.putText(frame, txt, (30, 80), FONT, 1.5, (0, 0, 255), 3)
            writer.write(frame)

        frames += 1
        if frames % 50 == 0:
            print(f"  {frames}/{total} кадрів, треків: {len(seen_ids)}")

    elapsed = time.perf_counter() - started
    cap.release()
    if writer:
        writer.release()

    if events_csv:
        Path(events_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(events_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            # знеособлені метадані — саме те, що за пропозицією виходить за межі пристрою
            wr.writerow(["t_seconds", "class", "direction", "track_id"])
            for ev in events:
                wr.writerow([round(ev.frame / fps_in, 2), ev.label, ev.direction, ev.track_id])

    c = counter.counts
    get = lambda lbl, key: c.get(lbl, {}).get(key, 0)
    if tracker == "none":
        per = {"person_total": get("person", "total"), "cyclist_total": get("cyclist", "total")}
    else:
        per = {"person_A": get("person", "A"), "person_B": get("person", "B"),
               "cyclist_A": get("cyclist", "A"), "cyclist_B": get("cyclist", "B")}
        per["person_total"] = per["person_A"] + per["person_B"]
        per["cyclist_total"] = per["cyclist_A"] + per["cyclist_B"]

    return {
        "clip": Path(source).name,
        "tracker": tracker,
        "frames": frames,
        "seconds": round(elapsed, 1),
        "fps": round(frames / elapsed, 1) if elapsed else 0.0,
        "tracks": len(seen_ids),
        **per,
        "device": device,
        "imgsz": imgsz,
        "line": f"{int(a[0])} {int(a[1])} {int(b[0])} {int(b[1])}",
        "output": output,
    }


def append_result(path: str, stats: dict) -> None:
    """Дописати прогін у CSV, з яким працює eval_count.py."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if new:
            wr.writeheader()
        wr.writerow({"time": datetime.now().isoformat(timespec="seconds"), **stats})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="відеофайл")
    ap.add_argument("-o", "--output", help="куди записати розмічене відео")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--tracker", choices=["bytetrack", "sort", "none"], default="bytetrack")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None, help="розмір входу; за замовчуванням із clips.csv або 640")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu | 0; за замовчуванням автовибір")
    ap.add_argument("--line", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="лінія підрахунку в пікселях; за замовчуванням із clips.csv або горизонталь посередині")
    ap.add_argument("--hysteresis", type=float, default=8.0, help="напівширина мертвої зони, px")
    ap.add_argument("--min-track-len", type=int, default=3)
    ap.add_argument("--zone-px", type=float, default=40.0, help="напівширина смуги для --tracker none")
    ap.add_argument("--events", help="CSV подій (t_seconds, class, direction, track_id)")
    ap.add_argument("--results", help="дописати підсумок прогону в цей CSV (для eval_count.py)")
    args = ap.parse_args()

    stats = track_video(
        args.source, args.output, args.weights, args.tracker,
        args.conf, args.imgsz, args.device,
        tuple(args.line) if args.line else None,
        CounterConfig(hysteresis_px=args.hysteresis, min_track_len=args.min_track_len),
        args.events, args.zone_px,
    )
    if args.results:
        append_result(args.results, stats)
    print()
    for k, v in stats.items():
        print(f"  {k:<14} {v}")


if __name__ == "__main__":
    main()
