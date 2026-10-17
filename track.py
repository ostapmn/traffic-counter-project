"""Детекція + трекінг на відео.

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

Дані (не входять до репозиторію):
    ваги yolo11n.pt      AGPL-3.0
    тестові відео        ліцензії їхніх власників
    MOT17 / MOT20        CC BY-NC-SA 4.0 — лише інференс, без донавчання

Запуск:
    python track.py data/samples/people-walking.mp4
    python track.py data/samples/people-walking.mp4 -o out.mp4 --classes 0 1
"""

from __future__ import annotations

import argparse
import time

import cv2
import torch
from ultralytics import YOLO

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.5
THICK = 1


def pick_device() -> str:
    """mps на Apple Silicon, інакше cpu."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


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


def side(p, a, b) -> float:
    """Півплощина точки p відносно лінії a→b: >0 з одного боку, <0 з іншого."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed(p0, p1, a, b) -> bool:
    """Чи відрізок руху p0→p1 перетинає відрізок лінії a→b (а не лише її продовження)."""
    return side(p0, a, b) * side(p1, a, b) < 0 and side(a, p0, p1) * side(b, p0, p1) < 0


def track_video(
    source: str,
    output: str | None = None,
    weights: str = "yolo11n.pt",
    classes: tuple[int, ...] = (0, 1),
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    line: tuple[int, int, int, int] | None = None,
) -> dict:
    """Прогнати відео через детектор + ByteTrack, намалювати рамки й порахувати перетини лінії.

    classes: id класів COCO. 0=person, 1=bicycle, 2=car, 5=bus, 7=truck.
    line: (x1, y1, x2, y2); за замовчуванням горизонталь посередині кадру.
    Напрямок A: рух із боку side<0 у бік side>0, B — навпаки.
    Повертає статистику прогону.
    """
    device = device or pick_device()
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
        writer = cv2.VideoWriter(
            output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError(f"не пишеться: {output}")

    a, b = (line[:2], line[2:]) if line else ((0, h // 2), (w, h // 2))
    last_point: dict[int, tuple[float, float]] = {}
    counted: set[int] = set()  # кожен трек рахуємо один раз; гістерезис — задача T3
    counts = {"A": 0, "B": 0}

    seen_ids: set[int] = set()
    frames = 0
    started = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # persist=True — трекер зберігає стан між кадрами. Без нього трекінгу немає.
        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=list(classes), conf=conf, imgsz=imgsz,
            device=device, verbose=False,
        )[0]

        boxes = result.boxes
        # .id буває None (на відміну від .conf, яка просто порожня) — перевірка обов'язкова
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for box, tid, c, p in zip(xyxy, ids, cls, confs):
                seen_ids.add(int(tid))

                # точка опори — центр низу рамки (ноги), вона стабільніша за центр рамки
                point = ((box[0] + box[2]) / 2, box[3])
                prev = last_point.get(int(tid))
                if prev and int(tid) not in counted and crossed(prev, point, a, b):
                    counts["A" if side(point, a, b) > 0 else "B"] += 1
                    counted.add(int(tid))
                last_point[int(tid)] = point

                label = f"#{tid} {result.names[c]} {p:.2f}"
                draw_box(frame, box, label, color_for(int(tid)))

        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 0, 255), 3)
        cv2.putText(frame, f"A: {counts['A']}  B: {counts['B']}", (30, 80),
                    FONT, 2, (0, 0, 255), 4)

        frames += 1
        if writer:
            writer.write(frame)
        if frames % 50 == 0:
            print(f"  {frames}/{total} кадрів, треків: {len(seen_ids)}")

    elapsed = time.perf_counter() - started
    cap.release()
    if writer:
        writer.release()

    return {
        "frames": frames,
        "seconds": round(elapsed, 1),
        "fps": round(frames / elapsed, 1) if elapsed else 0.0,
        "tracks": len(seen_ids),
        "A": counts["A"],
        "B": counts["B"],
        "device": device,
        "imgsz": imgsz,
        "output": output,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="відеофайл")
    ap.add_argument("-o", "--output", help="куди записати розмічене відео")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1],
                    help="id класів COCO (0=person, 1=bicycle)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="mps | cpu")
    ap.add_argument("--line", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="лінія підрахунку в пікселях; за замовчуванням горизонталь посередині")
    args = ap.parse_args()

    stats = track_video(
        args.source, args.output, args.weights, tuple(args.classes),
        args.conf, args.imgsz, args.device,
        tuple(args.line) if args.line else None,
    )
    print()
    for k, v in stats.items():
        print(f"  {k:<10} {v}")


if __name__ == "__main__":
    main()
