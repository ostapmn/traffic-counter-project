"""Малює лінію підрахунку та стрілки напрямків A/B на відео — для ручного підрахунку.

Запуск:
    python draw_line.py data/samples/people-walking.mp4 -o data/people_line.mp4
    python draw_line.py data/samples/market-square.mp4 -o data/market_line.mp4 \
        --line 0 2300 2160 2300 --scale 0.5

Лінія та сама, що в `track.py --line`. Напрямок A — рух із боку side<0 у бік side>0
(для лінії зліва направо — вниз по кадру), B — навпаки.
"""

import argparse

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)


def side(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--line", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--scale", type=float, default=1.0, help="зменшити відео, напр. 0.5")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # лінія в координатах оригіналу, як у track.py; за замовчуванням горизонталь посередині
    x1, y1, x2, y2 = args.line or (0, h // 2, w, h // 2)
    a, b = (x1, y1), (x2, y2)

    # стрілки ставимо посередині лінії: нормаль напрямку side<0 → side>0 — це напрямок A
    mid = ((x1 + x2) // 2, (y1 + y2) // 2)
    # side(p) = (x2-x1)*(py-y1) - (y2-y1)*(px-x1), тож зростає вздовж (-(y2-y1), x2-x1)
    nx, ny = -(y2 - y1), (x2 - x1)
    norm = (nx**2 + ny**2) ** 0.5
    ux, uy = nx / norm, ny / norm
    size = h // 10
    to_a = (int(mid[0] + ux * size), int(mid[1] + uy * size))
    to_b = (int(mid[0] - ux * size), int(mid[1] - uy * size))
    assert side(to_a, a, b) > 0 > side(to_b, a, b)

    out_w, out_h = int(w * args.scale), int(h * args.scale)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    thick = max(2, h // 400)
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.line(frame, a, b, RED, thick * 2)
        cv2.arrowedLine(frame, mid, to_a, RED, thick * 2, tipLength=0.3)
        cv2.arrowedLine(frame, mid, to_b, RED, thick * 2, tipLength=0.3)
        scale = h / 1080
        cv2.putText(frame, "A", (to_a[0] + 15, to_a[1]), FONT, 2.5 * scale, RED, thick * 2)
        cv2.putText(frame, "B", (to_b[0] + 15, to_b[1]), FONT, 2.5 * scale, RED, thick * 2)
        # час допомагає рахувати з паузами: «на 0:07 перетнув»
        cv2.putText(frame, f"{frame_no / fps:5.1f} s", (30, int(90 * scale)),
                    FONT, 2.5 * scale, RED, thick * 2)
        writer.write(cv2.resize(frame, (out_w, out_h)) if args.scale != 1 else frame)
        frame_no += 1

    cap.release()
    writer.release()
    print(f"готово: {args.output}, кадрів: {frame_no}")


if __name__ == "__main__":
    main()
