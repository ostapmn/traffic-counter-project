"""Шар підрахунку: геометрія лінії + машина станів із гістерезисом.

Відновлено з guide.md (розділи 18–20). Чиста логіка — без torch, OpenCV і камери,
тому покрита тестами в tests/test_counter.py.

Напрямки: A — рух із боку side<0 у бік side>0, B — навпаки (як у draw_line.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

Point = tuple[float, float]

NO_CROSS = 0
TO_POSITIVE = 1   # напрямок A
TO_NEGATIVE = -1  # напрямок B
DIRECTION_NAME = {TO_POSITIVE: "A", TO_NEGATIVE: "B"}


def _orientation(a: Point, b: Point, c: Point) -> float:
    """Векторний добуток (b - a) x (c - a); знак каже, з якого боку прямої a→b точка c."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    """Колінеарна точка c лежить у межах обмежувального прямокутника a-b."""
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    )


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point, eps: float = 1e-9) -> bool:
    """Чи мають замкнені відрізки a1-a2 і b1-b2 спільну точку.

    Саме ця перевірка відрізняє справжній перетин від руху повз *продовження* лінії:
    знак signed_side змінюється і там, бо пряма нескінченна, а ворота — ні.
    """
    d1 = _orientation(b1, b2, a1)
    d2 = _orientation(b1, b2, a2)
    d3 = _orientation(a1, a2, b1)
    d4 = _orientation(a1, a2, b2)

    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    ):
        return True

    # колінеарні випадки / дотик
    if abs(d1) <= eps and _on_segment(b1, b2, a1):
        return True
    if abs(d2) <= eps and _on_segment(b1, b2, a2):
        return True
    if abs(d3) <= eps and _on_segment(a1, a2, b1):
        return True
    if abs(d4) <= eps and _on_segment(a1, a2, b2):
        return True
    return False


@dataclass(frozen=True)
class Line:
    """Лінія підрахунку — відрізок p1→p2 у пікселях кадру."""

    p1: Point
    p2: Point

    @property
    def length(self) -> float:
        return ((self.p2[0] - self.p1[0]) ** 2 + (self.p2[1] - self.p1[1]) ** 2) ** 0.5

    def signed_side(self, pt: Point) -> float:
        return _orientation(self.p1, self.p2, pt)

    def perp_distance(self, pt: Point) -> float:
        """Перпендикулярна відстань зі знаком, у пікселях — не залежить від довжини лінії."""
        length = self.length
        if length < 1e-9:
            return 0.0
        return self.signed_side(pt) / length

    def crossing(self, prev: Point, curr: Point) -> int:
        """Напрямок, у якому рух prev→curr перетинає відрізок. Спершу геометрія, потім знак."""
        if not segments_intersect(prev, curr, self.p1, self.p2):
            return NO_CROSS
        before = self.signed_side(prev)
        after = self.signed_side(curr)
        if before <= 0.0 < after:
            return TO_POSITIVE
        if before >= 0.0 > after:
            return TO_NEGATIVE
        return NO_CROSS

    def near(self, pt: Point, band_px: float) -> bool:
        """Точка в смузі ±band_px навколо відрізка (а не його продовження)."""
        if abs(self.perp_distance(pt)) > band_px:
            return False
        dx, dy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        t = ((pt[0] - self.p1[0]) * dx + (pt[1] - self.p1[1]) * dy) / max(self.length ** 2, 1e-9)
        return 0.0 <= t <= 1.0


def anchor_point(bbox: Sequence[float], mode: str = "bottom_center") -> Point:
    """Рамка xyxy → одна точка, траєкторію якої рахуємо.

    bottom_center ≈ точка контакту з землею. Центроїд висить на пів зросту над ногами,
    і при погляді під кутом зсуває момент перетину в різні боки для двох напрямків.
    """
    x1, y1, x2, y2 = bbox[:4]
    if mode == "center":
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if mode == "bottom_center":
        return ((x1 + x2) / 2.0, float(y2))
    raise ValueError(f"unknown anchor mode: {mode!r}")


@dataclass
class CounterConfig:
    """Параметри підрахунку.

    hysteresis_px — напівширина мертвої зони в перпендикулярних пікселях. Пішохід на 720p
    має 137–205 px зросту, тож 8 px ≈ 5 % — більше за дрижання детектора, менше за крок.

    min_track_len — скільки спостережень потрібно треку, щоб його перетин рахувався.
    Має лишатись нижче бюджету кадрів на перетин у *найшвидшого* класу, інакше перетини
    зникають мовчки. Піднімати лише разом із виміряним FPS.

    debounce_frames — кулдаун на трек після події (проти топтання на лінії); 0 = вимкнено.
    """

    hysteresis_px: float = 8.0
    min_track_len: int = 3
    debounce_frames: int = 0
    anchor: str = "bottom_center"
    max_track_idle: int = 150


@dataclass
class Event:
    frame: int
    track_id: int
    label: str
    direction: str  # "A" | "B"


@dataclass
class _TrackState:
    n_obs: int = 0
    last_frame: int = 0
    stable_side: int = 0
    stable_point: Point | None = None
    last_event_frame: int | None = None


@dataclass
class LineCounter:
    """Машина станів із гістерезисом для однієї лінії (guide.md, розділ 19)."""

    line: Line
    config: CounterConfig = field(default_factory=CounterConfig)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    _tracks: dict[int, _TrackState] = field(default_factory=dict)

    def update(self, frame_idx: int, track_id: int, bbox: Sequence[float], label: str = "person") -> Event | None:
        """Одне спостереження треку. Повертає подію, якщо трек щойно перетнув лінію."""
        cfg = self.config
        st = self._tracks.setdefault(track_id, _TrackState())
        st.n_obs += 1
        st.last_frame = frame_idx
        anchor = anchor_point(bbox, cfg.anchor)

        distance = self.line.perp_distance(anchor)
        if distance > cfg.hysteresis_px:
            side = 1
        elif distance < -cfg.hysteresis_px:
            side = -1
        else:
            return None  # у мертвій зоні стабільний бік не змінюється

        if st.stable_side == 0:
            # Перший раз однозначно з одного боку: порівнювати нема з чим, тож трек,
            # народжений за лінією після фрагментації, не дасть подвійного підрахунку.
            st.stable_side = side
            st.stable_point = anchor
            return None

        if side == st.stable_side:
            st.stable_point = anchor  # освіжаємо точку відліку
            return None

        # Бік перевернувся: перевіряємо, що рух перетнув саме відрізок, а не продовження.
        direction = self.line.crossing(st.stable_point, anchor) if st.stable_point else NO_CROSS
        event = None
        if direction != NO_CROSS and self._may_count(st, frame_idx):
            st.last_event_frame = frame_idx
            name = DIRECTION_NAME[direction]
            per_label = self.counts.setdefault(label, {"A": 0, "B": 0})
            per_label[name] += 1
            event = Event(frame_idx, track_id, label, name)

        # Фіксуємо новий бік у будь-якому разі, щоб трек, що обійшов лінію з торця,
        # не перевірявся щокадру.
        st.stable_side = side
        st.stable_point = anchor
        return event

    def _may_count(self, st: _TrackState, frame_idx: int) -> bool:
        cfg = self.config
        if st.n_obs < cfg.min_track_len:
            return False
        if cfg.debounce_frames > 0 and st.last_event_frame is not None:
            if frame_idx - st.last_event_frame < cfg.debounce_frames:
                return False
        return True

    def prune(self, frame_idx: int) -> None:
        """Забути треки, яких не було max_track_idle кадрів (пам'ять на довгих прогонах)."""
        idle = self.config.max_track_idle
        for tid in [t for t, s in self._tracks.items() if frame_idx - s.last_frame > idle]:
            del self._tracks[tid]


class NaiveZoneCounter:
    """Baseline (a): без трекінгу. Рахує прихід детекцій у смугу біля лінії.

    Кожен кадр рахуємо, скільки опорних точок у смузі ±band_px. Якщо їх стало більше, ніж
    на попередньому кадрі, різницю вважаємо новими людьми. Без треків немає ні ідентичності,
    ні напрямку, тому baseline звітує лише суму обох напрямків.
    """

    def __init__(self, line: Line, band_px: float = 40.0, anchor: str = "bottom_center"):
        self.line, self.band_px, self.anchor = line, band_px, anchor
        self.prev: dict[str, int] = {}
        self.counts: dict[str, dict[str, int]] = {}

    def update(self, boxes_by_label: dict[str, list[Sequence[float]]]) -> None:
        for label in set(boxes_by_label) | set(self.prev):
            n = int(sum(self.line.near(anchor_point(b, self.anchor), self.band_px)
                        for b in boxes_by_label.get(label, [])))
            if n > self.prev.get(label, 0):
                per_label = self.counts.setdefault(label, {"total": 0})
                per_label["total"] += n - self.prev.get(label, 0)
            self.prev[label] = n
