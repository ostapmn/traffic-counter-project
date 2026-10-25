"""SORT — baseline (b) для трекінгу (Bewley et al., ICIP 2016; guide.md, розділи 10–12).

Калман із постійною швидкістю + IoU + угорський алгоритм, без зовнішнього вигляду.
Параметри за замовчуванням — як в оригінальній статті: max_age=1, min_hits=3, iou=0.3.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Попарний IoU рамок xyxy: (N, 4) x (M, 4) → (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def _to_z(box: np.ndarray) -> np.ndarray:
    """xyxy → [cx, cy, s, r]: центр, площа, співвідношення сторін (стан SORT)."""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-9)])


def _to_box(x: np.ndarray) -> np.ndarray:
    s, r = max(x[2], 1e-9), max(x[3], 1e-9)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-9)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class _KalmanBox:
    """Стан [cx, cy, s, r, vcx, vcy, vs]; r вважаємо сталим (як у SORT)."""

    def __init__(self, box: np.ndarray):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # швидкість невідома — велика невизначеність
        self.x = np.zeros(7)
        self.x[:4] = _to_z(box)

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _to_box(self.x)

    def update(self, box: np.ndarray) -> None:
        y = _to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


class _Track:
    def __init__(self, box: np.ndarray, track_id: int):
        self.kf = _KalmanBox(box)
        self.id = track_id
        self.hits = 0
        self.hit_streak = 0  # як в оригіналі: трек підтверджується після min_hits збігів
        self.time_since_update = 0
        self.box = box


class Sort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.tracks: list[_Track] = []
        self.frame_count = 0
        self._next_id = 1

    def update(self, dets: np.ndarray) -> np.ndarray:
        """dets: (N, 4) xyxy цього кадру. Повертає (K, 5): xyxy + track_id підтверджених треків."""
        self.frame_count += 1
        dets = np.asarray(dets, float).reshape(-1, 4)

        predicted = np.array([t.kf.predict() for t in self.tracks]).reshape(-1, 4)
        for t in self.tracks:
            t.time_since_update += 1
            if t.time_since_update > 1:
                t.hit_streak = 0

        matches, unmatched = [], set(range(len(dets)))
        if len(dets) and len(self.tracks):
            iou = iou_matrix(dets, predicted)
            rows, cols = linear_sum_assignment(-iou)  # угорський: максимізуємо сумарний IoU
            for d, t in zip(rows, cols):
                if iou[d, t] >= self.iou_threshold:  # гейт: слабкі пари не приймаємо
                    matches.append((d, t))
                    unmatched.discard(d)

        for d, t in matches:
            tr = self.tracks[t]
            tr.kf.update(dets[d])
            tr.box = _to_box(tr.kf.x)  # SORT віддає згладжений стан, а не сиру детекцію
            tr.time_since_update = 0
            tr.hits += 1
            tr.hit_streak += 1

        for d in sorted(unmatched):
            self.tracks.append(_Track(dets[d], self._next_id))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        out = [
            [*t.box, t.id]
            for t in self.tracks
            if t.time_since_update == 0
            and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
        ]
        return np.array(out, float).reshape(-1, 5)
