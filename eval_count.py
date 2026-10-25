"""Похибка підрахунку проти ручного ground truth: MAPE, WAPE, bias (guide.md, розділ 23).

Бакет — (кліп, клас, напрямок). Для baseline (a) без трекінгу напрямку немає, тому він
оцінюється лише по сумі обох напрямків; для чесного порівняння ту саму «суму» показуємо
й для трекерів.

    MAPE = mean |pred − gt| / gt     по бакетах з gt > 0 — заявлена метрика пропозиції
    WAPE = Σ|pred − gt| / Σgt        кожна людина важить однаково — стабільніша на малих числах
    bias = Σ(pred − gt) / Σgt        знак: > 0 перераховуємо, < 0 недораховуємо

Бакети з gt = 0 MAPE оцінити не може; їх не викидаємо, а звітуємо окремо разом із хибними
спрацюваннями в них — інакше саме тихі періоди сховали б помилки.

Запуск:
    python eval_count.py                         # results/runs.csv проти gt/gt.csv
    python eval_count.py --results results/runs.csv --gt gt/gt.csv --target 10
Код виходу 1, якщо MAPE пішоходів у bytetrack вища за --target (придатне для CI).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

TRACKERS = ["bytetrack", "sort", "none"]


@dataclass
class Report:
    buckets_scored: int = 0
    empty_buckets: int = 0
    false_positives_in_empty_buckets: int = 0
    gt: int = 0
    pred: int = 0
    mape: float | None = None
    wape: float | None = None
    bias: float | None = None

    def meets_target(self, target: float = 10.0) -> bool:
        # немає оцінених бакетів — немає MAPE, і це не може читатися як успіх
        return self.mape is not None and self.mape <= target


def compare(pred: dict, gt: dict) -> Report:
    """pred, gt: {ключ бакета: кількість}. Ключі, яких немає в gt, вважаються gt = 0."""
    r = Report()
    ape, abs_err, signed = [], 0, 0
    for key in sorted(set(pred) | set(gt), key=str):
        p, g = pred.get(key, 0), gt.get(key, 0)
        if g == 0:
            r.empty_buckets += 1
            r.false_positives_in_empty_buckets += p
            continue
        r.buckets_scored += 1
        r.gt += g
        r.pred += p
        ape.append(abs(p - g) / g)
        abs_err += abs(p - g)
        signed += p - g
    if r.buckets_scored:
        r.mape = 100 * sum(ape) / len(ape)
        r.wape = 100 * abs_err / r.gt
        r.bias = 100 * signed / r.gt
    return r


def load_gt(path: str) -> dict[str, dict]:
    """gt.csv → {clip: {"person_A":…, "person_B":…, ["cyclist_A", "cyclist_B"]}}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("clip") or not row.get("dir_a"):
                continue
            g = {"person_A": int(row["dir_a"]), "person_B": int(row["dir_b"])}
            if row.get("cyclist_a", "").strip():
                g["cyclist_A"] = int(row["cyclist_a"])
                g["cyclist_B"] = int(row["cyclist_b"])
            out[row["clip"]] = g
    return out


def load_runs(path: str) -> dict[tuple[str, str], dict]:
    """runs.csv → останній прогін для кожної пари (кліп, трекер)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["clip"], row["tracker"])] = row
    return out


def buckets(gt: dict, runs: dict, tracker: str, classes: list[str], by_direction: bool):
    pred_b, gt_b = {}, {}
    for clip, g in gt.items():
        run = runs.get((clip, tracker))
        if run is None:
            continue
        for c in classes:
            if f"{c}_A" not in g:
                continue
            if by_direction:
                for d in "AB":
                    gt_b[(clip, c, d)] = g[f"{c}_{d}"]
                    pred_b[(clip, c, d)] = int(run[f"{c}_{d}"])
            else:
                gt_b[(clip, c)] = g[f"{c}_A"] + g[f"{c}_B"]
                pred_b[(clip, c)] = int(run[f"{c}_total"])
    return pred_b, gt_b


def fmt(v: float | None) -> str:
    return "   —   " if v is None else f"{v:6.2f} %"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/runs.csv")
    ap.add_argument("--gt", default="gt/gt.csv")
    ap.add_argument("--target", type=float, default=10.0, help="ціль MAPE, %")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    if not gt:
        print(f"{args.gt}: немає жодного заповненого рядка — спершу ручний підрахунок (protocol.md)")
        return 1
    if not Path(args.results).exists():
        print(f"{args.results} не існує — спершу python benchmark.py")
        return 1
    runs = load_runs(args.results)

    print(f"кліпів з ground truth: {len(gt)} ({', '.join(gt)})\n")
    print("1) Пішоходи, по напрямках (бакет = кліп × напрямок)")
    print(f"   {'трекер':<10} {'бакетів':>7} {'gt':>5} {'pred':>5}   {'MAPE':>8}  {'WAPE':>8}  {'bias':>8}  FP у порожніх")
    main_report = None
    for t in ["bytetrack", "sort"]:
        r = compare(*buckets(gt, runs, t, ["person"], True))
        main_report = r if t == "bytetrack" else main_report
        print(f"   {t:<10} {r.buckets_scored:>7} {r.gt:>5} {r.pred:>5}   {fmt(r.mape)}  {fmt(r.wape)}  {fmt(r.bias)}  "
              f"{r.false_positives_in_empty_buckets} (у {r.empty_buckets})")

    print("\n2) Пішоходи, сума напрямків (так можна порівняти й baseline без трекінгу)")
    for t in TRACKERS:
        r = compare(*buckets(gt, runs, t, ["person"], False))
        print(f"   {t:<10} {r.buckets_scored:>7} {r.gt:>5} {r.pred:>5}   {fmt(r.mape)}  {fmt(r.wape)}  {fmt(r.bias)}")

    if any("cyclist_A" in g for g in gt.values()):
        print("\n3) Велосипедисти, по напрямках")
        for t in ["bytetrack", "sort"]:
            r = compare(*buckets(gt, runs, t, ["cyclist"], True))
            print(f"   {t:<10} {r.buckets_scored:>7} {r.gt:>5} {r.pred:>5}   {fmt(r.mape)}  {fmt(r.wape)}  {fmt(r.bias)}  "
                  f"{r.false_positives_in_empty_buckets} (у {r.empty_buckets})")

    print("\n   по кліпах (пішоходи A/B: gt → bytetrack / sort; сума: none)")
    for clip, g in gt.items():
        bt, so, no = (runs.get((clip, t), {}) for t in TRACKERS)
        print(f"   {clip:<24} A {g['person_A']:>3} → {bt.get('person_A', '—'):>3} / {so.get('person_A', '—'):>3}"
              f"   B {g['person_B']:>3} → {bt.get('person_B', '—'):>3} / {so.get('person_B', '—'):>3}"
              f"   сума {g['person_A'] + g['person_B']:>3} → none {no.get('person_total', '—')}"
              f"   | FPS {bt.get('fps', '—')} @ {bt.get('imgsz', '—')} на {bt.get('device', '—')}")

    ok = main_report is not None and main_report.meets_target(args.target)
    print(f"\nціль MAPE ≤ {args.target:g} % для пішоходів (bytetrack): {'так' if ok else 'ні'}")
    print("примітка: похибка валідна лише разом із виміряним FPS і розміром входу (plan.md)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
