# Лічильник учасників дорожнього руху

Семестровий проєкт курсу «Штучний інтелект», УКУ, 2026. Детектуємо, трекаємо й рахуємо
пішоходів і велосипедистів, що перетинають віртуальну лінію; назовні йдуть лише знеособлені
метадані `(час, клас, напрямок, track_id)`. Ціль — похибка підрахунку ≤ 10 % (MAPE) проти
ручного підрахунку на денних записах.

Пропозиція — [proposal.pdf](proposal.pdf), покроковий план — [plan.md](plan.md), теорія й
обґрунтування кожного рішення — [guide.md](guide.md), правила ручного підрахунку —
[protocol.md](protocol.md).

## Як запустити після клонування

Потрібен Python 3.10+.

```bash
git clone https://github.com/ostapmn/traffic-counter-project.git
cd traffic-counter-project

# 1. PyTorch. На NVIDIA — збірка з CUDA (для RTX 50xx потрібна cu128 або новіша):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#    На Mac і на CPU-машинах (зокрема Raspberry Pi) достатньо звичайного:
#    pip install torch torchvision

# 2. Решта залежностей
pip install -r requirements.txt

# 3. Тестові відео (у git їх немає)
python download_samples.py

# 4. Перевірка
pytest -q
python track.py data/samples/people-walking.mp4 -o out/people-walking.mp4
```

Пристрій обирається сам: `cuda` → `mps` → `cpu`. Ваги `yolo11n.pt` ultralytics скачає
під час першого запуску.

## Що де лежить

| Файл | Що робить |
|---|---|
| [track.py](track.py) | детекція (YOLO11n, COCO) + трекінг + підрахунок перетинів лінії на одному відео |
| [counter.py](counter.py) | геометрія лінії й машина станів із гістерезисом (guide.md, розділи 18–20); baseline без трекінгу |
| [sort_tracker.py](sort_tracker.py) | SORT — baseline (b) для трекінгу (guide.md, розділи 10–12) |
| [eval_count.py](eval_count.py) | MAPE / WAPE / bias проти ручного підрахунку (guide.md, розділ 23) |
| [benchmark.py](benchmark.py) | усі кліпи × усі варіанти → `results/runs.csv` → оцінка |
| [clips.csv](clips.csv) | лінія підрахунку й розмір входу для кожного кліпу — спільні для всіх скриптів |
| [draw_line.py](draw_line.py) | відео з лінією, стрілками A/B і таймером — для ручного підрахунку |
| [download_samples.py](download_samples.py) | завантажує тестові відео в `data/samples/` |
| [gt/gt.csv](gt/gt.csv) | ручні підрахунки (ground truth) |
| [tests/](tests/) | тести на чисту логіку: геометрія, гістерезис, SORT, метрика |
| [test_detect.ipynb](test_detect.ipynb) | перша проба детектора на одному кадрі |

## Три варіанти підрахунку

```bash
python track.py data/samples/people-walking.mp4 --tracker bytetrack   # основний
python track.py data/samples/people-walking.mp4 --tracker sort        # baseline (b): YOLO + SORT
python track.py data/samples/people-walking.mp4 --tracker none        # baseline (a): без трекінгу
```

Корисні параметри: `--line X1 Y1 X2 Y2` (інакше з `clips.csv`), `--imgsz` (інакше з
`clips.csv`), `--device`, `--hysteresis`, `--min-track-len`, `--events events.csv`.

Напрямок **A** — рух із боку `side < 0` у бік `side > 0`; для лінії зліва направо це рух
униз по кадру. Перед підрахунком перевірте на відео з `draw_line.py`, куди дивиться стрілка A.

**Велосипедисти.** COCO дає людину й велосипед окремими рамками, тож людина, під якою є
велосипед, рахується як `cyclist`, а сам велосипед не рахується — інакше один велосипедист
давав би два перетини.

**Розмір входу.** `market-square.mp4` і `subway.mp4` — вертикальне 4K. На вході 640 люди там
стискаються до кількох пікселів і детектор їх не бачить, тому в `clips.csv` для них 1920 і 1280.

## Оцінка похибки

1. Згенерувати відео з лінією в `data/gt_videos/`: `python download_samples.py --gt-videos`.
2. Порахувати перетини вручну за [protocol.md](protocol.md) і вписати в [gt/gt.csv](gt/gt.csv).
3. `python benchmark.py` — прогнати всі варіанти й порахувати MAPE / WAPE / bias.
   Або лише оцінка останніх прогонів: `python eval_count.py`.

Цифри похибки валідні лише разом із виміряним FPS і розміром входу, тому `results/runs.csv`
зберігає їх у кожному рядку.

## Ліцензія

AGPL-3.0 — бо детектор з Ultralytics під AGPL-3.0 (деталі в шапці [track.py](track.py)).
Сирі відео не публікуються; відкритими стають лише агреговані метадані (CC BY 4.0).
