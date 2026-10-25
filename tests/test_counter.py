from counter import CounterConfig, Line, LineCounter, NaiveZoneCounter

GATE = Line((0.0, 100.0), (200.0, 100.0))  # униз по кадру = напрямок A


def vertical(x, ys):
    """Рамки, у яких ноги (низ-центр) проходять через точки (x, y)."""
    return [(x - 10.0, y - 50.0, x + 10.0, float(y)) for y in ys]


def walk(path, track_id=1, config=None, counter=None, label="person"):
    counter = counter or LineCounter(GATE, config or CounterConfig())
    events = [counter.update(f, track_id, box, label) for f, box in enumerate(path)]
    return [e for e in events if e]


def test_straight_crossing_counts_once_with_direction():
    events = walk(vertical(100.0, [40, 60, 80, 120, 140]))
    assert [e.direction for e in events] == ["A"]


def test_reverse_crossing_has_opposite_direction():
    events = walk(vertical(100.0, [160, 140, 120, 80, 60]))
    assert [e.direction for e in events] == ["B"]


def test_parallel_walk_does_not_count():
    path = [(x - 10.0, 0.0, x + 10.0, 50.0) for x in range(0, 200, 20)]
    assert walk(path) == []


def test_approach_and_turn_back_does_not_count():
    assert walk(vertical(100.0, [40, 60, 80, 95, 80, 60])) == []


def test_turn_inside_dead_band_does_not_count():
    assert walk(vertical(100.0, [40, 60, 80, 104, 96, 80, 60])) == []


def test_crossing_the_line_extension_does_not_count():
    assert walk(vertical(400.0, [40, 60, 80, 120, 140])) == []


def test_jitter_in_dead_band_does_not_count():
    assert walk(vertical(100.0, [40, 60, 80, 97, 103, 97, 103, 97])) == []


def test_track_born_on_the_far_side_does_not_count():
    """Фрагментований трек, підхоплений уже за лінією, не дає подвійного підрахунку.

    Новий track_id не має стабільного боку на першому однозначному спостереженні,
    тож перетинати йому нічого. Міняємо подвійний підрахунок на недорахунок.
    """
    assert walk(vertical(100.0, [120, 140, 160, 180]), track_id=77) == []


def test_min_track_len_suppresses_a_short_track():
    path = vertical(100.0, [40, 60, 80, 120, 140])
    assert len(walk(path, config=CounterConfig(min_track_len=3))) == 1
    # Перетин стається на 4-му спостереженні, тож поріг 5 його відкидає. Підняти цей
    # параметр вище бюджету кадрів найшвидшого класу = мовчки губити перетини.
    assert walk(path, config=CounterConfig(min_track_len=5)) == []


def test_back_and_forth_gives_two_events_net_zero():
    events = walk(vertical(100.0, [40, 60, 80, 120, 140, 120, 80, 60]))
    assert [e.direction for e in events] == ["A", "B"]


def test_debounce_suppresses_loitering_astride_the_gate():
    """Людина стоїть на лінії й хитається ширше за мертву зону.

    Без debounce кожне хитання — окрема подія. Чиста сума правильна в обох випадках;
    debounce не дає роздуватись загальній кількості.
    """
    path = vertical(100.0, [40, 80, 130, 80, 130, 80, 130])
    assert len(walk(path, config=CounterConfig(debounce_frames=0))) > 1
    assert len(walk(path, config=CounterConfig(debounce_frames=100))) == 1


def test_two_lines_count_independently():
    upper, lower = LineCounter(GATE), LineCounter(Line((0.0, 200.0), (200.0, 200.0)))
    path = vertical(100.0, [40, 60, 80, 120, 160, 180, 220, 240])
    assert len(walk(path, counter=upper)) == 1
    assert len(walk(path, counter=lower)) == 1


def test_two_tracks_in_the_same_frame_both_count():
    counter = LineCounter(GATE)
    a, b = vertical(50.0, [40, 60, 80, 120, 140]), vertical(150.0, [160, 140, 120, 80, 60])
    events = []
    for f, (ba, bb) in enumerate(zip(a, b)):
        events += [counter.update(f, 1, ba), counter.update(f, 2, bb)]
    assert sorted(e.direction for e in events if e) == ["A", "B"]
    assert counter.counts["person"] == {"A": 1, "B": 1}


def test_counts_are_kept_per_label():
    counter = LineCounter(GATE)
    walk(vertical(50.0, [40, 60, 80, 120, 140]), 1, counter=counter, label="person")
    walk(vertical(150.0, [40, 60, 80, 120, 140]), 2, counter=counter, label="cyclist")
    assert counter.counts == {"person": {"A": 1, "B": 0}, "cyclist": {"A": 1, "B": 0}}


def test_prune_forgets_idle_tracks():
    counter = LineCounter(GATE, CounterConfig(max_track_idle=10))
    counter.update(0, 1, vertical(100.0, [40])[0])
    counter.prune(5)
    assert 1 in counter._tracks
    counter.prune(20)
    assert 1 not in counter._tracks


def test_naive_counts_arrivals_into_the_band():
    naive = NaiveZoneCounter(GATE, band_px=40)
    near, far = vertical(100.0, [110])[0], vertical(100.0, [300])[0]
    for boxes in [[far], [near], [near], [near, near], [], [near]]:
        naive.update({"person": boxes})
    # прихід: +1 (кадр 2), +1 (кадр 4), +1 (кадр 6) — без ідентичності й напрямку
    assert naive.counts["person"]["total"] == 3
