import pytest

from eval_count import compare


def test_mape_wape_bias_match_the_guide_example():
    # guide.md, розділ 23: три бакети 13/10/4, ми 12/9/3
    gt = {"b1": 13, "b2": 10, "b3": 4}
    pred = {"b1": 12, "b2": 9, "b3": 3}
    r = compare(pred, gt)
    assert r.buckets_scored == 3
    assert r.mape == pytest.approx(14.23, abs=0.01)
    assert r.wape == pytest.approx(11.11, abs=0.01)
    assert r.bias == pytest.approx(-11.11, abs=0.01)
    assert not r.meets_target(10)


def test_perfect_count_meets_target():
    r = compare({"b": 5}, {"b": 5})
    assert r.mape == 0 and r.bias == 0 and r.meets_target(10)


def test_false_positives_in_empty_buckets_are_reported_not_dropped():
    """MAPE не може оцінити бакет із gt = 0; помилка все одно має бути видна."""
    r = compare({"b": 4}, {})
    assert r.buckets_scored == 0
    assert r.empty_buckets == 1
    assert r.false_positives_in_empty_buckets == 4
    # немає оцінених бакетів — немає MAPE, і це не читається як успіх
    assert r.mape is None
    assert not r.meets_target()
