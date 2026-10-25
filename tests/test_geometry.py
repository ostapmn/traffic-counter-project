import pytest

from counter import NO_CROSS, TO_NEGATIVE, TO_POSITIVE, Line, anchor_point, segments_intersect

GATE = Line((0.0, 100.0), (200.0, 100.0))


def test_perp_distance_is_in_pixels_and_signed():
    # Відстань не залежить від довжини лінії: у цих одиницях задана мертва зона.
    assert GATE.perp_distance((100.0, 130.0)) == pytest.approx(30.0)
    assert GATE.perp_distance((100.0, 70.0)) == pytest.approx(-30.0)
    long_gate = Line((0.0, 100.0), (5000.0, 100.0))
    assert long_gate.perp_distance((100.0, 130.0)) == pytest.approx(30.0)


def test_degenerate_line_has_zero_distance():
    assert Line((5.0, 5.0), (5.0, 5.0)).perp_distance((10.0, 10.0)) == 0.0


def test_crossing_direction():
    assert GATE.crossing((100.0, 50.0), (100.0, 150.0)) == TO_POSITIVE
    assert GATE.crossing((100.0, 150.0), (100.0, 50.0)) == TO_NEGATIVE


def test_crossing_line_extension_is_not_a_crossing():
    """Пряма продовжується за кінці відрізка, ворота — ні.

    Пішохід на x = 400 перетинає y = 100 і змінює знак signed_side, але через ворота
    не проходить. Саме цей випадок рахує наївна перевірка зміни знаку.
    """
    assert GATE.crossing((400.0, 50.0), (400.0, 150.0)) == NO_CROSS
    assert GATE.crossing((-50.0, 50.0), (-50.0, 150.0)) == NO_CROSS


def test_parallel_motion_is_not_a_crossing():
    assert GATE.crossing((0.0, 50.0), (200.0, 50.0)) == NO_CROSS


def test_segments_intersect_touching_endpoint():
    assert segments_intersect((200.0, 50.0), (200.0, 150.0), (0.0, 100.0), (200.0, 100.0))
    assert not segments_intersect((201.0, 50.0), (201.0, 150.0), (0.0, 100.0), (200.0, 100.0))


def test_near_respects_segment_ends():
    assert GATE.near((100.0, 120.0), band_px=40)
    assert not GATE.near((100.0, 150.0), band_px=40)
    assert not GATE.near((300.0, 100.0), band_px=40)


def test_anchor_bottom_center_tracks_the_feet():
    bbox = (100.0, 200.0, 140.0, 340.0)
    assert anchor_point(bbox, "bottom_center") == (120.0, 340.0)
    assert anchor_point(bbox, "center") == (120.0, 270.0)
    with pytest.raises(ValueError):
        anchor_point(bbox, "top")
