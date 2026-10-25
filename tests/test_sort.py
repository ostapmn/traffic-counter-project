import numpy as np

from sort_tracker import Sort, iou_matrix


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert np.allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_moving_box_keeps_one_id():
    sort = Sort()
    ids = set()
    for k in range(10):
        out = sort.update(np.array([[10 + 3 * k, 10, 40 + 3 * k, 90]], float))
        ids |= set(out[:, 4].astype(int))
    assert ids == {1}


def test_two_objects_get_two_ids():
    sort = Sort()
    for k in range(5):
        out = sort.update(np.array([[10 + 2 * k, 10, 40 + 2 * k, 90],
                                    [300 - 2 * k, 10, 330 - 2 * k, 90]], float))
    assert sorted(out[:, 4].astype(int)) == [1, 2]


def test_track_is_dropped_after_max_age_and_reborn_with_new_id():
    sort = Sort(max_age=1)
    box = np.array([[10, 10, 40, 90]], float)
    for _ in range(4):
        sort.update(box)
    sort.update(np.zeros((0, 4)))
    sort.update(np.zeros((0, 4)))  # пропуск довший за max_age — трек видалено
    for _ in range(4):
        out = sort.update(box)
    assert out[:, 4].astype(int).tolist() == [2]


def test_new_track_is_hidden_until_min_hits():
    sort = Sort(min_hits=3)
    box = np.array([[10, 10, 40, 90]], float)
    shown = [len(sort.update(np.zeros((0, 4)))) for _ in range(5)]  # минаємо перші кадри
    shown = [len(sort.update(box)) for _ in range(4)]
    assert shown == [0, 0, 0, 1]
