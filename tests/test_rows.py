"""row-set and row-comparison helpers."""
import awkward as ak
import numpy as np

from akstream import rows as R


def test_union_paths():
    rng = np.random.default_rng(0)
    for n, k in ((1000, 5), (1000, 400), (50, 50)):
        sets = [np.unique(rng.integers(0, n, k)) for _ in range(3)]
        ref = np.unique(np.concatenate(sets))
        np.testing.assert_array_equal(R.union(*sets), ref)
    a = np.array([1, 5, 9])
    assert R.union(a, a, R.EMPTY) is a


def test_row_differs_jagged_fast_and_generic():
    a = ak.Array([[1.0, 2.0], [], [3.0], [np.nan, 4.0], [5.0]])
    b = ak.Array([[1.0, 2.5], [], [3.0], [np.nan, 4.0], [6.0]])
    np.testing.assert_array_equal(R.row_differs(a, b), [True, False, False, False, True])
    c = ak.Array([[1.0, 2.0], [7.0], [3.0], [np.nan, 4.0], []])  # different counts: generic path
    np.testing.assert_array_equal(R.row_differs(a, c), [False, True, False, False, True])
    np.testing.assert_array_equal(R.row_differs(a[[4, 0, 2]], b[[4, 0, 2]]), [True, True, False])  # ListArray


def test_row_differs_records_options():
    a = ak.Array([{"x": 1, "y": [1]}, {"x": 2, "y": []}, {"x": 3, "y": [2, 3]}])
    b = ak.Array([{"x": 1, "y": [1]}, {"x": 5, "y": []}, {"x": 3, "y": [2, 4]}])
    np.testing.assert_array_equal(R.row_differs(a, b), [False, True, True])
    o1 = ak.Array([1.0, None, 3.0, None])
    o2 = ak.Array([1.0, None, 4.0, 2.0])
    np.testing.assert_array_equal(R.row_differs(o1, o2), [False, False, True, True])


def test_flat_values_non_contiguous():
    a = ak.Array([[1.0, 2.0], [3.0], [], [4.0, 5.0, 6.0]])
    b = ak.Array([[1.0, 2.0], [3.0], [], [4.0, 5.5, 6.0]])
    idx = np.array([3, 0, 2, 1])
    np.testing.assert_array_equal(R.row_differs(a[idx], b[idx]), [True, False, False, False])
    np.testing.assert_array_equal(R.row_differs(a[idx], a[idx]), [False] * 4)
