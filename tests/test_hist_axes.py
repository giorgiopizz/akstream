"""Our bin index equals boost-histogram's, flow and edge cases included."""
import boost_histogram as bh
import numpy as np

from akstream.results import Regular, Variable


def boost_index(axis, x):
    h = bh.Histogram(axis)
    out = []
    for v in x:
        h.reset()
        h.fill([v])
        out.append(int(np.flatnonzero(h.view(flow=True))[0]))
    return np.array(out)


def test_regular_matches_boost():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-1, 11, 300), np.linspace(0, 10, 101), [np.nan, np.inf, -np.inf, 10 - 1e-15,
                                                                            1e-300, -1e-300]])
    for bins, lo, hi in ((10, 0, 10), (7, 0.1, 9.7), (40, 0, 400), (3, -1.5, 2.25)):
        ours = Regular(bins, lo, hi).index(x)
        np.testing.assert_array_equal(ours, boost_index(bh.axis.Regular(bins, lo, hi), x))


def test_variable_matches_boost():
    edges = [0, 1, 2.5, 7, 10]
    x = np.array([-1, 0, 0.5, 1, 2.5, 3, 7, 9.99, 10, 11, np.nan, np.inf, -np.inf])
    np.testing.assert_array_equal(Variable(edges).index(x), boost_index(bh.axis.Variable(edges), x))


def test_root_convention_matches_findbin():
    """ROOT TAxis::FindBin arithmetic: 1 + int(n * (x - lo) / (hi - lo)) inside [lo, hi)."""
    x = np.concatenate([np.random.default_rng(1).uniform(60, 120, 2000), np.linspace(70, 110, 401), [np.nan, 110, 70]])
    lo, hi, n = 70.0, 110.0, 400
    with np.errstate(invalid="ignore"):
        inside = (x >= lo) & (x < hi)
        ref = np.where(inside, 1 + (n * (np.where(inside, x, lo) - lo) / (hi - lo)).astype(np.int64), 0)
        ref = np.where(x < lo, 0, np.where(inside, ref, n + 1))
    np.testing.assert_array_equal(Regular(n, lo, hi, convention="root").index(x), ref)


def test_accumulate_boost_path_equals_numpy_path():
    """boost-convention axes are binned and accumulated by boost-histogram: identical to the numpy index + bincount."""
    from akstream.results import _fill_arrays, accumulate
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.uniform(60, 160, 5000), np.linspace(70, 150, 81), [np.nan, np.inf, -np.inf]])
    y = rng.uniform(-1, 11, len(x))
    w = rng.normal(1, 0.3, len(x))
    t = rng.integers(0, 4, len(x))
    axes = [Regular(40, 70, 150), Variable([0, 1, 2.5, 7, 10])]
    count, sumw, sumw2 = accumulate(axes, [x, y], w, target=t, ntargets=4)
    index, ww = _fill_arrays(axes, [x, y], w)
    size = axes[0].size * axes[1].size
    idx = index + t * size
    np.testing.assert_array_equal(count, np.bincount(idx, minlength=4 * size))
    np.testing.assert_allclose(sumw, np.bincount(idx, weights=ww, minlength=4 * size), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(sumw2, np.bincount(idx, weights=ww * ww, minlength=4 * size), rtol=1e-13, atol=1e-13)
