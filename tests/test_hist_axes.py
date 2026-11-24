"""Our bin index equals boost-histogram's, flow and edge cases included."""
import boost_histogram as bh
import numpy as np

import awkward as ak
import hist
import pytest

import akstream as aks
from akstream.results import IntCategory, Regular, Variable


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


def test_intcategory_matches_boost():
    cats = [3, 0, 7, -2]  # unsorted: the bins follow the given order, as boost
    x = np.array([0, 3, 7, -2, 1, 8, -5, 3])
    np.testing.assert_array_equal(IntCategory(cats).index(x), boost_index(bh.axis.IntCategory(cats), x))
    np.testing.assert_array_equal(IntCategory([0, 1]).index(np.array([True, False])), [1, 0])
    with pytest.raises(TypeError, match="integer"):
        IntCategory(cats).index(np.array([3.0, 3.5]))  # a float column is refused (as boost), not truncated
    with pytest.raises(ValueError, match="repeated"):
        IntCategory([1, 1])
    with pytest.raises(TypeError, match="growth"):
        aks.results.make_axis(axis=hist.axis.IntCategory([], growth=True))


@pytest.mark.parametrize("convention", ["boost", "root"])  # root: no boost equivalent, every axis binned by numpy
def test_nd_hist_axes(events_file, convention):
    """an n-dim histogram with (n, lo, hi), a list of edges, aks.IntCategory and a hist IntCategory axis (per-jet columns
    next to per-event ones), nominal and under a sparse jet variation, equals hist filled on the full arrays."""
    df = aks.read_root(events_file, step=3_000)
    df["njet"] = aks.num(df.Jet.pt[df.Jet.pt > 30], axis=1)
    df["cat"] = df.event % 4
    df["cat2"] = (df.event // 7) % 3
    df.vary("Jet_pt", up="Jet_pt_jer4_up", name="jer4")
    h = aks.varied(df.hist(["mll", "njet", "cat", "cat2", "Jet_pt"],
                           axes=[Regular(8, 70, 150, convention=convention), [0, 1, 2, 4], aks.IntCategory([2, 0, 1]),
                                 hist.axis.IntCategory([0, 1]), (5, 0, 250)], weight="genWeight"))
    got, = aks.compute(h)
    a = aks.read_root(events_file).collect(["mll", "event", "genWeight", "Jet_pt", "Jet_pt_jer4_up"]).value
    for v, pt in (("nominal", a.Jet_pt), ("jer4:up", a.Jet_pt_jer4_up)):
        njet = ak.num(pt[pt > 30], axis=1)
        b = ak.broadcast_arrays(a.mll, njet, a.event % 4, (a.event // 7) % 3, pt, a.genWeight)
        x = [ak.to_numpy(ak.flatten(c)) for c in b]
        ref = hist.Hist(hist.axis.Regular(8, 70, 150), hist.axis.Variable([0, 1, 2, 4]), hist.axis.IntCategory([2, 0, 1]),
                        hist.axis.IntCategory([0, 1]), hist.axis.Regular(5, 0, 250), storage=hist.storage.Weight())
        ref.fill(*x[:5], weight=x[5])
        assert isinstance(got[v].axes[2], hist.axis.IntCategory) and list(got[v].axes[2]) == [2, 0, 1]
        np.testing.assert_allclose(got[v].view(flow=True).value, ref.view(flow=True).value, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(got[v].view(flow=True).variance, ref.view(flow=True).variance, rtol=1e-12, atol=1e-9)
