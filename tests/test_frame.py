"""Frame basics: selections, defined columns, results, against plain awkward/numpy on the full arrays."""
import awkward as ak
import hist
import numpy as np
import pytest

import akstream as aks


def ref_hist(x, w, bins, lo, hi):
    h = hist.Hist(hist.axis.Regular(bins, lo, hi), storage=hist.storage.Weight())
    h.fill(x, weight=w)
    return h


def test_filter_define_hist(events_file, raw):
    df = aks.read_root(events_file, step=3_000)
    df["ht"] = aks.sum(df.Jet.pt[df.Jet.pt > 25], axis=1)
    df["w"] = df.genWeight * df.pu
    sel = df[(df.mll > 110) & (df.mll < 150)]
    h = sel.hist("ht", bins=40, range=(0, 400), weight="w")
    n = sel.count()
    s = sel.sum("w")
    m = (raw.mll > 110) & (raw.mll < 150)
    ht = ak.to_numpy(ak.sum(raw.Jet.pt[raw.Jet.pt > 25], axis=1))[m]
    w = ak.to_numpy(raw.genWeight * raw.pu)[m]
    ref = ref_hist(ht, w, 40, 0, 400)
    np.testing.assert_allclose(h.value.view(flow=True).value, ref.view(flow=True).value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h.value.view(flow=True).variance, ref.view(flow=True).variance, rtol=1e-12)
    assert n.value == int(ak.sum(m))
    assert s.value == pytest.approx(w.sum(), rel=1e-12)


def test_one_pass_for_every_booked_result(events_file):
    df = aks.read_root(events_file, step=5_000)
    a = df.count()
    b = df[df.mll > 100].count()
    _ = a.value
    assert b._done  # booked results are computed in the same pass
    assert aks.last_stats["chunks"] == 4


def test_python_syntax_filter_and_eval(events_file, raw):
    df = aks.read_root(events_file)
    q = df.filter("100 < mll <= 130 and not (mu1_pt < 30)")
    ref = (raw.mll > 100) & (raw.mll <= 130) & ~(raw.mu1_pt < 30)
    assert q.count().value == int(ak.sum(ref))
    e = df.eval("mu1_pt / mll + abs(mu2_pt - 40)")
    df["e"] = e
    out = df[["e"]].collect().value
    np.testing.assert_allclose(out.e, ak.to_numpy(raw.mu1_pt / raw.mll + abs(raw.mu2_pt - 40)))


def test_nested_selections_and_inherited_columns(events_file, raw):
    df = aks.read_root(events_file, step=4_000)
    df["r"] = df.mu1_pt / df.mll
    s1 = df[df.mll > 90]
    s1["njet"] = aks.num(s1.Jet.pt, axis=1)
    s2 = s1[s1.njet >= 2]
    df["late"] = df.mll * 2  # defined after the selection: not visible in s1 (pandas copy semantics)
    assert "late" not in s1.columns
    out = s2.collect(["r", "njet", "mll"]).value
    m1 = raw.mll > 90
    nj = ak.num(raw.Jet.pt, axis=1)
    m = ak.to_numpy(m1 & (nj >= 2))
    np.testing.assert_allclose(out.r, ak.to_numpy(raw.mu1_pt / raw.mll)[m])
    np.testing.assert_array_equal(out.njet, ak.to_numpy(nj)[m])


def test_collections_mask_sort_index(events_file, raw):
    df = aks.read_root(events_file, step=7_000)
    jets = df.Jet[df.Jet.pt > 30]
    order = aks.argsort(jets.pt, axis=1, ascending=False)
    jets = jets[order]
    df["j1_eta"] = aks.fill_none(aks.firsts(jets.eta, axis=1), -99.0)
    df["nj"] = aks.num(jets.pt, axis=1)
    out = df.collect(["j1_eta", "nj"]).value
    rj = raw.Jet[raw.Jet.pt > 30]
    rj = rj[ak.argsort(rj.pt, axis=1, ascending=False)]
    np.testing.assert_array_equal(out.j1_eta, ak.to_numpy(ak.fill_none(ak.firsts(rj.eta, axis=1), -99.0)))
    np.testing.assert_array_equal(out.nj, ak.to_numpy(ak.num(rj, axis=1)))


def test_jagged_hist_and_2d(events_file, raw):
    df = aks.read_root(events_file, step=6_000)
    h = df.hist(df.Jet.pt, bins=30, range=(0, 300), weight="genWeight")
    h2 = df.hist(["mll", "mu1_pt"], axes=[(10, 70, 150), (10, 0, 200)])
    counts = ak.num(raw.Jet.pt, axis=1)
    ref = ref_hist(ak.to_numpy(ak.flatten(raw.Jet.pt)), np.repeat(ak.to_numpy(raw.genWeight), counts), 30, 0, 300)
    np.testing.assert_allclose(h.value.view(flow=True).value, ref.view(flow=True).value, rtol=1e-12)
    r2 = hist.Hist(hist.axis.Regular(10, 70, 150), hist.axis.Regular(10, 0, 200))
    r2.fill(ak.to_numpy(raw.mll), ak.to_numpy(raw.mu1_pt))
    np.testing.assert_array_equal(h2.value.view(flow=True).value, r2.view(flow=True))


def test_groupby(events_file, raw):
    df = aks.read_root(events_file, step=6_000)
    df["nj"] = aks.num(df.Jet.pt, axis=1)
    g = df.groupby("nj").sum("genWeight")
    c = df.groupby("nj").count()
    nj = ak.to_numpy(ak.num(raw.Jet.pt, axis=1))
    for k, v in g.value.items():
        assert v == pytest.approx(ak.to_numpy(raw.genWeight)[nj == k].sum(), rel=1e-12)
        assert c.value[k] == (nj == k).sum()


def test_head(events_file, raw):
    df = aks.read_root(events_file, step=6_000)
    df["x"] = df.mll + 1
    out = df.head(7, ["x"])
    assert len(out) == 7
    np.testing.assert_allclose(out.x, ak.to_numpy(raw.mll[:7] + 1))


def test_numpy_functions(events_file, raw):
    df = aks.read_root(events_file)
    df["a"] = np.where(df.mll > 100, np.sqrt(df.mu1_pt), -1.0)
    df["b"] = np.clip(df.mu2_pt, 20, 50)
    out = df.collect(["a", "b"]).value
    np.testing.assert_allclose(out.a, np.where(ak.to_numpy(raw.mll) > 100, np.sqrt(ak.to_numpy(raw.mu1_pt)), -1.0))
    np.testing.assert_allclose(out.b, np.clip(ak.to_numpy(raw.mu2_pt), 20, 50))


def test_constant_column_and_define_keywords(events_file):
    df = aks.read_root(events_file)
    df2 = df.define(one=1.0, two=lambda d: d.mll * 0 + 2)
    out = df2.collect(["one", "two"]).value
    assert np.all(out.one == 1.0) and np.all(out.two == 2.0)
    assert "one" not in df.columns


def test_cpp_masks_and_local_names(events_file, raw):
    df = aks.read_root(events_file)
    base = df.eval("mll > 70 && mll < 110")
    sel = df.filter(df.eval("base && !(mu1_pt < 30) || false", local={"base": base}))
    ref = (raw.mll > 70) & (raw.mll < 110) & ~(raw.mu1_pt < 30)
    assert sel.count().value == int(ak.sum(ref))
    assert df.filter("true").count().value == 20_000


def test_peek(events_file, raw):
    df = aks.read_root(events_file, step=3_000)
    rec = aks.apply(lambda pt, m: ak.zip({"n": ak.num(pt, axis=1), "m2": m * 2}, depth_limit=1), df.Jet.pt, df.mll)
    v = df.peek(rec, 10)
    assert v.fields == ["n", "m2"] and len(v) == 10
    np.testing.assert_allclose(v.m2, ak.to_numpy(raw.mll[:10]) * 2)
