"""The RDataFrame-like chain API (define / filter / report / apply / hists / vary with expression strings)."""
import awkward as ak
import numpy as np
import pytest

import akstream as aks


def test_define_filter_chain_equals_pandas_style(events_file, raw):
    df = aks.read_root(events_file, step=3000)
    chained = (df.define("ht", aks.sum(df.Jet.pt, axis=1))
                 .define(ratio="mu2_pt / mu1_pt", one=1.0)
                 .filter("mll > 110 && ratio < 0.8", name="window")
                 .hist("ht", (20, 0, 400), weight="genWeight * one"))
    df["ht"] = aks.sum(df.Jet.pt, axis=1)
    df["ratio"] = df.mu2_pt / df.mu1_pt
    pandas_style = df[(df.mll > 110) & (df.ratio < 0.8)].hist("ht", bins=20, range=(0, 400), weight="genWeight")
    a, b = aks.compute(chained, pandas_style)
    np.testing.assert_array_equal(a.values(), b.values())
    keep = (raw.mll > 110) & (raw.mu2_pt / raw.mu1_pt < 0.8)
    ref = np.histogram(np.asarray(ak.sum(raw.Jet.pt, axis=1))[keep], bins=20, range=(0, 400),
                       weights=np.asarray(raw.genWeight)[keep])[0]
    np.testing.assert_allclose(a.values(), ref, rtol=1e-12)


def test_define_does_not_change_the_original(events_file):
    df = aks.read_root(events_file)
    df2 = df.define("x", "mll * 2")
    assert "x" in df2.columns and "x" not in df.columns


def test_report_cutflow(events_file, raw):
    df = aks.read_root(events_file, step=3000)
    z = df.filter("mll > 80", name="mll > 80").filter("mll < 100", name="mll < 100")
    h = z.hist("mll", (10, 80, 100))
    rep = z.report()
    rows = rep.value
    n = len(raw)
    n1 = int(np.sum(raw.mll > 80))
    n2 = int(np.sum((raw.mll > 80) & (raw.mll < 100)))
    assert [r[0] for r in rows] == ["all entries", "mll > 80", "mll < 100"]
    assert [r[1] for r in rows] == [n, n1, n2]
    assert rows[2][2] == pytest.approx(n2 / n1) and rows[2][3] == pytest.approx(n2 / n)
    assert h.value.values().sum() == n2  # filled in the same pass
    assert "mll < 100" in repr(rep)


def test_apply_chunk_function(events_file, raw):
    df = aks.read_root(events_file, step=3000)
    calls = []

    def leading(chunk, threshold):
        calls.append(len(chunk["Jet_pt"]))
        pt = chunk["Jet_pt"][chunk["Jet_pt"] > threshold]
        return {"n": np.asarray(ak.num(pt, axis=1)), "lead": np.asarray(ak.fill_none(ak.max(pt, axis=1), 0.0))}

    df2 = df.apply(leading, ["Jet_pt"], threshold=30.0, prefix="j_")
    assert {"j_n", "j_lead"} <= set(df2.columns)
    out = df2.filter("j_n >= 1").collect(["j_lead"]).value
    pt = raw.Jet.pt[raw.Jet.pt > 30]
    ref = np.asarray(ak.fill_none(ak.max(pt, axis=1), 0.0))[np.asarray(ak.num(pt, axis=1)) >= 1]
    np.testing.assert_array_equal(np.asarray(out["j_lead"]), ref)

    single = df.apply(lambda c: c["mll"] * 2, {"mll": "mll"}, outputs="mll2")
    np.testing.assert_array_equal(np.asarray(single.collect(["mll2"]).value["mll2"]), np.asarray(raw.mll) * 2)


def test_apply_under_variations_recomputes_only_changed_rows(events_file):
    df = aks.read_root(events_file, step=5000)
    df.vary("Jet_pt", up="Jet_pt_jer4_up", down="Jet_pt_jer4_down", name="jer4")
    rows = []

    def ht(chunk):
        rows.append(len(chunk["pt"]))
        return {"ht": np.asarray(ak.sum(chunk["pt"], axis=1))}

    df2 = df.apply(ht, {"pt": "Jet_pt"}, outputs=["ht"])
    h = aks.varied(df2.hist("ht", (20, 0, 400)))
    out, = aks.compute(h)
    assert set(out) == {"nominal", "jer4:up", "jer4:down"}
    assert 20_000 < sum(rows) < 2 * 20_000  # nominal on every row, the variations only on the rows they move


def test_hists_and_vary_with_expression_strings(events_file, raw):
    df = aks.read_root(events_file, step=4000)
    df["w"] = df.genWeight * df.pu
    df.vary("w", up="genWeight * pu_up", name="pu")
    hs = df.hists({"mll": (16, 70, 150), "lead": ("mu1_pt", [20, 40, 80, 200])}, weight="w", varied=True)
    out = aks.compute(*hs.values())
    mll, lead = out
    assert set(mll) == {"nominal", "pu:up"}
    w = np.asarray(raw.genWeight * raw.pu)
    wu = np.asarray(raw.genWeight * raw.pu_up)
    np.testing.assert_allclose(mll["pu:up"].values(), np.histogram(raw.mll, 16, (70, 150), weights=wu)[0], rtol=1e-12)
    np.testing.assert_allclose(lead["nominal"].values(),
                               np.histogram(raw.mu1_pt, [20, 40, 80, 200], weights=w)[0], rtol=1e-12)


def test_hist_of_an_expression_string(events_file, raw):
    df = aks.read_root(events_file)
    h = df.hist("mu1_pt + mu2_pt", (10, 0, 300))
    np.testing.assert_array_equal(h.value.values(), np.histogram(raw.mu1_pt + raw.mu2_pt, 10, (0, 300))[0])
