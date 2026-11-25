"""Flat-ntuple parquet input (aks.read_parquet) and output (to_parquet) against the same data read from ROOT."""
import awkward as ak
import numpy as np
import pytest
import uproot

import akstream as aks

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def root_to_parquet(path, out, row_group_size):
    """the ROOT tree of `path` as a flat parquet file written by pyarrow itself (nJet + Jet_* lists)."""
    t = uproot.open(path)["Events"]
    cols = {}
    for k in t.keys():
        a = t[k].array(library="ak")
        if a.ndim > 1:
            cols[k] = pa.ListArray.from_arrays(pa.array(np.asarray(a.layout.offsets).astype(np.int32)),
                                               pa.array(ak.to_numpy(ak.flatten(a))))
        else:
            cols[k] = pa.array(ak.to_numpy(a))
    pq.write_table(pa.table(cols), out, row_group_size=row_group_size)
    return str(out)


@pytest.fixture(scope="module")
def pq_file(events_file, tmp_path_factory):
    # row groups of 3_100 rows read in chunks of 2_000: chunks straddle row-group boundaries
    return root_to_parquet(events_file, tmp_path_factory.mktemp("pq") / "events.parquet", 3_100)


def analysis(df):
    df["ht"] = aks.sum(df.Jet.pt[df.Jet.pt > 30], axis=1)
    sel = df[df.mll > 100]
    sel["jets"] = sel.Jet[sel.Jet.pt > 25][["pt", "eta"]]
    return sel, [sel.hist("ht", bins=30, range=(0, 600), weight="genWeight"),
                 sel.hist(sel.jets.eta, bins=20, range=(-5, 5)), sel.count(), sel.sum("mll")]


def test_read_parquet_equals_read_root(events_file, pq_file):
    df = aks.read_parquet(pq_file, step=2_000)
    assert set(df.Jet.fields) >= {"pt", "eta", "phi", "mass", "pt_jes_up"}
    _, got = analysis(df)
    _, ref = analysis(aks.read_root(events_file, step=2_000))
    got, ref = aks.compute(*got), aks.compute(*ref)
    for g, r in zip(got, ref):
        if hasattr(g, "values"):
            np.testing.assert_array_equal(g.values(flow=True), r.values(flow=True))
            np.testing.assert_array_equal(g.variances(flow=True), r.variances(flow=True))
        else:
            assert g == r


def test_read_parquet_under_variations_and_workers(events_file, pq_file):
    def run(df, workers=1):
        df.vary("Jet_pt", up="Jet_pt_jes_up", down="Jet_pt_jes_down", name="jes")
        _, res = analysis(df)
        return aks.compute(*(aks.varied(r) for r in res[:2]), workers=workers)
    for workers in (1, 2):  # the same partial sums in the same order: identical
        ref = run(aks.read_root(events_file, step=2_500), workers)
        got = run(aks.read_parquet(pq_file, step=2_500), workers)
        for g, r in zip(got, ref):
            assert set(g) == set(r)
            for v in g:
                np.testing.assert_array_equal(g[v].values(flow=True), r[v].values(flow=True))


def test_read_parquet_chunks_and_columns(pq_file, raw, monkeypatch):
    """only the needed columns are read; chunk boundaries anywhere in the row groups; entry range; one row group decoded
    once even when several chunks fall in it."""
    df = aks.read_parquet(pq_file, step=777, entry_start=1_234, entry_stop=17_001)
    out = df.collect(["event", "Jet_pt", "rdfentry_"]).value
    np.testing.assert_array_equal(out["event"], ak.to_numpy(raw.event[1_234:17_001]))
    assert ak.all(out["Jet_pt"] == raw.Jet.pt[1_234:17_001])
    np.testing.assert_array_equal(out["rdfentry_"], np.arange(1_234, 17_001))
    assert str(out["Jet_pt"].type.content) == "var * float64"  # plain lists, not option types
    reader = aks.read_parquet(pq_file)._dataset.reader({"mll"})
    reads = []
    real = pq.ParquetFile.read_row_group

    def spy(self, i, **kw):
        reads.append((i, tuple(kw["columns"])))
        return real(self, i, **kw)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy)
    for s in range(0, 9_300, 1_000):
        c = reader.read(pq_file, s, min(s + 1_000, 9_300))
        np.testing.assert_array_equal(c.get("mll"), ak.to_numpy(raw.mll[s:min(s + 1_000, 9_300)]))
    assert reads == [(0, ("mll",)), (1, ("mll",)), (2, ("mll",))]


def test_read_parquet_several_files_nulls_and_other_columns(tmp_path):
    a = pa.table({"x": pa.array([1.0, 2.0, None]), "n": pa.array([1, 2, 3], pa.int16()),
                  "v": pa.array([[1, 2], None, [3]], pa.list_(pa.int32())),
                  "s": pa.array(["a", "b", "c"]), "rec": pa.array([{"q": 1}, {"q": 2}, {"q": 3}])})
    pq.write_table(a, tmp_path / "a.parquet")
    pq.write_table(a.slice(1), tmp_path / "b.parquet")
    with pytest.warns(RuntimeWarning, match="not flat-ntuple columns"):
        df = aks.read_parquet(str(tmp_path / "*.parquet"))
    assert "s" not in df.columns and "rec" not in df.columns
    out = df.collect(["x", "n", "v"]).value
    assert out["x"].tolist() == [1.0, 2.0, None, 2.0, None]
    assert out["n"].tolist() == [1, 2, 3, 2, 3] and ak.to_numpy(out["n"]).dtype == np.int16
    assert out["v"].tolist() == [[1, 2], None, [3], None, [3]]
    assert df[df.x > 1.5].count().value == 2  # None fails the selection


def test_to_parquet_roundtrip(events_file, raw, tmp_path):
    df = aks.read_root(events_file, step=3_000)
    sel, _ = analysis(df)
    path = tmp_path / "out.parquet"
    info = sel.to_parquet(path, ["mll", "ht", "jets", "event"], row_group_entries=4_000).value
    m = ak.to_numpy(raw.mll > 100)
    assert info["entries"] == int(m.sum())
    t = pq.read_table(path)
    assert t.column_names == ["mll", "ht", "njets", "jets_pt", "jets_eta", "event"]
    assert t.schema.field("njets").type == pa.int32() and pq.ParquetFile(path).metadata.num_row_groups > 1
    back = aks.read_parquet(str(path)).collect(["mll", "event", "jets", "njets"]).value
    np.testing.assert_array_equal(back["event"], ak.to_numpy(raw.event[m]))
    rj = raw.Jet[m][raw.Jet[m].pt > 25]
    assert ak.all(back["jets"].pt == rj.pt) and ak.all(back["jets"].eta == rj.eta)
    np.testing.assert_array_equal(back["njets"], ak.to_numpy(ak.num(rj, axis=1)))
    # the same columns as to_root writes, bit for bit
    root = tmp_path / "out.root"
    aks.compute(sel.to_root(root, ["mll", "ht", "jets", "event"]))
    r = uproot.open(root)["Events"].arrays(library="ak")
    for c in ("mll", "ht", "njets", "jets_pt", "jets_eta", "event"):
        x, y = t.column(c).to_pylist(), r[c].tolist()
        assert x == y, c
    with pytest.raises(ValueError, match="written twice"):
        sel.to_parquet(tmp_path / "x.parquet", ["jets", "jets_pt"])
    assert "njets" not in pq.read_table(sel.to_parquet(tmp_path / "nc.parquet", ["jets"], counters=False)
                                        .value["files"][0]).column_names


def test_varied_to_parquet_equals_varied_to_root(events_file, tmp_path):
    def build():
        df = aks.read_root(events_file, step=3_000)
        df.vary("Jet_pt", up="Jet_pt_jer4_up", down="Jet_pt_jer4_down", name="jer4")
        df["ht"] = aks.sum(df.Jet.pt[df.Jet.pt > 30], axis=1)
        sel = df.filter("ht > 60 && mll > 80")
        sel["jets"] = sel.Jet[sel.Jet.pt > 30][["pt"]]
        return sel
    cols = ["ht", "jets", "event"]
    sel = build()
    p, r = aks.compute(aks.varied(sel.to_parquet(str(tmp_path / "o_{variation}.parquet"), cols)),
                       aks.varied(sel.to_root(str(tmp_path / "o_{variation}.root"), cols)))
    assert set(p) == set(r) == {"nominal", "jer4:up", "jer4:down"}
    for v in p:
        assert p[v]["entries"] == r[v]["entries"]
        t = pq.read_table(p[v]["files"][0])
        u = uproot.open(r[v]["files"][0])["Events"].arrays(library="ak")
        for c in ("ht", "event", "njets", "jets_pt"):
            assert t.column(c).to_pylist() == u[c].tolist(), (v, c)


def test_to_parquet_empty_selection_and_options(events_file, tmp_path):
    df = aks.read_root(events_file)
    empty = df[df.mll > 1e9]
    info = empty.to_parquet(tmp_path / "e.parquet", ["mll", "Jet"]).value
    t = pq.read_table(info["files"][0])
    assert info["entries"] == 0 and t.num_rows == 0 and "Jet_pt" in t.column_names
    from akstream import hep
    df["j3"] = hep.nth(df.Jet.pt, 2)  # None where there is no third jet
    t = pq.read_table(df.to_parquet(tmp_path / "n.parquet", ["j3"]).value["files"][0])
    raw = uproot.open(events_file)["Events"].arrays(["Jet_pt"], library="ak")
    assert t.column("j3").to_pylist() == ak.pad_none(raw.Jet_pt, 3)[:, 2].tolist()
