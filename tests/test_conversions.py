"""aks.from_arrays (frames over arrays in memory) and df.to_awkward / to_numpy / to_pandas, round trips included."""
import awkward as ak
import numpy as np
import pytest
import uproot

import akstream as aks


@pytest.fixture(scope="module")
def arrays():
    rng = np.random.default_rng(7)
    n = 3000
    counts = rng.integers(0, 5, n)
    pt = ak.unflatten(rng.uniform(20, 200, counts.sum()).astype(np.float32), counts)
    return {"mll": rng.uniform(70, 150, n), "w": rng.uniform(0.5, 1.5, n).astype(np.float32),
            "Jet_pt": pt, "Jet_eta": ak.unflatten(rng.uniform(-4.7, 4.7, counts.sum()), counts),
            "Jet_pt_up": pt * np.float32(1.05)}


def test_from_arrays_is_a_frame(arrays):
    df = aks.from_arrays(arrays, step=700)
    assert "Jet" in df.collections and {"pt", "eta", "pt_up"} <= set(df.Jet.fields)
    df["Jet"] = df.Jet[df.Jet.pt > 50]
    df["ht"] = aks.sum(df.Jet.pt, axis=1)
    sel = df.filter("mll > 100", name="mll")
    h = sel.hist("ht", (20, 0, 600), weight="w").value
    keep = arrays["mll"] > 100
    j = arrays["Jet_pt"][arrays["Jet_pt"] > 50]
    ref = np.histogram(np.asarray(ak.sum(j, axis=1))[keep], 20, (0, 600), weights=arrays["w"][keep])[0]
    np.testing.assert_allclose(h.values(), ref, rtol=1e-6)
    assert sel.report().value[1][1] == int(keep.sum())
    assert np.array_equal(np.asarray(df.peek(df.entry, 5)), np.arange(5))
    assert "in-memory arrays" in repr(df)


def test_variations_and_workers_on_arrays(arrays):
    df = aks.from_arrays(arrays, step=500)
    df.vary(("Jet", "pt"), up="Jet_pt_up", name="jes")
    df["ht"] = aks.sum(df.Jet.pt, axis=1)
    h = aks.varied(df.hist("ht", (20, 0, 800)))
    one, = aks.compute(h)
    df2 = aks.from_arrays(arrays, step=500)
    df2.vary(("Jet", "pt"), up="Jet_pt_up", name="jes")
    df2["ht"] = aks.sum(df2.Jet.pt, axis=1)
    two, = aks.compute(aks.varied(df2.hist("ht", (20, 0, 800))), workers=2)
    ref = np.histogram(np.asarray(ak.sum(arrays["Jet_pt_up"], axis=1)), 20, (0, 800))[0]
    np.testing.assert_array_equal(one["jes:up"].values(), ref)
    np.testing.assert_array_equal(two["jes:up"].values(), ref)


def test_round_trip_file_to_awkward_to_frame(events_file, raw):
    df = aks.read_root(events_file, step=4000)
    df["ht"] = aks.sum(df.Jet.pt, axis=1)
    sel = df.filter("mll > 110")
    table = sel.to_awkward(["mll", "ht", "Jet"])
    assert table.fields == ["mll", "ht", "Jet"] and len(table) == int(np.sum(raw.mll > 110))
    back = aks.from_arrays(table, step=1000)
    assert "Jet" in back.collections and "pt" in back.Jet.fields
    back["njet"] = aks.num(back.Jet.pt, axis=1)
    out = back.to_awkward(["ht", "njet"])
    keep = raw.mll > 110
    np.testing.assert_array_equal(np.asarray(out["njet"]), np.asarray(ak.num(raw.Jet.pt, axis=1)[keep]))
    np.testing.assert_allclose(np.asarray(out["ht"]), np.asarray(ak.sum(raw.Jet.pt, axis=1)[keep]))


def test_to_numpy_and_a_scaler(arrays):
    df = aks.from_arrays(arrays, step=700)
    df["njet"] = aks.num(df.Jet.pt, axis=1)
    x = df.to_numpy(["mll", "w", "njet"])
    assert x.shape == (3000, 3) and x.dtype == np.float64
    np.testing.assert_array_equal(x[:, 0], arrays["mll"])
    assert df.to_numpy(["w"], dtype=np.float32).dtype == np.float32
    mean, std = x.mean(axis=0), x.std(axis=0)  # e.g. a StandardScaler, fitted on the table ...
    df["mll_scaled"] = (df.mll - mean[0]) / std[0]  # ... applied back as constants of the graph
    np.testing.assert_allclose(df.to_numpy(["mll_scaled"])[:, 0], (arrays["mll"] - mean[0]) / std[0])
    with pytest.raises(TypeError, match="Jet_pt"):
        df.to_numpy(["Jet_pt"])


def test_to_pandas(arrays):
    pd = pytest.importorskip("pandas")
    df = aks.from_arrays(arrays, step=700)
    df["njet"] = aks.num(df.Jet.pt, axis=1)
    t = df.to_pandas(["mll", "njet"])
    assert isinstance(t, pd.DataFrame) and list(t.columns) == ["mll", "njet"] and len(t) == 3000
    assert t["njet"].dtype == np.int64
    back = aks.from_arrays(t)  # and back
    np.testing.assert_array_equal(back.to_numpy(["njet"])[:, 0], t["njet"].to_numpy())
    jets = df.to_pandas(["mll", "Jet_pt"])  # jagged: one row per object
    assert jets.index.nlevels == 2 and len(jets) == int(ak.sum(ak.num(arrays["Jet_pt"])))
    with pytest.raises(ValueError, match="MultiIndex"):
        aks.from_arrays(jets)


def test_from_arrays_errors(arrays):
    with pytest.raises(ValueError, match="lengths"):
        aks.from_arrays({"a": np.zeros(3), "b": np.zeros(4)})
    with pytest.raises(ValueError, match="shape"):
        aks.from_arrays({"a": np.zeros((3, 2))})
    with pytest.raises(TypeError, match="string columns are planned"):
        aks.from_arrays({"a": np.array(["x", "y"])})


def test_to_root_from_arrays(arrays, tmp_path):
    df = aks.from_arrays(arrays, step=1000)
    path = tmp_path / "mem.root"
    info = aks.compute(df.to_root(str(path), ["mll", "Jet"]))[0]
    assert info["entries"] == 3000
    t = uproot.open(path)["Events"].arrays(library="ak")
    np.testing.assert_array_equal(np.asarray(t.mll), arrays["mll"])
    assert ak.all(t.Jet_pt == arrays["Jet_pt"])
