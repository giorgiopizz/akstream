"""Collections as awkward-like records over the X_<field> columns: field assignment, broadcasting, replacing the
collection, field-wise awkward functions, lazy hep kernels, vary by field, friend matching, aks.checked."""
import awkward as ak
import numpy as np
import pytest
import uproot

import akstream as aks
from akstream import hep


def _write(path, data, tree="Events"):
    with uproot.recreate(path) as f:
        f.mktree(tree, {k: (v.type.content if isinstance(v, ak.Array) else v.dtype) for k, v in data.items()})
        f[tree].extend(data)
    return str(path)


@pytest.fixture(scope="module")
def nocounter(tmp_path_factory):
    """jets as separate jagged branches vJet_* without a counter branch (as some skims write them)."""
    rng = np.random.default_rng(3)
    n = 5000
    counts = rng.integers(0, 6, n)
    total = int(counts.sum())
    data = {"event": np.arange(n, dtype=np.uint64), "w": rng.uniform(0.5, 1.5, n)}
    for f, (lo, hi) in {"pt": (20, 200), "eta": (-4.7, 4.7), "phi": (-3.14, 3.14), "mass": (2, 20)}.items():
        data[f"vJet_{f}"] = ak.unflatten(rng.uniform(lo, hi, total).astype(np.float32), counts)
    data["vJet_good"] = ak.unflatten(rng.uniform(size=total) > 0.3, counts)
    data["vJet_pt_up"] = data["vJet_pt"] * np.float32(1.1)
    path = _write(tmp_path_factory.mktemp("nc") / "nc.root", data)
    return path, uproot.open(path)["Events"].arrays(library="ak")


def test_collection_without_counter(nocounter):
    path, raw = nocounter
    df = aks.read_root(path, step=1500)
    assert "vJet" in df.collections and {"pt", "eta", "good", "pt_up"} <= set(df.vJet.fields)
    df["ht"] = aks.sum(df.vJet.pt, axis=1)
    np.testing.assert_array_equal(np.asarray(df.collect(["ht"]).value["ht"]), np.asarray(ak.sum(raw.vJet_pt, axis=1)))


def test_field_assignment_is_one_column(nocounter):
    path, raw = nocounter
    df = aks.read_root(path, step=1500)
    df["vJet_pt"] = df.vJet.pt * 2  # the flat name ...
    df.vJet["pt3"] = df.vJet.pt + 1  # ... a field ...
    df["vJet", "pt4"] = df["vJet", "pt3"] - 1  # ... the awkward-style key: all the same columns
    out = df.collect(["vJet_pt4"]).value
    assert ak.all(out["vJet_pt4"] == raw.vJet_pt * 2 + 1 - 1)  # float32 arithmetic, as the lazy one
    assert "pt4" in df.vJet.fields and "vJet_pt3" in df.columns


def test_field_broadcast_and_count_check(nocounter):
    path, raw = nocounter
    df = aks.read_root(path, step=1500)
    df.vJet["minus"] = np.float32(-1)
    df.vJet["flag"] = False
    df.vJet["w"] = df.w  # one value per row -> every object of the row
    out = df.collect(["vJet_minus", "vJet_flag", "vJet_w"]).value
    assert str(ak.type(out["vJet_minus"].layout)) == str(ak.type(raw.vJet_pt.layout))  # var * float32
    assert ak.all(ak.num(out["vJet_minus"]) == ak.num(raw.vJet_pt)) and ak.all(out["vJet_minus"] == -1)
    assert ak.flatten(out["vJet_flag"]).layout.dtype == np.bool_ and not ak.any(out["vJet_flag"])
    assert ak.all(out["vJet_w"] == ak.broadcast_arrays(raw.w, raw.vJet_pt)[0])
    bad = aks.read_root(path, step=1500)
    bad.vJet["x"] = bad.vJet.pt[bad.vJet.pt > 50]  # other object counts
    with pytest.raises(ValueError, match="vJet.x"):
        aks.compute(bad.collect(["vJet_x"]))


def test_replace_collection_mask_and_sort(nocounter):
    path, raw = nocounter
    df = aks.read_root(path, step=1500)
    df.vJet["idx"] = aks.local_index(df.vJet.pt)
    df["vJet"] = df.vJet[df.vJet.good]
    df["vJet"] = df.vJet[aks.argsort(df.vJet.pt)[:, ::-1]]  # Reverse(Argsort(pt))
    sel = df[aks.num(df.vJet.pt, axis=1) > 0]  # a selection sees the replaced collection
    out = sel.collect(["vJet_pt", "vJet_eta", "vJet_idx"]).value
    good = {f: raw[f"vJet_{f}"][raw.vJet_good] for f in ("pt", "eta")}
    good["idx"] = ak.local_index(raw.vJet_pt)[raw.vJet_good]
    order = ak.argsort(good["pt"])[:, ::-1]
    keep = ak.num(good["pt"]) > 0
    for f in ("pt", "eta", "idx"):
        assert ak.all(out[f"vJet_{f}"] == good[f][order][keep]), f


def test_counter_follows_the_collection(events_file, raw):
    df = aks.read_root(events_file, step=3000)
    df["Jet"] = df.Jet[df.Jet.pt > 40]
    out = df.collect(["nJet", "Jet_pt"]).value
    np.testing.assert_array_equal(np.asarray(out["nJet"]), np.asarray(ak.sum(raw.Jet.pt > 40, axis=1)))
    df["goodJet"] = df.Jet[abs(df.Jet.eta) < 2.5]  # a new collection
    df.goodJet["one"] = 1
    assert "goodJet" in df.collections and "one" in df.goodJet.fields
    j = raw.Jet[raw.Jet.pt > 40]
    got = df.collect(["goodJet_pt", "goodJet_one"]).value
    assert ak.all(got["goodJet_pt"] == j.pt[abs(j.eta) < 2.5]) and ak.all(got["goodJet_one"] == 1)


def test_to_root_a_subset_of_a_collection(events_file, raw, tmp_path):
    df = aks.read_root(events_file, step=3000)
    df["Jet"] = df.Jet[df.Jet.pt > 30]
    df.Jet["double_pt"] = df.Jet.pt * 2
    path = str(tmp_path / "sub.root")
    aks.compute(df.to_root(path, ["mll", ("Jet", ["pt", "double_pt"])]))
    t = uproot.open(path)["Events"]
    assert set(t.keys()) == {"mll", "nJet", "Jet_pt", "Jet_double_pt"}
    j = raw.Jet.pt[raw.Jet.pt > 30]
    out = t.arrays(library="ak")
    assert ak.all(out.Jet_pt == j) and ak.all(out.Jet_double_pt == j * 2)
    np.testing.assert_array_equal(np.asarray(out.nJet), np.asarray(ak.num(j)))


def test_fieldwise_awkward_and_leading_pair(nocounter):
    path, raw = nocounter
    df = aks.read_root(path, step=1500, rdf_compatible=True)
    jets = df.vJet
    lead = aks.fill_none(aks.pad_none(jets, 2, clip=True), 0.0)
    df["mjj"] = (lead[:, 0].p4 + lead[:, 1].p4).mass
    df["eta0"] = lead[:, 0].eta
    out = df.collect(["mjj", "eta0"]).value
    from akstream import vector
    pad = {f: np.asarray(ak.fill_none(ak.pad_none(raw[f"vJet_{f}"], 2, clip=True), 0.0)) for f in ("pt", "eta", "phi", "mass")}
    j0, j1 = (vector.p4(*(pad[f][:, k] for f in ("pt", "eta", "phi", "mass")), rdf_compatible=True) for k in (0, 1))
    np.testing.assert_array_equal(np.asarray(out["mjj"]), (j0 + j1).mass)
    np.testing.assert_array_equal(np.asarray(out["eta0"]), pad["eta"][:, 0])


def test_lazy_hep_kernels(nocounter):
    path, raw = nocounter
    df = aks.read_root(path, step=1500, rdf_compatible=True)
    found, i1, i2, mjj, deta = hep.max_mass_pair(df.vJet, min_mass=200.0)
    df["i1"], df["mjj"] = i1, mjj
    df["pt1"] = hep.pick(df.vJet.pt, i1, 0)
    out = df.collect(["i1", "mjj", "pt1"]).value
    ref = hep.max_mass_pair(*(raw[f"vJet_{f}"] for f in ("pt", "eta", "phi", "mass")), min_mass=200.0,
                            rdf_compatible=True)
    np.testing.assert_array_equal(np.asarray(out["i1"]), ref[1])
    np.testing.assert_array_equal(np.asarray(out["mjj"]), ref[3])
    np.testing.assert_array_equal(np.asarray(out["pt1"]), hep.pick(raw.vJet_pt, ref[1], 0))


def test_vary_a_field_and_a_defined_column(nocounter):
    path, raw = nocounter
    df = aks.read_root(path, step=1500)
    df.vary(("vJet", "pt"), up="vJet_pt_up", name="jes")
    df["order"] = aks.local_index(df.vJet.eta)
    df["also_order"] = aks.local_index(df.vJet.eta)  # the same expression (one hash-consed node)
    df.vary("order", by_pt=aks.argsort(df.vJet.pt), name="sort")
    df["first"] = aks.fill_none(aks.firsts(df.vJet.eta[df.order]), 0.0)
    df["first2"] = aks.fill_none(aks.firsts(df.vJet.eta[df.also_order]), 0.0)
    df["ht"] = aks.sum(df.vJet.pt, axis=1)
    h1 = aks.varied(df.sum("ht"))
    h2 = aks.varied(df.sum("first"))
    h3 = aks.varied(df.sum("first2"))
    ht, first, first2 = aks.compute(h1, h2, h3)
    per_row = np.asarray(ak.sum(raw.vJet_pt_up, axis=1)).astype(np.float64)
    assert ht["jes:up"] == pytest.approx(float(per_row.sum()), rel=1e-12)
    assert set(first) == {"nominal", "sort:by_pt", "jes:up"} or set(first) == {"nominal", "sort:by_pt"}
    ref = ak.firsts(raw.vJet_eta[ak.argsort(raw.vJet_pt)])
    assert first["sort:by_pt"] == pytest.approx(float(ak.sum(ak.fill_none(ref, 0.0))), rel=1e-6)
    assert set(first2) == {"nominal"}  # the variation of `order` does not leak into the identical expression


def test_friend_match(tmp_path):
    n = 400
    main = _write(tmp_path / "m.root", {"event": np.arange(n, dtype=np.uint64), "x": np.ones(n)})
    good = _write(tmp_path / "g.root", {"f_event": np.arange(n, dtype=np.uint64), "f_y": np.ones(n)}, "F")
    shifted = _write(tmp_path / "s.root", {"f_event": np.arange(n, dtype=np.uint64) + (np.arange(n) > 250),
                                           "f_y": np.ones(n)}, "F")
    df = aks.read_root(main, step=100, friends={"F": aks.Friend(good, match={"f_event": "event"})})
    assert aks.compute(df.sum("f_y"))[0] == n
    df = aks.read_root(main, step=100, friends={"F": aks.Friend(shifted, match={"f_event": "event"})})
    with pytest.raises(ValueError, match="not aligned"):
        aks.compute(df.sum("f_y"))
    with pytest.raises(ValueError, match="match"):
        aks.read_root(main, friends={"F": aks.Friend(good, match={"event": "f_event"})})


def test_checked(nocounter):
    path, raw = nocounter
    df = aks.read_root(path, step=1500)
    df["ok"] = aks.checked(df.w * 2, df.w > 0, "negative weight")
    assert np.asarray(df.collect(["ok"]).value["ok"]).sum() == pytest.approx(2 * float(ak.sum(raw.w)))
    df["bad"] = aks.checked(df.w, df.w > 1, "w <= 1")
    with pytest.raises(ValueError, match="w <= 1"):
        aks.compute(df.collect(["bad"]))
