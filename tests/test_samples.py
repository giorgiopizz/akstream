"""aks.read_samples: several samples in one frame, per-sample constants and results, sums inside expressions."""
import warnings

import awkward as ak
import numpy as np
import pytest
import uproot

import akstream as aks
from conftest import make_events

HIST = dict(bins=20, range=(70, 150))


@pytest.fixture(scope="module")
def samples(tmp_path_factory, two_files):
    """two simulated samples and a data-like one: no genWeight, no varied jet branches, no pu_up."""
    d = tmp_path_factory.mktemp("samples")
    data, njet = make_events(4_000, seed=7)
    branches = {k: v for k, v in data.items() if k in ("event", "mll", "mu1_pt", "mu2_pt", "pu")}
    branches["Jet"] = ak.zip({f: ak.unflatten(data[f"Jet_{f}"], njet) for f in ("pt", "eta", "phi", "mass")})
    path = str(d / "data.root")
    with uproot.recreate(path) as f:
        f.mktree("Events", {k: (v.type.content if isinstance(v, ak.Array) else v.dtype) for k, v in branches.items()})
        f["Events"].extend(branches)
    return {"DY": {"files": [two_files[0]], "xs": 6077.0, "is_data": False},
            "EWKZ": {"files": [two_files[1]], "xs": 1.2, "step": 1_500, "is_data": False},
            "data": {"files": [path], "genWeight": 1.0, "xs": 1.0, "is_data": True}}


def raw(spec):
    return uproot.open(spec["files"][0])["Events"].arrays(library="ak")


def same_hist(a, b):
    np.testing.assert_array_equal(a.values(flow=True), b.values(flow=True))
    np.testing.assert_array_equal(a.variances(flow=True), b.variances(flow=True))


def test_histograms_and_constants_per_sample(samples):
    df = aks.read_samples(samples, step=2_000)
    assert set(df.branches) >= {"sample", "xs", "is_data", "genWeight", "Jet_pt_jes_up"}
    h = df.hist("mll", **HIST, weight="genWeight")
    xs = df.sum("xs")
    n = df.count()
    ewkz = df[df.sample == "EWKZ"].count()
    data_only = df[df.is_data].count()
    hv = h.value
    assert list(hv) == ["DY", "EWKZ", "data"]
    for name, spec in samples.items():
        ref = aks.read_root(spec["files"], step=spec.get("step", 2_000))  # the same chunks: the same sums
        same_hist(hv[name], ref.hist("mll", **HIST, weight="genWeight").value if name != "data"
                  else ref.hist("mll", **HIST, weight=ref.mll * 0 + 1.0).value)
        entries = len(raw(spec))
        assert n.value[name] == entries
        assert xs.value[name] == pytest.approx(spec["xs"] * entries, rel=1e-12)
    assert ewkz.value == {"DY": 0, "EWKZ": len(raw(samples["EWKZ"])), "data": 0}
    assert data_only.value == {"DY": 0, "EWKZ": 0, "data": len(raw(samples["data"]))}
    import hist
    one = aks.to_hist(hv)
    assert [a.name for a in one.axes][-1] == "dataset"
    np.testing.assert_array_equal(one[:, hist.loc("EWKZ")].values(), hv["EWKZ"].values())


def test_variations_only_where_the_branches_are(samples):
    def run(df):
        df.vary("Jet_pt", up="Jet_pt_jes_up", down="Jet_pt_jes_down", name="jes")
        df["ht"] = aks.sum(df.Jet.pt[df.Jet.pt > 30], axis=1)
        return aks.varied(df[df.ht > 40].hist("ht", bins=30, range=(0, 600), weight="genWeight"))
    got = run(aks.read_samples(samples, step=2_000)).value
    assert set(got["DY"]) == set(got["EWKZ"]) == {"nominal", "jes:up", "jes:down"} and set(got["data"]) == {"nominal"}
    for name in ("DY", "EWKZ"):
        ref = run(aks.read_root(samples[name]["files"], step=samples[name].get("step", 2_000))).value
        for v in ref:
            same_hist(got[name][v], ref[v])
    import hist
    one = aks.to_hist(got)
    np.testing.assert_array_equal(one[:, hist.loc("data"), hist.loc("jes:up")].values(), got["data"]["nominal"].values())


def test_workers_per_sample(samples):
    def run(workers):
        df = aks.read_samples(samples, step=1_000)
        return aks.compute(df.hist("mll", **HIST, weight="genWeight"), df.count(), workers=workers)
    (h1, n1), (h2, n2) = run(1), run(3)
    assert n1 == n2
    for name in h1:
        np.testing.assert_allclose(h1[name].values(flow=True), h2[name].values(flow=True), rtol=1e-12)


def normalized_reference(samples, cut=None):
    out = {}
    for name, spec in samples.items():
        a = raw(spec)
        w = np.asarray(a.genWeight) if "genWeight" in a.fields else np.ones(len(a))
        w = w * spec["xs"] / w.sum()
        mll = np.asarray(a.mll)
        keep = np.ones(len(a), bool) if cut is None else w > cut
        out[name] = (mll[keep], w[keep])
    return out


def test_sum_in_the_weight_is_one_pass(samples):
    """df.sum in a histogram weight: no warning, one pass, the normalized histogram per sample."""
    df = aks.read_samples(samples, step=2_000)
    df["weight"] = df.genWeight * df.xs / df.sum("genWeight")
    h = df.hist("mll", **HIST, weight="weight")
    norm = df.sum("weight")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        hv, nv = aks.compute(h, norm)
    assert aks.last_stats["entries"] == sum(len(raw(s)) for s in samples.values())  # a single pass
    for name, (mll, w) in normalized_reference(samples).items():
        ref = np.histogram(mll, bins=20, range=(70, 150), weights=w)[0]
        np.testing.assert_allclose(hv[name].values(), ref, rtol=1e-12)
        np.testing.assert_allclose(hv[name].variances(), np.histogram(mll, 20, (70, 150), weights=w * w)[0], rtol=1e-12)
        assert nv[name] == pytest.approx(samples[name]["xs"], rel=1e-12)


def test_sum_in_a_selection_needs_a_pass_before(samples):
    df = aks.read_samples(samples, step=2_000)
    df["weight"] = df.genWeight * df.xs / df.sum("genWeight")
    sel = df[df.weight > 1e-4]
    c = sel.count()
    with pytest.warns(RuntimeWarning, match="its own pass"):
        got = c.value
    for name, (mll, _) in normalized_reference(samples, cut=1e-4).items():
        assert got[name] == len(mll)


def test_to_root_with_samples(samples, tmp_path):
    df = aks.read_samples(samples, step=2_000)
    one = df[df.mll > 100].to_root(tmp_path / "all.root", ["mll", "xs"])
    each = df[df.mll > 100].to_root(str(tmp_path / "{sample}.root"), ["mll"])
    a, b = aks.compute(one, each)
    t = uproot.open(a["files"][0])["Events"].arrays(library="np")
    assert set(t) == {"mll", "xs", "sample"}
    for i, (name, spec) in enumerate(samples.items()):
        ref = np.asarray(raw(spec).mll)
        np.testing.assert_array_equal(t["mll"][t["sample"] == i], ref[ref > 100])
        assert set(t["xs"][t["sample"] == i]) == {spec["xs"]}
        files = b[name]["files"]
        assert files == [str(tmp_path / f"{name}.root")]
        np.testing.assert_array_equal(uproot.open(files[0])["Events"]["mll"].array(library="np"), ref[ref > 100])


def test_a_branch_missing_in_a_sample(samples):
    """read on every chunk of the sample that needs it (a selection does not avoid the read): an error naming the
    sample, unless the sample gives it as a constant."""
    df = aks.read_samples(samples)
    with pytest.raises(KeyError, match="sample 'data'"):
        df[~df.is_data].sum("pu_up").value
    with_constant = dict(samples, data=dict(samples["data"], pu_up=0.0))
    df = aks.read_samples(with_constant)
    got = df[~df.is_data].sum("pu_up").value
    assert got["data"] == 0 and got["DY"] == pytest.approx(float(np.sum(raw(samples["DY"]).pu_up)), rel=1e-12)
