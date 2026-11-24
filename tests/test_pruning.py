"""Only the branches a result needs are read."""
import akstream as aks
from akstream.engine import Plan


def test_bulk_read_set(events_file):
    df = aks.read_root(events_file)
    df["ht"] = aks.sum(df.Jet.pt[df.Jet.eta > 0], axis=1)
    h = df[df.mll > 100].hist("ht", bins=10, range=(0, 100), weight="genWeight")
    assert Plan([h]).prefetch == ["Jet_eta", "Jet_pt", "genWeight", "mll"]


def test_variation_branches_only_when_varied_result(events_file):
    df = aks.read_root(events_file)
    df.vary("Jet_pt", up="Jet_pt_jer4_up", down="Jet_pt_jer4_down", name="jer4")
    df["ht"] = aks.sum(df.Jet.pt, axis=1)
    h = df.hist("ht", bins=10, range=(0, 100))
    assert Plan([h]).prefetch == ["Jet_pt"]
    hv = aks.varied(df.hist("ht", bins=10, range=(0, 100)))
    assert Plan([hv]).prefetch == ["Jet_pt", "Jet_pt_jer4_down", "Jet_pt_jer4_up"]


def test_record_fields_read_lazily(events_file):
    df = aks.read_root(events_file, step=5_000)
    jets = df.Jet[df.Jet.pt > 20]
    pairs = aks.combinations(jets, 2, axis=1)  # the whole record: fields are loaded on touch
    df["mjj_proxy"] = aks.max(pairs["0"].pt + pairs["1"].pt, axis=1)
    df["mjj_proxy"] = aks.fill_none(df.mjj_proxy, -1.0)
    out = df.collect(["mjj_proxy"]).value
    assert len(out) == 20_000
    assert set(aks.last_stats["lazy_branches"]) == set()  # pt is in the bulk set (mask), nothing else touched
    plan = Plan([df.collect(["mjj_proxy"])])
    assert "Jet_eta" not in plan.prefetch and "Jet_mass" not in plan.prefetch


def test_nominal_pass_computes_each_node_once(events_file):
    """nominal-only passes free values after their last consumer: nothing may be recomputed or re-read."""
    calls = []

    def counted(pt):
        calls.append(len(pt))
        return aks_np_sum(pt)

    import awkward as ak
    import numpy as np

    def aks_np_sum(pt):
        return ak.to_numpy(ak.sum(pt, axis=1)).astype(np.float64)

    df = aks.read_root(events_file, step=5_000)
    sel = df[df.mll > 90]
    sel["ht"] = aks.apply(counted, sel.Jet.pt)
    regions = {k: sel[sel.ht > cut] for k, cut in (("a", 0), ("b", 50), ("c", 100))}
    hs = [r.hist(v, bins=10, range=(0, 300), weight="genWeight") for r in regions.values() for v in ("ht", "mu1_pt")]
    aks.compute(*hs)
    assert len(calls) == 4  # one call per chunk
    assert aks.last_stats["lazy_branches"] == []  # no branch read twice
