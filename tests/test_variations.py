"""Systematic variations: the sparse (delta) evaluation must equal a full recomputation on the alternative branches.

The analysis has a jet-dependent selection (rows enter and leave it under a jet variation), option types (pad_none,
firsts), sorting / indexing of a collection, a user function (a fake DNN) and a weight variation.
"""
import awkward as ak
import hist
import numpy as np
import pytest

import akstream as aks

VARS = ["jer4", "jes"]


def fake_dnn(j1, j2, mll):
    return 1.0 / (1.0 + np.exp(-(0.01 * j1 + 0.02 * j2 - 0.01 * (mll - 100))))


def build(df):
    df["w"] = df.genWeight * df.pu
    jets = df.Jet[(df.Jet.pt > 25) & (abs(df.Jet.eta) < 4.7)]
    jets = jets[aks.argsort(jets.pt, axis=1, ascending=False)]
    df["njet"] = aks.num(jets.pt, axis=1)
    padded = aks.pad_none(jets.pt, 2, axis=1)
    df["j1"] = aks.fill_none(padded[:, 0], -1.0)
    df["j2"] = aks.fill_none(padded[:, 1], -1.0)
    df["ht"] = aks.sum(jets.pt, axis=1)
    df["mj"] = aks.fill_none(aks.firsts(jets.mass, axis=1), -1.0)
    df["score"] = aks.apply(fake_dnn, df.j1, df.j2, df.mll)
    for s in VARS:
        df.vary(["Jet_pt", "Jet_mass"], name=s, **{t: [f"Jet_pt_{s}_{t}", f"Jet_mass_{s}_{t}"] for t in ("up", "down")})
    df.vary("w", up=df.genWeight * df.pu_up, name="pu")
    sel = df[df.njet >= 2]
    sel2 = sel[sel.score > 0.3]
    return {
        "ht": aks.varied(sel.hist("ht", bins=40, range=(0, 600), weight="w")),
        "score": aks.varied(sel2.hist("score", bins=25, range=(0, 1), weight="w")),
        "jetpt": aks.varied(df.hist(jets.pt, bins=30, range=(0, 300), weight="w")),
        "j1j2": aks.varied(sel.hist(["j1", "j2"], axes=[(10, 0, 300), (10, 0, 200)], weight="w")),
        "mj": aks.varied(df.hist("mj", bins=20, range=(-1, 39))),
        "count": aks.varied(sel2.count()),
        "sumw": aks.varied(sel.sum("w")),
    }


def reference(a, variation):
    """the same analysis in plain awkward on the full arrays of one variation."""
    jet = a.Jet
    w = ak.to_numpy(a.genWeight * a.pu)
    if variation and variation.startswith(tuple(VARS)):
        syst, tag = variation.split(":")
        jet = ak.zip({"pt": jet[f"pt_{syst}_{tag}"], "eta": jet.eta, "mass": jet[f"mass_{syst}_{tag}"]})
    if variation == "pu:up":
        w = ak.to_numpy(a.genWeight * a.pu_up)
    jets = jet[(jet.pt > 25) & (abs(jet.eta) < 4.7)]
    jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
    njet = ak.to_numpy(ak.num(jets, axis=1))
    padded = ak.pad_none(jets.pt, 2, axis=1)
    j1 = ak.to_numpy(ak.fill_none(padded[:, 0], -1.0))
    j2 = ak.to_numpy(ak.fill_none(padded[:, 1], -1.0))
    ht = ak.to_numpy(ak.sum(jets.pt, axis=1))
    mj = ak.to_numpy(ak.fill_none(ak.firsts(jets.mass, axis=1), -1.0))
    score = fake_dnn(j1, j2, ak.to_numpy(a.mll))
    s1 = njet >= 2
    s2 = s1 & (score > 0.3)

    def h1(x, wt, bins, lo, hi):
        h = hist.Hist(hist.axis.Regular(bins, lo, hi), storage=hist.storage.Weight())
        h.fill(x, weight=wt)
        return h

    j1j2 = hist.Hist(hist.axis.Regular(10, 0, 300), hist.axis.Regular(10, 0, 200), storage=hist.storage.Weight())
    j1j2.fill(j1[s1], j2[s1], weight=w[s1])
    counts = ak.num(jets, axis=1)
    return {
        "ht": h1(ht[s1], w[s1], 40, 0, 600),
        "score": h1(score[s2], w[s2], 25, 0, 1),
        "jetpt": h1(ak.to_numpy(ak.flatten(jets.pt)), np.repeat(w, counts), 30, 0, 300),
        "j1j2": j1j2,
        "mj": h1(mj, np.ones(len(mj)), 20, -1, 39),
        "count": int(s2.sum()),
        "sumw": float(w[s1].sum()),
    }


def check(results, raw, rtol=1e-10):
    variations = ["nominal"] + [f"{s}:{t}" for s in VARS for t in ("up", "down")] + ["pu:up"]
    for v in variations:
        ref = reference(raw, None if v == "nominal" else v)
        for name, value in results.items():
            got = value[v] if v in value else value["nominal"]
            exp = ref[name]
            if isinstance(exp, hist.Hist):
                np.testing.assert_allclose(got.view(flow=True).value, exp.view(flow=True).value, rtol=rtol,
                                           atol=1e-9, err_msg=f"{name} {v}")
                np.testing.assert_allclose(got.view(flow=True).variance, exp.view(flow=True).variance, rtol=rtol,
                                           atol=1e-9, err_msg=f"{name} {v}")
            else:
                assert got == pytest.approx(exp, rel=rtol), (name, v)


def run(events_file, **kw):
    df = aks.read_root(events_file, step=kw.pop("step", 3_000))
    r = build(df)
    values = aks.compute(*r.values(), **kw)
    return dict(zip(r, values))


def test_variation_names(events_file):
    out = run(events_file)
    assert set(out["ht"]) == {"nominal", "jer4:up", "jer4:down", "jes:up", "jes:down", "pu:up"}
    assert set(out["mj"]) == {"nominal", "jer4:up", "jer4:down", "jes:up", "jes:down"}  # no weight


@pytest.mark.parametrize("opts", [dict(), dict(validate=100), dict(engine="delta"), dict(engine="delta", validate=100),
                                  dict(engine="delta", dense_fraction=0.0), dict(engine="delta", dense_fraction=1.0),
                                  dict(engine="delta", cutoff="all"), dict(exact=True), dict(step=20_000),
                                  dict(step=777), dict(step=777, engine="delta")],
                         ids=["fused", "fused-validate", "delta", "delta-validate", "delta-all-dense", "delta-all-sparse",
                              "delta-cutoff-all", "exact", "one-chunk", "small-chunks", "small-chunks-delta"])
def test_matches_full_recomputation(events_file, raw, opts):
    check(run(events_file, **opts), raw)


def test_exact_mode_is_bitwise_brute_force(events_file):
    """exact=True refills every row: identical to evaluating each variation as its own nominal pass."""
    exact = run(events_file, exact=True)
    sparse = run(events_file)
    for name in ("ht", "score", "jetpt"):
        for v, h in exact[name].items():
            assert np.allclose(h.view(flow=True).value, sparse[name][v].view(flow=True).value, rtol=1e-12)


def test_sparse_variation_recomputes_few_rows(events_file):
    df = aks.read_root(events_file, step=20_000)
    df["ht"] = aks.sum(df.Jet.pt[df.Jet.pt > 25], axis=1)
    df["s"] = aks.apply(fake_dnn, df.ht, df.ht, df.mll)
    df.vary("Jet_pt", up="Jet_pt_jer4_up", down="Jet_pt_jer4_down", name="jer4")
    aks.compute(aks.varied(df.hist("s", bins=10, range=(0, 1))))
    # fused: the DNN-like function reruns only on the rows whose (masked) jets move
    assert 0 < aks.last_stats["fused_rows"] < 0.3 * 2 * 20_000
    aks.compute(aks.varied(df.hist("s", bins=10, range=(0, 1))), engine="delta")
    st = aks.last_stats
    # jer4 moves only jets with 2.5 < |eta| < 3: far fewer rows than 2 variations x 20k events x nodes
    assert st["dense_nodes"] == 0
    assert st["recomputed_rows"] < 0.3 * 2 * 20_000 * st["sparse_nodes"]


def test_parallel_workers_match(two_files):
    def go(workers):
        df = aks.read_root(two_files, step=2_000)
        r = build(df)
        return dict(zip(r, aks.compute(*r.values(), workers=workers)))
    one, two = go(1), go(2)
    for name in one:
        for v in one[name]:
            a, b = one[name][v], two[name][v]
            if isinstance(a, hist.Hist):
                np.testing.assert_allclose(a.view(flow=True).value, b.view(flow=True).value, rtol=1e-12, atol=1e-12)
            else:
                assert a == pytest.approx(b, rel=1e-12)


def test_apply_returning_a_dict(events_file, raw):
    """an aks.apply function may return several columns as a dict, also under variations."""
    def jets_summary(pt):
        return {"n": ak.to_numpy(ak.num(pt[pt > 25], axis=1)), "ht": ak.to_numpy(ak.sum(pt[pt > 25], axis=1))}
    df = aks.read_root(events_file, step=3_000)
    df.vary("Jet_pt", up="Jet_pt_jer4_up", down="Jet_pt_jer4_down", name="jer4")
    out = aks.apply(jets_summary, df.Jet.pt)
    df["ht"] = out["ht"]
    h = aks.varied(df.hist("ht", bins=20, range=(0, 400)))
    got = aks.compute(h, validate=10)[0]
    for v, pt in (("nominal", raw.Jet.pt), ("jer4:up", raw.Jet.pt_jer4_up)):
        ref = np.histogram(ak.to_numpy(ak.sum(pt[pt > 25], axis=1)), bins=20, range=(0, 400))[0]
        np.testing.assert_array_equal(got[v].values(), ref)


def test_variation_attached_in_a_selection(events_file, raw):
    """a variation registered on a selection's column (not on the parent's) reaches the histograms of sub-selections."""
    df = aks.read_root(events_file, step=4_000)
    df["w"] = df.genWeight * 1.0
    sel = df[df.mll > 90]
    sel.vary("w", up=sel.w * 2.0, name="extra")
    sub = sel[sel.mu1_pt > 30]
    h = aks.varied(sub.hist("mll", bins=10, range=(70, 150), weight="w"))
    got = aks.compute(h)[0]
    assert set(got) == {"nominal", "extra:up"}
    np.testing.assert_allclose(got["extra:up"].values(), 2 * got["nominal"].values(), rtol=1e-12)


@pytest.mark.parametrize("engine", ["fused", "delta"])
def test_apply_when_trigger(events_file, raw, engine):
    """apply(when=...): recomputed only where the selected jets move, still exact (validated by brute force)."""
    def ht_selected(pt, eta):
        sel = (pt > 30) & (abs(eta) < 2.5)
        return ak.to_numpy(ak.sum(pt[sel], axis=1))
    df = aks.read_root(events_file, step=4_000)
    for s in ("jer4", "jes"):
        df.vary("Jet_pt", up=f"Jet_pt_{s}_up", down=f"Jet_pt_{s}_down", name=s)
    keep = (df.Jet.pt > 30) & (abs(df.Jet.eta) < 2.5)
    df["ht"] = aks.apply(ht_selected, df.Jet.pt, df.Jet.eta, when=[keep, df.Jet.pt[keep]])
    h = aks.varied(df.hist("ht", bins=30, range=(0, 600)))
    got = aks.compute(h, validate=100, engine=engine)[0]
    for v, pt in (("nominal", raw.Jet.pt), ("jer4:up", raw.Jet.pt_jer4_up), ("jes:down", raw.Jet.pt_jes_down)):
        sel = (pt > 30) & (abs(raw.Jet.eta) < 2.5)
        ref = np.histogram(ak.to_numpy(ak.sum(pt[sel], axis=1)), bins=30, range=(0, 600))[0]
        np.testing.assert_array_equal(got[v].values(), ref)


def test_batched_variations_match_unbatched(events_file, raw):
    """the jer4 / jes variations replace the same branches: their phase 2 runs stacked, with the same result."""
    batched = run(events_file, validate=100)
    st = dict(aks.last_stats)
    single = run(events_file, batch_rows=1)
    assert st["fused_batches"] > 0 and aks.last_stats.get("fused_batches", 0) == 0
    for name in batched:
        for v in batched[name]:
            a, b = batched[name][v], single[name][v]
            if isinstance(a, hist.Hist):
                np.testing.assert_allclose(a.view(flow=True).value, b.view(flow=True).value, rtol=1e-12, atol=1e-12)
            else:
                assert a == pytest.approx(b, rel=1e-12)
    check(batched, raw)


def test_vary_forms(events_file):
    """vary(column(s), name=, **tags): a list of columns varied together = one vary per column under the same name;
    name defaults to the column."""
    def ht(df):
        jets = df.Jet[df.Jet.pt > 25]
        df["ht"] = aks.sum(jets.pt * jets.mass, axis=1)
        df["w"] = df.genWeight * 1.0
        return aks.varied(df.hist("ht", bins=20, range=(0, 4000), weight="w"))

    df = aks.read_root(events_file, step=5_000)
    df.vary("Jet_pt", up="Jet_pt_jer4_up", down="Jet_pt_jer4_down", name="jer4")
    df.vary(("Jet", "mass"), up="Jet_mass_jer4_up", down="Jet_mass_jer4_down", name="jer4")
    by_column = ht(df)
    df = aks.read_root(events_file, step=5_000)
    df.vary(["Jet_pt", ("Jet", "mass")], name="jer4", up=["Jet_pt_jer4_up", "Jet_mass_jer4_up"],
            down=[df.Jet_pt_jer4_down, "Jet_mass_jer4_down"])
    by_list = ht(df)
    df2 = aks.read_root(events_file, step=5_000)
    df2.vary("genWeight", up="genWeight * 2.0")             # name defaults to the column
    by_default = aks.varied(df2.sum("mll", weight="genWeight"))
    (a,), (b,), (c,) = aks.compute(by_column), aks.compute(by_list), aks.compute(by_default)
    assert set(a) == set(b) == {"nominal", "jer4:up", "jer4:down"}
    assert set(c) == {"nominal", "genWeight:up"} and c["genWeight:up"] == pytest.approx(2 * c["nominal"])
    for v in a:
        np.testing.assert_array_equal(a[v].values(), b[v].values())


def test_vary_errors(events_file):
    df = aks.read_root(events_file, step=5_000)
    df["w"] = df.genWeight * 1.0
    with pytest.raises(TypeError, match="name="):
        df.vary(["Jet_pt", "Jet_mass"], up=["Jet_pt_jes_up", "Jet_mass_jes_up"])
    with pytest.raises(TypeError, match="list of 2"):
        df.vary(["Jet_pt", "Jet_mass"], up="Jet_pt_jes_up", name="jes")
    with pytest.raises(TypeError, match="tags"):
        df.vary("w", name="pu")
    df.vary("w", up="genWeight * 2.0", name="pu")
    df.vary("w", up="genWeight * 2.0", name="pu")         # the same alternative again: a no-op
    with pytest.raises(ValueError, match="already registered"):
        df.vary("w", up="genWeight * 3.0", name="pu")
