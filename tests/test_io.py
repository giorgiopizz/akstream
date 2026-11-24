"""ROOT output with uproot, multiple input files, workers."""
import awkward as ak
import numpy as np
import pytest
import uproot

import akstream as aks


def test_to_root_roundtrip(events_file, raw, tmp_path):
    df = aks.read_root(events_file, step=3_000)
    df["ht"] = aks.sum(df.Jet.pt, axis=1)
    sel = df[df.mll > 120]
    sel["goodjets"] = sel.Jet[sel.Jet.pt > 30][["pt", "eta"]]
    out = tmp_path / "out.root"
    r = sel.to_root(out, ["mll", "ht", "goodjets"], basket_entries=4_000)
    info = r.value
    assert info["entries"] == int(ak.sum(raw.mll > 120))
    t = uproot.open(out)["Events"]
    assert set(t.keys()) == {"mll", "ht", "ngoodjets", "goodjets_pt", "goodjets_eta"}
    a = t.arrays(library="ak")
    m = raw.mll > 120
    np.testing.assert_allclose(a.mll, raw.mll[m])
    np.testing.assert_allclose(a.ht, ak.sum(raw.Jet.pt, axis=1)[m])
    rj = raw.Jet[m][raw.Jet[m].pt > 30]
    assert ak.all(a.goodjets_pt == rj.pt) and ak.all(a.ngoodjets == ak.num(rj, axis=1))


def test_multiple_files_and_glob(two_files):
    df = aks.read_root(two_files, step=2_500)
    assert df.count().value == 12_000
    glob = two_files[0].rsplit("/", 1)[0] + "/*.root"
    assert aks.read_root(glob).count().value == 12_000


def test_tree_in_path(events_file):
    assert aks.read_root(events_file + ":Events").count().value == 20_000


def test_to_root_with_workers(two_files, tmp_path):
    df = aks.read_root(two_files, step=2_000)
    r = df[df.mll > 100].to_root(tmp_path / "w.root", ["mll"])
    info = aks.compute(r, workers=2)[0]
    total = sum(uproot.open(f)["Events"].num_entries for f in info["files"])
    ref = aks.read_root(two_files)
    assert total == info["entries"] == ref[ref.mll > 100].count().value


def test_threads_option(events_file):
    df = aks.read_root(events_file, step=5_000, threads=2)
    assert df.hist("mll", bins=10, range=(70, 150)).value.sum(flow=True).value == 20_000


def test_record_under_variation(events_file, raw):
    """a whole collection passed to an awkward function (combinations): fields loaded lazily, also in a variation."""
    df = aks.read_root(events_file, step=4_000)
    df.vary_branches(["Jet_pt"], ["jer4"])
    jets = df.Jet[df.Jet.pt > 20]
    pairs = aks.combinations(jets, 2, axis=1)
    df["best"] = aks.fill_none(aks.max(pairs["0"].pt + pairs["1"].pt, axis=1), -1.0)
    h = aks.varied(df.hist("best", bins=30, range=(0, 600)))
    got = aks.compute(h, validate=10)[0]
    for v in ("nominal", "jer4:up"):
        pt = raw.Jet.pt if v == "nominal" else raw.Jet.pt_jer4_up
        j = pt[pt > 20]
        p = ak.combinations(j, 2, axis=1)
        best = ak.to_numpy(ak.fill_none(ak.max(p["0"] + p["1"], axis=1), -1.0))
        ref = np.histogram(best, bins=30, range=(0, 600))[0]
        np.testing.assert_array_equal(got[v].values(), ref)


def test_workers_split_large_chunks(two_files):
    """two chunks (one per file) and three workers: chunks are split so that every worker gets work."""
    df = aks.read_root(two_files, step=100_000)
    h = df.hist("mll", bins=20, range=(70, 150), weight="genWeight")
    one = aks.compute(h, workers=1)[0]
    df2 = aks.read_root(two_files, step=100_000)
    h2 = df2.hist("mll", bins=20, range=(70, 150), weight="genWeight")
    three = aks.compute(h2, workers=3)[0]
    assert aks.last_stats["chunks"] >= 6
    np.testing.assert_allclose(one.view(flow=True).value, three.view(flow=True).value, rtol=1e-12)


def test_friend_trees_and_entry(tmp_path):
    import uproot
    rng = np.random.default_rng(5)
    n = 1000
    main, friend = [], []
    for i, size in enumerate((n, 700)):
        x = rng.normal(size=size)
        m, f = tmp_path / f"main{i}.root", tmp_path / f"friend{i}.root"
        cols = {"x": x, "event": np.arange(size, dtype=np.uint64) + 10 * i}
        fcols = {"y": 2 * x, "fr_event": np.arange(size, dtype=np.uint64) + 10 * i,
                 "jets": ak.Array([[v] * (k % 3) for k, v in enumerate(x)])}
        for path, tree, data in ((m, "Events", cols), (f, "Fr", fcols)):
            with uproot.recreate(path) as out:
                out.mktree(tree, {k: (v.type.content if isinstance(v, ak.Array) else v.dtype) for k, v in data.items()})
                out[tree].extend(data)
        main.append(str(m))
        friend.append(str(f))
    df = aks.read_root(main, step=300, friends={"Fr": friend})
    assert "y" in df.branches and "rdfentry_" not in df.branches
    sel = df[df.y - 2 * df.x == 0]
    sel["k"] = sel.entry
    sel["nj"] = aks.num(sel.jets, axis=1)
    out = sel.collect(["k", "event", "fr_event", "nj"]).value
    assert len(out) == n + 700
    np.testing.assert_array_equal(np.asarray(out["event"]), np.asarray(out["fr_event"]))
    np.testing.assert_array_equal(np.asarray(out["k"]), np.concatenate([np.arange(n), np.arange(700)]))
    np.testing.assert_array_equal(np.asarray(out["nj"]), np.asarray(out["k"]) % 3)
    with pytest.raises(ValueError):
        aks.read_root(main, friends={"Fr": friend[:1]})


def test_cpp_conditional_expressions(events_file, raw):
    df = aks.read_root(events_file)
    df = df.define("reg", "mll > 120 ? 1 : (mll > 90 && mu1_pt > 40 ? 2 : (min(mll, 100.0) > 95 ? 3 : 0))")
    df = df.define("w", "abs(mu2_pt - 30) < 5 ? genWeight : 0.0")
    out = df.collect(["reg", "w"]).value
    mll, mu1, mu2, gw = (np.asarray(raw[k]) for k in ("mll", "mu1_pt", "mu2_pt", "genWeight"))
    ref = np.where(mll > 120, 1, np.where((mll > 90) & (mu1 > 40), 2, np.where(np.minimum(mll, 100.0) > 95, 3, 0)))
    np.testing.assert_array_equal(np.asarray(out["reg"]), ref)
    np.testing.assert_array_equal(np.asarray(out["w"]), np.where(np.abs(mu2 - 30) < 5, gw, 0.0))


def test_varied_to_root_one_file_per_variation(events_file, tmp_path):
    """aks.varied(to_root): each variation's file equals a nominal-only run with the varied branches swapped in."""
    def build(df):  # the Jet_pt column (the reference below replaces it by a varied branch)
        pt = df["Jet_pt"]
        df["ht"] = aks.sum(pt[pt > 30], axis=1)
        df["lead"] = aks.fill_none(aks.max(pt, axis=1), -1.0)
        sel = df.filter("ht > 60 && mll > 80")
        sel["jets"] = sel.Jet_pt[sel.Jet_pt > 30]
        return sel

    df = aks.read_root(events_file, step=3_000)
    df.vary("jer4", up={"Jet_pt": "Jet_pt_jer4_up"}, down={"Jet_pt": "Jet_pt_jer4_down"})
    df.vary("jes", up={"Jet_pt": "Jet_pt_jes_up"})
    out = aks.varied(build(df).to_root(str(tmp_path / "out_{group}_{tag}.root"), ["ht", "lead", "jets", "event"]))
    h = aks.varied(build(df).hist("ht", (20, 0, 600)))  # alongside, in the same pass
    files, hist = aks.compute(out, h)
    assert set(files) == {"nominal", "jer4:up", "jer4:down", "jes:up"}
    for v, branch in (("nominal", "Jet_pt"), ("jer4:up", "Jet_pt_jer4_up"), ("jer4:down", "Jet_pt_jer4_down"),
                      ("jes:up", "Jet_pt_jes_up")):
        ref_df = aks.read_root(events_file, step=3_000)
        ref_df["Jet_pt"] = ref_df[branch]
        ref = build(ref_df).collect(["ht", "lead", "jets", "event"]).value
        got = uproot.open(files[v]["files"][0])["Events"].arrays(library="ak")
        assert files[v]["entries"] == len(ref) == len(got)
        for c in ("ht", "lead", "event"):
            np.testing.assert_array_equal(np.asarray(got[c]), np.asarray(ref[c]), err_msg=f"{v} {c}")
        assert ak.all(got["jets"] == ref["jets"]), v
        np.testing.assert_array_equal(hist[v].values(), np.histogram(np.asarray(ref["ht"]), 20, (0, 600))[0])
    assert (tmp_path / "out_nominal_nominal.root").exists() and (tmp_path / "out_jer4_up.root").exists()
