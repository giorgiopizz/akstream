"""hep.match_dr / clean_dr / ancestor / best_pair / met_shift / LumiMask against plain Python + numpy references,
eager and lazy."""
import awkward as ak
import numpy as np
import pytest
import uproot

import akstream as aks
from akstream import hep


def _jag(rng, counts, lo, hi, dtype=np.float32):
    return ak.unflatten(rng.uniform(lo, hi, counts.sum()).astype(dtype), counts)


@pytest.fixture(scope="module")
def two_collections():
    rng = np.random.default_rng(21)
    n = 2000
    na, nb = rng.integers(0, 4, n), rng.integers(0, 6, n)
    a = {"eta": _jag(rng, na, -2.4, 2.4), "phi": _jag(rng, na, -np.pi, np.pi)}
    b = {"eta": _jag(rng, nb, -2.4, 2.4), "phi": _jag(rng, nb, -np.pi, np.pi), "ok": _jag(rng, nb, 0, 1) > 0.3}
    return a, b


def _dr2(ea, pa, eb, pb):
    de = np.float64(ea) - np.float64(eb)
    dp = hep.delta_phi(np.float64(pb), np.float64(pa))
    return de * de + dp * dp


def _ref_match(a, b, max_dr, mask=None, unique=False):
    out_i, out_d = [], []
    for ea, pa, eb, pb, ok in zip(a["eta"], a["phi"], b["eta"], b["phi"], b["ok"] if mask else b["eta"]):
        ok = list(ok) if mask else [True] * len(eb)
        d = [[_dr2(ea[i], pa[i], eb[j], pb[j]) for j in range(len(eb))] for i in range(len(ea))]
        idx, dr = [-1] * len(ea), [-1.0] * len(ea)
        if not unique:
            for i in range(len(ea)):
                cands = [(d[i][j], j) for j in range(len(eb)) if ok[j] and d[i][j] < max_dr * max_dr]
                if cands:
                    best = min(cands, key=lambda c: c[0])  # first minimum
                    idx[i], dr[i] = best[1], float(np.sqrt(best[0]))
        else:
            pairs = sorted(((d[i][j], i, j) for i in range(len(ea)) for j in range(len(eb))
                            if ok[j] and d[i][j] < max_dr * max_dr), key=lambda p: p[0])
            used_a, used_b = set(), set()
            for d2, i, j in pairs:
                if i in used_a or j in used_b:
                    continue
                used_a.add(i)
                used_b.add(j)
                idx[i], dr[i] = j, float(np.sqrt(d2))
        out_i.append(idx)
        out_d.append(dr)
    return out_i, out_d


@pytest.mark.parametrize("unique", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_match_dr(two_collections, unique, masked):
    a, b = two_collections
    index, dr = hep.match_dr(a["eta"], a["phi"], b["eta"], b["phi"], 0.8, b_mask=b["ok"] if masked else None,
                             unique=unique)
    ri, rd = _ref_match(a, b, 0.8, masked, unique)
    assert index.tolist() == ri
    assert dr.tolist() == rd
    if unique:  # every B object used at most once
        for row in index.tolist():
            got = [j for j in row if j >= 0]
            assert len(got) == len(set(got))


def test_clean_dr(two_collections):
    a, b = two_collections
    keep = hep.clean_dr(a["eta"], a["phi"], b["eta"], b["phi"], 0.4)
    ref = [[all(_dr2(ea[i], pa[i], eb[j], pb[j]) >= 0.16 for j in range(len(eb))) for i in range(len(ea))]
           for ea, pa, eb, pb in zip(a["eta"], a["phi"], b["eta"], b["phi"])]
    assert keep.tolist() == ref


def _gen_tree(rng, n):
    counts = rng.integers(1, 12, n)
    pdg, mother = [], []
    for c in counts:
        pdg.append(list(rng.choice([1, 2, 13, -13, 22, 23, 24, -24, 25], c)))
        mother.append([int(rng.integers(-1, i)) if i else -1 for i in range(c)])
    return ak.Array(pdg), ak.Array(mother)


def test_ancestor():
    rng = np.random.default_rng(5)
    pdg, mother = _gen_tree(rng, 1500)
    got = hep.ancestor(pdg, mother, [23, 24])
    ref = []
    for p, m in zip(pdg.tolist(), mother.tolist()):
        row = []
        for j in range(len(p)):
            k, hit = m[j], -1
            while k >= 0:
                if abs(p[k]) in (23, 24):
                    hit = k
                    break
                k = m[k]
            row.append(hit)
        ref.append(row)
    assert got.tolist() == ref
    signed = hep.ancestor(pdg, mother, -24, abs_id=False).tolist()
    assert all(pdg.tolist()[e][i] == -24 for e, row in enumerate(signed) for i in row if i >= 0)


def test_best_pair():
    rng = np.random.default_rng(9)
    n = 1500
    c = rng.integers(0, 5, n)
    lep = {"pt": _jag(rng, c, 10, 120), "eta": _jag(rng, c, -2.4, 2.4), "phi": _jag(rng, c, -np.pi, np.pi),
           "mass": _jag(rng, c, 0.0, 0.2), "charge": ak.unflatten(rng.choice([-1, 1], c.sum()), c),
           "flav": ak.unflatten(rng.choice([11, 13], c.sum()), c)}
    found, i1, i2, mass = hep.best_pair(lep, flavour=lep["flav"])
    for ev in range(n):
        f = {k: np.asarray(v[ev]) for k, v in lep.items()}
        best = (np.inf, -1, -1, 0.0)
        for a in range(len(f["pt"])):
            for b in range(a + 1, len(f["pt"])):
                if f["charge"][a] * f["charge"][b] >= 0 or f["flav"][a] != f["flav"][b]:
                    continue
                v = [aks.vector.p4(*(f[k][[i]].astype(np.float64) for k in ("pt", "eta", "phi", "mass"))) for i in (a, b)]
                m = float((v[0] + v[1]).mass[0])
                if abs(m - 91.1876) < best[0]:
                    best = (abs(m - 91.1876), a, b, m)
        assert (bool(found[ev]), int(i1[ev]), int(i2[ev])) == (best[1] >= 0, best[1], best[2]), ev
        assert float(mass[ev]) == best[3]


def test_met_shift():
    rng = np.random.default_rng(3)
    c = rng.integers(0, 5, 500)
    old, phi = _jag(rng, c, 20, 100), _jag(rng, c, -np.pi, np.pi)
    new = old * np.float32(1.03)
    mpt, mphi = rng.uniform(0, 80, 500), rng.uniform(-np.pi, np.pi, 500)
    pt, ph = hep.met_shift(mpt, mphi, new, old, phi)
    d = ak.values_astype(new, np.float64) - ak.values_astype(old, np.float64)
    x = mpt * np.cos(mphi) - np.asarray(ak.sum(d * np.cos(ak.values_astype(phi, np.float64)), axis=1))
    y = mpt * np.sin(mphi) - np.asarray(ak.sum(d * np.sin(ak.values_astype(phi, np.float64)), axis=1))
    np.testing.assert_allclose(pt, np.hypot(x, y), rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(ph, np.arctan2(y, x), rtol=1e-13, atol=1e-12)


def test_lumi_mask(tmp_path):
    golden = {"355100": [[1, 10], [20, 25]], "355200": [[5, 5]]}
    m = hep.LumiMask(golden)
    run = np.array([355100, 355100, 355100, 355100, 355200, 355200, 355300, 355099])
    lumi = np.array([1, 10, 11, 22, 5, 6, 1, 3])
    np.testing.assert_array_equal(m(run, lumi), [True, True, False, True, True, False, False, False])
    path = tmp_path / "golden.json"
    path.write_text(__import__("json").dumps(golden))
    np.testing.assert_array_equal(hep.LumiMask(str(path))(run, lumi), m(run, lumi))


def test_kernels_lazily(tmp_path, two_collections):
    a, b = two_collections
    rng = np.random.default_rng(1)
    pdg, mother = _gen_tree(rng, 2000)
    data = {"run": np.full(2000, 355100), "lumi": np.arange(2000) % 30,
            "A_eta": a["eta"], "A_phi": a["phi"], "B_eta": b["eta"], "B_phi": b["phi"],
            "Gen_pdgId": pdg, "Gen_mother": mother}
    path = str(tmp_path / "k.root")
    with uproot.recreate(path) as f:
        f.mktree("Events", {k: (v.type.content if isinstance(v, ak.Array) else v.dtype) for k, v in data.items()})
        f["Events"].extend(data)
    df = aks.read_root(path, step=600)
    index, dr = hep.match_dr(df.A.eta, df.A.phi, df.B.eta, df.B.phi, 0.8)
    df.A["match"], df.A["dr"] = index, dr
    df.A["clean"] = hep.clean_dr(df.A.eta, df.A.phi, df.B.eta, df.B.phi, 0.4)
    df.Gen["from_v"] = hep.ancestor(df.Gen.pdgId, df.Gen.mother, [23, 24])
    df["good_lumi"] = hep.LumiMask({"355100": [[1, 10]]})(df.run, df.lumi)
    out = df.to_awkward(["A", "Gen", "good_lumi"])
    ei, ed = hep.match_dr(a["eta"], a["phi"], b["eta"], b["phi"], 0.8)
    assert out.A.match.tolist() == ei.tolist() and out.A.dr.tolist() == ed.tolist()
    assert out.A.clean.tolist() == hep.clean_dr(a["eta"], a["phi"], b["eta"], b["phi"], 0.4).tolist()
    assert out.Gen.from_v.tolist() == hep.ancestor(pdg, mother, [23, 24]).tolist()
    np.testing.assert_array_equal(np.asarray(out.good_lumi), (np.arange(2000) % 30 >= 1) & (np.arange(2000) % 30 <= 10))
