"""akstream.vector: the C++ 4-vector kernels against the plain numpy formulas (eager), and lazily on frame columns."""
import math

import awkward as ak
import numpy as np
import pytest

import akstream as aks
from akstream import vector


# ---------------------------------------------------------------- the numpy formulas the kernels replace
def np_p4(pt, eta, phi, mass):
    pt, eta, phi, mass = (np.asarray(x, dtype=np.float64) for x in (pt, eta, phi, mass))
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return dict(px=px, py=py, pz=pz, e=np.sqrt(px * px + py * py + pz * pz + mass * mass))


def np_add(*vs):
    out = {k: sum(v[k] for v in vs) for k in ("px", "py", "pz", "e")}
    pt = np.hypot(out["px"], out["py"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["eta"] = np.arcsinh(np.where(pt > 0, out["pz"] / np.where(pt > 0, pt, 1), 0.0))
    out["pt"], out["phi"] = pt, np.arctan2(out["py"], out["px"])
    return out


def np_mass(v):
    m2 = v["e"] ** 2 - (v["px"] ** 2 + v["py"] ** 2 + v["pz"] ** 2)
    return np.where(m2 >= 0, np.sqrt(np.abs(m2)), -np.sqrt(np.abs(m2)))


def _boost(v, beta):
    bx, by, bz = beta
    b2 = bx * bx + by * by + bz * bz
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 1.0 / np.sqrt(1.0 - b2)
        gamma2 = np.where(b2 > 0, (gamma - 1.0) / np.where(b2 > 0, b2, 1), 0.0)
    bp = bx * v[0] + by * v[1] + bz * v[2]
    return (v[0] + gamma2 * bp * bx + gamma * bx * v[3], v[1] + gamma2 * bp * by + gamma * by * v[3],
            v[2] + gamma2 * bp * bz + gamma * bz * v[3], gamma * (v[3] + bp))


def _unit(x, y, z):
    n = np.sqrt(x * x + y * y + z * z)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, x / n, x), np.where(n > 0, y / n, y), np.where(n > 0, z / n, z)


def _cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def np_collins_soper(mu1, mu2, charge1):
    d = np_add(mu1, mu2)
    beta = (-d["px"] / d["e"], -d["py"] / d["e"], -d["pz"] / d["e"])
    one, zero = np.ones_like(d["e"]), np.zeros_like(d["e"])
    b1 = _boost((mu1["px"], mu1["py"], mu1["pz"], mu1["e"]), beta)
    b2 = _boost((mu2["px"], mu2["py"], mu2["pz"], mu2["e"]), beta)
    u1, u2 = _unit(*_boost((zero, zero, one, one), beta)[:3]), _unit(*_boost((zero, zero, -one, one), beta)[:3])
    z = _unit(u1[0] - u2[0], u1[1] - u2[1], u1[2] - u2[2])
    y = _unit(*_cross(u1, u2))
    x = _unit(*_cross(y, z))
    positive = tuple(np.where(charge1 > 0, b1[i], b2[i]) for i in range(3))
    return np.arctan2(_dot(positive, y), _dot(positive, x)), _dot(_unit(*positive), z)


def _numpy_uses_libm():
    """numpy's float64 transcendental functions are the C library's: then the kernels are exact. Not on AVX-512 CPUs
    (numpy's own SIMD sinh / arcsinh, last-ulp differences)."""
    x = np.random.default_rng(0).normal(0, 3, 2000)
    return all(np.array_equal(f(x), np.array([g(v) for v in x])) for f, g in
               ((np.cos, math.cos), (np.sin, math.sin), (np.sinh, math.sinh), (np.arcsinh, math.asinh)))


def assert_same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if _numpy_uses_libm():
        np.testing.assert_array_equal(a, b)
    else:  # pragma: no cover
        # an ulp of pz grows through the cancellation in E^2 - p^2 (mass) and in the boosts (Collins-Soper angles)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_libm_kernels_are_the_c_library():
    """_hep.cos / sin / sinh / cosh / log / atan2 / hypot: the C math library bit for bit, on any CPU (numpy's own SIMD
    versions on AVX-512 are not), keeping the shape."""
    import ctypes
    import ctypes.util
    from akstream import _hep
    libm = ctypes.CDLL(ctypes.util.find_library("m"))  # the C library itself (Python's math.hypot is CPython's own)
    rng = np.random.default_rng(3)
    x, y = rng.normal(0, 4, (40, 50)), rng.normal(0, 4, (40, 50))
    pos = np.abs(x) + 1e-3
    for name, args in (("cos", (x,)), ("sin", (x,)), ("sinh", (x,)), ("cosh", (x,)), ("log", (pos,)),
                       ("atan2", (y, x)), ("hypot", (x, y))):
        c = getattr(libm, name)
        c.restype, c.argtypes = ctypes.c_double, [ctypes.c_double] * len(args)
        out = getattr(_hep, name)(*args)
        assert out.shape == x.shape
        np.testing.assert_array_equal(out, np.vectorize(c)(*args), err_msg=name)
    assert float(_hep.cos(np.float64(0.5))) == math.cos(0.5)


def _vectors(n, seed):
    rng = np.random.default_rng(seed)
    pt = rng.uniform(0, 200, n).astype(np.float32)
    pt[:5] = 0  # zero pt, a vector along the beam
    eta = rng.uniform(-4.7, 4.7, n).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    mass = rng.uniform(0, 10, n).astype(np.float32)
    mass[5:10] = 0
    return pt, eta, phi, mass


def test_p4_add_mass_angles_match_numpy():
    a, b, c = _vectors(5000, 1), _vectors(5000, 2), _vectors(5000, 3)
    va, vb, vc = vector.p4(*a), vector.p4(*b), vector.p4(*c)
    ra, rb, rc = np_p4(*a), np_p4(*b), np_p4(*c)
    for k in ("px", "py", "pz", "e"):
        assert_same(va[k], ra[k])
    for vs, rs in (((va, vb), (ra, rb)), ((va, vb, vc), (ra, rb, rc))):
        s, r = vector.add(*vs), np_add(*rs)
        assert_same(s.mass, np_mass(r))
        for k in ("px", "py", "pz", "e", "pt", "eta", "phi"):
            assert_same(getattr(s, k), r[k])
    assert_same((va + vb).mass, np_mass(np_add(ra, rb)))
    assert_same(sum([va, vb, vc]).pt, np_add(ra, rb, rc)["pt"])
    assert va.mass is va["mass"]  # built from (pt, eta, phi, m): the input mass


def test_space_like_mass_is_negative():
    v = vector.P4.from_cartesian(np.array([3.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    np.testing.assert_array_equal(v.mass, [-np.sqrt(8.0), 0.0])


def test_collins_soper_matches_numpy():
    a, b = _vectors(5000, 4), _vectors(5000, 5)
    charge = np.random.default_rng(6).choice([-1, 1], 5000).astype(np.int32)
    phi, cos = vector.collins_soper(vector.p4(*a), vector.p4(*b), charge)
    rphi, rcos = np_collins_soper(np_p4(*a), np_p4(*b), charge)
    assert_same(phi, rphi)
    assert_same(cos, rcos)
    assert np.all(np.abs(cos[np.isfinite(cos)]) <= 1 + 1e-12)


def test_jagged_vectors():
    pt = ak.Array([[30.0, 20.0], [], [50.0]])
    eta, phi, m = ak.Array([[0.1, -1.0], [], [2.0]]), ak.Array([[0.5, 2.0], [], [-1.0]]), ak.Array([[1.0, 2.0], [], [3]])
    v = vector.p4(pt, eta, phi, m)
    assert ak.to_list(ak.num(v.px)) == [2, 0, 1]
    flat = np_p4(ak.flatten(pt), ak.flatten(eta), ak.flatten(phi), ak.flatten(m))
    assert_same(ak.flatten(v.e), flat["e"])
    s = v + v
    assert_same(ak.flatten(s.mass), np_mass(np_add(flat, flat)))
    with pytest.raises(ValueError):
        vector.add(v, vector.p4(np.ones(3), np.ones(3), np.ones(3), np.ones(3)))


def test_delta_r():
    a = vector.p4(np.array([10.0]), np.array([0.5]), np.array([3.0]), np.array([0.0]))
    b = vector.p4(np.array([10.0]), np.array([-0.5]), np.array([-3.0]), np.array([0.0]))
    np.testing.assert_allclose(vector.delta_r(a, b), np.hypot(1.0, 2 * np.pi - 6.0))


# ---------------------------------------------------------------- lazy, on frame columns
@pytest.fixture(scope="module")
def dimuon_file(tmp_path_factory):
    import uproot
    rng = np.random.default_rng(7)
    n = 3000
    data = {}
    for i in (1, 2):
        data[f"mu{i}_pt"] = (rng.exponential(30, n) + 20).astype(np.float32)
        data[f"mu{i}_eta"] = rng.uniform(-2.4, 2.4, n).astype(np.float32)
        data[f"mu{i}_phi"] = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
        data[f"mu{i}_mass"] = np.full(n, 0.10566, np.float32)
    data["mu1_charge"] = rng.choice([-1, 1], n).astype(np.int32)
    njet = rng.poisson(2, n)
    jets = {v: ak.unflatten(rng.uniform(lo, hi, int(njet.sum())).astype(np.float32), njet)
            for v, lo, hi in (("pt", 20, 200), ("eta", -4.7, 4.7), ("phi", -3.14, 3.14), ("mass", 1, 20))}
    path = str(tmp_path_factory.mktemp("vec") / "dimuon.root")
    with uproot.recreate(path) as f:
        f.mktree("Events", {**{k: v.dtype for k, v in data.items()}, "Jet": ak.zip(jets).type.content})
        f["Events"].extend({**data, "Jet": ak.zip(jets)})
    return path


def _reference(path):
    import uproot
    return uproot.open(path)["Events"].arrays(library="np")


def test_lazy_p4_columns_selections_and_variations(dimuon_file):
    df = aks.read_root(dimuon_file, step=700)
    df.vary("mu1_pt", up=df.mu1_pt * np.float32(1.01), down=df.mu1_pt * np.float32(0.99), name="scale")
    df["mu1"] = aks.p4(df.mu1_pt, df.mu1_eta, df.mu1_phi, df.mu1_mass)
    df["mu2"] = aks.p4(df.mu2_pt, df.mu2_eta, df.mu2_phi, df.mu2_mass)
    df["mll"] = (df.mu1 + df.mu2).mass
    sel = df[df.mll > 60]
    sel["cos_cs"] = aks.collins_soper(sel.mu1, sel.mu2, sel.mu1_charge)[1]
    h = sel.hist("cos_cs", bins=20, range=(-1, 1))
    got, = aks.compute(aks.varied(h))

    a = _reference(dimuon_file)
    for name, scale in (("nominal", None), ("scale:up", 1.01), ("scale:down", 0.99)):
        pt1 = a["mu1_pt"] if scale is None else a["mu1_pt"] * np.float32(scale)
        mu1 = np_p4(pt1, a["mu1_eta"], a["mu1_phi"], a["mu1_mass"])
        mu2 = np_p4(a["mu2_pt"], a["mu2_eta"], a["mu2_phi"], a["mu2_mass"])
        keep = np_mass(np_add(mu1, mu2)) > 60
        cut = [{k: v[keep] for k, v in m.items()} for m in (mu1, mu2)]
        cos = np_collins_soper(cut[0], cut[1], a["mu1_charge"][keep])[1]
        np.testing.assert_array_equal(got[name].values(), np.histogram(cos, bins=20, range=(-1, 1))[0], err_msg=name)


def test_collection_p4(dimuon_file):
    df = aks.read_root(dimuon_file)
    jets = df.Jet[df.Jet.pt > 30]
    df["sum_pt"] = aks.sum(jets.p4.pt, axis=1)
    out = df.collect(["sum_pt"]).value
    j = __import__("uproot").open(dimuon_file)["Events"].arrays(["Jet_pt"], library="ak")["Jet_pt"]
    ref = ak.sum(ak.values_astype(j[j > 30], np.float64), axis=1)
    np.testing.assert_array_equal(np.asarray(out["sum_pt"]), np.asarray(ref))


@pytest.mark.parametrize("rdf", [False, True])
def test_p4_extra_fields(rdf):
    pt, eta, phi, m = (np.array(x) for x in ([30.0, 40.0, 50.0], [0.1, -1.0, 2.0], [0.5, 2.0, -1.0], [0.1, 0.1, 0.1]))
    pdg = np.array([13, -13, 11], np.int32)
    v, plain = vector.p4(pt, eta, phi, m, rdf, pdgId=pdg, w=1.5), vector.p4(pt, eta, phi, m, rdf)
    assert v.pdgId.dtype == np.int32 and v.pdgId.tolist() == [13, -13, 11] and v.w.tolist() == [1.5] * 3
    for k in ("pt", "eta", "phi", "mass", "px", "e"):
        np.testing.assert_array_equal(getattr(v, k), getattr(plain, k))
    s = v + v  # a sum has no extra fields
    np.testing.assert_array_equal(s.mass, (plain + plain).mass)
    with pytest.raises(AttributeError, match="pdgId"):
        s.pdgId
    # jagged: a jagged field, a per-row one repeated per object, a scalar
    jpt, jeta = ak.Array([[30.0, 20.0], [], [50.0]]), ak.Array([[0.1, -1.0], [], [2.0]])
    jv = vector.p4(jpt, jeta, jeta, jpt * 0, rdf, charge=ak.Array([[1, -1], [], [1]]), ev=np.array([7, 8, 9]), flav=13)
    assert ak.to_list(jv.charge) == [[1, -1], [], [1]] and ak.to_list(jv.ev) == [[7, 7], [], [9]]
    assert ak.to_list(jv.flav) == [[13, 13], [], [13]]
    for bad in ("px", "e", "energy", "rapidity", "values", "_x"):
        with pytest.raises(ValueError, match="4-vector attribute"):
            vector.p4(pt, eta, phi, m, rdf, **{bad: pdg})
    with pytest.raises(ValueError, match="rows"):
        vector.p4(pt, eta, phi, m, rdf, pdgId=pdg[:2])
    with pytest.raises(ValueError, match="layout"):
        vector.p4(jpt, jeta, jeta, jpt * 0, rdf, charge=ak.Array([[1], [], [1]]))


def test_lazy_p4_extra_fields(dimuon_file):
    """the user's dimuon selection with mu1.pdgId * mu2.pdgId < 0, under a variation, against numpy."""
    df = aks.read_root(dimuon_file, step=700)
    df.vary("mu1_pt", up=df.mu1_pt * np.float32(1.01), down=df.mu1_pt * np.float32(0.99), name="scale")
    df["mu1"] = aks.p4(df.mu1_pt, df.mu1_eta, df.mu1_phi, df.mu1_mass, pdgId=-13 * df.mu1_charge)
    df["mu2"] = aks.p4(df.mu2_pt, df.mu2_eta, df.mu2_phi, df.mu2_mass, pdgId=13)
    df["mll"] = (df.mu1 + df.mu2).mass
    sel = df[(df.mll > 60) & (df.mu1.pt > 26) & (abs(df.mu2.eta) < 2.0) & (df.mu1.pdgId * df.mu2.pdgId < 0)]
    n, out = aks.varied(sel.count()), sel.collect(["mll"])
    got_n, got = aks.compute(n, out)

    a = _reference(dimuon_file)
    for name, scale in (("nominal", None), ("scale:up", 1.01), ("scale:down", 0.99)):
        pt1 = a["mu1_pt"] if scale is None else a["mu1_pt"] * np.float32(scale)
        mll = np_mass(np_add(np_p4(pt1, a["mu1_eta"], a["mu1_phi"], a["mu1_mass"]),
                             np_p4(a["mu2_pt"], a["mu2_eta"], a["mu2_phi"], a["mu2_mass"])))
        keep = (mll > 60) & (pt1 > 26) & (np.abs(a["mu2_eta"]) < 2.0) & (a["mu1_charge"] > 0)
        assert got_n[name] == keep.sum(), name
        if name == "nominal":
            np.testing.assert_array_equal(np.asarray(got["mll"]), mll[keep])


# ---------------------------------------------------------------- ROOT GenVector conventions
def test_root_p4_matches_genvector():
    ROOT = pytest.importorskip("ROOT")
    rng = np.random.default_rng(11)
    n = 2000
    a = [rng.uniform(1, 300, n).astype(np.float32), rng.uniform(-5, 5, n).astype(np.float32),
         rng.uniform(-np.pi, np.pi, n).astype(np.float32), rng.uniform(-0.01, 20, n).astype(np.float32)]
    b = [rng.uniform(1, 300, n).astype(np.float32), rng.uniform(-5, 5, n).astype(np.float32),
         rng.uniform(-np.pi, np.pi, n).astype(np.float32), rng.uniform(0, 20, n).astype(np.float32)]
    s = vector.p4(*a, rdf_compatible=True) + vector.p4(*b, rdf_compatible=True)
    ref = {k: np.empty(n) for k in ("mass", "pt", "eta", "phi", "rapidity")}
    for i in range(n):
        v = (ROOT.Math.PtEtaPhiMVector(*(float(x[i]) for x in a)) + ROOT.Math.PtEtaPhiMVector(*(float(x[i]) for x in b)))
        ref["mass"][i], ref["pt"][i], ref["eta"][i], ref["phi"][i], ref["rapidity"][i] = \
            v.M(), v.Pt(), v.Eta(), v.Phi(), v.Rapidity()
    # ROOT 6.36's GenVector (LCG_108) is built / JIT-compiled with fused multiply-adds: ~1e-16 relative differences
    # (invisible once stored as float, as RDataFrame outputs mostly are); from 6.40 identical
    exact = tuple(int(x) for x in ROOT.gROOT.GetVersion().split("/")[0].split(".")[:2]) >= (6, 40)
    for k, r in ref.items():
        if exact:
            assert_same(getattr(s, k), r)
        else:
            np.testing.assert_allclose(getattr(s, k), r, rtol=1e-11, atol=1e-12)
            if k != "rapidity":
                np.testing.assert_array_equal(np.asarray(getattr(s, k), np.float32), r.astype(np.float32))


def test_max_mass_pair_genvector_matches_root():
    ROOT = pytest.importorskip("ROOT")
    from akstream import hep
    rng = np.random.default_rng(12)
    counts = rng.integers(0, 7, 3000)
    cols = [ak.unflatten(rng.uniform(lo, hi, counts.sum()).astype(np.float32), counts)
            for lo, hi in ((20, 300), (-4.7, 4.7), (-3.14, 3.14), (2, 30))]
    found, i1, i2, mjj, deta = hep.max_mass_pair(*cols, rdf_compatible=True)
    for ev in range(len(counts)):
        jets = [ROOT.Math.PtEtaPhiMVector(*(float(c[ev][j]) for c in cols)) for j in range(counts[ev])]
        best = (-1, -1, 0.0, 0.0)
        top = np.float32(-1)
        for i in range(len(jets)):
            for j in range(i + 1, len(jets)):
                m = np.float32((jets[i] + jets[j]).M())
                d = np.float32(abs(jets[i].Eta() - jets[j].Eta()))
                if m > 400 and d > 2.5 and max(jets[i].Pt(), jets[j].Pt()) > 35 and m > top:
                    top, best = m, (i, j, m, d)
        assert (i1[ev], i2[ev], mjj[ev], deta[ev]) == best


def test_rdf_compatible_double_literals(tmp_path):
    import uproot
    x = np.array([0.0199999995529651, 0.02, 0.1, 0.3], np.float32)  # float32(0.02) < 0.02
    path = str(tmp_path / "f.root")
    with uproot.recreate(path) as f:
        f.mktree("Events", {"x": np.float32})
        f["Events"].extend({"x": x})
    for double, expected in ((True, [False, False, True, True]), (False, [False, False, True, True])):
        df = aks.read_root(path, rdf_compatible=double)
        out = df.define(gt="x > 0.02", half="0.5 * x").collect(["gt", "half"]).value
        assert np.asarray(out["half"]).dtype == (np.float64 if double else np.float32)
    df = aks.read_root(path, rdf_compatible=True)
    out = df.define(le="x <= 0.02").collect(["le"]).value
    np.testing.assert_array_equal(np.asarray(out["le"]), x.astype(np.float64) <= 0.02)
