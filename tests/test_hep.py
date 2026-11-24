"""akstream.hep kernels against the equivalent awkward code: same values and dtypes, bit for bit."""
import awkward as ak
import numpy as np
import pytest

from akstream import hep


def jagged_random(n, seed, dtype=np.float32, mean=3.0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(mean, n)
    values = rng.exponential(40, counts.sum()).astype(dtype)
    return ak.unflatten(values, counts)


def test_flat_and_jagged_roundtrip():
    x = jagged_random(200, 1)
    offsets, values = hep.flat(x[50:120])  # a sliced (non-zero offset) array
    assert offsets[0] == 0
    assert ak.all(hep.jagged(offsets, values) == x[50:120])
    sel = x[np.array([5, 3, 199, 0])]  # a ListArray (carried)
    o, v = hep.flat(sel)
    assert ak.all(hep.jagged(o, v) == sel)


@pytest.mark.parametrize("fill", [0.0, -1, -999.0])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_nth_matches_pad_fill(k, fill):
    x = jagged_random(500, 2)
    ref = ak.to_numpy(ak.fill_none(ak.pad_none(x, k + 1)[:, k], fill))
    got = hep.nth(x, k, fill)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)


def test_pick_matches_mask_firsts():
    x = jagged_random(500, 3)
    idx = np.random.default_rng(4).integers(-1, 5, 500)
    local = ak.local_index(x)
    ref = ak.to_numpy(ak.fill_none(ak.firsts(x[local == idx]), 0))
    got = hep.pick(x, idx, 0)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)


def test_select_sort_matches_mask_argsort():
    x = jagged_random(1000, 5)
    values = ak.to_numpy(ak.flatten(x)).copy()
    values[np.random.default_rng(6).random(len(values)) < 0.1] = 30.0  # ties: the stable order must be kept
    x = ak.unflatten(values, ak.num(x))
    keep = x > 20
    y = x * 2
    pt, other = hep.select_sort(keep, x, y)
    sel = x[keep]
    order = ak.argsort(sel, ascending=False, stable=True)
    assert ak.all(pt == sel[order]) and ak.all(other == y[keep][order])
    assert ak.flatten(pt).type == ak.flatten(sel[order]).type


def test_count_and_sums_match_awkward():
    x = jagged_random(800, 7)
    m = x > 30
    assert np.array_equal(hep.count(m), ak.to_numpy(ak.sum(m, axis=1)))
    ref = ak.to_numpy(ak.sum(ak.where(m, x, 0), axis=1))
    got = hep.per_event_sum(x, m)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)
    plain = ak.to_numpy(ak.sum(x, axis=1))  # float32 accumulation, as awkward
    assert np.array_equal(hep.per_event_sum(x), plain)


def test_isolated_and_between():
    rng = np.random.default_rng(8)
    eta, phi = jagged_random(300, 9), jagged_random(300, 9)
    eta, phi = eta / 20 - 2, phi / 20 - 3
    refs = [(rng.uniform(-2, 2, 300), rng.uniform(-3, 3, 300)) for _ in range(2)]
    keep = ak.ones_like(eta, dtype=bool)
    for re, rp in refs:
        d = phi - rp
        d = ak.where(d > np.pi, d - 2 * np.pi, d)
        d = ak.where(d <= -np.pi, d + 2 * np.pi, d)
        keep = keep & ((eta - re) ** 2 + d ** 2 >= 0.16)
    assert np.array_equal(hep.isolated(eta, phi, refs, 0.16), ak.to_numpy(ak.flatten(keep)))
    lo, hi = refs[0][0] - 1, refs[0][0] + 1
    assert np.array_equal(hep.between(eta, lo, hi), ak.to_numpy(ak.flatten((eta > lo) & (eta < hi))))


def _max_mass_pair_reference(pt, eta, phi, mass):
    """the plain awkward implementation."""
    i1, i2 = ak.unzip(ak.argcombinations(pt, 2))

    def p4(i):
        a = [np.asarray(ak.flatten(v[i]), dtype=np.float64) for v in (pt, eta, phi, mass)]
        px, py, pz = a[0] * np.cos(a[2]), a[0] * np.sin(a[2]), a[0] * np.sinh(a[1])
        return px, py, pz, np.sqrt(px * px + py * py + pz * pz + a[3] * a[3]), a[1], a[0]
    f, s = p4(i1), p4(i2)
    sx, sy, sz, se = (0 + f[0]) + s[0], (0 + f[1]) + s[1], (0 + f[2]) + s[2], (0 + f[3]) + s[3]
    m2 = se ** 2 - (sx ** 2 + sy ** 2 + sz ** 2)
    mjj = np.where(m2 >= 0, np.sqrt(np.abs(m2)), -np.sqrt(np.abs(m2))).astype(np.float32)
    detajj = np.abs(f[4] - s[4]).astype(np.float32)
    passes = (mjj > 400) & (detajj > 2.5) & (np.maximum(f[5], s[5]) > 35)
    n = ak.num(i1)
    cand = ak.unflatten(np.where(passes, mjj, np.float32(-1)), n)
    best = ak.argmax(cand, axis=1, keepdims=True)
    best_mjj = ak.to_numpy(ak.fill_none(ak.firsts(cand[best]), -1.0))
    found = best_mjj > 0
    return (found, np.where(found, ak.to_numpy(ak.fill_none(ak.firsts(i1[best]), -1)), -1),
            np.where(found, ak.to_numpy(ak.fill_none(ak.firsts(i2[best]), -1)), -1), best_mjj,
            np.where(found, ak.to_numpy(ak.fill_none(ak.firsts(ak.unflatten(detajj, n)[best]), 0.0)), 0))


def test_max_mass_pair_matches_awkward():
    rng = np.random.default_rng(10)
    pt = jagged_random(3000, 11, mean=3.5)
    n = int(ak.sum(ak.num(pt)))
    counts = ak.num(pt)
    eta = ak.unflatten(rng.uniform(-4.7, 4.7, n).astype(np.float32), counts)
    phi = ak.unflatten(rng.uniform(-np.pi, np.pi, n).astype(np.float32), counts)
    mass = ak.unflatten(rng.exponential(8, n).astype(np.float32), counts)
    found, i1, i2, mjj, detajj = hep.max_mass_pair(pt, eta, phi, mass)
    rf, r1, r2, rm, rd = _max_mass_pair_reference(pt, eta, phi, mass)
    assert found.sum() > 100
    assert np.array_equal(found, rf) and np.array_equal(i1, r1) and np.array_equal(i2, r2)
    assert np.array_equal(mjj[found], rm[found].astype(np.float32))
    assert np.array_equal(detajj[found].astype(np.float64), rd[found])
