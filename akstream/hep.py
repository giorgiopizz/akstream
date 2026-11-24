"""Fast kernels for the jagged-collection operations most HEP analyses need.

awkward expresses these with intermediate jagged arrays (combinations, padding, masks) and pays a fixed cost per
operation; here each one is a single pass over the flat contents and offsets (numpy where a gather is enough, the C++
extension akstream._hep where an event loop is needed). All are row-local, so they work inside aks.apply and under
variations.

Every function also takes lazy frame columns (and max_mass_pair a collection): the result is then lazy too (a tuple of
lazy columns where the function returns a tuple).

Floating-point results follow the plain numpy formulas (transcendental functions are evaluated with numpy; the C++
loops only add, multiply, compare and take square roots, which are exactly rounded, built without fused multiply-add),
so they reproduce the equivalent awkward / numpy code bit for bit.

    offsets, flat = hep.flat(jagged)          # ListOffsetArray -> (int64 offsets, numpy content)
    hep.nth(jagged, k, fill)                  # k-th object of each event, `fill` where there is none
    hep.pick(jagged, index, fill)             # the object at a per-event index
    hep.count(mask)                           # number of True per event (jagged boolean)
    hep.per_event_sum(values, mask=None)      # sum per event
    hep.broadcast(per_event, jagged)          # one value per object
    hep.isolated(eta, phi, refs, min_dr2)     # objects away from per-event reference directions
    hep.between(x, lo, hi)                    # lo < x < hi per object, with per-event bounds
    hep.select_sort(keep, key, *fields)       # objects passing `keep`, sorted by `key` (descending, stable)
    hep.max_mass_pair(pt, eta, phi, mass, ...)  # the max-invariant-mass pair passing cuts (VBF tagging); or (jets, ...)
    hep.delta_phi(phi1, phi2)                 # wrapped to (-pi, pi]
    hep.match_dr(a_eta, a_phi, b_eta, b_phi, max_dr)  # nearest object of B within Delta R (or one-to-one)
    hep.clean_dr(eta, phi, ref_eta, ref_phi, min_dr)  # objects away from every object of another collection
    hep.ancestor(pdg_id, mother_idx, targets) # the nearest ancestor with one of the pdgIds (gen trees)
    hep.best_pair(leptons, target=91.1876)    # the (opposite-charge, same-flavour) pair closest to a mass
    hep.met_shift(met_pt, met_phi, pt_new, pt_old, phi)  # object pT changes propagated to MET
    hep.LumiMask(golden_json)(run, lumi)      # certified luminosity sections
"""
import awkward as ak
import numpy as np

try:
    from . import _hep
except ImportError as err:  # pragma: no cover
    raise ImportError("akstream.hep needs the compiled extension akstream._hep: install the package with "
                      "`pip install -e .` (or `python setup.py build_ext --inplace`)") from err

_SUM_TYPES = (np.float32, np.float64, np.int64)


def _is_lazy(args, kwargs):
    from .expr import Collection, Expr
    from .graph import _nodes_in
    from .expr import _unwrap
    return any(isinstance(a, (Expr, Collection)) for a in list(args) + list(kwargs.values())) and \
        any(True for _ in _nodes_in(_unwrap((args, kwargs))))


def lazy(outputs=None):
    """the function, lazy when given frame columns: one lazy column, or a tuple of them (`outputs` names the elements of
    a tuple result)."""
    def deco(fn):
        import functools

        def as_dict(*args, **kwargs):
            return dict(zip(outputs, fn(*args, **kwargs)))
        as_dict.__name__ = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _is_lazy(args, kwargs):
                return fn(*args, **kwargs)
            from .expr import call
            if outputs is None:
                return call(fn, *args, _name=fn.__name__, **kwargs)
            record = call(as_dict, *args, _name=fn.__name__, **kwargs)
            return tuple(record[k] for k in outputs)
        wrapper.eager = fn
        return wrapper
    return deco


# ---------------------------------------------------------------- jagged <-> flat
def flat(x):
    """(offsets, flat values) of a one-level jagged array of numbers; the offsets start at 0."""
    layout = x.layout if isinstance(x, ak.Array) else ak.to_layout(x)
    while isinstance(layout, ak.contents.IndexedArray):
        layout = layout.project()
    if isinstance(layout, ak.contents.ListOffsetArray):
        offsets = np.asarray(layout.offsets, dtype=np.int64)
        content = layout.content
        if isinstance(content, ak.contents.NumpyArray):
            data = np.asarray(content.data)
            return offsets - offsets[0], data[offsets[0]:offsets[-1]]
    packed = ak.to_packed(ak.Array(layout))
    return flat(packed)


def jagged(offsets, values):
    """a jagged array from offsets (int64, starting at 0) and flat numpy values."""
    return ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(offsets), ak.contents.NumpyArray(values)))


def counts_of(offsets):
    return np.diff(offsets)


def event_of(offsets):
    """the event index of every object."""
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def _filled_dtype(dtype, fill):
    """the dtype of ak.fill_none(x, fill) for numbers x: numpy promotion with the fill as a 64-bit value (float32 filled
    with 0.0 becomes float64, as in awkward)."""
    return np.result_type(dtype, np.asarray(fill).dtype)


@lazy()
def nth(x, k, fill):
    """the k-th object (0 = first) of every event, `fill` for events with fewer objects; the dtype is that of
    ak.fill_none(ak.pad_none(x, k + 1)[:, k], fill)."""
    offsets, values = flat(x)
    n = np.diff(offsets)
    out = np.full(len(n), fill, dtype=_filled_dtype(values.dtype, fill))
    has = n > k
    out[has] = values[offsets[:-1][has] + k]
    return out


@lazy()
def pick(x, index, fill):
    """x[i, index[i]] for every event (local index), `fill` where index < 0 or out of range; dtype as ak.fill_none."""
    offsets, values = flat(x)
    index = np.asarray(index, dtype=np.int64)
    n = np.diff(offsets)
    out = np.full(len(n), fill, dtype=_filled_dtype(values.dtype, fill))
    ok = (index >= 0) & (index < n)
    out[ok] = values[offsets[:-1][ok] + index[ok]]
    return out


@lazy()
def count(mask):
    """number of True objects per event."""
    offsets, values = flat(mask)
    return per_event_sum_flat(offsets, values.astype(np.int64))


@lazy()
def per_event_sum(values, mask=None):
    """sum over the objects of each event, in object order: ak.sum(values, axis=1), or with a mask
    ak.sum(ak.where(mask, values, 0), axis=1) -- whose values are promoted as awkward does (float32 -> float64)."""
    offsets, v = flat(values)
    if mask is not None:
        _, m = flat(mask)
        dtype = np.result_type(v.dtype, np.int64)
        v = np.where(m, v.astype(dtype), dtype.type(0))
    return per_event_sum_flat(offsets, v)


def per_event_sum_flat(offsets, values):
    """sums in object order, accumulated in the values' type (as ak.sum); bool / small ints are summed as int64."""
    values = np.asarray(values)
    if values.dtype.kind in "biu" and values.dtype != np.int64:
        values = values.astype(np.int64)
    if values.dtype.type not in _SUM_TYPES:
        raise TypeError(f"per-event sums of {values.dtype} are not supported")
    return _hep.segment_sum(np.ascontiguousarray(offsets, dtype=np.int64), np.ascontiguousarray(values))


@lazy()
def broadcast(per_event, x):
    """one value per object of `x` from one value per event."""
    offsets, _ = flat(x)
    return np.repeat(np.asarray(per_event), np.diff(offsets))


@lazy()
def delta_phi(phi1, phi2):
    """phi2 - phi1 wrapped to (-pi, pi] (ROOT::Math::VectorUtil::DeltaPhi)."""
    d = phi2 - phi1
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    return np.where(d <= -np.pi, d + 2 * np.pi, d)


@lazy()
def isolated(eta, phi, references, min_dr2):
    """flat boolean per object: Delta R^2 >= min_dr2 to every reference direction. references: [(eta, phi)] with one
    value per event each; Delta R^2 = (eta - eta_ref)^2 + delta_phi(phi_ref, phi)^2, as in the equivalent awkward code."""
    offsets, e = flat(eta)
    _, p = flat(phi)
    n = np.diff(offsets)
    keep = np.ones(len(e), dtype=bool)
    for ref_eta, ref_phi in references:
        re, rp = np.repeat(np.asarray(ref_eta), n), np.repeat(np.asarray(ref_phi), n)
        keep &= (e - re) ** 2 + delta_phi(rp, p) ** 2 >= min_dr2
    return keep


@lazy()
def between(x, lo, hi):
    """flat boolean per object: lo < x < hi with one (lo, hi) per event."""
    offsets, v = flat(x)
    n = np.diff(offsets)
    return (v > np.repeat(np.asarray(lo), n)) & (v < np.repeat(np.asarray(hi), n))


# ---------------------------------------------------------------- selection and sorting
def select_sort(keep, key, *fields, descending=True):
    """the objects with keep True, ordered by key (stable; descending by default), and the same selection / order of
    every field. Returns (key, *fields) as jagged arrays."""
    offsets, k = flat(key)
    _, m = flat(keep)
    flats = [flat(f)[1] for f in fields]
    sort_key = k if k.dtype.type in _SUM_TYPES else k.astype(np.float64 if k.dtype.kind == "f" else np.int64)
    new_offsets, order = _hep.select_sort(offsets, np.ascontiguousarray(sort_key),
                                          np.ascontiguousarray(m, dtype=np.bool_), bool(descending))
    return tuple(jagged(new_offsets, v[order]) for v in [k] + flats)


# ---------------------------------------------------------------- pairs
def max_mass_pair(pt, eta=None, phi=None, mass=None, min_mass=400.0, min_deta=2.5, min_lead_pt=35.0, leading_only=False,
                  rdf_compatible=None):
    """see _max_mass_pair; the first argument may be a collection (its pt, eta, phi, mass fields). Lazy columns: lazy
    results, with rdf_compatible from read_root by default."""
    if eta is None:
        pt, eta, phi, mass = (pt[f] if isinstance(pt, dict) else getattr(pt, f) for f in ("pt", "eta", "phi", "mass"))
    if rdf_compatible is None:
        rdf_compatible = False
        if _is_lazy((pt, eta, phi, mass), {}):
            from .expr import _space_of, _unwrap
            rdf_compatible = _space_of(_unwrap((pt, eta, phi, mass)), {}).dataset.rdf_compatible
    return _max_mass_pair_lazy(pt, eta, phi, mass, min_mass=min_mass, min_deta=min_deta, min_lead_pt=min_lead_pt,
                               leading_only=leading_only, rdf_compatible=rdf_compatible)


@lazy(outputs=("found", "i1", "i2", "mjj", "detajj"))
def _max_mass_pair_lazy(pt, eta, phi, mass, min_mass, min_deta, min_lead_pt, leading_only, rdf_compatible):
    return _max_mass_pair(pt, eta, phi, mass, min_mass, min_deta, min_lead_pt, leading_only, rdf_compatible)


def _max_mass_pair(pt, eta, phi, mass, min_mass=400.0, min_deta=2.5, min_lead_pt=35.0, leading_only=False,
                   rdf_compatible=False):
    """per event, the pair (i < j, in object order) with the largest invariant mass among the pairs passing
    mass > min_mass, |delta eta| > min_deta and max(pt) > min_lead_pt (the first such pair on ties).

    The mass is computed as the plain numpy formulation: 4-vectors from (pt, eta, phi, mass) in double, summed,
    m = sqrt(|E^2 - p^2|) with the sign of E^2 - p^2, rounded to float32; delta eta in double rounded to float32.
    rdf_compatible=True: as the C++ GetVBFJetIndices on ROOT::Math::PtEtaPhiMVector (E = sqrt((pt cosh eta)^2 + m^2),
    m^2 = E^2 - px^2 - py^2 - pz^2 in that order), bit for bit.

    Returns (found, i1, i2, mjj (float32, 0 if not found), detajj (float32, 0 if not found)); i1 / i2 are local indices
    (-1 if not found)."""
    offsets, pt_ = flat(pt)
    _, eta_ = flat(eta)
    _, phi_ = flat(phi)
    _, m_ = flat(mass)
    pt64, eta64, phi64, m64 = (np.asarray(v, dtype=np.float64) for v in (pt_, eta_, phi_, m_))
    px, py, pz = pt64 * np.cos(phi64), pt64 * np.sin(phi64), pt64 * np.sinh(eta64)
    if rdf_compatible:
        p = pt64 * np.cosh(eta64)
        e2 = p * p + m64 * m64
        e = np.sqrt(np.where(e2 > 0, e2, 0.0))
    else:
        e = np.sqrt(px * px + py * py + pz * pz + m64 * m64)
    return _hep.max_mass_pair(offsets, px, py, pz, e, eta64, pt64, float(np.float32(min_mass)),
                              float(np.float32(min_deta)), float(min_lead_pt), bool(leading_only), bool(rdf_compatible))


# ---------------------------------------------------------------- matching and cleaning
def _flat64(x):
    offsets, v = flat(x)
    return offsets, np.ascontiguousarray(v, dtype=np.float64)


@lazy(outputs=("index", "dr"))
def match_dr(a_eta, a_phi, b_eta, b_phi, max_dr, b_mask=None, unique=False):
    """for every object of A, the nearest object of B in the same event within Delta R < max_dr: (local index into B,
    -1 if none; Delta R, -1 if none), jagged like A. Delta R^2 = (eta_a - eta_b)^2 + delta_phi^2 in double; the first B
    object wins on ties. b_mask: only these B objects are candidates. unique=True: one-to-one, the pairs taken in
    increasing Delta R (ties: A order, then B order), each B object matched at most once (greedy matching)."""
    oa, ea = _flat64(a_eta)
    _, pa = _flat64(a_phi)
    ob, eb = _flat64(b_eta)
    _, pb = _flat64(b_phi)
    if len(oa) != len(ob):
        raise ValueError("match_dr: A and B have different numbers of events")
    mb = np.ones(len(eb), bool) if b_mask is None else np.ascontiguousarray(flat(b_mask)[1], dtype=bool)
    index, dr2 = _hep.match_dr(oa, ea, pa, ob, eb, pb, mb, float(max_dr) * float(max_dr), bool(unique))
    dr = np.where(index >= 0, np.sqrt(np.where(dr2 >= 0, dr2, 0.0)), -1.0)
    return jagged(oa, index), jagged(oa, dr)


@lazy()
def clean_dr(eta, phi, ref_eta, ref_phi, min_dr, ref_mask=None):
    """jagged boolean per object: no object of the reference collection (where ref_mask) within Delta R < min_dr
    (lepton-jet cleaning)."""
    index, _ = match_dr.eager(eta, phi, ref_eta, ref_phi, min_dr, ref_mask)
    offsets, i = flat(index)
    return jagged(offsets, i < 0)


# ---------------------------------------------------------------- generator trees
@lazy()
def ancestor(pdg_id, mother_idx, targets, abs_id=True):
    """for every particle (NanoAOD GenPart), the local index of its nearest ancestor whose |pdgId| (pdgId with
    abs_id=False) is in `targets` (an int or a list), following the mother indices; -1 if there is none. A particle is
    not its own ancestor. E.g. muons from a Z: hep.ancestor(gen.pdgId, gen.genPartIdxMother, 23) >= 0."""
    offsets, p = flat(pdg_id)
    _, m = flat(mother_idx)
    t = np.atleast_1d(np.asarray(targets, dtype=np.int64))
    out = _hep.ancestor(offsets, np.ascontiguousarray(p, dtype=np.int64), np.ascontiguousarray(m, dtype=np.int64),
                        np.ascontiguousarray(t), bool(abs_id))
    return jagged(offsets, out)


# ---------------------------------------------------------------- pairs closest to a mass
def best_pair(pt, eta=None, phi=None, mass=None, charge=None, target=91.1876, flavour=None, opposite_charge=True,
              rdf_compatible=None):
    """per event, the pair (i < j) whose invariant mass is closest to `target` (the first on ties), among the pairs
    with opposite charges (opposite_charge) and, with `flavour` (e.g. |pdgId|), equal flavour: (found, i1, i2, mass;
    -1 / 0 where there is no pair). The first argument may be a collection (pt, eta, phi, mass, charge fields). The
    mass as in max_mass_pair (in double, ROOT's PtEtaPhiMVector sum with rdf_compatible)."""
    if eta is None:
        coll = pt
        get = (lambda f: coll[f]) if isinstance(coll, dict) else (lambda f: getattr(coll, f))
        pt, eta, phi, mass = (get(f) for f in ("pt", "eta", "phi", "mass"))
        if charge is None:
            charge = get("charge")
    if charge is None:
        raise ValueError("best_pair: give the charges (charge=...), or opposite_charge=False with charge=0")
    if rdf_compatible is None:
        rdf_compatible = False
        if _is_lazy((pt, eta, phi, mass), {}):
            from .expr import _space_of, _unwrap
            rdf_compatible = _space_of(_unwrap((pt, eta, phi, mass)), {}).dataset.rdf_compatible
    return _best_pair(pt, eta, phi, mass, charge, flavour, target=float(target), opposite_charge=opposite_charge,
                      rdf_compatible=rdf_compatible)


@lazy(outputs=("found", "i1", "i2", "mass"))
def _best_pair(pt, eta, phi, mass, charge, flavour, target, opposite_charge, rdf_compatible):
    offsets, _ = flat(pt)
    pt64, eta64, phi64, m64 = (np.asarray(flat(v)[1], dtype=np.float64) for v in (pt, eta, phi, mass))
    px, py, pz = pt64 * np.cos(phi64), pt64 * np.sin(phi64), pt64 * np.sinh(eta64)
    if rdf_compatible:
        p = pt64 * np.cosh(eta64)
        e2 = p * p + m64 * m64
        e = np.sqrt(np.where(e2 > 0, e2, 0.0))
    else:
        e = np.sqrt(px * px + py * py + pz * pz + m64 * m64)
    q = _per_object_int(charge, offsets)
    fl = np.zeros(len(pt64), np.int64) if flavour is None else _per_object_int(flavour, offsets)
    return _hep.best_pair(offsets, px, py, pz, e, q, fl, target, bool(opposite_charge), flavour is not None,
                          bool(rdf_compatible))


def _per_object_int(x, offsets):
    if np.isscalar(x):
        return np.full(offsets[-1], int(x), np.int64)
    return np.ascontiguousarray(flat(x)[1], dtype=np.int64)


# ---------------------------------------------------------------- MET and luminosity
@lazy(outputs=("pt", "phi"))
def met_shift(met_pt, met_phi, pt_new, pt_old, phi, mask=None):
    """MET after objects changed pT (Type-1-like propagation): MET_xy - sum over objects (where mask) of
    (pt_new - pt_old) * (cos phi, sin phi), in double. Returns (pt, phi) per event."""
    offsets, new = flat(pt_new)
    _, old = flat(pt_old)
    _, ph = flat(phi)
    d = np.asarray(new, np.float64) - np.asarray(old, np.float64)
    if mask is not None:
        d = np.where(np.asarray(flat(mask)[1], dtype=bool), d, 0.0)
    ph = np.asarray(ph, np.float64)
    dx = per_event_sum_flat(offsets, d * np.cos(ph))
    dy = per_event_sum_flat(offsets, d * np.sin(ph))
    mpt, mphi = np.asarray(met_pt, np.float64), np.asarray(met_phi, np.float64)
    x = mpt * np.cos(mphi) - dx
    y = mpt * np.sin(mphi) - dy
    return np.hypot(x, y), np.arctan2(y, x)


class LumiMask:
    """certified luminosity sections from a golden JSON ({"run": [[first, last], ...]}, a path or a dict):
    mask = LumiMask(path)(run, luminosityBlock), on arrays or lazy columns."""

    def __init__(self, golden):
        import json
        if not isinstance(golden, dict):
            with open(golden) as f:
                golden = json.load(f)
        starts, stops = [], []
        for run, ranges in golden.items():
            for lo, hi in ranges:
                starts.append((int(run) << 32) | int(lo))
                stops.append((int(run) << 32) | int(hi))
        order = np.argsort(starts, kind="stable")
        self.starts = np.asarray(starts, np.int64)[order]
        self.stops = np.asarray(stops, np.int64)[order]
        self.__name__ = "lumi_mask"

    def mask(self, run, lumi):
        key = (np.asarray(run, np.int64) << 32) | np.asarray(lumi, np.int64)
        i = np.searchsorted(self.starts, key, side="right") - 1
        ok = i >= 0
        return ok & (key <= self.stops[np.where(ok, i, 0)])

    def __call__(self, run, lumi):
        if _is_lazy((run, lumi), {}):
            from .expr import call
            return call(self.mask, run, lumi, _name="lumi_mask")
        return self.mask(run, lumi)
