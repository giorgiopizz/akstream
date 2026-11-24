"""Lorentz vectors: one 4-vector per row (an event) or per object (jagged), in double precision.

    mu1 = vector.p4(pt, eta, phi, mass)        # px = pt cos(phi), py = pt sin(phi), pz = pt sinh(eta), E from p and m
    ll = mu1 + mu2                             # cartesian sums
    ll.mass, ll.pt, ll.eta, ll.phi             # computed when asked for (an unused eta / phi costs nothing)
    phi_cs, cos_cs = vector.collins_soper(mu1, mu2, charge1)

The same code is lazy on frame columns: aks.p4(df.mu1_pt, ...) is a column whose value is a P4, so (mu1 + mu2).mass is
a column too, and df.Muon.p4 gives the 4-vectors of a collection.

The loops are C++ (akstream._hep) and follow the plain numpy formulas operation by operation (no fused multiply-add):
where numpy evaluates cos / sin / sinh / asinh / atan2 / hypot with the C math library (any machine without AVX-512) the
results are bit-identical to the numpy version, elsewhere within a few ulp. A P4 is a dict of its components (px, py,
pz, e; pt, eta, phi, mass when built from them), so selections and variations handle it like any multi-column value.
"""
import awkward as ak
import numpy as np

from . import jagged

try:
    from . import _hep
except ImportError as err:  # pragma: no cover
    raise ImportError("akstream.vector needs the compiled extension akstream._hep: install the package with "
                      "`pip install -e .`") from err

CARTESIAN = ("px", "py", "pz", "e")


def _flat(x):
    """(offsets or None, float64 numpy) of a per-row numpy array or a one-level jagged array."""
    if isinstance(x, ak.Array) and x.ndim == 1:
        return None, np.ascontiguousarray(ak.to_numpy(x), dtype=np.float64)
    if isinstance(x, ak.Array):
        s = jagged.simple(x)
        if s is None:
            s = jagged.simple(ak.to_packed(x))
            if s is None:
                raise TypeError(f"4-vector components must be numbers or lists of numbers, got {x.type}")
        return s[0], np.ascontiguousarray(s[1], dtype=np.float64)
    return None, np.ascontiguousarray(x, dtype=np.float64)


def _flatten(arrays):
    """(offsets or None, [flat float64]) of components with one common layout."""
    parts = [_flat(a) for a in arrays]
    offsets = parts[0][0]
    for o, _ in parts[1:]:
        if (o is None) != (offsets is None) or (o is not None and o is not offsets and not np.array_equal(o, offsets)):
            raise ValueError("4-vector components of different layouts (one per row vs one per object, or different "
                             "object counts)")
    return offsets, [d for _, d in parts]


def _wrap(offsets, flat):
    return flat if offsets is None else jagged.wrap(offsets, flat)


class P4(dict):
    """a Lorentz vector (arrays of components); see the module docstring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._derived = {}

    @classmethod
    def from_ptetaphim(cls, pt, eta, phi, mass):
        offsets, (pt, eta, phi, mass) = _flatten([pt, eta, phi, mass])
        px, py, pz, e = _hep.p4_cartesian(pt, eta, phi, mass)
        return cls({k: _wrap(offsets, v) for k, v in
                    zip(("px", "py", "pz", "e", "pt", "eta", "phi", "mass"), (px, py, pz, e, pt, eta, phi, mass))})

    @classmethod
    def from_cartesian(cls, px, py, pz, e):
        return cls(px=px, py=py, pz=pz, e=e)

    # ------------------------------------------------------------ components
    px = property(lambda self: self["px"])
    py = property(lambda self: self["py"])
    pz = property(lambda self: self["pz"])
    e = property(lambda self: self["e"])
    energy = e

    def _cartesian(self):
        return _flatten([self[k] for k in CARTESIAN])

    @property
    def mass(self):
        if "mass" in self:
            return self["mass"]
        hit = self._derived.get("mass")
        if hit is None:
            offsets, c = self._cartesian()
            hit = self._derived["mass"] = _wrap(offsets, _hep.p4_mass(*c))
        return hit

    def _angles(self, name):
        if name in self:
            return self[name]
        hit = self._derived.get(name)
        if hit is None:
            offsets, (px, py, pz) = _flatten([self["px"], self["py"], self["pz"]])
            want_eta, want_phi = name == "eta", name == "phi"
            pt, eta, phi = _hep.p4_angles(px, py, pz, want_eta, want_phi)
            self._derived["pt"] = _wrap(offsets, pt)
            if want_eta:
                self._derived["eta"] = _wrap(offsets, eta)
            if want_phi:
                self._derived["phi"] = _wrap(offsets, phi)
            hit = self._derived[name]
        return hit

    pt = property(lambda self: self._angles("pt"))
    eta = property(lambda self: self._angles("eta"))
    phi = property(lambda self: self._angles("phi"))

    # ------------------------------------------------------------ arithmetic
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:  # sum(vectors)
            return self
        return add(other, self)

    def __repr__(self):
        return f"P4({', '.join(self)})"

    # rows (take / concat / combine in the engine) build a new P4 from a dict of components
    def __reduce__(self):
        return (P4, (dict(self),))


_ETA_MAX = 22756.0  # ROOT::Math::etaMax<double>()
_BIG_Z_SCALED = np.finfo(np.float64).eps ** -0.25


class RootP4(P4):
    """ROOT::Math::PtEtaPhiMVector, operation for operation (GenVector, double; ROOT's GenVector library is the reference
    implementation these formulas follow): stored as (pt, eta, phi, mass) with
    E = sqrt((pt cosh eta)^2 + m^2); a sum adds the cartesian components and converts back to (pt, eta, phi, mass), as
    LorentzVector<PtEtaPhiM4D>::operator+ does. Use it to reproduce RDataFrame code bit for bit (numpy evaluates the
    transcendental functions)."""

    @classmethod
    def from_ptetaphim(cls, pt, eta, phi, mass):
        with np.errstate(all="ignore"):  # placeholders (eta = -999): inf / nan, as ROOT computes them
            return cls._from_ptetaphim(pt, eta, phi, mass)

    @classmethod
    def _from_ptetaphim(cls, pt, eta, phi, mass):
        offsets, (pt, eta, phi, mass) = _flatten([pt, eta, phi, mass])
        if np.any(mass < 0):  # PtEtaPhiM4D::RestrictNegMass: m < 0 is kept (space-like) unless p^2 - m^2 < 0
            p = pt * np.cosh(eta)
            mass = np.where((mass < 0) & (p * p - mass * mass < 0), -p, mass)
        outside = (phi <= -np.pi) | (phi > np.pi)  # PtEtaPhiM4D::RestrictPhi
        if np.any(outside):
            phi = np.where(outside, phi - np.floor(phi / (2 * np.pi) + .5) * 2 * np.pi, phi)
        return cls({k: _wrap(offsets, v) for k, v in zip(("pt", "eta", "phi", "mass"), (pt, eta, phi, mass))})

    def _flat4(self):
        return _flatten([self[k] for k in ("pt", "eta", "phi", "mass")])

    def _cart(self):
        hit = self._derived.get("cart")
        if hit is None:
            offsets, (pt, eta, phi, m) = self._flat4()
            with np.errstate(all="ignore"):
                hit = self._derived["cart"] = self._cartesian_root(offsets, pt, eta, phi, m)
        return hit

    @staticmethod
    def _cartesian_root(offsets, pt, eta, phi, m):
        p = pt * np.cosh(eta)
        e2 = p * p + np.where(m >= 0, m * m, -m * m)  # PtEtaPhiM4D::E2 (M2 keeps the sign of m)
        e = np.sqrt(np.where(e2 > 0, e2, 0.0))
        return offsets, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), e

    def __getitem__(self, k):
        if k in CARTESIAN and not dict.__contains__(self, k):
            offsets, *c = self._cart()
            return _wrap(offsets, c[CARTESIAN.index(k)])
        return dict.__getitem__(self, k)

    mass = property(lambda self: self["mass"])
    pt = property(lambda self: self["pt"])
    eta = property(lambda self: self["eta"])
    phi = property(lambda self: self["phi"])

    @property
    def rapidity(self):
        """0.5 log((E + pz) / (E - pz)) (LorentzVector::Rapidity)."""
        offsets, _, _, z, e = self._cart()
        with np.errstate(all="ignore"):
            return _wrap(offsets, 0.5 * np.log((e + z) / (e - z)))

    def __add__(self, other):
        if not isinstance(other, RootP4):
            return NotImplemented
        o1, x1, y1, z1, t1 = self._cart()
        o2, x2, y2, z2, t2 = other._cart()
        if (o1 is None) != (o2 is None) or (o1 is not None and not np.array_equal(o1, o2)):
            raise ValueError("4-vectors of different layouts")
        with np.errstate(all="ignore"):
            return self._sum(o1, x1 + x2, y1 + y2, z1 + z2, t1 + t2)

    @staticmethod
    def _sum(o1, x, y, z, t):
        # PxPyPzE4D -> PtEtaPhiM4D: Pt, Impl::Eta_FromRhoZ, Phi, M (with the sign of M2)
        pt = np.sqrt(x * x + y * y)
        zs = z / np.where(pt > 0, pt, 1.0)
        small = np.log(zs + np.sqrt(zs * zs + 1.0))
        large = np.where(z > 0, np.log(2.0 * zs + 0.5 / zs), -np.log(-2.0 * zs))
        eta = np.where(np.abs(zs) < _BIG_Z_SCALED, small, large)
        eta = np.where(pt > 0, eta, np.where(z == 0, 0.0, np.where(z > 0, z + _ETA_MAX, z - _ETA_MAX)))
        phi = np.where((x == 0) & (y == 0), 0.0, np.arctan2(y, x))
        m2 = t * t - x * x - y * y - z * z
        m = np.where(m2 >= 0, np.sqrt(np.abs(m2)), -np.sqrt(np.abs(m2)))
        return RootP4({k: _wrap(o1, v) for k, v in zip(("pt", "eta", "phi", "mass"), (pt, eta, phi, m))})

    __radd__ = P4.__radd__

    def __reduce__(self):
        return (RootP4, (dict(self),))


def p4(pt, eta, phi, mass, rdf_compatible=False):
    """4-vectors from (pt, eta, phi, mass) arrays (numpy per row, or jagged per object); rdf_compatible=True: RootP4
    (ROOT's PtEtaPhiMVector arithmetic, to reproduce RDataFrame code exactly)."""
    return (RootP4 if rdf_compatible else P4).from_ptetaphim(pt, eta, phi, mass)


def add(*vectors):
    """the sum of 4-vectors: each cartesian component summed in order, starting from 0 (as Python's sum)."""
    out = {}
    for k in CARTESIAN:
        offsets, parts = _flatten([v[k] for v in vectors])
        total = 0
        for p in parts:
            total = total + p
        out[k] = _wrap(offsets, total)
    return P4(out)


def collins_soper(l1, l2, charge1):
    """(phi_CS, cos theta_CS) of the pair: the l+ direction in the Collins-Soper frame of l1 + l2 (l1 is the positive
    lepton where charge1 > 0)."""
    offsets, c = _flatten([l1[k] for k in CARTESIAN] + [l2[k] for k in CARTESIAN] + [charge1])
    phi, cos_theta = _hep.collins_soper(*c)
    return _wrap(offsets, phi), _wrap(offsets, cos_theta)


def delta_phi(phi1, phi2):
    """phi2 - phi1 wrapped to (-pi, pi] (ROOT::Math::VectorUtil::DeltaPhi)."""
    d = phi2 - phi1
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    return np.where(d <= -np.pi, d + 2 * np.pi, d)


def delta_r(a, b):
    """Delta R between two vectors (P4s, or dicts with eta and phi)."""
    eta = lambda v: v.eta if isinstance(v, P4) else v["eta"]  # noqa: E731
    phi = lambda v: v.phi if isinstance(v, P4) else v["phi"]  # noqa: E731
    return np.sqrt((eta(a) - eta(b)) ** 2 + delta_phi(phi(a), phi(b)) ** 2)
