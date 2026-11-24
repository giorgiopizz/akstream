// akstream._hep: the event loops of akstream.hep in C++.
//
// Every kernel works on the flat contents of jagged arrays and their int64 offsets (starting at 0). The arithmetic
// follows the numpy formulations operation by operation (no fused multiply-add: built with -ffp-contract=off), with the
// C math library's cos / sin / sinh / asinh / atan2 / hypot: the results are bit-identical to the equivalent numpy code
// wherever numpy uses that library too (any CPU without AVX-512), and to the RDataFrame code everywhere.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

// pybind11 < 2.12 predates NumPy 2: against NumPy 2 its arrays are silently wrong (bool outputs read as garbage).
// pip builds get a recent pybind11 from pyproject.toml; this stops a direct setup.py build on an old one (LCG_108).
#if PYBIND11_VERSION_HEX < 0x020C0000
#error "akstream._hep needs pybind11 >= 2.12 (NumPy 2 support): build with pip (pip install / pip wheel)"
#endif

namespace py = pybind11;
using i64 = std::int64_t;

template <typename T>
using carray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// sum per event, in object order, accumulated in the values' own type (as awkward's sum)
template <typename T>
py::array_t<T> segment_sum(carray<i64> offsets, carray<T> values) {
    const i64 n = offsets.size() - 1;
    auto o = offsets.template unchecked<1>();
    auto v = values.template unchecked<1>();
    py::array_t<T> out(n);
    auto r = out.template mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for (i64 i = 0; i < n; ++i) {
            T s = T(0);
            for (i64 j = o(i); j < o(i + 1); ++j) s += v(j);
            r(i) = s;
        }
    }
    return out;
}

// objects with keep true, ordered by key (stable insertion; descending or ascending): new offsets and the order (indices
// into the flat arrays)
template <typename K>
std::tuple<py::array_t<i64>, py::array_t<i64>> select_sort(carray<i64> offsets, carray<K> key, carray<bool> keep,
                                                            bool descending) {
    const i64 n = offsets.size() - 1;
    auto o = offsets.template unchecked<1>();
    auto k = key.template unchecked<1>();
    auto m = keep.template unchecked<1>();
    py::array_t<i64> new_offsets(n + 1);
    py::array_t<i64> order_full(key.size());
    auto no = new_offsets.template mutable_unchecked<1>();
    i64 *ord = order_full.mutable_data();
    i64 pos = 0;
    {
        py::gil_scoped_release release;
        no(0) = 0;
        for (i64 i = 0; i < n; ++i) {
            const i64 start = pos;
            for (i64 j = o(i); j < o(i + 1); ++j) {
                if (!m(j)) continue;
                i64 q = pos;
                if (descending) {
                    while (q > start && k(ord[q - 1]) < k(j)) { ord[q] = ord[q - 1]; --q; }
                } else {
                    while (q > start && k(ord[q - 1]) > k(j)) { ord[q] = ord[q - 1]; --q; }
                }
                ord[q] = j;
                ++pos;
            }
            no(i + 1) = pos;
        }
    }
    py::array_t<i64> order(pos);
    std::copy(ord, ord + pos, order.mutable_data());
    return {new_offsets, order};
}

// per event the pair (a < b) with the largest float32 invariant mass among those with mass > min_mass,
// |delta eta| > min_deta, max(pt) > min_lead_pt (first pair on ties); 4-vector components precomputed in double;
// genvector: the mass as ROOT's PtEtaPhiMVector sum (E from pt cosh eta, PxPyPzE4D::M2)
py::tuple max_mass_pair(carray<i64> offsets, carray<double> px, carray<double> py_, carray<double> pz, carray<double> e,
                        carray<double> eta, carray<double> pt, float min_mass, float min_deta, double min_lead_pt,
                        bool leading_only, bool genvector) {
    const i64 n = offsets.size() - 1;
    auto o = offsets.unchecked<1>();
    auto X = px.unchecked<1>(), Y = py_.unchecked<1>(), Z = pz.unchecked<1>(), E = e.unchecked<1>();
    auto H = eta.unchecked<1>(), P = pt.unchecked<1>();
    py::array_t<bool> found(n);
    py::array_t<i64> i1(n), i2(n);
    py::array_t<float> mjj(n), detajj(n);
    auto f = found.mutable_unchecked<1>();
    auto a1 = i1.mutable_unchecked<1>(), a2 = i2.mutable_unchecked<1>();
    auto M = mjj.mutable_unchecked<1>(), D = detajj.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for (i64 ev = 0; ev < n; ++ev) {
            const i64 a0 = o(ev), nj = o(ev + 1) - o(ev);
            float best = -1.0f, best_de = 0.0f;
            i64 b1 = -1, b2 = -1;
            for (i64 a = 0; a < nj; ++a) {
                for (i64 b = a + 1; b < nj; ++b) {
                    if (leading_only && !(a == 0 && b == 1)) continue;
                    const i64 ia = a0 + a, ib = a0 + b;
                    // sum(v for v in vectors) = (0 + v1) + v2, then m^2 = E^2 - (px^2 + py^2 + pz^2)
                    const double sx = (0.0 + X(ia)) + X(ib);
                    const double sy = (0.0 + Y(ia)) + Y(ib);
                    const double sz = (0.0 + Z(ia)) + Z(ib);
                    const double se = (0.0 + E(ia)) + E(ib);
                    // ROOT GenVector (PxPyPzE4D::M2): t^2 - x^2 - y^2 - z^2; numpy formula: E^2 - (px^2 + py^2 + pz^2)
                    const double m2 = genvector ? ((se * se - sx * sx) - sy * sy) - sz * sz
                                                : se * se - ((sx * sx + sy * sy) + sz * sz);
                    const double m = m2 >= 0 ? std::sqrt(std::fabs(m2)) : -std::sqrt(std::fabs(m2));
                    const float m32 = static_cast<float>(m);
                    const float de32 = static_cast<float>(std::fabs(H(ia) - H(ib)));
                    const double lead = P(ia) >= P(ib) ? P(ia) : P(ib);
                    const float cand = (m32 > min_mass && de32 > min_deta && lead > min_lead_pt) ? m32 : -1.0f;
                    if (cand > best) {  // first maximum wins (ak.argmax)
                        best = cand;
                        b1 = a;
                        b2 = b;
                        best_de = de32;
                    }
                }
            }
            if (best > 0) {
                f(ev) = true;
                a1(ev) = b1;
                a2(ev) = b2;
                M(ev) = best;
                D(ev) = best_de;
            } else {
                f(ev) = false;
                a1(ev) = -1;
                a2(ev) = -1;
                M(ev) = 0.0f;
                D(ev) = 0.0f;
            }
        }
    }
    return py::make_tuple(found, i1, i2, mjj, detajj);
}

// ---------------------------------------------------------------- matching, ancestry, pairs
// phi2 - phi1 wrapped to (-pi, pi], as hep.delta_phi
inline double wrapped_dphi(double phi1, double phi2) {
    double d = phi2 - phi1;
    if (d > M_PI) d = d - 2.0 * M_PI;
    if (d <= -M_PI) d = d + 2.0 * M_PI;
    return d;
}

// for every object a of A, the object b of B (same event, mask_b true) with the smallest
// dR^2 = (eta_a - eta_b)^2 + dphi^2 below max_dr2 (the first on ties); unique: one-to-one, the pairs taken in increasing
// dR^2 (ties: A order, then B order), each object used once. Returns (local index in B or -1, dR^2 or -1), flat over A.
py::tuple match_dr(carray<i64> offsets_a, carray<double> eta_a, carray<double> phi_a, carray<i64> offsets_b,
                   carray<double> eta_b, carray<double> phi_b, carray<bool> mask_b, double max_dr2, bool unique) {
    const i64 n = offsets_a.size() - 1;
    auto oa = offsets_a.unchecked<1>(), ob = offsets_b.unchecked<1>();
    auto EA = eta_a.unchecked<1>(), PA = phi_a.unchecked<1>(), EB = eta_b.unchecked<1>(), PB = phi_b.unchecked<1>();
    auto MB = mask_b.unchecked<1>();
    py::array_t<i64> index(eta_a.size());
    py::array_t<double> dr2(eta_a.size());
    auto I = index.mutable_unchecked<1>();
    auto R = dr2.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        std::vector<std::tuple<double, i64, i64>> pairs;
        std::vector<char> used_a, used_b;
        for (i64 ev = 0; ev < n; ++ev) {
            const i64 a0 = oa(ev), na = oa(ev + 1) - oa(ev), b0 = ob(ev), nb = ob(ev + 1) - ob(ev);
            for (i64 a = 0; a < na; ++a) {
                I(a0 + a) = -1;
                R(a0 + a) = -1.0;
            }
            if (!unique) {
                for (i64 a = 0; a < na; ++a) {
                    double best = max_dr2;
                    i64 bi = -1;
                    for (i64 b = 0; b < nb; ++b) {
                        if (!MB(b0 + b)) continue;
                        const double de = EA(a0 + a) - EB(b0 + b);
                        const double dp = wrapped_dphi(PB(b0 + b), PA(a0 + a));
                        const double d2 = de * de + dp * dp;
                        if (d2 < best) {
                            best = d2;
                            bi = b;
                        }
                    }
                    if (bi >= 0) {
                        I(a0 + a) = bi;
                        R(a0 + a) = best;
                    }
                }
                continue;
            }
            pairs.clear();
            for (i64 a = 0; a < na; ++a)
                for (i64 b = 0; b < nb; ++b) {
                    if (!MB(b0 + b)) continue;
                    const double de = EA(a0 + a) - EB(b0 + b);
                    const double dp = wrapped_dphi(PB(b0 + b), PA(a0 + a));
                    const double d2 = de * de + dp * dp;
                    if (d2 < max_dr2) pairs.emplace_back(d2, a, b);
                }
            std::stable_sort(pairs.begin(), pairs.end(),
                             [](const auto &x, const auto &y) { return std::get<0>(x) < std::get<0>(y); });
            used_a.assign(na, 0);
            used_b.assign(nb, 0);
            for (const auto &[d2, a, b] : pairs) {
                if (used_a[a] || used_b[b]) continue;
                used_a[a] = used_b[b] = 1;
                I(a0 + a) = b;
                R(a0 + a) = d2;
            }
        }
    }
    return py::make_tuple(index, dr2);
}

// for every particle, the local index of its nearest ancestor (following mother indices) whose pdgId (|pdgId| with
// use_abs) is one of `targets`, or -1; the particle itself is not its own ancestor; a mother index outside the event
// (or a loop) ends the search
py::array_t<i64> ancestor(carray<i64> offsets, carray<i64> pdg, carray<i64> mother, carray<i64> targets, bool use_abs) {
    const i64 n = offsets.size() - 1;
    auto o = offsets.unchecked<1>();
    auto P = pdg.unchecked<1>(), M = mother.unchecked<1>(), T = targets.unchecked<1>();
    const i64 nt = targets.size();
    py::array_t<i64> out(pdg.size());
    auto A = out.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for (i64 ev = 0; ev < n; ++ev) {
            const i64 p0 = o(ev), np_ = o(ev + 1) - o(ev);
            for (i64 j = 0; j < np_; ++j) {
                i64 found = -1, m = M(p0 + j), steps = 0;
                while (m >= 0 && m < np_ && steps++ < np_) {
                    const i64 id = use_abs ? std::llabs(P(p0 + m)) : P(p0 + m);
                    bool hit = false;
                    for (i64 t = 0; t < nt; ++t)
                        if (T(t) == id) hit = true;
                    if (hit) {
                        found = m;
                        break;
                    }
                    m = M(p0 + m);
                }
                A(p0 + j) = found;
            }
        }
    }
    return out;
}

// per event the pair (a < b) whose invariant mass is closest to target (the first on ties), among the pairs with
// opposite charges (opposite_charge) and equal flavour (same_flavour); the mass as in max_mass_pair, in double.
// Returns (found, i1, i2, mass (0 if not found)).
py::tuple best_pair(carray<i64> offsets, carray<double> px, carray<double> py_, carray<double> pz, carray<double> e,
                    carray<i64> charge, carray<i64> flavour, double target, bool opposite_charge, bool same_flavour,
                    bool genvector) {
    const i64 n = offsets.size() - 1;
    auto o = offsets.unchecked<1>();
    auto X = px.unchecked<1>(), Y = py_.unchecked<1>(), Z = pz.unchecked<1>(), E = e.unchecked<1>();
    auto Q = charge.unchecked<1>(), F = flavour.unchecked<1>();
    py::array_t<bool> found(n);
    py::array_t<i64> i1(n), i2(n);
    py::array_t<double> mass(n);
    auto f = found.mutable_unchecked<1>();
    auto a1 = i1.mutable_unchecked<1>(), a2 = i2.mutable_unchecked<1>();
    auto M = mass.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for (i64 ev = 0; ev < n; ++ev) {
            const i64 a0 = o(ev), nl = o(ev + 1) - o(ev);
            double best = INFINITY, best_m = 0.0;
            i64 b1 = -1, b2 = -1;
            for (i64 a = 0; a < nl; ++a)
                for (i64 b = a + 1; b < nl; ++b) {
                    const i64 ia = a0 + a, ib = a0 + b;
                    if (opposite_charge && !(Q(ia) * Q(ib) < 0)) continue;
                    if (same_flavour && F(ia) != F(ib)) continue;
                    const double sx = (0.0 + X(ia)) + X(ib);
                    const double sy = (0.0 + Y(ia)) + Y(ib);
                    const double sz = (0.0 + Z(ia)) + Z(ib);
                    const double se = (0.0 + E(ia)) + E(ib);
                    const double m2 = genvector ? ((se * se - sx * sx) - sy * sy) - sz * sz
                                                : se * se - ((sx * sx + sy * sy) + sz * sz);
                    const double m = m2 >= 0 ? std::sqrt(std::fabs(m2)) : -std::sqrt(std::fabs(m2));
                    const double d = std::fabs(m - target);
                    if (d < best) {
                        best = d;
                        best_m = m;
                        b1 = a;
                        b2 = b;
                    }
                }
            f(ev) = b1 >= 0;
            a1(ev) = b1;
            a2(ev) = b2;
            M(ev) = b1 >= 0 ? best_m : 0.0;
        }
    }
    return py::make_tuple(found, i1, i2, mass);
}

// ---------------------------------------------------------------- 4-vectors (double precision, libm functions)
// The formulas of the numpy 4-vector code they replace, operation by operation: on a machine where numpy evaluates
// cos / sin / sinh / asinh / atan2 / hypot with the C math library (no AVX-512 SVML), the results are bit-identical.

// (pt, eta, phi, mass) -> (px, py, pz, E) with E = sqrt(px^2 + py^2 + pz^2 + m^2)
py::tuple p4_cartesian(carray<double> pt, carray<double> eta, carray<double> phi, carray<double> mass) {
    const i64 n = pt.size();
    auto P = pt.unchecked<1>(), H = eta.unchecked<1>(), F = phi.unchecked<1>(), M = mass.unchecked<1>();
    py::array_t<double> px(n), py_(n), pz(n), e(n);
    double *X = px.mutable_data(), *Y = py_.mutable_data(), *Z = pz.mutable_data(), *E = e.mutable_data();
    {
        py::gil_scoped_release release;
        for (i64 i = 0; i < n; ++i) {
            const double x = P(i) * std::cos(F(i)), y = P(i) * std::sin(F(i)), z = P(i) * std::sinh(H(i));
            X[i] = x;
            Y[i] = y;
            Z[i] = z;
            E[i] = std::sqrt(((x * x + y * y) + z * z) + M(i) * M(i));
        }
    }
    return py::make_tuple(px, py_, pz, e);
}

// the C math library, elementwise (any shape): numpy's own SIMD versions (on AVX-512 CPUs) round differently from it,
// and the RDataFrame code these formulas reproduce calls it
template <double (*F)(double)>
py::array_t<double> libm1(carray<double> x) {
    py::array_t<double> out(x.request().shape);
    const double *a = x.data();
    double *r = out.mutable_data();
    const i64 n = x.size();
    {
        py::gil_scoped_release release;
        for (i64 i = 0; i < n; ++i) r[i] = F(a[i]);
    }
    return out;
}

template <double (*F)(double, double)>
py::array_t<double> libm2(carray<double> x, carray<double> y) {
    if (x.size() != y.size()) throw std::invalid_argument("arrays of different sizes");
    py::array_t<double> out(x.request().shape);
    const double *a = x.data(), *b = y.data();
    double *r = out.mutable_data();
    const i64 n = x.size();
    {
        py::gil_scoped_release release;
        for (i64 i = 0; i < n; ++i) r[i] = F(a[i], b[i]);
    }
    return out;
}

namespace {
double cos_(double x) { return std::cos(x); }
double sin_(double x) { return std::sin(x); }
double sinh_(double x) { return std::sinh(x); }
double cosh_(double x) { return std::cosh(x); }
double log_(double x) { return std::log(x); }
double atan2_(double y, double x) { return std::atan2(y, x); }
double hypot_(double x, double y) { return std::hypot(x, y); }
}  // namespace

// m = sqrt(|E^2 - p^2|) with the sign of E^2 - p^2 (ROOT's convention for space-like vectors)
py::array_t<double> p4_mass(carray<double> px, carray<double> py_, carray<double> pz, carray<double> e) {
    const i64 n = px.size();
    auto X = px.unchecked<1>(), Y = py_.unchecked<1>(), Z = pz.unchecked<1>(), E = e.unchecked<1>();
    py::array_t<double> out(n);
    double *r = out.mutable_data();
    {
        py::gil_scoped_release release;
        for (i64 i = 0; i < n; ++i) {
            const double m2 = E(i) * E(i) - ((X(i) * X(i) + Y(i) * Y(i)) + Z(i) * Z(i));
            r[i] = m2 >= 0 ? std::sqrt(std::fabs(m2)) : -std::sqrt(std::fabs(m2));
        }
    }
    return out;
}

// pt = hypot(px, py), eta = asinh(pz / pt) (0 when pt = 0), phi = atan2(py, px); each only when asked for
py::tuple p4_angles(carray<double> px, carray<double> py_, carray<double> pz, bool want_eta, bool want_phi) {
    const i64 n = px.size();
    auto X = px.unchecked<1>(), Y = py_.unchecked<1>(), Z = pz.unchecked<1>();
    py::array_t<double> pt(n), eta(want_eta ? n : 0), phi(want_phi ? n : 0);
    double *T = pt.mutable_data(), *H = eta.mutable_data(), *F = phi.mutable_data();
    {
        py::gil_scoped_release release;
        for (i64 i = 0; i < n; ++i) {
            const double t = std::hypot(X(i), Y(i));
            T[i] = t;
            if (want_eta) H[i] = std::asinh(t > 0 ? Z(i) / t : 0.0);
            if (want_phi) F[i] = std::atan2(Y(i), X(i));
        }
    }
    return py::make_tuple(pt, eta, phi);
}

namespace {
// VectorUtil::boost of (x, y, z, t) by the velocity (bx, by, bz)
inline void boost(double v0, double v1, double v2, double v3, double bx, double by, double bz, double *out) {
    const double b2 = (bx * bx + by * by) + bz * bz;
    const double gamma = 1.0 / std::sqrt(1.0 - b2);
    const double bp = (bx * v0 + by * v1) + bz * v2;
    const double gamma2 = b2 > 0 ? (gamma - 1.0) / b2 : 0.0;
    out[0] = (v0 + (gamma2 * bp) * bx) + (gamma * bx) * v3;
    out[1] = (v1 + (gamma2 * bp) * by) + (gamma * by) * v3;
    out[2] = (v2 + (gamma2 * bp) * bz) + (gamma * bz) * v3;
    out[3] = gamma * (v3 + bp);
}

inline void unit(double x, double y, double z, double *out) {
    const double n = std::sqrt((x * x + y * y) + z * z);
    out[0] = n > 0 ? x / n : x;
    out[1] = n > 0 ? y / n : y;
    out[2] = n > 0 ? z / n : z;
}

inline void cross(const double *a, const double *b, double *out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

inline double dot(const double *a, const double *b) { return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]; }
}  // namespace

// Collins-Soper angles (phi_CS, cos theta_CS) of the pair (l1, l2); the positive lepton is l1 where positive1 > 0
py::tuple collins_soper(carray<double> px1, carray<double> py1, carray<double> pz1, carray<double> e1,
                        carray<double> px2, carray<double> py2, carray<double> pz2, carray<double> e2,
                        carray<double> positive1) {
    const i64 n = px1.size();
    auto X1 = px1.unchecked<1>(), Y1 = py1.unchecked<1>(), Z1 = pz1.unchecked<1>(), E1 = e1.unchecked<1>();
    auto X2 = px2.unchecked<1>(), Y2 = py2.unchecked<1>(), Z2 = pz2.unchecked<1>(), E2 = e2.unchecked<1>();
    auto Q = positive1.unchecked<1>();
    py::array_t<double> phi(n), cos_theta(n);
    double *F = phi.mutable_data(), *C = cos_theta.mutable_data();
    {
        py::gil_scoped_release release;
        for (i64 i = 0; i < n; ++i) {
            const double sx = (0.0 + X1(i)) + X2(i), sy = (0.0 + Y1(i)) + Y2(i), sz = (0.0 + Z1(i)) + Z2(i);
            const double se = (0.0 + E1(i)) + E2(i);
            const double bx = -sx / se, by = -sy / se, bz = -sz / se;
            double b1[4], b2[4], beam1[4], beam2[4], u1[3], u2[3], z[3], y[3], x[3], c[3], pos[3], upos[3];
            boost(X1(i), Y1(i), Z1(i), E1(i), bx, by, bz, b1);
            boost(X2(i), Y2(i), Z2(i), E2(i), bx, by, bz, b2);
            boost(0.0, 0.0, 1.0, 1.0, bx, by, bz, beam1);
            boost(0.0, 0.0, -1.0, 1.0, bx, by, bz, beam2);
            unit(beam1[0], beam1[1], beam1[2], u1);
            unit(beam2[0], beam2[1], beam2[2], u2);
            unit(u1[0] - u2[0], u1[1] - u2[1], u1[2] - u2[2], z);
            cross(u1, u2, c);
            unit(c[0], c[1], c[2], y);
            cross(y, z, c);
            unit(c[0], c[1], c[2], x);
            const double *src = Q(i) > 0 ? b1 : b2;
            pos[0] = src[0];
            pos[1] = src[1];
            pos[2] = src[2];
            F[i] = std::atan2(dot(pos, y), dot(pos, x));
            unit(pos[0], pos[1], pos[2], upos);
            C[i] = dot(upos, z);
        }
    }
    return py::make_tuple(phi, cos_theta);
}

PYBIND11_MODULE(_hep, m) {
    m.doc() = "akstream.hep event loops (C++)";
    m.def("segment_sum", &segment_sum<float>);
    m.def("segment_sum", &segment_sum<double>);
    m.def("segment_sum", &segment_sum<i64>);
    m.def("select_sort", &select_sort<float>);
    m.def("select_sort", &select_sort<double>);
    m.def("select_sort", &select_sort<i64>);
    m.def("max_mass_pair", &max_mass_pair);
    m.def("match_dr", &match_dr);
    m.def("ancestor", &ancestor);
    m.def("best_pair", &best_pair);
    m.def("cos", &libm1<cos_>);
    m.def("sin", &libm1<sin_>);
    m.def("sinh", &libm1<sinh_>);
    m.def("cosh", &libm1<cosh_>);
    m.def("log", &libm1<log_>);
    m.def("atan2", &libm2<atan2_>);
    m.def("hypot", &libm2<hypot_>);
    m.def("p4_cartesian", &p4_cartesian);
    m.def("p4_mass", &p4_mass);
    m.def("p4_angles", &p4_angles);
    m.def("collins_soper", &collins_soper);
}
