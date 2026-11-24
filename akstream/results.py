"""Results: what a pass over the data produces (histograms, sums, collected columns, ROOT files).

A result is a definition (the nodes it reads, its space, its options) plus a *state* (the accumulators), kept apart so
that worker processes fill their own states and the parent merges them.

Histograms (and sums / counts, histograms without axes) are filled with numpy bincount on a flat bin index. Varied
histograms are corrected per chunk: correction += fill(varied values of changed rows) - fill(nominal values of the
same rows), and varied = nominal + correction at the end. A bin with no entry in the variation is set to exactly 0.
"""
import functools

import awkward as ak
import numpy as np

from . import rows as R
from .graph import lift

# ---------------------------------------------------------------- axes


class Regular:
    """uniform bins. convention "boost": index = floor((x - lo) / (hi - lo) * n), as boost-histogram / hist;
    "root": floor(n * (x - lo) / (hi - lo)), as ROOT's TAxis::FindBin (the two can differ by one bin at an edge)."""

    def __init__(self, bins, start, stop, name=None, label=None, convention="boost"):
        self.bins, self.start, self.stop = int(bins), float(start), float(stop)
        self.name, self.label = name, label
        if convention not in ("boost", "root"):
            raise ValueError("convention: 'boost' or 'root'")
        self.convention = convention

    def index(self, x):
        """bin index with flow: 0 underflow, 1..bins, bins+1 overflow (NaN -> overflow)."""
        if self.convention == "root":
            x = np.asarray(x, np.float64)
            inside = (x >= self.start) & (x < self.stop)
            with np.errstate(invalid="ignore"):  # NaN / inf: replaced below
                z = (self.bins * (x - self.start) / (self.stop - self.start)).astype(np.int64)
            z += 1
            return np.where(inside, z, np.where(x < self.start, 0, self.bins + 1))
        z = (np.asarray(x, np.float64) - self.start) / (self.stop - self.start)
        out = np.full(z.shape, self.bins + 1, np.int64)
        inside = (z >= 0) & (z < 1)
        out[inside] = (z[inside] * self.bins).astype(np.int64) + 1
        out[z < 0] = 0
        return out

    @property
    def size(self):
        return self.bins + 2

    def key(self):
        return ("r", self.bins, self.start, self.stop, self.convention)

    def to_hist(self, default="x"):
        import hist
        return hist.axis.Regular(self.bins, self.start, self.stop, name=self.name or default, label=self.label)


class Variable:
    def __init__(self, edges, name=None, label=None):
        self.edges = np.asarray(edges, np.float64)
        self.name, self.label = name, label

    def index(self, x):
        x = np.asarray(x, np.float64)
        out = np.searchsorted(self.edges, x, side="right")  # 0 underflow ... len(edges) overflow
        out[np.isnan(x)] = len(self.edges)
        return out.astype(np.int64)

    @property
    def size(self):
        return len(self.edges) + 1

    def key(self):
        return ("v",) + tuple(self.edges.tolist())

    def to_hist(self, default="x"):
        import hist
        return hist.axis.Variable(self.edges, name=self.name or default, label=self.label)


class IntCategory:
    """integer categories (as hist.axis.IntCategory): one bin per category, in the given order, plus an overflow bin
    for every other value."""

    def __init__(self, categories, name=None, label=None):
        self.categories = np.asarray(categories, np.int64).reshape(-1)
        if len(np.unique(self.categories)) != len(self.categories):
            raise ValueError(f"IntCategory: repeated categories in {self.categories.tolist()}")
        self.name, self.label = name, label
        self._order = np.argsort(self.categories, kind="stable")
        self._sorted = self.categories[self._order]

    def values(self, x):
        """the column as int64; a float column is refused (as boost-histogram), not truncated."""
        x = np.asarray(x)
        if x.dtype.kind not in "iub":
            raise TypeError(f"IntCategory axis {self.name or ''}: needs an integer (or bool) column, got {x.dtype} "
                            "(cast it, e.g. aks.values_astype(x, np.int64), or bin it with a Regular / Variable axis)")
        return x.astype(np.int64, copy=False)

    def index(self, x):
        x = self.values(x)
        n = len(self.categories)
        if n == 0:
            return np.zeros(x.shape, np.int64)
        pos = np.minimum(np.searchsorted(self._sorted, x), n - 1)
        out = self._order[pos].astype(np.int64)
        out[self._sorted[pos] != x] = n  # not a category: overflow
        return out

    @property
    def size(self):
        return len(self.categories) + 1

    def key(self):
        return ("c",) + tuple(self.categories.tolist())

    def to_hist(self, default="x"):
        import hist
        return hist.axis.IntCategory(self.categories.tolist(), name=self.name or default, label=self.label)


AXES = (Regular, Variable, IntCategory)


def make_axis(bins=None, range=None, edges=None, name=None, axis=None):
    if axis is not None:
        return _from_hist_axis(axis) if not isinstance(axis, AXES) else axis
    if edges is not None:
        return Variable(edges, name)
    if isinstance(bins, (list, tuple, np.ndarray)):
        return Variable(bins, name)
    if bins is None or range is None:
        raise ValueError("a histogram axis needs bins + range, or edges")
    return Regular(bins, range[0], range[1], name)


def _from_hist_axis(a):
    import boost_histogram as bh

    if isinstance(a, bh.axis.Regular) and a.transform is None:
        return Regular(a.size, a.edges[0], a.edges[-1], getattr(a, "name", None))
    if isinstance(a, (bh.axis.Regular, bh.axis.Variable)):
        return Variable(a.edges, getattr(a, "name", None))
    if isinstance(a, bh.axis.IntCategory):
        if a.traits.growth or not a.traits.overflow:
            raise TypeError(f"unsupported axis {a!r}: an IntCategory with an overflow bin and no growth")
        return IntCategory(list(a), getattr(a, "name", None), getattr(a, "label", None))
    raise TypeError(f"unsupported axis {a!r}: Regular, Variable or IntCategory (StrCategory needs string columns)")


# ---------------------------------------------------------------- accumulators
class Acc:
    """sum of weights, sum of squared weights and entry count per bin (flow included)."""
    __slots__ = ("sumw", "sumw2", "count")

    def __init__(self, size):
        self.sumw = np.zeros(size)
        self.sumw2 = np.zeros(size)
        self.count = np.zeros(size, np.int64)

    def add(self, other, sign=1):
        if sign > 0:
            self.sumw += other.sumw
            self.sumw2 += other.sumw2
            self.count += other.count
        else:
            self.sumw -= other.sumw
            self.sumw2 -= other.sumw2
            self.count -= other.count

    def reset(self):
        self.sumw[:] = 0
        self.sumw2[:] = 0
        self.count[:] = 0


class Result:
    """base class. `nodes()` are the graph nodes read; `space` the selection.

    A result is lazy: `result.value` runs the pass (for every result booked on the dataset) and returns the value
    (a hist.Hist, a float, an awkward array, ...); `result[...]` indexes the value."""
    variations = None  # set by aks.varied: the variation names to fill ("auto" = every one that affects the result)

    def __init__(self, frame):
        self.space = frame._space
        self.dataset = frame._dataset
        self.frame = frame
        self._value = None
        self._done = False
        self.dataset.booked.append(self)

    def __init_subclass__(cls, **kwargs):
        """a result whose construction fails (e.g. an unknown column) is not left booked: it would break every later
        compute of the dataset."""
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is None:
            return

        @functools.wraps(init)
        def checked_init(self, *args, **kw):
            try:
                init(self, *args, **kw)
            except BaseException:
                booked = getattr(getattr(self, "dataset", None), "booked", None)
                if booked is not None and self in booked:
                    booked.remove(self)
                raise
        cls.__init__ = checked_init

    def __getitem__(self, item):
        return self.value[item]

    def __getattr__(self, name):
        # only for attributes a result does not have: no computation here (the engine probes results with hasattr)
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(f"{type(self).__name__!r} is a lazy akstream result and has no attribute {name!r}: "
                             f"`.value` computes it (e.g. h.value.{name})")

    def __repr__(self):
        state = f"computed: {type(self._value).__name__}" if self._done else "not computed yet; .value computes it"
        return f"<akstream {type(self).__name__} result, {state}>"

    def nodes(self):
        return []

    def all_nodes(self):
        return list(self.nodes()) + [s.mask for s in self.space.ancestors() if s.mask is not None]

    # --- state protocol
    def init_state(self):
        raise NotImplementedError

    def fill(self, state, ctx):
        raise NotImplementedError

    def fill_variation(self, state, vctx):
        raise NotImplementedError(f"{type(self).__name__} has no variations")

    def end_chunk(self, state):
        pass

    def merge(self, a, b):
        raise NotImplementedError

    def finalize(self, state):
        raise NotImplementedError

    # --- lazy value (computes every booked result of the dataset in one pass, as RDataFrame)
    @property
    def value(self):
        if not self._done:
            from .engine import compute
            pending = [r for r in self.dataset.booked if not r._done]
            compute(*pending)
        return self._value

    def _set(self, value):
        self._value, self._done = value, True
        if self in self.dataset.booked:
            self.dataset.booked.remove(self)


def _varied_ids(vctx, sd):
    return sd.ids if sd.ids is not None else np.arange(vctx.ctx.n)


def _bincount_sel(size, bins, sel):
    """the accumulator of the entries `sel` (None: all) of a binned set of rows."""
    index, weight = bins
    if sel is not None:
        index = index[sel]
        weight = weight[sel] if weight is not None else None
    acc = Acc(size)
    acc.count += np.bincount(index, minlength=size)
    if weight is None:
        acc.sumw += acc.count
        acc.sumw2 += acc.count
    else:
        acc.sumw += np.bincount(index, weights=weight, minlength=size)
        acc.sumw2 += np.bincount(index, weights=weight * weight, minlength=size)
    return acc


def _varied_rows(vctx, node, space, sd):
    """values of `node` on the varied rows of `space` (node's own space or a descendant selection)."""
    return vctx.on_rows(node, space, sd)


def _boost_axis(a):
    """the boost-histogram axis with the same binning as an akstream axis, or None (ROOT convention)."""
    import boost_histogram as bh
    if isinstance(a, Variable):
        return bh.axis.Variable(a.edges)
    if isinstance(a, Regular) and a.convention == "boost":
        return bh.axis.Regular(a.bins, a.start, a.stop)
    if isinstance(a, IntCategory):
        return bh.axis.IntCategory(a.categories.tolist(), overflow=True)
    return None


_BOOST_CACHE = {}


def _boost_hists(axes, ntargets):
    """(Weight-storage, Int64-storage) boost histograms with a leading target axis, cached and reset; None when an
    axis has no boost equivalent."""
    key = (tuple(a.key() for a in axes), ntargets)
    hit = _BOOST_CACHE.get(key)
    if hit is None:
        import boost_histogram as bh
        baxes = [_boost_axis(a) for a in axes]
        if any(b is None for b in baxes):
            hit = _BOOST_CACHE[key] = False
        else:
            target = bh.axis.Integer(0, ntargets, underflow=False, overflow=False)
            hit = _BOOST_CACHE[key] = (bh.Histogram(target, *baxes, storage=bh.storage.Weight()),
                                       bh.Histogram(target, *baxes, storage=bh.storage.Int64()))
    if hit is False:
        return None
    hit[0].reset()
    hit[1].reset()
    return hit


def _flat_columns(columns, weight):
    """flat columns and weights of one fill: jagged columns give one entry per object, per-row ones are repeated."""
    counts = None
    for c in columns:
        if not isinstance(c, np.ndarray):
            counts = np.asarray(ak.num(c, axis=1))
            break
    flat = []
    for c in columns:
        if isinstance(c, np.ndarray):
            flat.append(c if counts is None else np.repeat(c, counts))
        else:
            v = ak.flatten(c, axis=1)
            flat.append(np.asarray(ak.fill_none(v, np.nan)) if v.layout.is_option else np.asarray(v))
    if weight is not None:
        weight = np.asarray(weight, np.float64)
        if counts is not None:
            weight = np.repeat(weight, counts)
    return flat, weight, counts


def accumulate(axes, columns, weight, target=None, ntargets=1, n=None):
    """(entries, sum w, sum w^2) per bin (flow included) of ntargets histograms sharing the axes, flattened with the
    target outermost; `target` gives each row's histogram (None: all in histogram 0). Boost-convention axes are binned
    and accumulated by boost-histogram (C++); ROOT-convention axes by numpy (bin index + bincount)."""
    size = int(np.prod([a.size for a in axes])) if axes else 1
    total = size * ntargets
    flat, w, counts = _flat_columns(columns, weight)
    flat = [a.values(x) if isinstance(a, IntCategory) else x for a, x in zip(axes, flat)]
    if target is not None and counts is not None:
        target = np.repeat(target, counts)
    rows = len(flat[0]) if flat else (len(w) if w is not None else int(n or 0))
    hists = _boost_hists(axes, ntargets) if axes and rows > 0 else None
    if hists is not None:
        hw, hc = hists
        t = np.zeros(rows, np.int64) if target is None else np.asarray(target, np.int64)
        if w is None:
            hc.fill(t, *flat)
            count = np.asarray(hc.view(flow=True)).reshape(-1).astype(np.int64)
            return count, count.astype(np.float64), count.astype(np.float64)
        hw.fill(t, *flat, weight=w)
        hc.fill(t, *flat)
        view = hw.view(flow=True)
        return (np.asarray(hc.view(flow=True)).reshape(-1).astype(np.int64), np.asarray(view.value).reshape(-1).copy(),
                np.asarray(view.variance).reshape(-1).copy())
    index = np.zeros(rows, np.int64)
    stride = 1
    for axis, x in zip(reversed(axes), reversed(flat)):
        index += axis.index(x) * stride
        stride *= axis.size
    if target is not None:
        index += np.asarray(target, np.int64) * size
    count = np.bincount(index, minlength=total)
    if w is None:
        f = count.astype(np.float64)
        return count, f, f.copy()
    return count, np.bincount(index, weights=w, minlength=total), np.bincount(index, weights=w * w, minlength=total)


def _fill_arrays(axes, columns, weight):
    """flat bin index + flat weights of one fill; jagged columns (one entry per object) broadcast the per-row ones."""
    counts = None
    for c in columns:
        if not isinstance(c, np.ndarray):
            counts = np.asarray(ak.num(c, axis=1))
            break
    flat = []
    for c in columns:
        if isinstance(c, np.ndarray):
            flat.append(c if counts is None else np.repeat(c, counts))
        else:
            v = ak.flatten(c, axis=1)
            flat.append(np.asarray(ak.fill_none(v, np.nan)) if v.layout.is_option else np.asarray(v))
    if weight is not None:
        weight = np.asarray(weight, np.float64)
        if counts is not None:
            weight = np.repeat(weight, counts)
    index = np.zeros(len(flat[0]) if flat else (len(weight) if weight is not None else 0), np.int64)
    stride = 1
    for axis, x in zip(reversed(axes), reversed(flat)):
        index += axis.index(x) * stride
        stride *= axis.size
    return index, weight


_lift = lift


class Hist(Result):
    """a weighted histogram (no axes: a sum / count)."""

    def __init__(self, frame, columns, axes, weight=None, name=None):
        super().__init__(frame)
        self.columns = [_lift(c.node) for c in columns]
        self.weight = _lift(weight.node) if weight is not None else None
        self.axes = axes
        self.size = int(np.prod([a.size for a in axes])) if axes else 1
        self.name = name

    def nodes(self):
        return self.columns + ([self.weight] if self.weight is not None else [])

    def init_state(self):
        # corr: variation -> correction to nominal; direct: variation -> varied histogram (exact mode)
        return {"nominal": Acc(self.size), "chunk": Acc(self.size), "corr": {}, "direct": {}}

    def _acc(self, columns, weight, n):
        acc = Acc(self.size)
        count, sumw, sumw2 = accumulate(self.axes, columns, weight, n=n)
        acc.count += count
        acc.sumw += sumw
        acc.sumw2 += sumw2
        return acc

    def _fill_with(self, get, n):
        cols = [get(c) for c in self.columns]
        return self._acc(cols, get(self.weight) if self.weight is not None else None, n)

    def fill(self, state, ctx):
        acc = self._fill_with(lambda node: ctx.value_at(node, self.space), ctx.size(self.space))
        state["chunk"] = acc
        state["nominal"].add(acc)

    def fill_variation(self, state, vctx):
        Hist.fill_variation_group([self], {id(self): state}, vctx)

    @staticmethod
    def fill_variation_group(hists, states, vctx):
        """the variation of histograms on the same selection with the same weight. Sparse: correction = -nominal
        +varied on the changed rows (computed once for the group); dense (or exact mode): every row refilled."""
        first = hists[0]
        v, space, ctx = vctx.variation, first.space, vctx.ctx
        sd = vctx.space(space)
        nodes = list({n.id: n for h in hists for n in h.nodes()}.values())
        deltas = [vctx.delta(n) for n in nodes]
        unchanged = sd.trivial and all(d.clean for d in deltas)
        if vctx.exact:
            for h in hists:
                st = states[id(h)]
                direct = st["direct"].get(v)
                if direct is None:
                    direct = st["direct"][v] = Acc(h.size)
                if unchanged:
                    direct.add(st["chunk"])
                else:
                    direct.add(h._fill_with(lambda node: _varied_rows(vctx, node, space, sd), sd.size()))
            return
        if unchanged:
            return
        corrs = []
        for h in hists:
            st = states[id(h)]
            c = st["corr"].get(v)
            if c is None:
                c = st["corr"][v] = Acc(h.size)
            corrs.append(c)
        dense = any(d.full for d in deltas) or sd.size() == 0
        if not dense:
            changed = R.union(*[d.ids for d in deltas])
            if len(changed) and not space.is_root:  # the parent columns' changes on this selection's varied rows
                inside = ctx.is_row(space, changed)
                if len(sd.removed):
                    inside &= ~R.member(changed, sd.removed)
                if len(sd.added):
                    inside |= R.member(changed, sd.added)
                changed = changed[inside]
            changed = R.union(changed, sd.removed, sd.added)
            dense = len(changed) > vctx.dense_fraction * max(sd.size(), 1)
        if dense:
            for h, c in zip(hists, corrs):
                c.add(h._fill_with(lambda node: _varied_rows(vctx, node, space, sd), sd.size()))
                c.add(states[id(h)]["chunk"], -1)
            return
        old = changed[ctx.is_row(space, changed)] if not space.is_root else changed
        new = R.difference(changed, sd.removed)
        cache = {}

        def nominal(node):
            k = ("old", node.id)
            if k not in cache:
                cache[k] = R.take(ctx.value(node), ctx.positions(node.space, old))
            return cache[k]

        def varied(node):
            k = ("new", node.id)
            if k not in cache:
                cache[k] = vctx.gather(node, new)
            return cache[k]

        for h, c in zip(hists, corrs):
            if len(old):
                c.add(h._fill_with(nominal, len(old)), -1)
            if len(new):
                c.add(h._fill_with(varied, len(new)))

    @staticmethod
    def fill_variation_shared(hists, states, vctx):
        """the variation of every histogram with the same weight (any selection, any columns). Per selection: either
        -nominal +varied on its changed rows, or (when fewer rows) all its varied rows minus its nominal chunk fill. The
        changed rows are found once per selection, every column is gathered once on the union of the rows needed, the
        bins once per (columns, axes), then one bincount per histogram."""
        v, ctx = vctx.variation, vctx.ctx
        jagged = [h for h in hists if not all(isinstance(ctx.value(n), np.ndarray) for n in h.nodes())]
        if jagged:  # jagged columns: per selection
            by_space = {}
            for h in jagged:
                by_space.setdefault(h.space.id, []).append(h)
            for group in by_space.values():
                Hist.fill_variation_group(group, states, vctx)
            hists = [h for h in hists if h not in jagged]
            if not hists:
                return
        nodes = list({n.id: n for h in hists for n in h.nodes()}.values())
        deltas = {n.id: vctx.delta(n) for n in nodes}
        plans = {}  # space id -> (old rows or None = subtract the chunk fill, new rows), or False: unchanged

        def plan(h):
            p = plans.get(h.space.id)
            if p is not None:
                return p
            sd = vctx.space(h.space)
            ds = [deltas[n.id] for n in h.nodes()]
            any_full = any(d.full for d in ds)
            base = R.union(*[d.ids for d in ds if not d.full])
            if vctx.exact:
                p = False if (sd.trivial and not any_full and not len(base)) else (R.EMPTY, _varied_ids(vctx, sd))
            elif sd.trivial and not any_full and not len(base):
                p = False
            elif any_full:
                p = (None, _varied_ids(vctx, sd))
            else:
                changed = R.union(vctx.restrict(base, h.space, sd), sd.removed, sd.added)
                if not len(changed):
                    p = False
                else:
                    old = changed[ctx.is_row(h.space, changed)] if not h.space.is_root else changed
                    new = R.difference(changed, sd.removed)
                    all_rows = _varied_ids(vctx, sd)
                    p = (None, all_rows) if len(all_rows) <= len(old) + len(new) else (old, new)
            plans[h.space.id] = p
            return p

        todo = []
        for h in hists:
            st = states[id(h)]
            p = plan(h)
            kind = "direct" if vctx.exact else "corr"
            if p is False:
                if vctx.exact:
                    acc = st["direct"].setdefault(v, Acc(h.size))
                    acc.add(st["chunk"])
                continue
            acc = st[kind].get(v)
            if acc is None:
                acc = st[kind][v] = Acc(h.size)
            todo.append((h, p, acc))
        if not todo:
            return
        all_old = R.union(*[p[0] for _, p, _ in todo if p[0] is not None])
        all_new = R.union(*[p[1] for _, p, _ in todo])
        old_vals = {n.id: R.take(ctx.value(n), ctx.positions(n.space, all_old)) for n in nodes} if len(all_old) else {}
        new_vals = {n.id: vctx.gather(n, all_new) for n in nodes} if len(all_new) else {}
        bins, sels = {}, {}

        def binned(h, which, values, all_rows, rows):
            k = (which, tuple(c.id for c in h.columns), h.weight.id if h.weight is not None else None,
                 tuple(id(a) for a in h.axes))
            b = bins.get(k)
            if b is None:
                b = bins[k] = _fill_arrays(h.axes, [values[c.id] for c in h.columns],
                                           values[h.weight.id] if h.weight is not None else None)
            sk = (which, h.space.id)
            sel = sels.get(sk)
            if sel is None:
                sel = sels[sk] = None if len(rows) == len(all_rows) else np.searchsorted(all_rows, rows)
            return b, sel

        for h, (old, new), acc in todo:
            if old is None:
                acc.add(states[id(h)]["chunk"], -1)
            elif len(old):
                b, sel = binned(h, "old", old_vals, all_old, old)
                acc.add(_bincount_sel(h.size, b, sel), -1)
            if len(new):
                b, sel = binned(h, "new", new_vals, all_new, new)
                acc.add(_bincount_sel(h.size, b, sel))

    def merge(self, a, b):
        a["nominal"].add(b["nominal"])
        for kind in ("corr", "direct"):
            for v, c in b[kind].items():
                if v in a[kind]:
                    a[kind][v].add(c)
                else:
                    a[kind][v] = c
        return a

    def _finish(self, acc):
        if not self.axes:
            return {"sumw": float(acc.sumw[0]), "sumw2": float(acc.sumw2[0]), "count": int(acc.count[0])}
        import hist
        h = hist.Hist(*[a.to_hist(f"x{i}" if len(self.axes) > 1 else "x") for i, a in enumerate(self.axes)], storage=hist.storage.Weight(), name=self.name)
        shape = tuple(a.size for a in self.axes)
        view = h.view(flow=True)
        view.value = acc.sumw.reshape(shape)
        view.variance = acc.sumw2.reshape(shape)
        h._akstream_count = acc.count.reshape(shape)  # unweighted entries per bin (ROOT fEntries)
        return h

    def finalize(self, state):
        nominal = self._finish(state["nominal"])
        if self.variations is None:
            return nominal
        out = {"nominal": nominal}
        for v in self.variation_names:
            if v in state["direct"]:
                out[v] = self._finish(state["direct"][v])
                continue
            acc = Acc(self.size)
            acc.add(state["nominal"])
            if v in state["corr"]:
                acc.add(state["corr"][v])
                empty = acc.count == 0
                acc.sumw[empty] = 0.0
                acc.sumw2[empty] = 0.0
            out[v] = self._finish(acc)
        return out


class Sum(Hist):
    def __init__(self, frame, column=None, weight=None):
        # sum(column * weight): a histogram without axes, weighted by the product
        if column is not None and weight is not None:
            weight = column * weight
        elif column is not None:
            weight = column
        super().__init__(frame, [], [], weight)

    def finalize(self, state):
        out = super().finalize(state)
        pick = (lambda d: d["sumw"]) if self.weight is not None else (lambda d: d["count"])
        if self.variations is None:
            return pick(out)
        return {k: pick(v) for k, v in out.items()}


class Report:
    """a cutflow: entries (or sum of weights) after each filter of a chain, filled with every other result."""

    def __init__(self, steps):
        self.steps = steps  # [(label, Sum)]

    @property
    def value(self):
        out, prev = [], None
        first = None
        for label, r in self.steps:
            n = r.value
            first = n if first is None else first
            out.append((label, n, n / prev if prev else float("nan"), n / first if first else float("nan")))
            prev = n
        return out

    def __repr__(self):
        rows = self.value
        width = max(len(r[0]) for r in rows)
        lines = [f"{label:<{width}}  {n:>14.6g}  {eff:8.2%}  {cum:8.2%}" if i else f"{label:<{width}}  {n:>14.6g}"
                 for i, (label, n, eff, cum) in enumerate(rows)]
        return "\n".join(lines)


class GroupSum(Result):
    """sum of a column (or count) per distinct value of a key column (nominal only): {key: value}."""

    def __init__(self, frame, key, column=None, weight=None):
        super().__init__(frame)
        self.key = key.node
        if column is not None and weight is not None:
            weight = column * weight
        elif column is not None:
            weight = column
        self.weight = weight.node if weight is not None else None

    def nodes(self):
        return [self.key] + ([self.weight] if self.weight is not None else [])

    def init_state(self):
        return {}

    def fill(self, state, ctx):
        keys = np.asarray(ctx.value(self.key))
        uniq, inverse = np.unique(keys, return_inverse=True)
        if self.weight is None:
            sums = np.bincount(inverse, minlength=len(uniq))
        else:
            sums = np.bincount(inverse, weights=np.asarray(ctx.value(self.weight), np.float64), minlength=len(uniq))
        for k, v in zip(uniq.tolist(), sums.tolist()):
            state[k] = state.get(k, 0) + v

    def merge(self, a, b):
        for k, v in b.items():
            a[k] = a.get(k, 0) + v
        return a

    def finalize(self, state):
        return dict(sorted(state.items()))


class Collect(Result):
    """the values of some columns, concatenated over all chunks (nominal only)."""

    def __init__(self, frame, columns, names=None):
        super().__init__(frame)
        self.names = list(names) if names is not None else list(columns)
        self.columns = [frame._column_node(c) for c in columns]

    def nodes(self):
        return self.columns

    def init_state(self):
        return {n: [] for n in self.names}

    def fill(self, state, ctx):
        for name, node in zip(self.names, self.columns):
            state[name].append(_copy(ctx.value(node)))

    def merge(self, a, b):
        for k in a:
            a[k].extend(b[k])
        return a

    def finalize(self, state):
        out = {name: _concat(parts) for name, parts in state.items()}
        if any(isinstance(v, dict) for v in out.values()):
            return out  # a dict-valued column (aks.apply returning a dict): a dict of columns
        return ak.zip(out, depth_limit=1) if out else ak.Array([])


def _copy(v):
    if isinstance(v, dict):
        return {k: _copy(x) for k, x in v.items()}
    return ak.to_packed(v) if isinstance(v, ak.Array) else np.array(v, copy=True)


def _concat(parts):
    if not parts:
        return np.empty(0)
    if isinstance(parts[0], dict):
        return {k: _concat([p[k] for p in parts]) for k in parts[0]}
    if all(isinstance(p, np.ndarray) for p in parts):
        return np.concatenate(parts)
    return ak.concatenate(parts)


class ToRoot(Result):
    """write columns to a ROOT TTree with uproot (collections as nX + X_field branches; ("X", [fields]) writes a
    collection with only those fields).

    With aks.varied(...), one file per variation in the same pass: `path` must contain {variation} (or {group} /
    {tag}: "jet:jes_up" -> group "jet", tag "jes_up"; the nominal is "nominal" for all three). A variation's file holds
    the rows of its selection with its values, in entry order: per chunk, the nominal rows with the rows the variation
    changes replaced by their recomputation (fused engine)."""

    def __init__(self, frame, path, columns, tree="Events", basket_entries=200_000, compression=None):
        super().__init__(frame)
        self.path, self.tree = str(path), tree
        self.basket_entries = basket_entries
        self.compression = compression
        self.names, self.columns = [], []
        from .expr import Collection
        for c in columns:
            if isinstance(c, tuple):  # ("Jet", ["pt", "eta", ...]): a collection with only these fields
                c, fields = c
                col = frame[c][list(fields)]
            else:
                col = frame[c]
            if isinstance(col, Collection):
                for f in col.fields:
                    self.names.append((c, f))
                    self.columns.append(col.field(f).node)
            else:
                self.names.append((c, None))
                self.columns.append(frame._column_node(c))

    def nodes(self):
        return self.columns

    def init_state(self):
        return {"writers": {}, "part": None, "chunk": None, "corr": {}}

    # ------------------------------------------------------------ writers (one per variation)
    def _file_path(self, variation, part):
        path = self.path
        if self.variations is not None:
            if not any(k in path for k in ("{variation}", "{group}", "{tag}")):
                raise ValueError(f"a varied to_root path needs {{variation}} (or {{group}} / {{tag}}): {path}")
            group, _, tag = variation.partition(":")
            path = path.format(variation=variation, group=group, tag=tag or group)
        return path if part is None else path.replace(".root", f"_part{part}.root")

    def _writer(self, state, variation):
        w = state["writers"].get(variation)
        if w is None:
            w = state["writers"][variation] = {"path": None, "file": None, "tree": None, "buffer": [], "buffered": 0,
                                               "entries": 0}
        return w

    def _append(self, state, variation, values, n):
        w = self._writer(state, variation)
        w["buffer"].append([ak.to_packed(v) if isinstance(v, ak.Array) else np.array(v, copy=True) for v in values])
        w["buffered"] += n
        if w["buffered"] >= self.basket_entries:
            self._flush(state, variation)

    # ------------------------------------------------------------ nominal
    def fill(self, state, ctx):
        values = [ctx.value(n) for n in self.columns]
        n = ctx.size(self.space)
        self._append(state, "nominal", values, n)
        if self.variations is not None:
            ids = ctx.index(self.space)
            state["chunk"] = (np.arange(ctx.n) if ids is None else ids, values)
            state["corr"] = {}

    def fill_variation(self, state, vctx):
        raise NotImplementedError("a varied to_root needs the fused engine (compute(exact=False, engine='fused'))")

    # ------------------------------------------------------------ variations
    def record_variation(self, state, variation, changed, ids, values, replace_all=False):
        """the rows `ids` (root ids, sorted) of this chunk's varied selection with their varied values, computed for the
        `changed` root ids (replace_all: every row was recomputed)."""
        state["corr"][variation] = (changed, ids, values, replace_all)

    def end_chunk(self, state):
        if self.variations is None or state["chunk"] is None:
            return
        ids, nominal = state["chunk"]
        for v in self.variation_names:
            hit = state["corr"].get(v)
            if hit is None:  # nothing changes in this chunk
                self._append(state, v, nominal, len(ids))
                continue
            changed, new_ids, new_values, replace_all = hit
            if replace_all:
                self._append(state, v, new_values, len(new_ids))
                continue
            keep = np.flatnonzero(~R.member(ids, changed))
            order = np.argsort(np.concatenate([ids[keep], new_ids]), kind="stable")
            from_new = order >= len(keep)
            values = [R.combine(R.take(a, keep), b, from_new) for a, b in zip(nominal, new_values)]
            self._append(state, v, values, len(order))
        state["chunk"], state["corr"] = None, {}

    # ------------------------------------------------------------ output
    def _branches(self, parts):
        groups = {}
        for (col, field), i in zip(self.names, range(len(self.names))):
            vals = [p[i] for p in parts]
            v = np.concatenate(vals) if all(isinstance(x, np.ndarray) for x in vals) else ak.concatenate(vals)
            if field is None:
                groups[col] = v
            else:
                groups.setdefault(col, {})[field] = v
        return {k: (ak.zip(v) if isinstance(v, dict) else v) for k, v in groups.items()}

    def _flush(self, state, variation):
        w = self._writer(state, variation)
        if not w["buffer"]:
            return
        import uproot
        data = self._branches(w["buffer"])
        if w["file"] is None:
            path = self._file_path(variation, state["part"])
            kw = {} if self.compression is None else {"compression": self.compression}
            w["file"] = uproot.recreate(path, **kw)
            w["path"] = path
            types = {k: (v.type.content if isinstance(v, ak.Array) else v.dtype) for k, v in data.items()}
            w["tree"] = w["file"].mktree(self.tree, types)
        if w["buffered"]:
            w["tree"].extend(data)
        w["entries"] += w["buffered"]
        w["buffer"], w["buffered"] = [], 0

    def end_task(self, state):
        names = ["nominal"] + list(getattr(self, "variation_names", []) if self.variations is not None else [])
        for v in names:
            self._flush(state, v)  # zero-row chunks are buffered too: an empty selection still writes its typed tree
            w = state["writers"].get(v)
            if w is None:
                continue
            if w["file"] is not None:
                w["file"].close()
        out = {v: {"path": [w["path"]] if w["path"] else [], "entries": w["entries"]}
               for v, w in state["writers"].items()}
        state.clear()
        state.update({"outputs": out})
        return state

    def merge(self, a, b):
        out = {}
        for v in set(a["outputs"]) | set(b["outputs"]):
            x, y = a["outputs"].get(v, {"path": [], "entries": 0}), b["outputs"].get(v, {"path": [], "entries": 0})
            out[v] = {"path": x["path"] + y["path"], "entries": x["entries"] + y["entries"]}
        return {"outputs": out}

    def finalize(self, state):
        out = {v: {"files": o["path"], "entries": o["entries"]} for v, o in state["outputs"].items()}
        return out if self.variations is not None else out["nominal"]
