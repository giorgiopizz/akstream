"""Per-chunk evaluation: the nominal values, and the variations as deltas with respect to nominal.

Nominal (ChunkContext): every needed node is computed once per chunk and memoised. A selection space stores the root
row ids of its rows (`index`), so that rows can be matched across spaces.

Variation (VariationContext): a node that the variation affects is represented by a Delta -- the rows (root ids) whose
value differs from nominal, with their values. A row-local node is recomputed only on the union of the changed rows of
its inputs; every other row keeps its nominal value. Selections are tracked the same way: a SpaceDelta holds the rows a
variation removes from / adds to a selection, so added rows get their values computed and removed rows are dropped.
Results consume deltas directly (histograms are corrected by -nominal +varied on the changed rows only).

A Delta that covers most rows switches to a dense representation (`full`), computed on all rows at once, so variations
that move every event (a muon scale) cost one plain recomputation and no row bookkeeping.
"""
import operator
import sys

import awkward as ak
import numpy as np

from . import jagged
from . import rows as R
from .graph import Call, Source, Take, Zip, _nodes_in, lift, resolve

sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

_MISSING = object()


class NonLocalError(RuntimeError):
    pass


def _call_args(node, value):
    """resolved arguments of a Call node."""
    return resolve(node.args, value), resolve(node.kwargs, value)


def _apply(fn, args, kwargs):
    """fn(*args, **kwargs), on the flat contents when the jagged arguments allow it (jagged.py: same result as awkward,
    without its broadcasting machinery)."""
    if fn is operator.getitem and len(args) == 2 and isinstance(args[0], ak.Array) and isinstance(args[1], ak.Array):
        out = jagged.mask(args[0], args[1])
        if out is None:
            out = jagged.index(args[0], args[1])
        if out is not None:
            return out
    out = jagged.elementwise(fn, args, kwargs)
    return fn(*args, **kwargs) if out is None else out


def _check_length(node, value, n):
    if isinstance(value, dict):  # several columns (aks.apply returning a dict)
        for v in value.values():
            _check_length(node, v, n)
        return value
    if isinstance(value, (np.ndarray, ak.Array)):
        if len(value) != n:
            raise NonLocalError(f"{node!r} returned {len(value)} rows for {n} input rows: expressions must be row-local "
                                f"(axis=1 reductions, elementwise maths); use a result (df.sum, df.hist) for reductions")
        return value
    if np.isscalar(value) or isinstance(value, np.generic):
        raise NonLocalError(f"{node!r} returned a single value for {n} rows: expressions must be row-local; use a "
                            f"result (df.sum, df.hist) for reductions")
    raise TypeError(f"{node!r} returned a {type(value).__name__}, not an array")


def _static_dtype(node):
    """the numpy dtype of a node's (flat) values when it is known without computing it."""
    while True:
        if isinstance(node, Source):
            return node.dtype
        if isinstance(node, Take):
            node = node.parent
            continue
        if isinstance(node, Call) and node.name == "getitem" and node.args and hasattr(node.args[0], "space"):
            node = node.args[0]
            continue
        return None


def make_record(fields, length, counts, dtypes):
    """an awkward record array whose fields are only computed when touched.

    fields: name -> callable returning the field values (numpy per row, or jagged awkward per object)
    counts: numpy counts per row for a jagged collection, None for a per-row record
    dtypes: name -> numpy dtype or None (unknown: computed now)
    """
    contents, container = [], {}
    for i, (name, fn) in enumerate(fields.items()):
        dtype = dtypes.get(name)
        if dtype is None:
            values = fn()
            flat = np.asarray(ak.flatten(values, axis=1) if counts is not None else values)
            dtype = flat.dtype
            container[f"f{i}-data"] = flat
        else:
            container[f"f{i}-data"] = _flat_loader(fn, counts is not None, dtype)
        contents.append({"class": "NumpyArray", "primitive": ak.types.numpytype.dtype_to_primitive(np.dtype(dtype)),
                         "form_key": f"f{i}"})
    record = {"class": "RecordArray", "fields": list(fields), "contents": contents}
    if counts is None:
        form = record
    else:
        offsets = np.zeros(len(counts) + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        container["o-offsets"] = offsets
        form = {"class": "ListOffsetArray", "offsets": "i64", "content": record, "form_key": "o"}
    return ak.from_buffers(ak.forms.from_dict(form), length, container, buffer_key="{form_key}-{attribute}")


def _flat_loader(fn, jagged, dtype):
    def load():
        values = fn()
        if jagged:
            values = ak.flatten(values, axis=1)
        return np.ascontiguousarray(np.asarray(values), dtype=dtype)
    return load


def _masks(space, known):
    """the selection masks still needed to know the rows of `space` (`known`: spaces whose rows are computed)."""
    return [s.mask for s in space.ancestors() if s.mask is not None and s.id not in known]


def _deps(node, known):
    if isinstance(node, Source):
        return []
    if isinstance(node, Call):
        return list(_nodes_in((node.args, node.kwargs))) + _masks(node.space, known)
    if isinstance(node, Take):
        return [node.parent] + _masks(node.space, known)
    if isinstance(node, Zip):
        return ([node.counts] if node.counts is not None else []) + _masks(node.space, known)
    return list(node.inputs())


class ChunkContext:
    """nominal values of one chunk. With `subst` (node id -> node) it evaluates a variation by brute force (every
    affected node recomputed on every row): the reference the delta evaluation is validated against."""

    def __init__(self, chunk, subst=None, consumers=None):
        self.chunk = chunk
        self.n = chunk.n
        self.values = {}
        self._remaining = dict(consumers) if consumers is not None else None
        self._local = {}
        self._index = {}
        self._inverse = {}
        self._rows_of = {}
        self.subst = subst or {}

    # ------------------------------------------------------------ spaces
    def local_index(self, space):
        """positions, in the parent space, of the rows of `space`."""
        li = self._local.get(space.id)
        if li is None:
            mask = self.value(space.mask)
            if not (isinstance(mask, np.ndarray) and mask.dtype == bool and mask.ndim == 1):
                raise TypeError(f"a selection mask must be one boolean per row, got {type(mask).__name__} "
                                f"{getattr(mask, 'dtype', '')}")
            li = self._local[space.id] = np.flatnonzero(mask)
        return li

    def index(self, space):
        """root row ids of the rows of `space` (None for the root space: the identity)."""
        if space.is_root:
            return None
        ids = self._index.get(space.id)
        if ids is None:
            li = self.local_index(space)
            parent = self.index(space.parent)
            ids = self._index[space.id] = li if parent is None else parent[li]
        return ids

    def size(self, space):
        return self.n if space.is_root else len(self.local_index(space))

    def value_at(self, node, space):
        """nominal values of `node` on the rows of `space` (node's own space, or a selection below it)."""
        if node.space is space:
            return self.value(node)
        key = (node.space.id, space.id)
        pos = self._rows_of.get(key)
        if pos is None:
            pos = self._rows_of[key] = self.positions(node.space, self.index(space))
        return R.take(self.value(node), pos)

    def inverse(self, space):
        """root id -> position in the space's rows (-1 if not a row): positions and membership as one gather."""
        inv = self._inverse.get(space.id)
        if inv is None:
            ids = self.index(space)
            inv = np.full(self.n, -1, np.int64)
            inv[ids] = np.arange(len(ids))
            self._inverse[space.id] = inv
        return inv

    def positions(self, space, ids):
        """positions of root ids (rows of `space`) in the space's nominal values."""
        if space.is_root:
            return ids
        if len(ids) < 64:
            return np.searchsorted(self.index(space), ids)
        return self.inverse(space)[ids]

    def is_row(self, space, ids):
        """mask over ids: the root id is a nominal row of `space`."""
        if space.is_root:
            return np.ones(len(ids), bool)
        if len(ids) < 64:
            return R.member(ids, self.index(space))
        return self.inverse(space)[ids] >= 0

    # ------------------------------------------------------------ values
    def value(self, node):
        """the value of `node`, its dependencies first, with an explicit stack: deep recursion would make the Python
        interpreter map and unmap 16 kB frame-stack chunks over and over (a page fault storm)."""
        v = self.values.get(node.id, _MISSING)
        if v is not _MISSING:
            return v
        values = self.values
        stack = [node]
        while stack:
            n = stack[-1]
            if n.id in values:
                stack.pop()
                continue
            missing = [d for d in self._deps(n) if d.id not in values]
            if missing:
                stack.extend(missing)
                continue
            stack.pop()
            values[n.id] = self._compute(n)
            if self._remaining is not None:
                self.release(n.inputs())
        return values[node.id]

    def _deps(self, node):
        """the nodes `node` is computed from (and the selection masks its rows come from)."""
        alt = self.subst.get(node.id)
        if alt is not None:
            return [alt]
        return _deps(node, self._local)

    def release(self, nodes):
        """one consumer of each node is done: free the values nobody else will read."""
        if self._remaining is None:
            return
        for i, n in {n.id: n for n in nodes if n is not None}.items():
            if i not in self._remaining:  # not counted (e.g. the inputs of a variation's alternative): kept
                continue
            left = self._remaining[i] - 1
            self._remaining[i] = left
            if left <= 0:
                self.values.pop(i, None)
                if isinstance(n, Source):
                    self.chunk.arrays.pop(n.branch, None)

    def _compute(self, node):
        alt = self.subst.get(node.id)
        if alt is not None:
            return self.value(alt)
        if isinstance(node, Source):
            return self.chunk.get(node.branch)
        if isinstance(node, Call):
            args, kwargs = _call_args(node, self.value)
            return _check_length(node, R.normalize(_apply(node.fn, args, kwargs)), self.size(node.space))
        if isinstance(node, Take):
            return R.take(self.value(node.parent), self.local_index(node.space))
        if isinstance(node, Zip):
            counts = None if node.counts is None else np.asarray(self.value(node.counts))
            fields = {f: (lambda n=n: self.value(n)) for f, n in node.fields.items()}
            if counts is None and node.fields:
                first = self.value(next(iter(node.fields.values())))
                if not isinstance(first, np.ndarray):
                    counts = np.asarray(ak.num(first, axis=1))
            dtypes = {f: _static_dtype(n) for f, n in node.fields.items()}
            return make_record(fields, self.size(node.space), counts, dtypes)
        raise TypeError(node)


class Delta:
    """a node's value under a variation, relative to nominal.

    sparse: `ids` (sorted root ids) are the rows whose value is `vals`; every other row of the varied space keeps its
            nominal value (and is a nominal row).
    full:   `vals` are the values on every row of the varied space, in order (ids None).
    """
    __slots__ = ("ids", "vals", "full", "_inv", "_lookups")

    def __init__(self, ids, vals, full=False):
        self.ids, self.vals, self.full = ids, vals, full
        self._inv = None
        self._lookups = 0

    @property
    def clean(self):
        return not self.full and len(self.ids) == 0

    def lookup(self, ids):
        """(mask: id is one of this delta's rows, position of those ids in vals). Binary search for a few lookups;
        from the second large lookup on, a dense inverse map (root id -> position) built once."""
        self._lookups += 1
        if self._inv is None and (len(ids) < 64 or self._lookups < 2 or not len(self.ids)):
            mask = R.member(ids, self.ids)
            return mask, np.searchsorted(self.ids, ids[mask])
        if self._inv is None:
            inv = np.full(int(max(self.ids[-1], ids[-1] if len(ids) else 0)) + 1, -1, np.int64)
            inv[self.ids] = np.arange(len(self.ids))
            self._inv = inv
        inv = self._inv
        if len(ids) and ids[-1] >= len(inv):
            pos = np.full(len(ids), -1, np.int64)
            ok = ids < len(inv)
            pos[ok] = inv[ids[ok]]
        else:
            pos = inv[ids]
        mask = pos >= 0
        return mask, pos[mask]


CLEAN = Delta(R.EMPTY, None)


class SpaceDelta:
    """how a variation changes the rows of a space: `removed` (nominal rows that leave) and `added` (rows that enter)."""
    __slots__ = ("nominal", "removed", "added", "_ids", "trivial", "root_n")

    def __init__(self, nominal, removed=R.EMPTY, added=R.EMPTY, ids=None, root_n=None):
        self.nominal, self.removed, self.added = nominal, removed, added
        self._ids = ids
        self.root_n = root_n
        self.trivial = not len(removed) and not len(added) and ids is None

    @property
    def ids(self):
        """root ids of the varied rows (None: identity, the root space)."""
        if self._ids is None:
            if self.trivial:
                return self.nominal
            self._ids = R.union(R.difference(self.nominal, self.removed), self.added)
        return self._ids

    def size(self):
        if self.nominal is None:
            return self.root_n
        if self._ids is not None:
            return len(self._ids)
        return len(self.nominal) - len(self.removed) + len(self.added)


class VariationContext:
    def __init__(self, ctx, variation, affected, dense_fraction=0.5, cutoff_all=False, exact=False):
        self.ctx = ctx
        self.exact = exact  # results refill every row of the variation (bitwise equal to a full recomputation)
        self.variation = variation
        self.affected = affected  # node ids downstream of a node varied by `variation`
        self.dense_fraction = dense_fraction
        self.cutoff_all = cutoff_all
        self.deltas = {}
        self.cut = {}
        self.spaces = {}
        self._positions = {}
        self.stats = {"recomputed_rows": 0, "dense_nodes": 0, "sparse_nodes": 0}

    # ------------------------------------------------------------ spaces
    def space(self, space):
        sd = self.spaces.get(space.id)
        if sd is None:
            sd = self.spaces[space.id] = self._space(space)
        return sd

    def _space(self, space):
        ctx = self.ctx
        if space.is_root:
            return SpaceDelta(None, root_n=ctx.n)
        psd = self.space(space.parent)
        md = self.delta(space.mask)
        nominal = ctx.index(space)
        if psd.trivial and md.clean:
            return SpaceDelta(nominal)
        parent_ids = psd.ids
        if md.full:
            ids = (np.arange(ctx.n) if parent_ids is None else parent_ids)[np.asarray(md.vals, bool)]
            return SpaceDelta(nominal, R.difference(nominal, ids), R.difference(ids, nominal), ids)
        if md.clean:
            changed, passing = R.EMPTY, np.zeros(0, bool)
        else:
            changed, passing = md.ids, np.asarray(md.vals, bool)
        failing, entering = changed[~passing], changed[passing]
        removed_parent = psd.removed[ctx.is_row(space, psd.removed)] if len(psd.removed) else R.EMPTY
        removed = R.union(removed_parent, failing[ctx.is_row(space, failing)] if len(failing) else R.EMPTY)
        added = entering[~ctx.is_row(space, entering)] if len(entering) else R.EMPTY
        return SpaceDelta(nominal, removed, added)

    # ------------------------------------------------------------ values
    def nominal_positions(self, space, ids):
        """positions of root ids in the space's nominal rows, cached per (space, row set): the inputs of a node are all
        gathered at the same rows."""
        key = (space.id, id(ids))
        hit = self._positions.get(key)
        if hit is not None and hit[0] is ids:
            return hit[1]
        pos = self.ctx.positions(space, ids)
        self._positions[key] = (ids, pos)
        return pos

    def _varied_positions(self, space, ids):
        """positions of root ids in the varied rows of a space (where a dense delta's values are)."""
        sd = self.space(space)
        if sd.trivial:
            return self.nominal_positions(space, ids)
        key = ("v", space.id, id(ids))
        hit = self._positions.get(key)
        if hit is not None and hit[0] is ids:
            return hit[1]
        pos = R.positions(ids, sd.ids)
        self._positions[key] = (ids, pos)
        return pos

    def gather(self, node, ids):
        """values of `node` on the rows `ids` (root ids, a subset of the varied rows of its space)."""
        d = self.delta(node)
        if d.full:
            return R.take(d.vals, self._varied_positions(node.space, ids))
        nominal = self.ctx.value(node)
        if d.clean:
            return R.take(nominal, self.nominal_positions(node.space, ids))
        if len(ids) == len(d.ids) and (ids is d.ids or np.array_equal(ids, d.ids)):
            return d.vals
        in_d, pos = d.lookup(ids)
        if len(pos) == len(ids):
            return R.take(d.vals, pos)
        if not len(pos):
            return R.take(nominal, self.nominal_positions(node.space, ids))
        a = R.take(nominal, self.nominal_positions(node.space, ids[~in_d]))
        return R.combine(a, R.take(d.vals, pos), in_d)

    def full(self, node):
        """values of `node` on every varied row of its space, in order."""
        d = self.delta(node)
        if d.full:
            return d.vals
        sd = self.space(node.space)
        if d.clean and sd.trivial:
            return self.ctx.value(node)
        ids = sd.ids if sd.ids is not None else np.arange(self.ctx.n)
        return self.gather(node, ids)

    def restrict(self, ids, space, sd):
        """the ids (rows of an ancestor space) that are varied rows of `space`."""
        if space.is_root or not len(ids):
            return ids
        inside = self.ctx.is_row(space, ids)
        if len(sd.removed):
            inside &= ~R.member(ids, sd.removed)
        if len(sd.added):
            inside |= R.member(ids, sd.added)
        return ids[inside]

    def on_rows(self, node, space, sd):
        """values of `node` (in `space` or an ancestor of it) on every varied row of `space`, in order."""
        if node.space is space:
            return self.full(node)
        if sd.trivial and self.delta(node).clean:
            return self.ctx.value_at(node, space)
        return self.gather(node, sd.ids if sd.ids is not None else np.arange(self.ctx.n))

    def delta(self, node):
        d = self.deltas.get(node.id)
        if d is None:
            d = self.deltas[node.id] = self._delta(node)
        return d

    def _delta(self, node):
        sd = self.space(node.space)
        if node.id not in self.affected and not len(sd.added):
            return CLEAN
        alt = node.variations.get(self.variation)
        if alt is not None:
            return self._varied(node, alt, sd)
        if isinstance(node, Source):
            return CLEAN  # root space: no added rows, not varied itself
        if isinstance(node, Take):
            return self._take(node, sd)
        if isinstance(node, Call):
            return self._call(node, sd)
        if isinstance(node, Zip):
            return self._zip(node, sd)
        raise TypeError(node)

    def _varied(self, node, alt, sd):
        """a node replaced by `alt` under this variation: the rows where alt differs from nominal."""
        if not sd.trivial:
            raise NotImplementedError(f"{node!r} is varied by {self.variation} inside a selection that the same "
                                      f"variation changes")
        alt_values = self.full(alt)
        nominal = self.ctx.value(node)
        differs = R.row_differs(alt_values, nominal)
        changed = np.flatnonzero(differs)
        if len(changed) > self.dense_fraction * len(differs):
            return Delta(None, alt_values, full=True)
        ids = changed if sd.nominal is None else sd.nominal[changed]
        return Delta(ids, R.take(alt_values, changed))

    def _take(self, node, sd):
        pd = self.delta(node.parent)
        if pd.full:
            return Delta(None, self.gather(node.parent, self._varied_ids(sd)), full=True)
        ids = R.union(R.intersect(pd.ids, sd.ids) if len(pd.ids) else R.EMPTY, sd.added)
        if not len(ids):
            return CLEAN
        return Delta(ids, self.gather(node.parent, ids))

    def _varied_ids(self, sd):
        return np.arange(self.ctx.n) if sd.ids is None else sd.ids

    def _input_delta(self, node, child):
        if self.cutoff_all or child.cutoff:
            return self.cutoff(child)
        if node.expensive and not self._skip_compare(child):
            return self.cutoff(child)
        return self.delta(child)

    def _skip_compare(self, node):
        """the output (or a field of the output) of an expensive node: it was recomputed because its inputs changed, so
        comparing it with nominal again costs more than it saves (measured; aks.checkpoint forces the comparison)."""
        parent = self._projected_parent(node)
        return node.expensive or (parent is not None and parent.expensive)

    def _projection(self, node):
        """x["name"] / x.name with a constant name: its delta is the parent's delta restricted to that field (no
        gather, no recomputation). None if the node is not such a projection."""
        if node.name not in ("getitem", ".") and not node.name.startswith("."):
            return None
        if len(node.args) != 2 or not isinstance(node.args[1], str) or node.kwargs:
            return None
        parent = node.args[0]
        if not hasattr(parent, "space") or parent.space is not node.space:
            return None
        return self._project(node, self.delta(parent))

    def _project(self, node, pd):
        if pd.clean:
            return CLEAN
        key = node.args[1]
        vals = pd.vals[key] if node.fn is operator.getitem else getattr(pd.vals, key)
        return Delta(pd.ids, R.normalize(vals), full=pd.full)

    def _projected_parent(self, node):
        """the parent of a projection node (x["name"] / x.name), else None."""
        if isinstance(node, Call) and len(node.args) == 2 and isinstance(node.args[1], str) and not node.kwargs \
                and (node.fn is operator.getitem or node.fn is getattr) and hasattr(node.args[0], "space") \
                and node.args[0].space is node.space:
            return node.args[0]
        return None

    def _call(self, node, sd):
        projected = self._projection(node)
        if projected is not None:
            return projected
        # inputs taken from a parent selection are read from the parent column directly (shared deltas)
        parts = []
        considered = node.trigger if node.trigger else node.inputs()
        for c in {c.id: c for c in considered}.values():
            a = lift(c)
            d = self._input_delta(node, a)
            if d.full:
                return self._dense(node, sd)
            parts.append(d.ids if a.space is node.space else self.restrict(d.ids, node.space, sd))
        ids = R.union(*parts, sd.added)
        if not len(ids):
            return CLEAN
        # dense above dense_fraction of the rows; an expensive node stays sparse unless (almost) every row changed:
        # recomputing it costs much more than the row bookkeeping
        limit = max(self.dense_fraction, 0.95) if node.expensive else self.dense_fraction
        if len(ids) > limit * sd.size():
            return self._dense(node, sd)
        args, kwargs = _call_args(node, lambda n: self.gather(lift(n), ids))
        values = _apply(node.fn, args, kwargs)
        values = _check_length(node, R.normalize(values), len(ids))
        self.stats["recomputed_rows"] += len(ids)
        self.stats["sparse_nodes"] += 1
        return Delta(ids, values)

    def _dense(self, node, sd):
        args, kwargs = _call_args(node, lambda n: self.on_rows(lift(n), node.space, sd))
        values = _apply(node.fn, args, kwargs)
        self.stats["dense_nodes"] += 1
        return Delta(None, _check_length(node, R.normalize(values), sd.size()), full=True)

    def _zip(self, node, sd):
        deltas = [self.delta(c) for c in node.inputs()]
        if any(d.full for d in deltas):
            ids = self._varied_ids(sd)
            full = True
        else:
            ids = R.union(*[d.ids for d in deltas], sd.added)
            if not len(ids):
                return CLEAN
            full = False
        counts = None if node.counts is None else np.asarray(self.gather(node.counts, ids))
        fields = {f: (lambda n=n: self.gather(n, ids)) for f, n in node.fields.items()}
        if counts is None and node.fields:
            first = self.gather(next(iter(node.fields.values())), ids)
            if not isinstance(first, np.ndarray):
                counts = np.asarray(ak.num(first, axis=1))
        dtypes = {f: _static_dtype(n) for f, n in node.fields.items()}
        record = make_record(fields, len(ids), counts, dtypes)
        return Delta(None, record, full=True) if full else Delta(ids, record)

    def cutoff(self, node):
        """the delta of `node` without the rows whose recomputed value equals nominal (early cutoff)."""
        c = self.cut.get(node.id)
        if c is not None:
            return c
        d = self.delta(node)
        if d.clean or isinstance(node, Source) and not node.variations:
            c = d
        elif d.full:  # a dense delta: back to the rows that really changed
            ids = self._varied_ids(self.space(node.space))
            in_nominal = self.ctx.is_row(node.space, ids)
            keep = ~in_nominal
            where = np.flatnonzero(in_nominal)
            if len(where):
                nominal = R.take(self.ctx.value(node), self.nominal_positions(node.space, ids[where]))
                keep[where] = R.row_differs(R.take(d.vals, where), nominal)
            sel = np.flatnonzero(keep)
            c = d if len(sel) == len(ids) else Delta(ids[sel], R.take(d.vals, sel))
        else:
            ids = d.ids
            in_nominal = self.ctx.is_row(node.space, ids)
            keep = ~in_nominal
            where = np.flatnonzero(in_nominal)
            if len(where):
                nominal = R.take(self.ctx.value(node), self.nominal_positions(node.space, ids[where]))
                keep[where] = R.row_differs(R.take(d.vals, where), nominal)
            if keep.all():
                c = d
            elif not keep.any():
                c = CLEAN
            else:
                sel = np.flatnonzero(keep)
                c = Delta(ids[sel], R.take(d.vals, sel))
        # the cut delta decides which rows a consumer recomputes; values are still gathered from the full delta (a
        # superset of those rows: one take instead of merging recomputed and nominal rows)
        self.cut[node.id] = c
        return c
