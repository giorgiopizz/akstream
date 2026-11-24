"""Row sets and row-wise array helpers.

A row set is a sorted, duplicate-free int64 numpy array of *root row ids*: the entry number inside the chunk, before any
selection. Column values are either numpy arrays (one value per row, possibly (n, k)) or awkward arrays (jagged, records,
options). Every helper here keeps numpy as numpy, so flat columns never pay the awkward dispatch cost.
"""
import awkward as ak
import numpy as np

from . import jagged

EMPTY = np.empty(0, np.int64)


def as_ids(x):
    return np.asarray(x, dtype=np.int64)


def union(*sets):
    uniq = {}
    for x in sets:
        if x is not None and len(x):
            uniq[id(x)] = x  # many deltas share the very same row array
    sets = list(uniq.values())
    if not sets:
        return EMPTY
    if len(sets) == 1:
        return sets[0]
    total = sum(len(x) for x in sets)
    hi = max(int(x[-1]) for x in sets) + 1
    if total * 8 > hi:  # dense enough: a boolean scatter beats sorting
        mask = np.zeros(hi, bool)
        for x in sets:
            mask[x] = True
        return np.flatnonzero(mask)
    c = np.sort(np.concatenate(sets))
    keep = np.empty(len(c), bool)
    keep[0] = True
    np.not_equal(c[1:], c[:-1], out=keep[1:])
    return c[keep]


def intersect(a, b):
    if not len(a) or not len(b):
        return EMPTY
    return a[member(a, b)]


def difference(a, b):
    if not len(a) or not len(b):
        return a
    return a[~member(a, b)]


def member(a, b):
    """mask over `a`: a[i] is in `b` (both sorted)."""
    if not len(b):
        return np.zeros(len(a), bool)
    pos = np.searchsorted(b, a)
    pos[pos == len(b)] = len(b) - 1
    return b[pos] == a


def positions(ids, universe):
    """index of each of `ids` inside `universe` (ids must be a subset of universe; universe None = identity)."""
    if universe is None:
        return ids
    return np.searchsorted(universe, ids)


# ---------------------------------------------------------------- values
def is_numpy(values):
    return isinstance(values, np.ndarray)


def normalize(values):
    """flat awkward arrays (one number per row) become numpy: the fast path for every per-event column. A dict (the
    columns returned by an aks.apply function) is normalised column by column."""
    if isinstance(values, dict):
        return _like(values, {k: normalize(v) for k, v in values.items()})
    if isinstance(values, ak.Array):
        layout = values.layout
        while isinstance(layout, (ak.contents.IndexedArray,)):
            layout = layout.project()
        if isinstance(layout, ak.contents.NumpyArray) and layout.data.ndim >= 1 and values.ndim == layout.data.ndim:
            if layout.parameter("__array__") is None:
                return np.asarray(layout.data)
    return values


def _like(template, columns):
    """a dict of columns of the same type as `template` (a dict subclass such as vector.P4 keeps its type)."""
    return columns if type(template) is dict else type(template)(columns)


def length(values):
    if isinstance(values, dict):
        return len(next(iter(values.values())))
    return len(values)


def take(values, index):
    """values[index] for numpy or awkward, index an int64 numpy array."""
    if isinstance(values, np.ndarray):
        return values[index]
    if isinstance(values, dict):
        return _like(values, {k: take(v, index) for k, v in values.items()})
    if np.isscalar(values):
        return values
    fast = jagged.take(values, index)
    return values[index] if fast is None else fast


def concat(pieces):
    """rows of several arrays one after the other (numpy, awkward or dicts of them)."""
    if len(pieces) == 1:
        return pieces[0]
    if isinstance(pieces[0], dict):
        return _like(pieces[0], {k: concat([p[k] for p in pieces]) for k in pieces[0]})
    if all(isinstance(p, np.ndarray) for p in pieces):
        return np.concatenate(pieces)
    fast = jagged.concat(pieces) if all(isinstance(p, ak.Array) for p in pieces) else None
    return ak.concatenate([_as_ak(p) for p in pieces]) if fast is None else fast


def combine(a, b, from_b):
    """array of len(from_b) rows: the rows where from_b is True come in order from `b`, the others from `a`."""
    if isinstance(a, dict):
        return _like(a, {k: combine(a[k], b[k], from_b) for k in a})
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        out = np.empty((len(from_b),) + a.shape[1:], dtype=np.result_type(a.dtype, b.dtype))
        out[~from_b] = a
        out[from_b] = b
        return out
    a, b = _as_ak(a), _as_ak(b)
    fast = jagged.combine(a, b, from_b)
    if fast is not None:
        return fast
    src = np.empty(len(from_b), np.int64)
    src[~from_b] = np.arange(len(a))
    src[from_b] = len(a) + np.arange(len(b))
    return ak.concatenate([a, b])[src]


def _as_ak(x):
    return x if isinstance(x, ak.Array) else ak.Array(x)


def row_differs(a, b):
    """boolean numpy mask: row i of `a` differs from row i of `b` (same length). NaN == NaN. Unknown layouts compare as
    'all rows differ', which is always correct (it only costs the recomputation)."""
    if isinstance(a, dict):
        if set(a) != set(b):
            return np.ones(length(a), bool)
        d = np.zeros(length(a), bool)
        for k in a:
            d |= row_differs(a[k], b[k])
        return d
    n = len(a)
    if n == 0:
        return np.zeros(0, bool)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return _numpy_differs(a, b)
    try:
        return _ak_differs(_as_ak(a).layout, _as_ak(b).layout)
    except _Unsupported:
        return np.ones(n, bool)


class _Unsupported(Exception):
    pass


def _list_parts(layout):
    if isinstance(layout, ak.contents.ListOffsetArray):
        o = np.asarray(layout.offsets)
        return o[:-1], o[1:], layout.content
    if isinstance(layout, ak.contents.ListArray):
        return np.asarray(layout.starts), np.asarray(layout.stops), layout.content
    return None


def _flat_values(starts, stops, content):
    """the elements of every list in order, as one numpy array (a slice when the lists are contiguous)."""
    data = np.asarray(content.data)
    n = len(starts)
    if n == 1 or np.array_equal(starts[1:], stops[:-1]):
        return data[starts[0]:stops[-1]] if n else data[:0]
    counts = stops - starts
    total = int(counts.sum())
    if total == 0:
        return data[:0]
    # element j of list i sits at starts[i] + j
    first = np.repeat(starts - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    return data[first + np.arange(total)]


def masked_row_differs(xa, xb, ma, mb):
    """rows where xa[ma] differs from xb[mb] (jagged numeric x, jagged boolean masks, same counts everywhere): the
    masks differ, or an element selected by the mask differs. None if the layouts do not allow the flat computation."""
    parts = []
    for v in (xa, xb, ma, mb):
        if not isinstance(v, ak.Array):
            return None
        lay = _strip(v.layout)
        p = _list_parts(lay)
        if p is None:
            return None
        s, e, c = p
        c = _strip(c)
        if not isinstance(c, ak.contents.NumpyArray):
            return None
        parts.append((s, e, c))
    counts = parts[0][1] - parts[0][0]
    if any(not np.array_equal(p[1] - p[0], counts) for p in parts[1:]):
        return None
    fa, fb, fma, fmb = (_flat_values(*p) for p in parts)
    diff = (fma != fmb) | (fma & _numpy_differs(fa, fb))
    c = np.zeros(len(diff) + 1, np.int64)
    np.cumsum(diff, out=c[1:])
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return c[bounds[1:]] > c[bounds[:-1]]


def _flat_list_differs(la, lb):
    """jagged lists of numbers with the same counts per row (a JES-like variation): compare the flat contents and
    reduce per row with a cumulative sum. None if the fast path does not apply."""
    pa, pb = _list_parts(la), _list_parts(lb)
    if pa is None or pb is None:
        return None
    (sa, ea, ca), (sb, eb, cb) = pa, pb
    ca, cb = _strip(ca), _strip(cb)
    if not (isinstance(ca, ak.contents.NumpyArray) and isinstance(cb, ak.contents.NumpyArray)):
        return None
    counts, counts_b = ea - sa, eb - sb
    n = len(counts)
    if n == 0:
        return np.zeros(0, bool)
    out = counts != counts_b  # a different number of objects: differs
    same = np.flatnonzero(~out)
    if len(same) < n:
        sa, ea, sb, eb, counts = sa[same], ea[same], sb[same], eb[same], counts[same]
    da, db = _flat_values(sa, ea, ca), _flat_values(sb, eb, cb)
    if da.dtype != db.dtype and (da.dtype.kind != db.dtype.kind):
        return None
    diff = _numpy_differs(da, db)
    c = np.zeros(len(diff) + 1, np.int64)
    np.cumsum(diff, out=c[1:])
    bounds = np.concatenate([[0], np.cumsum(counts)])
    out[same] = c[bounds[1:]] > c[bounds[:-1]]
    return out


def _numpy_differs(a, b):
    if a.shape != b.shape:
        return np.ones(len(a), bool)
    d = a != b
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        d &= ~(np.isnan(a) & np.isnan(b))
    if d.ndim > 1:
        d = d.reshape(len(d), -1).any(axis=1)
    return d


def _strip(layout):
    while isinstance(layout, ak.contents.IndexedArray):
        layout = layout.project()
    return layout


def _ak_differs(la, lb):
    la, lb = _strip(la), _strip(lb)
    n = la.length
    if isinstance(la, ak.contents.NumpyArray) and isinstance(lb, ak.contents.NumpyArray):
        return _numpy_differs(np.asarray(la.data), np.asarray(lb.data))
    if isinstance(la, ak.contents.RecordArray) and isinstance(lb, ak.contents.RecordArray):
        if la.fields != lb.fields:
            return np.ones(n, bool)
        d = np.zeros(n, bool)
        for f in la.fields:
            d |= _ak_differs(la.content(f), lb.content(f))
        return d
    if la.is_option or lb.is_option:
        na = np.asarray(ak.is_none(ak.Array(la), axis=0)) if la.is_option else np.zeros(n, bool)
        nb = np.asarray(ak.is_none(ak.Array(lb), axis=0)) if lb.is_option else np.zeros(n, bool)
        d = na != nb
        both = np.nonzero(~na & ~nb)[0]
        if len(both):
            va = ak.Array(la)[both]
            vb = ak.Array(lb)[both]
            va = ak.drop_none(va, axis=0) if la.is_option else va
            vb = ak.drop_none(vb, axis=0) if lb.is_option else vb
            d[both] |= _ak_differs(va.layout, vb.layout)
        return d
    if la.is_list and lb.is_list:
        fast = _flat_list_differs(la, lb)
        if fast is not None:
            return fast
        a, b = ak.Array(la), ak.Array(lb)
        ca, cb = np.asarray(ak.num(a, axis=1)), np.asarray(ak.num(b, axis=1))
        d = ca != cb
        same = np.nonzero(~d)[0]
        if len(same) and ca[same].sum():
            counts = ca[same]
            fa = ak.flatten(a[same], axis=1) if len(same) < n else ak.flatten(a, axis=1)
            fb = ak.flatten(b[same], axis=1) if len(same) < n else ak.flatten(b, axis=1)
            inner = _ak_differs(fa.layout, fb.layout)
            if inner.any():
                event = np.repeat(np.arange(len(same)), counts)
                d[same] |= np.bincount(event[inner], minlength=len(same)) > 0
        return d
    raise _Unsupported(f"{type(la).__name__} vs {type(lb).__name__}")
