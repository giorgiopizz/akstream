"""Fast paths for the most common awkward layout: one level of lists of plain numbers ("var * float32").

awkward handles any layout through its broadcasting machinery, at a fixed cost and with intermediate arrays per
operation; for this layout an elementwise operation is just the same numpy operation on the flat contents, and row
operations are one gather. Every function returns None (or falls back to awkward) when the layout is anything else,
so the results are always those of awkward (values, dtypes, and the "var * T" type).

    simple(x)                  -> (offsets starting at 0, flat numpy contents) or None
    wrap(offsets, contents)    -> a ListOffsetArray
    elementwise(fn, args, kw)  -> fn on the flat contents (jagged args with equal counts, per-row numpy, scalars) or None
    mask(x, m)                 -> x[m] for a jagged boolean m with the same counts, or None
    index(x, i)                -> x[i] for a jagged integer index i (per-row positions), or None
    take(x, index)             -> rows of x, contiguous, or None
    combine(a, b, from_b)      -> rows from a / b interleaved, or None
    concat(pieces)             -> rows of every piece one after the other, or None
"""
import operator

import awkward as ak
import numpy as np


def simple(x):
    if not isinstance(x, ak.Array):
        return None
    layout = x.layout
    if isinstance(layout, ak.contents.ListOffsetArray):
        content = layout.content
        if not _plain(content):
            return None
        offsets = np.asarray(layout.offsets, dtype=np.int64)
        data = np.asarray(content.data)
        if offsets[0] != 0 or offsets[-1] != len(data):
            data = data[offsets[0]:offsets[-1]]
            offsets = offsets - offsets[0]
        return offsets, data
    if isinstance(layout, ak.contents.ListArray):
        content = layout.content
        if not _plain(content):
            return None
        starts, stops = np.asarray(layout.starts, dtype=np.int64), np.asarray(layout.stops, dtype=np.int64)
        counts = stops - starts
        offsets = np.zeros(len(counts) + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        data = np.asarray(content.data)
        if len(starts) and np.array_equal(starts[1:], stops[:-1]):  # contiguous view
            return offsets, data[starts[0]:stops[-1]] if len(starts) else data[:0]
        return offsets, data[_gather_index(starts, counts, offsets)]
    return None


def _plain(content):
    return (isinstance(content, ak.contents.NumpyArray) and content.data.ndim == 1
            and content.parameter("__array__") is None)


def _gather_index(starts, counts, offsets):
    """flat source index of every element of rows (starts, counts) written at offsets."""
    total = int(offsets[-1])
    return np.repeat(starts - offsets[:-1], counts) + np.arange(total, dtype=np.int64)


def wrap(offsets, data):
    return ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(offsets), ak.contents.NumpyArray(data)))


# ---------------------------------------------------------------- elementwise
ELEMENTWISE = {operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.mod, operator.pow,
               operator.and_, operator.or_, operator.xor, operator.lt, operator.le, operator.gt, operator.ge,
               operator.eq, operator.ne, operator.neg, operator.invert, np.abs, np.where}


def elementwise(fn, args, kwargs):
    """fn(*args) computed on the flat contents, or None when not applicable."""
    if kwargs or not (fn in ELEMENTWISE or isinstance(fn, np.ufunc)):
        return None
    offsets, counts, n = None, None, None
    flat = []
    for a in args:
        if isinstance(a, ak.Array):
            s = simple(a)
            if s is None:
                return None
            o, d = s
            if offsets is None:
                offsets, counts, n = o, np.diff(o), len(o) - 1
            elif len(o) != len(offsets) or not (o is offsets or np.array_equal(o, offsets)):
                return None
            flat.append(d)
        else:
            flat.append(a)
    if offsets is None:
        return None
    for i, a in enumerate(flat):  # one value per row: repeated over the row's objects (awkward broadcasting)
        if isinstance(a, np.ndarray) and a.ndim == 1 and not isinstance(args[i], ak.Array):
            if len(a) != n:
                return None
            flat[i] = np.repeat(a, counts)
        elif isinstance(a, np.ndarray) and a.ndim != 1 and a.ndim != 0:
            return None
    if fn is np.where:  # ak.where promotes Python scalars as 64-bit values (not NumPy 2's weak scalars)
        flat = [np.asarray(a) if np.isscalar(a) else a for a in flat]
    out = fn(*flat)
    if not isinstance(out, np.ndarray) or out.ndim != 1 or len(out) != offsets[-1]:
        return None
    return wrap(offsets, out)


def mask(x, m):
    """x[m] for a jagged boolean m with the same counts as x, or None."""
    sx, sm = simple(x), simple(m)
    if sx is None or sm is None or sm[1].dtype != np.bool_:
        return None
    (ox, dx), (om, dm) = sx, sm
    if len(ox) != len(om) or not np.array_equal(ox, om):
        return None
    kept = np.zeros(len(dm) + 1, np.int64)  # number of selected objects before each object
    np.cumsum(dm, out=kept[1:])
    return wrap(kept[ox], dx[dm])


def index(x, i):
    """x[i] for a jagged integer index i (positions inside each row, negative from the end; any number per row), as a
    contiguous array, or None. IndexError where a position is out of its row, as awkward."""
    sx, si = simple(x), simple(i)
    if sx is None or si is None or si[1].dtype.kind not in "iu":
        return None
    (ox, dx), (oi, di) = sx, si
    if len(ox) != len(oi):
        return None
    ci = np.diff(oi)
    n = np.repeat(np.diff(ox), ci)
    pos = di.astype(np.int64)
    pos = np.where(pos < 0, pos + n, pos)
    if len(pos) and (pos.min() < 0 or np.any(pos >= n)):
        raise IndexError("jagged index out of range of its row")
    return wrap(oi, dx[pos + np.repeat(ox[:-1], ci)])


# ---------------------------------------------------------------- rows
def take(x, index):
    """x[index] (rows), as a contiguous array, or None."""
    s = simple(x)
    if s is None:
        return None
    offsets, data = s
    index = np.asarray(index, dtype=np.int64)
    starts, counts = offsets[:-1][index], np.diff(offsets)[index]
    out = np.zeros(len(index) + 1, np.int64)
    np.cumsum(counts, out=out[1:])
    return wrap(out, data[_gather_index(starts, counts, out)])


def concat(pieces):
    """the rows of every piece one after the other, or None."""
    parts = [simple(p) for p in pieces]
    if any(p is None for p in parts):
        return None
    dtypes = {p[1].dtype for p in parts}
    if len(dtypes) != 1:
        return None
    offsets, shift = [np.zeros(1, np.int64)], 0
    for o, d in parts:
        offsets.append(o[1:] + shift)
        shift += len(d)
    return wrap(np.concatenate(offsets), np.concatenate([d for _, d in parts]))


def combine(a, b, from_b):
    """rows of len(from_b): where from_b, the next row of b, else the next row of a; or None."""
    sa, sb = simple(a), simple(b)
    if sa is None or sb is None or sa[1].dtype != sb[1].dtype:
        return None
    (oa, da), (ob, db) = sa, sb
    from_b = np.asarray(from_b, dtype=bool)
    counts = np.empty(len(from_b), np.int64)
    counts[~from_b] = np.diff(oa)
    counts[from_b] = np.diff(ob)
    starts = np.empty(len(from_b), np.int64)  # in the concatenation [da, db]
    starts[~from_b] = oa[:-1]
    starts[from_b] = ob[:-1] + len(da)
    out = np.zeros(len(from_b) + 1, np.int64)
    np.cumsum(counts, out=out[1:])
    return wrap(out, np.concatenate([da, db])[_gather_index(starts, counts, out)])
