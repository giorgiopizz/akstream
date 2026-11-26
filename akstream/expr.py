"""User-facing lazy values: Expr (one value per row) and Collection (a jagged set of objects with named fields).

Expr supports the Python operators, numpy ufuncs (np.sqrt(e)), numpy functions (np.where(e, a, b)) and, through the
`akstream` namespace, every awkward function (aks.sum(e, axis=1)). Nothing is computed until a result is requested.
"""
import operator

import awkward as ak
import numpy as np

from .graph import Call, Node, Scalar, Source, Take, Zip, key

# numpy functions that are row-local (row i of the output only depends on row i of the inputs)
_NUMPY_LOCAL = {np.where, np.clip, np.select, np.isnan, np.isfinite, np.isinf, np.nan_to_num, np.round, np.around,
                np.digitize, np.interp, np.full_like, np.zeros_like, np.ones_like, np.stack, np.column_stack,
                np.searchsorted, np.abs, np.sign, np.maximum, np.minimum, np.logical_and, np.logical_or, np.copy}


def _space_of(args, kwargs):
    from .graph import _nodes_in

    spaces = {n.space.id: n.space for n in _nodes_in((args, kwargs))}
    if not spaces:
        raise ValueError("an expression needs at least one column argument")
    if len(spaces) > 1:
        raise ValueError("an expression mixes columns of different selections: take every column from the same frame "
                         "(df_sel.x, not df.x, after df_sel = df[mask])")
    return next(iter(spaces.values()))


def _unwrap(obj):
    """Expr -> Node, Collection -> Zip node, containers followed."""
    if isinstance(obj, Expr):
        return obj.node
    if isinstance(obj, Collection):
        return obj.zip_node()
    if isinstance(obj, tuple):
        return tuple(_unwrap(x) for x in obj)
    if isinstance(obj, list):
        return [_unwrap(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    return obj


def _scalars(obj, space):
    """a sum / count result in an expression -> its Scalar node in `space` (hash-consed)."""
    from .results import Result, Sum
    if isinstance(obj, Result):
        if not isinstance(obj, Sum) or obj.variations is not None:
            raise TypeError(f"only a sum or a count (df.sum(...), df.count()) can be used in an expression, not "
                            f"{type(obj).__name__}" + (" with variations" if isinstance(obj, Sum) else ""))
        if obj.dataset is not space.dataset:
            raise ValueError("a sum from another dataset in an expression")
        k = ("scalar", space.id, id(obj))
        node = space.dataset.cache.get(k)
        if node is None:
            node = space.dataset.cache[k] = Scalar(space, obj)
            space.dataset.keep.append(obj)
        return node
    if isinstance(obj, tuple):
        return tuple(_scalars(x, space) for x in obj)
    if isinstance(obj, list):
        return [_scalars(x, space) for x in obj]
    if isinstance(obj, dict):
        return {k: _scalars(v, space) for k, v in obj.items()}
    return obj


def call(fn, *args, _name=None, _expensive=False, _cutoff=False, **kwargs):
    """the lazy fn(*args, **kwargs): an Expr in the (single) space of its column arguments. Hash-consed."""
    args, kwargs = _unwrap(args), _unwrap(kwargs)
    space = _space_of(args, kwargs)
    args, kwargs = _scalars(args, space), _scalars(kwargs, space)
    cache = space.dataset.cache
    k = ("call", id(fn), key(args), key(kwargs))
    node = cache.get(k)
    if node is None:
        node = Call(space, fn, args, kwargs, _name)
        node.expensive, node.cutoff = _expensive, _cutoff
        cache[k] = node
        space.dataset.keep.append(fn)  # id(fn) stays unique while the dataset lives
    return Expr(node)


class Expr:
    __array_priority__ = 1000

    def __init__(self, node):
        assert isinstance(node, Node)
        self.node = node

    # ------------------------------------------------------------ introspection
    @property
    def space(self):
        return self.node.space

    def __repr__(self):
        return f"Expr({self.node!r})"

    def __bool__(self):
        raise TypeError("a lazy column has no truth value: use & | ~ instead of and / or / not, or aks.where")

    def __len__(self):
        raise TypeError("a lazy column has no length before it is computed")

    def __iter__(self):
        raise TypeError("a lazy column cannot be iterated")

    # ------------------------------------------------------------ operators
    def _bin(self, op, other, name):
        return call(op, self, other, _name=name)

    def _rbin(self, op, other, name):
        return call(op, other, self, _name=name)

    __add__ = lambda s, o: s._bin(operator.add, o, "add")
    __radd__ = lambda s, o: s._rbin(operator.add, o, "add")
    __sub__ = lambda s, o: s._bin(operator.sub, o, "sub")
    __rsub__ = lambda s, o: s._rbin(operator.sub, o, "sub")
    __mul__ = lambda s, o: s._bin(operator.mul, o, "mul")
    __rmul__ = lambda s, o: s._rbin(operator.mul, o, "mul")
    __truediv__ = lambda s, o: s._bin(operator.truediv, o, "div")
    __rtruediv__ = lambda s, o: s._rbin(operator.truediv, o, "div")
    __floordiv__ = lambda s, o: s._bin(operator.floordiv, o, "floordiv")
    __rfloordiv__ = lambda s, o: s._rbin(operator.floordiv, o, "floordiv")
    __mod__ = lambda s, o: s._bin(operator.mod, o, "mod")
    __rmod__ = lambda s, o: s._rbin(operator.mod, o, "mod")
    __pow__ = lambda s, o: s._bin(operator.pow, o, "pow")
    __rpow__ = lambda s, o: s._rbin(operator.pow, o, "pow")
    __and__ = lambda s, o: s._bin(operator.and_, o, "and")
    __rand__ = lambda s, o: s._rbin(operator.and_, o, "and")
    __or__ = lambda s, o: s._bin(operator.or_, o, "or")
    __ror__ = lambda s, o: s._rbin(operator.or_, o, "or")
    __xor__ = lambda s, o: s._bin(operator.xor, o, "xor")
    __lt__ = lambda s, o: s._bin(operator.lt, o, "lt")
    __le__ = lambda s, o: s._bin(operator.le, o, "le")
    __gt__ = lambda s, o: s._bin(operator.gt, o, "gt")
    __ge__ = lambda s, o: s._bin(operator.ge, o, "ge")
    __eq__ = lambda s, o: s._bin(operator.eq, s._sample_code(o), "eq")
    __ne__ = lambda s, o: s._bin(operator.ne, s._sample_code(o), "ne")

    def _sample_code(self, other):
        """df.sample == "DY": the sample name as its index (several samples, aks.read_samples)."""
        if not isinstance(other, str):
            return other
        node = self.node
        while isinstance(node, Take):
            node = node.parent
        names = getattr(self.node.space.dataset, "sample_names", None)
        if names is None or not (isinstance(node, Source) and node.branch == "sample"):
            return other
        if other not in names:
            raise KeyError(f"no sample {other!r}: {names}")
        return names.index(other)
    __neg__ = lambda s: call(operator.neg, s, _name="neg")
    __pos__ = lambda s: s
    __abs__ = lambda s: call(np.abs, s, _name="abs")
    __invert__ = lambda s: call(operator.invert, s, _name="invert")
    __hash__ = object.__hash__

    def __getitem__(self, item):
        return call(operator.getitem, self, item, _name="getitem", _cutoff=_is_selection(item))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return call(getattr, self, name, _name=f".{name}")

    def astype(self, dtype):
        return call(_astype, self, np.dtype(dtype).str, _name="astype")

    # ------------------------------------------------------------ numpy protocol
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented  # reductions / accumulations across rows are not row-local
        return call(ufunc, *inputs, _name=ufunc.__name__, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _NUMPY_LOCAL:
            raise TypeError(f"np.{func.__name__} on a lazy column: only row-local numpy functions are supported "
                            f"(use aks.apply for anything else)")
        if func in (np.stack, np.column_stack) and kwargs.get("axis", 0 if func is np.stack else 1) == 0:
            raise TypeError("np.stack(..., axis=0) mixes rows: use axis=1")
        return call(func, *args, _name=func.__name__, **kwargs)


def _selected_counts(key):
    """objects per row kept by a jagged boolean mask (count of True) or a jagged index (its length)."""
    flat = ak.flatten(key, axis=1)
    if flat.layout.is_numpy and flat.layout.dtype == np.bool_:
        return np.asarray(ak.sum(key, axis=1))
    return np.asarray(ak.num(key, axis=1))


def _astype(x, dtype):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    return ak.values_astype(x, np.dtype(dtype))


def _is_selection(item):
    """a jagged mask / index applied to a jagged column: worth comparing with nominal after a variation."""
    return isinstance(item, (Expr, Collection))


# ---------------------------------------------------------------- collections
class Collection:
    """objects with named fields (the Jet_* branches of a tree, or a selection of them). Fields are separate lazy
    columns: masking, sorting or indexing a collection only touches the fields that are used downstream."""

    def __init__(self, space, fields, field_fn, counts_fn, name=None):
        self._space = space
        self._fields = list(fields)
        self._field_fn = field_fn
        self._counts_fn = counts_fn
        self._cache = {}
        self._extra = {}
        self._name = name

    # the field names
    @property
    def fields(self):
        return list(self._fields) + [f for f in self._extra if f not in self._fields]

    @property
    def space(self):
        return self._space

    def field(self, name):
        if name in self._extra:
            return self._extra[name]
        if name not in self._fields:
            raise AttributeError(f"collection {self._name or ''} has no field {name!r} (fields: {self.fields})")
        e = self._cache.get(name)
        if e is None:
            e = self._cache[name] = self._field_fn(name)
        return e

    def counts(self):
        """number of objects per row (an Expr)."""
        if self._counts_fn is None:
            return call(ak.num, self.field(self.fields[0]), axis=1, _name="num")
        return self._counts_fn()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self.field(name)

    @property
    def p4(self):
        """the Lorentz vectors of the objects (from the pt, eta, phi, mass fields; ROOT GenVector arithmetic with
        read_root(rdf_compatible=True))."""
        from .vector import p4
        return call(p4, *[self.field(f) for f in ("pt", "eta", "phi", "mass")],
                    rdf_compatible=self._space.dataset.rdf_compatible, _name="p4")

    def __setitem__(self, name, value):
        if not isinstance(value, Expr):
            raise TypeError("a collection field must be a lazy column")
        self._extra[name] = value

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.field(item)
        if isinstance(item, list) and all(isinstance(x, str) for x in item):
            return Collection(self._space, item, self.field, self._counts_fn, self._name)
        parent = self
        key_node = item.node if isinstance(item, Expr) else None

        def field_fn(name):
            return call(operator.getitem, parent.field(name), item, _name="getitem", _cutoff=key_node is not None)

        def counts_fn():
            if isinstance(item, Expr):
                return call(_selected_counts, item, _name="counts")
            return call(ak.num, field_fn(parent.fields[0]), axis=1, _name="num")

        # jets[:, 0]: one object per row (a record per row, no counts); otherwise the counts follow the selection
        per_row = isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], (int, np.integer))
        jagged = parent._counts_fn is not None and not per_row
        return Collection(self._space, self.fields, field_fn, counts_fn if jagged else None, self._name)

    def __repr__(self):
        return f"Collection({self._name or ''}: {', '.join(self.fields)})"

    def __len__(self):
        raise TypeError("a lazy collection has no length before it is computed")

    def zip_node(self):
        """a record Node with every field (fields are loaded only when the record's code touches them)."""
        fields = {f: self.field(f).node for f in self.fields}
        counts = self.counts().node if self._counts_fn is not None else None
        cache = self._space.dataset.cache
        k = ("zip", key(fields), counts.id if counts is not None else None)
        node = cache.get(k)
        if node is None:
            node = cache[k] = Zip(self._space, fields, counts)
        return node

    def to_expr(self):
        return Expr(self.zip_node())

    def take_into(self, child_space, take_expr):
        """the same collection on the rows of a child space."""
        parent = self
        counts_fn = (lambda: take_expr(parent.counts())) if self._counts_fn is not None else None
        return Collection(child_space, self.fields, lambda f: take_expr(parent.field(f)), counts_fn, self._name)


def take(expr, child_space):
    """Expr of a parent space -> the same column on the rows of child_space (hash-consed)."""
    cache = child_space.dataset.cache
    k = ("take", expr.node.id, child_space.id)
    node = cache.get(k)
    if node is None:
        if expr.node.space is not child_space.parent:
            raise ValueError("take: the column is not in the parent selection")
        node = cache[k] = Take(child_space, expr.node)
    return Expr(node)


def source_expr(space, branch):
    ds = space.dataset
    k = ("source", branch)
    node = ds.cache.get(k)
    if node is None:
        node = ds.cache[k] = Source(space, branch, ds.branches[branch], ds.dtypes.get(branch))
    return Expr(node)


def zip_record(fields, counts=None):
    """aks.zip of lazy columns: a Collection (jagged, with counts) or a per-row record."""
    exprs = dict(fields)
    space = _space_of(tuple(e.node for e in exprs.values()), {})
    counts_fn = (lambda: counts) if counts is not None else None  # None: decided from the first field at run time
    return Collection(space, list(exprs), exprs.__getitem__, counts_fn)
