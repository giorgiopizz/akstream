"""Frame: the front end, between RDataFrame (define / filter / vary / hist chains), pandas (df["x"] = ..., df[mask]) and
awkward (collections, aks.<awkward function>, 4-vectors).

    df = aks.read_root("files/*.root", "Events")
    df = (df.define("mu1", aks.p4(df.mu1_pt, df.mu1_eta, df.mu1_phi, df.mu1_mass))
            .define("mu2", aks.p4(df.mu2_pt, df.mu2_eta, df.mu2_phi, df.mu2_mass))
            .define("mll", "(mu1 + mu2).mass")
            .define("ht", aks.sum(df.Jet.pt, axis=1)))
    df["njet"] = aks.num(df.Jet[df.Jet.pt > 25].pt, axis=1)        # pandas-style, the same thing
    z = df.filter("mll > 70 && mll < 110", name="Z window")       # or df[(df.mll > 70) & (df.mll < 110)]
    h = z.hist("ht", (50, 0, 500), weight="genWeight")
    h.value                                                         # one pass over the files for every booked result
    z.report()                                                      # the cutflow of the named filters

Collections are awkward-like records over the columns X_<field> (NanoAOD naming): df.Jet.pt is the column Jet_pt, so
the three assignments below are the same, and replacing the collection replaces every field:

    df.Jet["pt_raw"] = df.Jet.pt * (1 - df.Jet.rawFactor)   # = df["Jet", "pt_raw"] = ... = df["Jet_pt_raw"] = ...
    df.Jet["is_data"] = False                                 # a scalar (or a per-row column) is broadcast to the objects
    df["Jet"] = df.Jet[df.Jet.pt > 25]                        # every field (and the counter nJet) selected
    df["Jet"] = df.Jet[aks.argsort(df.Jet.pt)[:, ::-1]]       # sorted

Columns are lazy. A filtered frame is a new selection (space); every column of the parent is available on it, taken
on its rows only when used. Assigning to a frame after a selection was made from it does not change the selection's
columns (as a pandas copy); define / apply return a new frame and leave this one as it is.
"""
import ast

import awkward as ak
import numpy as np

from . import results as res
from .expr import Collection, Expr, call, source_expr, take
from .graph import Call, Space, Source


class Frame:
    def __init__(self, dataset, space, inherited=None, parent=None, collections=None):
        self._dataset = dataset
        self._space = space
        self._columns = {}
        self._inherited = dict(inherited or {})  # name -> Expr | Collection in the parent space
        self._parent = parent
        self._resolved = {}
        self._filter_name = None
        self._new_collections = list(collections or [])  # collections made on a frame (df["goodJet"] = df.Jet[...])

    # ------------------------------------------------------------ column lookup
    def _lookup(self, name):
        if name in self._columns:
            return self._columns[name]
        hit = self._resolved.get(name)
        if hit is not None:
            return hit
        if name in self._inherited:
            hit = self._into_me(self._inherited[name])
        elif self._is_collection(name):
            return FrameCollection(self, name)  # a view: its fields are looked up when used
        else:
            hit = self._schema_column(name)
        self._resolved[name] = hit
        return hit

    def _is_collection(self, name):
        return name in self._dataset.collections or name in self._new_collections

    def _fields_of(self, name):
        """the fields of collection `name`: its branches, then the columns name_<field> defined on the frame."""
        out = list(self._dataset.collections.get(name, []))
        prefix = name + "_"
        seen = set(out)
        for c in self.columns:
            if c.startswith(prefix) and c[len(prefix):] not in seen:
                out.append(c[len(prefix):])
                seen.add(c[len(prefix):])
        return out

    def _counter(self, name):
        """the counter column nX of collection X (a branch, or set when the collection is replaced), or None."""
        c = f"n{name}"
        if c in self.columns or self._dataset.counters.get(name) == c:
            return c
        return None

    def _schema_column(self, name):
        """a branch or collection of the tree, in this frame's space (ignores defined columns)."""
        k = ("schema", name)
        hit = self._resolved.get(k)
        if hit is None:
            hit = self._root_column(name) if self._space.is_root else self._into_me(self._parent._schema_column(name))
            self._resolved[k] = hit
        return hit

    def _root_column(self, name):
        ds = self._dataset
        if name in ds.branches:
            return source_expr(self._space, name)
        if name in ds.collections:
            space = self._space
            return Collection(space, ds.collections[name], lambda f: source_expr(space, f"{name}_{f}"),
                              lambda: source_expr(space, f"n{name}"), name)
        raise KeyError(f"no column or branch {name!r}")

    def _into_me(self, value):
        """a parent-space column on the rows of this frame."""
        space = self._space
        if isinstance(value, Collection):
            return value.take_into(space, lambda e: take(e, space))
        return take(value, space)

    def _column_node(self, name):
        col = self[name] if isinstance(name, str) else name
        if isinstance(col, Collection):
            return col.zip_node()
        return col.node

    @property
    def columns(self):
        """the defined columns (inherited from the parent selections and assigned on this frame)."""
        return list(dict.fromkeys(list(self._inherited) + list(self._columns)))

    @property
    def branches(self):
        from .io import ENTRY
        return [b for b in self._dataset.branches if b != ENTRY]

    @property
    def entry(self):
        """the entry number of each row in its file (RDataFrame's rdfentry_ for one file)."""
        from .io import ENTRY
        return self._schema_column(ENTRY)

    @property
    def dtypes(self):
        """{branch: numpy dtype} of the tree (the content type of jagged branches)."""
        return dict(self._dataset.dtypes)

    @property
    def collections(self):
        """{name: [fields]} of the collections (the tree's and the ones made on this frame)."""
        names = list(self._dataset.collections) + [c for c in self._new_collections if c not in self._dataset.collections]
        return {c: self._fields_of(c) for c in names}

    def __dir__(self):
        return list(super().__dir__()) + self.columns + list(self.collections)

    # ------------------------------------------------------------ pandas-like access
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._lookup(name)
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._lookup(key)
        if isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, str) for k in key):  # df["Jet", "pt"]
            return self._collection(key[0]).field(key[1])
        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            out = Frame(self._dataset, self._space, parent=self._parent, collections=self._new_collections)
            out._inherited = {k: v for k, v in self._inherited.items()}
            out._columns = {k: self[k] for k in key}
            out._resolved = self._resolved
            out._projection = list(key)
            return out
        if isinstance(key, Expr):
            return self._select(key)
        raise TypeError(f"cannot index a frame with {type(key).__name__}")

    def _select(self, mask):
        if mask.space is not self._space:
            raise ValueError("the selection mask must be built from this frame's columns")
        mask = call(_selection_mask, mask, _name="mask")  # None (e.g. mjj of an event without two jets): fails
        space = Space(self._dataset, parent=self._space, mask=mask.node)
        inherited = {k: self._lookup(k) for k in self.columns}  # every defined column, in this frame's space
        return Frame(self._dataset, space, inherited=inherited, parent=self, collections=self._new_collections)

    def _collection(self, name):
        col = self._lookup(name)
        if not isinstance(col, Collection):
            raise KeyError(f"{name!r} is not a collection")
        return col

    def __setitem__(self, name, value):
        if isinstance(name, tuple) and len(name) == 2 and all(isinstance(k, str) for k in name):  # df["Jet", "pt"] = ...
            return self._set_field(name[0], name[1], value)
        if callable(value) and not isinstance(value, (Expr, Collection)):
            value = value(self)
        if isinstance(value, Collection):
            return self._set_collection(name, value)
        if isinstance(value, Expr):
            if value.space is not self._space:
                raise ValueError(f"column {name!r}: the value is built from another selection's columns")
            self._columns[name] = value
            self._resolved.pop(name, None)
            return
        if np.isscalar(value):
            # TODO (planned, docs/api.md "Not supported yet"): string columns. A str here becomes a constant column of
            # text (pandas semantics) that nothing downstream handles yet: hist / to_root / comparisons fail on it
            self._columns[name] = call(_constant, self._row_reference(), value, _name="constant")
            return
        raise TypeError(f"column {name!r}: assign a lazy column (an expression of this frame's columns), "
                        f"not a {type(value).__name__}")

    def _set_collection(self, name, value):
        """df["Jet"] = <collection>: every field becomes the column Jet_<field> (and the counter nJet, if any)."""
        if value.space is not self._space:
            raise ValueError(f"collection {name!r}: the value is built from another selection's columns")
        if name in self._columns or (not self._is_collection(name) and self._has(name)):
            raise ValueError(f"{name!r} is a column, not a collection")
        fields = value.fields
        exprs = {f: value.field(f) for f in fields}  # before any assignment: the value may be a view of this collection
        counter = self._counter(name) if self._is_collection(name) else None
        counts = value.counts() if counter is not None else None
        for f, e in exprs.items():
            self._columns[f"{name}_{f}"] = e
            self._resolved.pop(f"{name}_{f}", None)
        if counter is not None:
            self._columns[counter] = counts
            self._resolved.pop(counter, None)
        if name not in self._dataset.collections and name not in self._new_collections:
            self._new_collections.append(name)

    def _set_field(self, name, field, value):
        """df.Jet["x"] = value: a jagged column with the objects of the collection (checked when it is computed), or a
        scalar / one value per row, broadcast to the objects."""
        col = self._collection(name)
        if callable(value) and not isinstance(value, (Expr, Collection)):
            value = value(self)
        if isinstance(value, Expr) and value.space is not self._space:
            raise ValueError(f"{name}.{field}: the value is built from another selection's columns")
        own = isinstance(value, Expr) and _is_field_of(value, name, self)
        if not own:
            if not (isinstance(value, Expr) or np.isscalar(value)):
                raise TypeError(f"{name}.{field}: assign a lazy column or a scalar, not a {type(value).__name__}")
            value = call(_as_field, value, col.counts(), label=f"{name}.{field}", _name="as_field")
        self._columns[f"{name}_{field}"] = value
        self._resolved.pop(f"{name}_{field}", None)

    def _row_reference(self):
        """a cheap column with one value per row (for constants)."""
        ds = self._dataset
        flat = [b for b, j in ds.branches.items() if not j]
        name = "event" if "event" in flat else flat[0]
        return self._schema_column(name)

    def copy(self):
        out = Frame(self._dataset, self._space, inherited=self._inherited, parent=self._parent,
                    collections=self._new_collections)
        out._columns = dict(self._columns)
        out._resolved = dict(self._resolved)
        out._filter_name = self._filter_name
        return out

    # ------------------------------------------------------------ RDataFrame-like chains
    def define(self, name=None, value=None, **columns):
        """a new frame with more columns: define("x", value) or define(x=value, y=...). A value is a lazy column, an
        expression string ("(mu1 + mu2).mass", "pt > 25 && abs(eta) < 2.4"), a constant, or a function of the frame."""
        out = self.copy()
        if name is not None:
            columns = {name: value, **columns}
        for k, v in columns.items():
            out[k] = out.eval(v) if isinstance(v, str) else v
        return out

    def filter(self, condition, name=None):
        """the rows passing `condition` (a boolean column or an expression string, C++ or Python syntax). A named
        filter is a line of report()."""
        mask = self.eval(condition) if isinstance(condition, str) else condition
        if callable(mask) and not isinstance(mask, Expr):
            mask = mask(self)
        out = self._select(mask)
        out._filter_name = name or (condition if isinstance(condition, str) else None)
        return out

    def report(self, weight=None):
        """the cutflow of the filters leading to this frame: a result whose value is [(filter, entries passing,
        efficiency w.r.t. the previous filter, cumulative efficiency)], all filled in the same pass as the rest."""
        chain = []
        f = self
        while f is not None:
            chain.append(f)
            f = f._parent
        chain.reverse()
        steps = [("all entries", chain[0])] + [(f._filter_name or f"selection {f._space.id}", f) for f in chain[1:]]
        return res.Report([(label, res.Sum(f, None, f._expr(weight) if weight is not None else None))
                           for label, f in steps])

    def apply(self, fn, columns, outputs=None, when=None, prefix="", name=None, **kwargs):
        """a new frame with the columns computed by fn(chunk, **kwargs), a function of a dict of arrays
        {name: numpy / awkward array} (one entry per row of this frame) that returns a dict of arrays with one entry
        per row: the usual "function of a chunk" of an awkward analysis, with akstream reading only the listed columns,
        and under a variation calling fn only on the rows whose inputs change.

        columns  the inputs: a list of column / branch names, or {key in the chunk: column name or lazy column}
        outputs  the output names (default: found by running fn on the first rows), or one name if fn returns an array
        when     columns that decide, row by row, whether the outputs can change (see aks.apply)
        prefix   prepended to the output names
        """
        import akstream
        if not isinstance(columns, dict):
            columns = {c: c for c in columns}
        keys = list(columns)
        args = [self[v] if isinstance(v, str) else v for v in columns.values()]
        when = None if when is None else [self.eval(w) if isinstance(w, str) else w for w in when]
        fn_name = name or getattr(fn, "__name__", "apply")
        value = akstream.apply(_ChunkFunction(fn, keys, kwargs, fn_name), *args, name=fn_name, when=when)
        out = self.copy()
        if isinstance(outputs, str):
            out[prefix + outputs] = value
            return out
        if outputs is None:
            sample = self.peek(value, 200)
            if not isinstance(sample, dict):
                raise TypeError(f"apply({fn_name}): the function returned a {type(sample).__name__}, not a dict: give "
                                f"outputs='name' for a single column")
            outputs = list(sample)
        for k in outputs:
            out[prefix + k] = value[k]
        return out

    def eval(self, expression, local=None):
        """an expression string of this frame's columns; `local` adds names (e.g. masks defined earlier)."""
        if not isinstance(expression, str):
            return expression
        tree = _BoolRewriter().visit(ast.parse(_from_cpp(expression), mode="eval"))
        if self._dataset.rdf_compatible:
            tree = _DoubleLiterals().visit(tree)
        ast.fix_missing_locations(tree)
        import akstream
        namespace = _Namespace(self, {"np": np, "aks": akstream, "ak": ak, "abs": abs, "min": np.minimum,
                                      "max": np.maximum, "sqrt": np.sqrt, "where": np.where, "_f64": np.float64,
                                      **(local or {})})
        out = eval(compile(tree, "<query>", "eval"), {"__builtins__": {}}, namespace)
        if isinstance(out, bool):  # a constant mask ("true")
            out = call(_constant, self._row_reference(), out, _name="constant")
        return out

    # ------------------------------------------------------------ variations
    def vary(self, column, name=None, **tags):
        """register a systematic variation (as RDataFrame Vary): every result that depends on the varied column(s) gets
        the variations "<name>:<tag>" (any tag names, not only up / down). Returns the frame (chainable).

            df.vary("w", up="genWeight * pu_up", down=df.genWeight * df.pu_down, name="pu")
            df.vary(("Jet", "pt"), up="Jet_pt_jes_up", down="Jet_pt_jes_down", name="jes")        # a collection field
            df.vary(["Jet_pt", "Jet_mass"], up=["Jet_pt_jer_up", "Jet_mass_jer_up"],
                    down=["Jet_pt_jer_down", "Jet_mass_jer_down"], name="jer")                   # columns varied together

        column: a column / branch name, a collection field ("Jet", "pt"), or a list of them varied together (each tag
        then gives a list of alternatives, in the same order). name defaults to the column name for one column (as in
        RDataFrame) and is required for a list. An alternative is a branch / column name, an expression string, or a
        lazy column of this frame. A varied branch varies everywhere it is read; a varied defined column varies from
        here on (it gets a node of its own, so an identical expression elsewhere is not affected).
        """
        many = isinstance(column, list)
        cols = column if many else [column]
        if name is None:
            if many:
                raise TypeError("vary of several columns: give name=")
            name = _key(column)
        if not tags:
            raise TypeError(f"vary {name}: give the alternatives as tags, e.g. up=..., down=...")
        for tag, alts in tags.items():
            if not many:
                alts = [alts]
            elif not isinstance(alts, list) or len(alts) != len(cols):
                raise TypeError(f"vary {name}:{tag}: give a list of {len(cols)} alternatives, one per column")
            for col, alt in zip(cols, alts):
                key = _key(col)
                target = self[col]
                if isinstance(alt, str):
                    alt = self[alt] if self._has(alt) else self.eval(alt)
                if isinstance(target, Collection) or isinstance(alt, Collection):
                    raise TypeError("vary a collection field ((\"Jet\", \"pt\") or Jet_pt), not the collection")
                if alt.space is not target.space:
                    raise ValueError(f"vary {name}:{tag}: {key} and its alternative are in different selections")
                if not isinstance(target.node, Source) and key in self._columns and not target.node.variations:
                    target = self._columns[key] = _own_node(target)  # hash-consed: shared with identical expressions
                    self._resolved.pop(key, None)
                variation = f"{name}:{tag}"
                previous = target.node.variations.get(variation)
                if previous is not None and previous is not alt.node:
                    raise ValueError(f"variation {variation} of {key} is already registered (with another alternative)")
                target.node.variations[variation] = alt.node
        return self

    def _has(self, name):
        try:
            self._lookup(name)
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------ results
    def _expr(self, x):
        if isinstance(x, str):
            x = self[x] if self._has(x) else self.eval(x)
        if isinstance(x, Collection):
            raise TypeError("histogram a field of the collection (jets.pt), not the collection")
        if not isinstance(x, Expr):
            raise TypeError(f"expected a column, got {type(x).__name__}")
        if x.space is not self._space:
            raise ValueError("a result must use the columns of the frame it is booked on")
        return x

    def hist(self, x, bins=None, range=None, weight=None, edges=None, axes=None, name=None):
        """a weighted histogram (hist.Hist with Weight storage) of a column (name, expression string or lazy column).
        Binning: hist("mll", (80, 70, 150)), hist("mll", [70, 110, 150]) (edges), hist("mll", bins=80, range=(70, 150))
        or an axis (aks.Regular, aks.Variable, aks.IntCategory, hist axes). A list of columns makes an n-dim histogram,
        with one axis per column in `axes` ((n, lo, hi), a list of edges, or an axis).
        Jagged columns fill one entry per object."""
        xs = x if isinstance(x, (list, tuple)) and axes is not None else [x]
        cols = [self._expr(c) for c in xs]
        if axes is None:
            if isinstance(bins, tuple) and len(bins) == 3:
                axes = [res.Regular(*bins, name=_name(xs[0]))]
            elif isinstance(bins, (list, np.ndarray)):
                axes = [res.Variable(list(bins), name=_name(xs[0]))]
            elif isinstance(bins, res.AXES) or hasattr(bins, "edges"):
                axes = [res.make_axis(axis=bins)]
            else:
                axes = [res.make_axis(bins, range, edges, name=_name(xs[0]))]
        else:
            axes = [res.Regular(*a, name=_name(c)) if isinstance(a, tuple)
                    else res.Variable(list(a), name=_name(c)) if isinstance(a, (list, np.ndarray))
                    else res.make_axis(axis=a) for a, c in zip(axes, xs)]
        if len(axes) != len(cols):
            raise ValueError("one axis per column")
        w = self._expr(weight) if weight is not None else None
        return res.Hist(self, cols, axes, w, name=name)

    def hists(self, specs, weight=None, varied=False):
        """several histograms at once: {name: binning} (the column of the same name) or {name: (column, binning)};
        returns {name: result} (with varied=True, each with its variations, as aks.varied)."""
        from .engine import varied as _varied
        out = {}
        for k, spec in specs.items():
            if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], (str, Expr)):
                column, binning = spec
            else:
                column, binning = k, spec
            h = self.hist(column, binning, weight=weight, name=k)
            out[k] = _varied(h) if varied else h
        return out

    def sum(self, column=None, weight=None):
        return res.Sum(self, self._expr(column) if column is not None else None,
                       self._expr(weight) if weight is not None else None)

    def count(self, *args, **kwargs):
        """the number of rows (entries passing the selections); a sum of weights is df.sum("weight")."""
        if args or kwargs:
            raise TypeError('count() counts the rows and takes no argument: for a sum of weights use df.sum("weight")')
        return res.Sum(self, None, None)

    def collect(self, columns=None):
        if columns is None:
            columns = getattr(self, "_projection", None) or self.columns
            if not columns:
                raise ValueError("collect: no defined columns; pass columns=[...]")
        return res.Collect(self, columns)

    def to_root(self, path, columns=None, tree="Events", basket_entries=200_000, compression=None):
        """write columns to a TTree: names of columns or collections (nX + X_<field>), or ("X", [fields]) for a
        collection with only some of its fields."""
        if columns is None:
            columns = getattr(self, "_projection", None) or self.columns
        return res.ToRoot(self, path, columns, tree, basket_entries, compression)

    # ------------------------------------------------------------ conversions (computed now, every row in memory)
    def to_awkward(self, columns=None):
        """the rows of this frame as an awkward record array (a collection is a field of jagged records); computed now.
        aks.from_arrays(...) turns it back into a frame."""
        from .engine import compute
        out, = compute(self.collect(columns))
        return out

    def to_numpy(self, columns=None, dtype=None):
        """the rows of this frame as a 2-d numpy array (rows x columns, in the order given; per-row columns only, the
        common dtype or `dtype`); computed now. For a scaler, a training matrix, sklearn."""
        names = self._conversion_columns(columns)
        values = self.to_awkward(names)
        cols = [_flat_numpy(values[c], c, "to_numpy") for c in names]
        dtype = dtype or (np.result_type(*cols) if cols else np.float64)
        return np.column_stack([c.astype(dtype, copy=False) for c in cols]) if cols else np.empty((0, 0), dtype)

    def to_pandas(self, columns=None):
        """the rows of this frame as a pandas DataFrame; computed now. Per-row columns only: one DataFrame row per row
        of the frame. With jagged columns or collections: awkward's ak.to_dataframe layout (one row per object, a
        (entry, subentry) MultiIndex, the per-row columns repeated)."""
        import pandas as pd
        names = self._conversion_columns(columns)
        values = self.to_awkward(names)
        if all(values[c].ndim == 1 and not values[c].fields for c in names):
            return pd.DataFrame({c: _flat_numpy(values[c], c, "to_pandas") for c in names})
        return ak.to_dataframe(values)

    def _conversion_columns(self, columns):
        if columns is None:
            columns = getattr(self, "_projection", None) or self.columns
            if not columns:
                raise ValueError("no defined columns on this frame: pass columns=[...]")
        return list(columns)

    def head(self, n=5, columns=None):
        """the rows of the selection among the first n entries of the input (at most n rows; runs immediately, on the
        first chunk(s) only)."""
        from .engine import compute
        r = self.collect(columns)
        out, = compute(r, max_entries=n)
        return out[:n]

    def peek(self, expr, n=1000):
        """the value of an expression (or column name) on the rows of this frame among the first n entries of the
        input (the selections apply after: fewer than n rows when they cut), computed now: to look at a column, or at
        the fields of a record returned by aks.apply."""
        from .engine import compute
        r = res.Collect(self, [expr], names=["value"])
        out, = compute(r, max_entries=n)
        return out["value"]

    def groupby(self, key):
        return GroupBy(self, self._expr(key))

    def __repr__(self):
        where = "root" if self._space.is_root else f"selection {self._space.id}"
        source = "in-memory arrays" if self._dataset.tree is None else f"{len(self._dataset.files)} files"
        return f"Frame({source}, {where}, columns={self.columns})"


class _ChunkFunction:
    """fn(chunk dict, **kwargs) called with positional column values (what aks.apply passes)."""

    def __init__(self, fn, keys, kwargs, name):
        self.fn, self.keys, self.kwargs = fn, keys, kwargs
        self.__name__ = name

    def __call__(self, *values):
        return self.fn(dict(zip(self.keys, values)), **self.kwargs)


class GroupBy:
    def __init__(self, frame, key):
        self.frame, self.key = frame, key

    def sum(self, column=None, weight=None):
        f = self.frame
        return res.GroupSum(f, self.key, f._expr(column) if column is not None else None,
                            f._expr(weight) if weight is not None else None)

    def count(self):
        return res.GroupSum(self.frame, self.key, None, None)


def _selection_mask(mask):
    """a selection mask with None (an option array: a comparison with a missing value) as numpy bool, None failing;
    anything else as it is (checked when the rows are taken)."""
    if isinstance(mask, ak.Array) and mask.ndim == 1 and mask.layout.is_option:
        return np.asarray(ak.fill_none(mask, False), bool)
    return mask


def _from_cpp(expr):
    """C++ syntax -> Python: &&, ||, !, true, false, std:: functions, and the conditional c ? a : b
    (-> where(c, a, b))."""
    import re
    s = re.sub(r"\bstd::", "", expr).replace("&&", " and ").replace("||", " or ")
    s = re.sub(r"!(?!=)", " not ", s)
    s = re.sub(r"\btrue\b", "True", s)
    s = re.sub(r"\bfalse\b", "False", s).strip()
    return _ternary(s) if "?" in s else s


_OPEN, _CLOSE = "([{", ")]}"


def _closing(s, i):
    """index of the bracket closing the one at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] in _OPEN:
            depth += 1
        elif s[j] in _CLOSE:
            depth -= 1
            if depth == 0:
                return j
    raise SyntaxError(f"unbalanced brackets in {s!r}")


def _top_level(s, chars):
    """positions of `chars` outside brackets (a C++ scope '::' is not a ':')."""
    out, depth = [], 0
    for j, c in enumerate(s):
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif depth == 0 and c in chars and not (c == ":" and ("::" in s[max(j - 1, 0):j + 2])):
            out.append(j)
    return out


def _ternary(s):
    """c ? a : b -> where(c, a, b): the lowest-precedence operator, right-associative, inside brackets and function
    arguments too."""
    parts, j = [], 0
    while j < len(s):  # bracket contents first, argument by argument
        if s[j] in _OPEN:
            k = _closing(s, j)
            inner = s[j + 1:k]
            cuts = [-1] + _top_level(inner, ",") + [len(inner)]
            args = [_ternary(inner[a + 1:b]) for a, b in zip(cuts, cuts[1:])]
            parts.append(s[j] + ",".join(args) + s[k])
            j = k + 1
        else:
            parts.append(s[j])
            j += 1
    s = "".join(parts)
    marks = _top_level(s, "?:")
    if not marks or s[marks[0]] != "?":
        return s
    nest = 0
    for m in marks[1:]:  # the ':' of the first '?' (nested conditionals in the middle operand pair up first)
        if s[m] == "?":
            nest += 1
        elif nest:
            nest -= 1
        else:
            cond, a, b = s[:marks[0]], s[marks[0] + 1:m], s[m + 1:]
            return f"where({cond.strip()}, {_ternary(a).strip()}, {_ternary(b).strip()})"
    raise SyntaxError(f"'?' without ':' in {s!r}")


def _constant(ref, value):
    return np.full(len(ref), value)


def _flat_numpy(v, name, what):
    """one value per row as numpy, or a TypeError naming the column."""
    if isinstance(v, ak.Array):
        if v.ndim > 1 or v.fields:
            raise TypeError(f"{what}: column {name!r} has lists or records ({v.type}): use to_awkward, or a per-row "
                            f"column (aks.num(...), aks.sum(..., axis=1), a field of one object)")
        v = ak.to_numpy(v)
    return np.asarray(v)


def _as_field(value, counts, label):
    """a value for a collection field: jagged with the collection's object counts, or broadcast from a scalar or one
    value per row."""
    counts = np.asarray(counts)
    if isinstance(value, ak.Array) and value.ndim > 1:
        n = np.asarray(ak.num(value, axis=1))
        if not np.array_equal(n, counts):
            bad = np.flatnonzero(n != counts)
            raise ValueError(f"{label}: {len(bad)} rows with another number of objects than the collection (first: row "
                             f"{bad[0]}, {n[bad[0]]} vs {counts[bad[0]]})")
        return value
    from .jagged import repeat, wrap
    offsets = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    if np.ndim(value) == 0:
        return wrap(offsets, np.full(offsets[-1], value))
    return wrap(offsets, repeat(value, counts))


def _is_field_of(value, name, frame):
    """the value already has the collection's objects: one of its branches, or a field column of this frame."""
    node = value.node
    if isinstance(node, Source):
        return node.branch.startswith(name + "_") and name in frame._dataset.collections
    prefix = name + "_"
    return any(c.startswith(prefix) and frame._lookup(c).node is node for c in frame.columns)


def _identity(x):
    return x


def _key(column):
    """the column name of a column key: ("Jet", "pt") -> "Jet_pt"."""
    return "_".join(column) if isinstance(column, tuple) else column


def _own_node(expr):
    """a node of its own for a column (not hash-consed), to attach variations to it alone."""
    return Expr(Call(expr.space, _identity, (expr.node,), {}, name="vary"))


class FrameCollection(Collection):
    """collection `name` of a frame: a view of the frame's columns name_<field> (see the module docstring)."""

    def __init__(self, frame, name):
        counter = frame._counter(name)
        super().__init__(frame._space, [], None, (lambda: frame._lookup(counter)) if counter else None, name)
        self._frame = frame

    @property
    def fields(self):
        return self._frame._fields_of(self._name)

    def field(self, name):
        if name not in self.fields:
            raise AttributeError(f"collection {self._name} has no field {name!r} (fields: {self.fields})")
        return self._frame._lookup(f"{self._name}_{name}")

    def __setitem__(self, name, value):
        self._frame._set_field(self._name, name, value)

    def __repr__(self):
        return f"Collection({self._name}: {', '.join(self.fields)})"


def _name(x):
    return x if isinstance(x, str) else None


class _Namespace(dict):
    """eval namespace: names resolve to the frame's columns."""

    def __init__(self, frame, extra):
        super().__init__(extra)
        self.frame = frame

    def __missing__(self, name):
        return self.frame[name]


class _DoubleLiterals(ast.NodeTransformer):
    """float literals as float64 scalars: with a float32 column they promote to float64, as in C++ (a Python float is a
    "weak" scalar for NumPy 2 and keeps the column's float32)."""

    def visit_Constant(self, node):
        if type(node.value) is float:
            return ast.Call(func=ast.Name(id="_f64", ctx=ast.Load()), args=[node], keywords=[])
        return node


class _BoolRewriter(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        out = node.values[0]
        for v in node.values[1:]:
            out = ast.BinOp(left=out, op=op, right=v)
        return out

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.UnaryOp(op=ast.Invert(), operand=node.operand)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        parts, left = [], node.left
        for op, right in zip(node.ops, node.comparators):
            parts.append(ast.Compare(left=left, ops=[op], comparators=[right]))
            left = right
        out = parts[0]
        for p in parts[1:]:
            out = ast.BinOp(left=out, op=ast.BitAnd(), right=p)
        return out
