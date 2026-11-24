"""akstream: streaming, pandas-like analysis of ROOT trees with awkward arrays.

    import akstream as aks

    df = aks.read_root("skims/*.root", "Events", step=200_000)
    df["ht"] = aks.sum(df.Jet.pt[df.Jet.pt > 25], axis=1)
    df.vary_branches(["Jet_pt", "Jet_mass"], ["jer_4", "jes_Total"])
    sel = df.filter("mll > 110 && mll < 150")
    h = aks.varied(sel.hist("ht", bins=50, range=(0, 500), weight="genWeight"))
    aks.compute(h, workers=8)            # {"nominal": Hist, "jer_4:up": Hist, ...}

Every awkward function is available lazily as aks.<name> (aks.sum, aks.num, aks.firsts, aks.pad_none, ...); the ones
that act on each field (pad_none, fill_none, firsts, ...) also take a collection and return one. akstream.hep kernels
take lazy columns too (hep.max_mass_pair(df.Jet), hep.pick(df.Jet.pt, idx, 0)).
"""
import awkward as _ak
import numpy as _np

from .engine import compute, last_failed, last_stats, varied
from .expr import Collection, Expr, call, zip_record
from .frame import Frame
from .io import ArraysDataset, Dataset, Friend
from .results import Regular, Variable
from . import vector as _vector

__all__ = ["read_root", "from_arrays", "compute", "varied", "apply", "onnx", "checkpoint", "checked", "Frame", "Expr", "Collection",
           "Friend", "Regular", "Variable", "last_stats", "last_failed", "zip", "p4", "collins_soper", "delta_r"]


def read_root(files, tree="Events", step=200_000, entry_start=0, entry_stop=None, threads=0, friends=None,
              rdf_compatible=False, remote_strategy=0, retries=3, retry_delay=10.0):
    """a Frame over the TTree `tree` of `files` (a path, a glob, a list, or 'file.root:tree').

    step     entries per chunk (memory scales with it)
    threads  uproot decompression threads per process
    friends  {tree name: files or aks.Friend(files, match={friend branch: main branch})}: friend trees
             (TTree::AddFriend), one file per main file in the same order, with the same entries; their branches are
             columns too; `match` branches are checked to agree entry by entry
    rdf_compatible  evaluate as C++ / RDataFrame code does, to reproduce it bit for bit: float literals in expression
             strings are doubles ("x > 0.02", "0.5 * x" on a float32 x are evaluated in double; NumPy's default keeps
             the column's float32), and 4-vectors / hep.max_mass_pair use ROOT's GenVector PtEtaPhiMVector arithmetic
             (vector.RootP4) instead of plain cartesian sums. Results differ only in the last bits.
    remote_strategy  what to do with a chunk (or file) that cannot be read, e.g. an xrootd timeout:
             0 / "raise"  raise at the first one (default)
             1 / "skip"   skip it and go on
             2 / "retry"  go on with the other chunks, then retry it: `retries` rounds, `retry_delay` seconds apart;
                          still failing: skipped
             The skipped chunks are in aks.last_failed after the run, as records {"file", "start", "stop", "error",
             ...}: read_root(aks.last_failed, ...) is a frame over exactly those chunks (to process them later and add
             the results). The results of a run with skipped chunks are those of the other chunks; with "retry", the
             rows of a retried chunk come after the others in to_root / collect.
    `files` may also be such a list of records.
    """
    ds = Dataset(files, tree, step, entry_start, entry_stop, threads, friends, rdf_compatible, remote_strategy, retries,
                 retry_delay)
    return Frame(ds, ds.root)


def from_arrays(data, step=200_000, rdf_compatible=False):
    """a Frame over arrays in memory: a dict {name: array} (numpy, or awkward for lists), an awkward record array, or a
    pandas DataFrame; one entry per row. Record fields become name_field columns, so df.to_awkward(["Jet", ...]) comes
    back with its collections. Everything works as on a file (vary, hist, to_root, workers); `step` rows per chunk."""
    ds = ArraysDataset(data, step, rdf_compatible)
    return Frame(ds, ds.root)


def apply(fn, *args, name=None, when=None, **kwargs):
    """a row-local user function of lazy columns: fn(*values, **kwargs) must return one value per input row (an array,
    or a dict of arrays: then e["name"] is one of its columns) and row i must depend only on row i of the inputs. Under
    a variation it is recomputed only on the rows whose inputs really changed (it is treated as expensive).

    when: optional list of expressions that decide, row by row, whether the output can change (e.g. the selected jets,
    for a function of all the jets that only uses the selected ones): under a variation fn is recomputed only where one
    of them changed, and still receives `args`. A promise about fn; compute(validate=...) checks it."""
    out = call(fn, *args, _name=name or getattr(fn, "__name__", "apply"), _expensive=True, **kwargs)
    if when is not None:
        triggers = [w.node for w in when]
        if any(t.space is not out.node.space for t in triggers):
            raise ValueError("apply(when=...): the trigger columns must be in the same selection as the inputs")
        out.node.set_trigger(triggers)
    return out


def _lazy(*args):
    from .graph import _nodes_in
    return any(True for _ in _nodes_in(_unwrap_all(args)))


def _unwrap_all(args):
    from .expr import _unwrap
    return _unwrap(args)


def p4(pt, eta, phi, mass, rdf_compatible=None):
    """Lorentz vectors from (pt, eta, phi, mass): a column of akstream.vector.P4 (lazy) for frame columns, a P4 for
    arrays. (mu1 + mu2).mass, .pt, .eta, .phi, .px ... are columns as well. rdf_compatible: ROOT's PtEtaPhiMVector
    arithmetic (vector.RootP4), bit for bit as RDataFrame code (default: the read_root setting for columns, else
    False)."""
    if _lazy(pt, eta, phi, mass):
        if rdf_compatible is None:
            rdf_compatible = _dataset_of(pt, eta, phi, mass).rdf_compatible
        return call(_vector.p4, pt, eta, phi, mass, rdf_compatible=rdf_compatible, _name="p4")
    return _vector.p4(pt, eta, phi, mass, rdf_compatible=bool(rdf_compatible))


def _dataset_of(*args):
    from .expr import _space_of
    return _space_of(_unwrap_all(args), {}).dataset


def _collins_soper(l1, l2, charge1):
    phi, cos_theta = _vector.collins_soper(l1, l2, charge1)
    return {"phi": phi, "cos_theta": cos_theta}


def collins_soper(l1, l2, charge1):
    """(phi_CS, cos theta_CS) of a lepton pair (l1 is the positive lepton where charge1 > 0)."""
    if _lazy(l1, l2, charge1):
        cs = call(_collins_soper, l1, l2, charge1, _name="collins_soper")
        return cs["phi"], cs["cos_theta"]
    return _vector.collins_soper(l1, l2, charge1)


def delta_r(a, b):
    """Delta R of two 4-vector columns."""
    if _lazy(a, b):
        return call(_vector.delta_r, a, b, _name="delta_r")
    return _vector.delta_r(a, b)


def _checked(value, condition, message):
    ok = _np.asarray(condition)
    if not ok.all():
        bad = int(_np.flatnonzero(~ok)[0])
        raise ValueError(f"{message} ({int((~ok).sum())} rows, the first at row {bad} of the chunk)")
    return value


def checked(value, condition, message):
    """value, after checking that `condition` holds on every row (else ValueError(message) when it is computed)."""
    return call(_checked, value, condition, message=message, _name="checked")


def checkpoint(expr):
    """mark a column as worth comparing with nominal after a variation (rows whose value did not change are not
    propagated downstream)."""
    expr.node.cutoff = True
    return expr


class OnnxModel:
    """an onnxruntime model applied to stacked float32 inputs; the session is created lazily in each process."""

    def __init__(self, path, input_name=None, output=0, threads=1, transform=None):
        self.path, self.input_name, self.output, self.threads = path, input_name, output, threads
        self.transform = transform
        self._session = None
        self._pid = None
        self.__name__ = f"onnx:{path}"

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_session"} | {"_session": None}

    def session(self):
        import os
        if self._session is None or self._pid != os.getpid():  # never reuse a session across fork
            import onnxruntime as ort
            self._pid = os.getpid()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.threads
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(self.path, opts)
        return self._session

    def __call__(self, *columns):
        import numpy as np
        x = np.stack([np.asarray(c, np.float32) for c in columns], axis=1)
        if self.transform is not None:
            x = self.transform(x)
        s = self.session()
        name = self.input_name or s.get_inputs()[0].name
        outputs = s.run(None, {name: x})
        out = outputs[self.output] if isinstance(self.output, int) else \
            outputs[[o.name for o in s.get_outputs()].index(self.output)]
        return out.reshape(len(x), -1)[:, 0] if out.ndim > 1 and out.shape[1] == 1 else out


def onnx(path, inputs, input_name=None, output=0, threads=1, transform=None):
    """score an ONNX model on the listed input columns (stacked as float32 in order)."""
    model = OnnxModel(path, input_name, output, threads, transform)
    return apply(model, *inputs, name=f"onnx:{path.rsplit('/', 1)[-1]}")


def zip(fields, counts=None):  # noqa: A001 (mirrors ak.zip)
    """lazy ak.zip of columns: a Collection whose fields stay separate until a record is really needed."""
    if all(isinstance(v, Expr) for v in fields.values()):
        return zip_record(fields, counts)
    return call(_ak.zip, fields, _name="zip")


_wrapped = {}
# awkward functions applied field by field when given a collection (the result is a collection)
_FIELDWISE = {"pad_none", "fill_none", "firsts", "singletons", "drop_none", "mask", "is_none", "values_astype"}
# awkward functions that mix rows when they act on the outermost axis (axis=None, 0, or -1 of a flat array)
_AXIS_FUNCTIONS = {"sum", "prod", "any", "all", "count", "count_nonzero", "min", "max", "argmin", "argmax", "mean",
                   "std", "var", "moment", "ptp", "softmax", "sort", "argsort", "local_index", "run_lengths",
                   "combinations", "argcombinations", "cartesian", "argcartesian", "flatten", "ravel", "cumsum",
                   "linear_fit", "corr", "covar", "nansum", "nanprod", "nanmin", "nanmax", "nanargmin", "nanargmax",
                   "nanmean", "nanstd", "nanvar", "firsts", "singletons", "pad_none", "num", "unflatten"}


class _AxisChecked:
    """ak.<fn> that refuses to act across rows (the outermost axis) at run time."""

    def __init__(self, name, fn):
        import inspect
        self.name, self.fn = name, fn
        self.__name__ = name
        self._sig = inspect.signature(fn)

    def __call__(self, *args, **kwargs):
        bound = self._sig.bind_partial(*args, **kwargs)
        axis = bound.arguments.get("axis", self._sig.parameters["axis"].default)
        first = next((a for a in bound.args if hasattr(a, "ndim")), None)
        ndim = first.ndim if first is not None else 1
        if axis is None or (axis if axis >= 0 else ndim + axis) == 0:
            from .execute import NonLocalError
            raise NonLocalError(f"aks.{self.name}(axis={axis}) on a {ndim}-dimensional column acts across rows: use "
                                f"axis=1 (per row), or a result (df.sum, df.hist) for totals")
        return self.fn(*args, **kwargs)


def _fieldwise(fn, coll, args, kwargs):
    """fn on every field of a collection: a collection (counts from its first field when computed)."""
    fields = coll.fields
    return Collection(coll.space, fields, lambda f: fn(coll.field(f), *args, **kwargs), None, getattr(coll, "_name", None))


def __getattr__(name):
    """aks.<awkward function>: the lazy version."""
    fn = getattr(_ak, name, None)
    if fn is None or not callable(fn) or name.startswith("_"):
        raise AttributeError(f"module 'akstream' has no attribute {name!r}")
    w = _wrapped.get(name)
    if w is None:
        import inspect
        target = fn
        if name in _AXIS_FUNCTIONS and "axis" in inspect.signature(fn).parameters:
            target = _AxisChecked(name, fn)

        def w(*args, **kwargs):
            if name in _FIELDWISE and args and isinstance(args[0], Collection):
                return _fieldwise(w, args[0], args[1:], kwargs)
            return call(target, *args, _name=name, **kwargs)
        w.__name__ = name
        w.__doc__ = f"lazy ak.{name}\n\n" + (fn.__doc__ or "")
        _wrapped[name] = w
    return w
