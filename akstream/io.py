"""Input: a set of ROOT files with one TTree, read in chunks with uproot.

Per chunk, the branches the plan needs are read in two bulk calls per tree (flat branches as numpy, jagged ones as
awkward). Anything touched lazily (a field of a materialised record) is read on demand and remembered, so that from the
next chunk on it is part of the bulk read too.

Friend trees (as TTree::AddFriend): trees in other files with the same entries as the main one, entry by entry; their
branches are columns like the main tree's. Friend(files, match={"gj_event": "event"}) also checks, chunk by chunk, that
the friend branch equals the main-tree one. ENTRY ("rdfentry_") is the entry number in the file.

Unreadable chunks (remote_strategy): 0 raises at the first one; 1 skips it; 2 processes the other chunks first and
retries it at the end (`retries` rounds, `retry_delay` seconds apart), then skips it. Skipped chunks are reported as
records {"file", "start", "stop", "error", "tree", "friends", "match"} (stop None: the whole file, which could not be
opened); a list of them is a valid `files` argument: read_root(failed) processes exactly those chunks.

Collections: the jagged branches X_<field> of a common prefix X (NanoAOD's Jet_pt, Jet_eta, ...), with the counter nX
when the tree has one.
"""
import glob
import os
import re
import threading

import numpy as np
import uproot

from .graph import Space

ENTRY = "rdfentry_"  # the pseudo-branch of the entry number (in its file)


def expand(files):
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    out = []
    for f in files:
        f = os.path.expanduser(str(f))
        hits = sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f]
        if not hits:
            raise FileNotFoundError(f)
        out.extend(hits)
    return out


def _split_tree(path, tree):
    """'file.root:Events' -> ('file.root', 'Events')."""
    m = re.match(r"^(.*\.root):(.+)$", path)
    if m:
        return m.group(1), m.group(2)
    return path, tree


STRATEGIES = {0: 0, 1: 1, 2: 2, "raise": 0, "skip": 1, "retry": 2}


def _is_records(files):
    return isinstance(files, (list, tuple)) and len(files) > 0 and all(isinstance(f, dict) for f in files)


def error_text(err):
    return f"{type(err).__name__}: {err}"[:500]


class Friend:
    """a friend tree's files (one per main file, same order and entries) and the branches that must agree entry by entry
    with the main tree's: match={friend branch: main branch}, checked on every chunk that reads the friend."""

    def __init__(self, files, match=None):
        self.files, self.match = files, dict(match or {})


class Dataset:
    """files + tree + chunking + the branch schema. Owns the graph cache of every frame built on it."""

    def __init__(self, files, tree="Events", step=200_000, entry_start=0, entry_stop=None, threads=0, friends=None,
                 rdf_compatible=False, remote_strategy=0, retries=3, retry_delay=10.0):
        if remote_strategy not in STRATEGIES:
            raise ValueError(f"remote_strategy: 0 / 'raise', 1 / 'skip' or 2 / 'retry', not {remote_strategy!r}")
        self.remote_strategy = STRATEGIES[remote_strategy]
        self.retries, self.retry_delay = int(retries), float(retry_delay)
        self._ranges = {}  # file -> [(start, stop or None)]: only these entries (files given as failed-chunk records)
        if _is_records(files):
            files, tree, friends = self._from_records(files, tree, friends)
        if isinstance(files, str):
            files, tree = _split_tree(files, tree)
        self.files = expand(files)
        self.tree = tree
        self.open_failures = []  # records of the files that could not be opened (remote_strategy 1 / 2)
        self.rdf_compatible = rdf_compatible
        # friends: {tree name: files | Friend, one file per main file (same order)} -> {tree name: {main file: friend file}}
        self.friends, self.friend_match = {}, {}
        for name, ffiles in (friends or {}).items():
            if isinstance(ffiles, Friend):
                if ffiles.match:
                    self.friend_match[name] = ffiles.match
                ffiles = ffiles.files
            ffiles = expand(ffiles)
            if len(ffiles) != len(self.files):
                raise ValueError(f"friend tree {name}: {len(ffiles)} files for {len(self.files)} main files")
            self.friends[name] = dict(zip(self.files, ffiles))
        self.source_tree = {}  # branch -> friend tree name (main-tree branches are absent)
        self.step = int(step)
        self.entry_start, self.entry_stop = entry_start, entry_stop
        self.threads = threads
        self.cache, self.keep = {}, []
        self.booked = []
        self.root = Space(self)
        self._entries = {}
        # the first readable file stays open: its TTree metadata is the schema and the first chunks read from it
        self.branches, self.dtypes = {}, {}
        for path in self.files:
            try:
                f = uproot.open(path)
                t = f[self.tree]
                friend_trees = {name: uproot.open(paths[path]) for name, paths in self.friends.items()}
                schema = [(t, None)] + [(ff[name], name) for name, ff in friend_trees.items()]
            except Exception as err:  # noqa: BLE001 (any read error: xrootd, deserialization, a missing file)
                if self.remote_strategy == 0 or path == self.files[-1]:
                    raise
                self._open_failed(path, err)
                continue
            break
        self._open = (os.getpid(), path, f, t)
        self._entries[path] = t.num_entries
        for tree_, name in schema:
            self._add_schema(tree_, name)
        for ff in friend_trees.values():
            ff.close()
        self.branches[ENTRY] = False
        self.dtypes[ENTRY] = np.dtype(np.int64)
        for name, match in self.friend_match.items():
            for fb, mb in match.items():
                if self.source_tree.get(fb) != name or mb not in self.branches or mb in self.source_tree:
                    raise ValueError(f"friend tree {name}: match {fb!r} -> {mb!r} needs a branch of the friend and one of "
                                     f"the main tree")
        self.collections, self.counters = self._collections()

    def _add_schema(self, t, friend):
        for name in t.keys():
            if name in self.branches:
                raise ValueError(f"branch {name} of friend tree {friend} is also in {self.source_tree.get(name, self.tree)}")
            interp = t[name].interpretation
            jagged = isinstance(interp, uproot.interpretation.jagged.AsJagged)
            self.branches[name] = jagged
            inner = interp.content if jagged else interp
            self.dtypes[name] = getattr(inner, "to_dtype", None)
            if friend is not None:
                self.source_tree[name] = friend

    def _collections(self):
        """({X: [fields]}, {X: counter branch or None}) for the jagged X_<field> branches of each prefix X."""
        out = {}
        for name, jagged in self.branches.items():
            if not jagged or "_" not in name:
                continue
            prefix, field = name.split("_", 1)
            if prefix and field and prefix not in self.branches:
                out.setdefault(prefix, []).append(field)
        counters = {c: (f"n{c}" if f"n{c}" in self.branches and not self.branches[f"n{c}"] else None) for c in out}
        return out, counters

    def chunks(self):
        """[(file, start, stop)] over every file, `step` entries each (the global entry range applied to the first
        file only when there is one file, as uproot.iterate; only the given ranges for failed-chunk records). A file
        that cannot be opened: raised (remote_strategy 0), or retried (2) and recorded in open_failures."""
        out = []
        failed = {r["file"] for r in self.open_failures}
        for path in self.files:
            if path in failed:
                continue
            n = self._entries.get(path)
            if n is None:
                n = self._num_entries(path)
                if n is None:
                    continue
            if path in self._ranges:
                ranges = [(s, n if e is None else min(e, n)) for s, e in self._ranges[path]]
            elif len(self.files) == 1:
                ranges = [(self.entry_start or 0, n if self.entry_stop is None else min(self.entry_stop, n))]
            else:
                ranges = [(0, n)]
            for start, stop in ranges:
                for s in range(start, stop, self.step):
                    out.append((path, s, min(s + self.step, stop)))
        return out

    def _num_entries(self, path):
        attempts = 1 + (self.retries if self.remote_strategy == 2 else 0)
        for attempt in range(attempts):
            try:
                with uproot.open(path) as f:
                    n = self._entries[path] = f[self.tree].num_entries
                return n
            except Exception as err:  # noqa: BLE001
                if self.remote_strategy == 0:
                    raise
                if attempt + 1 < attempts:
                    import time
                    time.sleep(self.retry_delay)
                    continue
                self._open_failed(path, err)
        return None

    def _open_failed(self, path, err):
        import warnings
        warnings.warn(f"akstream: {path} could not be opened ({error_text(err)}): skipped", RuntimeWarning, stacklevel=3)
        self.open_failures.append(self.record(path, 0, None, err))

    def record(self, path, start, stop, err):
        """a failed chunk, as read_root accepts it back."""
        return {"file": path, "start": int(start), "stop": None if stop is None else int(stop), "error": error_text(err),
                "tree": self.tree, "friends": {name: paths[path] for name, paths in self.friends.items()},
                "match": {name: dict(m) for name, m in self.friend_match.items()}}

    def _from_records(self, records, tree, friends):
        """files, tree and friends of a list of failed-chunk records (the ranges go to self._ranges)."""
        files, ffiles, match = [], {}, {}
        for r in records:
            if r["file"] not in self._ranges:
                files.append(r["file"])
                for name, p in (r.get("friends") or {}).items():
                    ffiles.setdefault(name, []).append(p)
                match.update(r.get("match") or {})
            self._ranges.setdefault(r["file"], []).append((int(r.get("start", 0)), r.get("stop")))
            tree = r.get("tree") or tree
        if friends is None and ffiles:
            friends = {name: Friend(p, match.get(name)) for name, p in ffiles.items()}
        return files, tree, friends

    def reader(self, prefetch):
        """the per-worker reader of this dataset."""
        return Reader(self, prefetch)

    def open_tree(self, path):
        """the already open TTree of `path` (same process only), else None."""
        o = self._open
        if o is not None and o[0] == os.getpid() and o[1] == path:
            return o[3]
        return None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["booked"] = []
        state["_open"] = None
        return state


class Reader:
    """per-worker reader: keeps the open files, reads one chunk at a time. Safe for one background read-ahead thread
    next to the main thread: the two most recent files stay open (the current chunk may still read lazily from its
    file while the next one is read), and the branch list is guarded by a lock."""

    def __init__(self, dataset, prefetch):
        self.dataset = dataset
        self.prefetch = set(prefetch)
        self.lazy_reads = set()
        self._lock = threading.Lock()
        self._open = {}  # path -> (file or None, tree), at most two
        self._executor = None
        if dataset.threads:
            self._executor = uproot.ThreadPoolExecutor(max_workers=dataset.threads)

    def open(self, path, friend=None):
        """the TTree of main file `path`, or of its friend tree `friend`."""
        ds = self.dataset
        with self._lock:
            key = (path, friend)
            hit = self._open.get(key)
            if hit is None:
                tree = ds.open_tree(path) if friend is None else None
                f = None
                if tree is None:
                    f = uproot.open(path if friend is None else ds.friends[friend][path])
                    tree = f[ds.tree if friend is None else friend]
                if friend is not None:
                    main = self._open.get((path, None))
                    n = main[1].num_entries if main is not None else ds._entries.get(path)
                    if n is not None and tree.num_entries != n:
                        raise ValueError(f"friend tree {friend} of {path}: {tree.num_entries} entries, the main tree {n}")
                hit = self._open[key] = (f, tree)
                while len(self._open) > 2 * (1 + len(ds.friends)):  # the two most recent files, with their friends
                    old = next(iter(self._open))
                    oldf, _ = self._open.pop(old)
                    if oldf is not None:
                        oldf.close()
            return hit[1]

    def forget(self, path):
        """drop the cached handles of a file after a read error (a retry opens it again); not closed here: the
        read-ahead thread may still hold it."""
        with self._lock:
            for key in [k for k in self._open if k[0] == path]:
                self._open.pop(key)
            if self.dataset._open is not None and self.dataset._open[1] == path:
                self.dataset._open = None

    def add_branch(self, branch):
        with self._lock:
            self.lazy_reads.add(branch)
            self.prefetch.add(branch)

    def read(self, path, start, stop):
        """the prefetched branches of one chunk: {branch: numpy or awkward}."""
        ds = self.dataset
        tree = self.open(path)
        with self._lock:
            wanted = sorted(self.prefetch)
        out = {}
        if ENTRY in wanted:
            out[ENTRY] = np.arange(start, stop, dtype=np.int64)
        by_tree = {}
        for b in wanted:
            if b != ENTRY:
                by_tree.setdefault(ds.source_tree.get(b), []).append(b)
        kw = {}
        if self._executor is not None:
            kw = dict(decompression_executor=self._executor, interpretation_executor=self._executor)
        for friend, names in by_tree.items():
            t = tree if friend is None else self.open(path, friend)
            present = set(t.keys())
            missing = [b for b in names if b not in present]
            if missing:
                raise KeyError(f"{path}: branches not in the tree {friend or ds.tree}: {missing[:10]}")
            flat = [b for b in names if not ds.branches.get(b, False)]
            jagged = [b for b in names if ds.branches.get(b, False)]
            if flat:
                out.update(t.arrays(flat, entry_start=start, entry_stop=stop, library="np", how=dict, **kw))
            if jagged:
                out.update(t.arrays(jagged, entry_start=start, entry_stop=stop, library="ak", how=dict, **kw))
        chunk = Chunk(self, path, tree, start, stop, out)
        for friend in by_tree:
            if friend is not None:
                chunk.check_friend(friend)
        return chunk


class ReadAhead:
    """iterate over the chunks with the next one read in a background thread while the current one is processed
    (decompression releases the GIL). depth=0: plain sequential reads; "auto": one chunk ahead when the process may run
    on at least two cores (on a single core the two threads only take turns, and two chunks are in memory)."""

    def __init__(self, reader, chunks, depth="auto"):
        if depth == "auto":  # a background read only pays off with a second core to run on
            try:
                depth = 1 if len(os.sched_getaffinity(0)) >= 2 else 0
            except AttributeError:  # not Linux
                depth = 1 if (os.cpu_count() or 1) >= 2 else 0
        self.reader, self.chunks, self.depth = reader, list(chunks), int(depth)

    def __iter__(self):
        if not self.depth or len(self.chunks) < 2:
            for c in self.chunks:
                yield c, (lambda c=c: self.reader.read(*c))
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(1, thread_name_prefix="akstream-read") as pool:
            pending = [pool.submit(self.reader.read, *c) for c in self.chunks[:self.depth]]
            for i, c in enumerate(self.chunks):
                future = pending.pop(0)
                nxt = i + self.depth
                if nxt < len(self.chunks):
                    pending.append(pool.submit(self.reader.read, *self.chunks[nxt]))
                yield c, future.result


class Chunk:
    def __init__(self, reader, path, tree, start, stop, arrays):
        self.reader, self.path, self.tree, self.start, self.stop = reader, path, tree, start, stop
        self.arrays = arrays
        self.n = stop - start
        self._checked = set()

    def check_friend(self, friend):
        """the Friend(match=...) branches of `friend` equal the main tree's on this chunk (once per chunk)."""
        if friend in self._checked:
            return
        self._checked.add(friend)
        for fb, mb in self.reader.dataset.friend_match.get(friend, {}).items():
            a, b = self.get(fb), self.get(mb)
            if not np.array_equal(a, b):
                bad = np.flatnonzero(np.asarray(a) != np.asarray(b))[0]
                raise ValueError(f"friend tree {friend} of {self.path} not aligned: {fb} != {mb} at entry "
                                 f"{self.start + bad} ({a[bad]} vs {b[bad]})")

    def get(self, branch):
        a = self.arrays.get(branch)
        if a is None:
            ds = self.reader.dataset
            jagged = ds.branches.get(branch)
            if jagged is None:
                raise KeyError(f"no branch {branch!r} in the tree")
            if branch == ENTRY:
                a = np.arange(self.start, self.stop, dtype=np.int64)
            else:
                friend = ds.source_tree.get(branch)
                t = self.tree if friend is None else self.reader.open(self.path, friend)
                a = t[branch].array(entry_start=self.start, entry_stop=self.stop, library="ak" if jagged else "np")
            self.arrays[branch] = a
            self.reader.add_branch(branch)
            friend = ds.source_tree.get(branch)
            if friend is not None:
                self.check_friend(friend)
        return a


# ---------------------------------------------------------------- in-memory arrays (aks.from_arrays)
ARRAYS = "<arrays>"  # the "file" of an in-memory dataset


def flatten_columns(data):
    """{name: one-dimensional numpy / awkward array} from a dict of arrays, an awkward record array or a pandas
    DataFrame. Record fields become name_field columns (one level: an awkward "Jet" field of jagged records gives
    Jet_pt, Jet_eta, ...), so that collections come back as collections."""
    import awkward as ak
    if hasattr(data, "columns") and hasattr(data, "index") and hasattr(data, "to_numpy"):  # pandas DataFrame
        if getattr(data.index, "nlevels", 1) > 1:
            raise ValueError("from_arrays: a pandas DataFrame with a MultiIndex (one row per object): give the per-row "
                             "table, or an awkward array for jagged columns")
        data = {str(c): data[c].to_numpy() for c in data.columns}
    elif isinstance(data, ak.Array):
        if not data.fields:
            raise ValueError("from_arrays: an awkward array of records (named fields) is needed")
        data = {f: data[f] for f in data.fields}
    out = {}
    for name, v in dict(data).items():
        if isinstance(v, ak.Array) and v.fields:  # a collection (jagged records) or a per-row record
            for f in v.fields:
                out[f"{name}_{f}"] = v[f]
            continue
        if not isinstance(v, ak.Array):
            v = np.asarray(v)
            if v.ndim != 1:
                raise ValueError(f"from_arrays: column {name!r} has shape {v.shape}: one value per row is needed "
                                 f"(split the columns, or pass an awkward array for lists)")
        out[name] = v
    return out


def _content_dtype(v):
    import awkward as ak
    if isinstance(v, np.ndarray):
        return v.dtype
    layout = v.layout
    while hasattr(layout, "content"):
        layout = layout.content
    return getattr(layout, "dtype", None) if isinstance(layout, ak.contents.NumpyArray) else None


class ArraysDataset(Dataset):
    """a dataset over arrays in memory (one entry per row), read in chunks of `step` rows like a file."""

    def __init__(self, data, step=200_000, rdf_compatible=False):
        import awkward as ak
        columns = flatten_columns(data)
        if not columns:
            raise ValueError("from_arrays: no columns")
        lengths = {k: len(v) for k, v in columns.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"from_arrays: columns of different lengths {lengths}")
        for k, v in columns.items():
            dt = _content_dtype(v)
            if dt is None or dt.kind not in "biuf":
                kind = v.type if isinstance(v, ak.Array) else dt
                raise TypeError(f"from_arrays: column {k!r} is of type {kind}: only numbers, booleans and lists of "
                                f"them (string columns are planned, see docs/api.md)")
        self.files, self.tree = [ARRAYS], None
        self.rdf_compatible = rdf_compatible
        self.remote_strategy, self.retries, self.retry_delay = 0, 0, 0.0
        self._ranges, self.open_failures = {}, []
        self.friends, self.friend_match, self.source_tree = {}, {}, {}
        self.step = int(step)
        self.entry_start, self.entry_stop = 0, None
        self.threads = 0
        self.cache, self.keep, self.booked = {}, [], []
        self.root = Space(self)
        self.arrays = columns
        self.n = next(iter(lengths.values()))
        self._entries = {ARRAYS: self.n}
        self._open = None
        self.branches = {k: isinstance(v, ak.Array) and v.ndim > 1 for k, v in columns.items()}
        self.dtypes = {k: _content_dtype(v) for k, v in columns.items()}
        self.branches[ENTRY] = False
        self.dtypes[ENTRY] = np.dtype(np.int64)
        self.collections, self.counters = self._collections()

    def chunks(self):
        return [(ARRAYS, s, min(s + self.step, self.n)) for s in range(0, self.n, self.step)]

    def reader(self, prefetch):
        return ArraysReader(self)

    def open_tree(self, path):
        return None


class ArraysReader:
    """the reader of an ArraysDataset: a chunk is a slice of every array (views, no copy)."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.lazy_reads = set()

    def read(self, path, start, stop):
        return ArraysChunk(self, path, start, stop)

    def forget(self, path):
        pass


class ArraysChunk(Chunk):
    def __init__(self, reader, path, start, stop):
        super().__init__(reader, path, None, start, stop, {})

    def get(self, branch):
        a = self.arrays.get(branch)
        if a is None:
            if branch == ENTRY:
                a = np.arange(self.start, self.stop, dtype=np.int64)
            else:
                source = self.reader.dataset.arrays.get(branch)
                if source is None:
                    raise KeyError(f"no column {branch!r} in the arrays")
                a = source[self.start:self.stop]
            self.arrays[branch] = a
        return a
