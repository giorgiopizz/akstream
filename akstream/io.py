"""Input: ROOT files with one TTree, read in chunks with uproot (or flat-ntuple parquet files, with pyarrow).

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

Timeout: every uproot.open gets timeout= (seconds per request: an open, a stat, one vector read), so that a server that
stops answering raises instead of hanging; uproot_options go to uproot.open as they are.

Replicas: an entry of `files` that is a list holds alternative URLs of one file; the first names it (records, samples,
friends, output parts). A failed open or read moves on to the next replica at once (no retry_delay) and a replica that
works stays in use (per process); a chunk fails only when every replica failed. A replica with another number of
entries than the one counted first raises ReplicaMismatch, whatever remote_strategy.

Collections: the jagged branches X_<field> of a common prefix X (NanoAOD's Jet_pt, Jet_eta, ...), with the counter nX
when the tree has one; X_<field>__<v> (a file written with every variation) form the collection X__<v>.
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


def expand_replicas(files):
    """(files, {file: [urls]}) of a `files` argument whose entries may be lists of replicas (alternative URLs of one
    file, the first naming it; no glob patterns in them)."""
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    out, replicas = [], {}
    for f in files:
        if not isinstance(f, (list, tuple)):
            out.extend(expand([f]))
            continue
        urls = [os.path.expanduser(str(u)) for u in f]
        if not urls:
            raise ValueError("an empty list of replicas in files")
        if any(c in u for u in urls for c in "*?["):
            raise ValueError(f"replicas {urls}: glob patterns are not expanded in a list of replicas")
        out.append(urls[0])
        if len(urls) > 1:
            replicas[urls[0]] = urls
    return out, replicas


class ReplicaMismatch(ValueError):
    """a replica of a file with another number of entries than the one counted first: never skipped or retried."""


def _timeout_like(err):
    text = f"{type(err).__name__}: {err}"
    return isinstance(err, TimeoutError) or any(s in text for s in ("TimeoutError", "Operation expired", "timed out"))


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
    notes = "".join(f" [{n}]" for n in getattr(err, "__notes__", ()))
    return f"{type(err).__name__}: {err}"[:500] + notes[:500]


class Friend:
    """a friend tree's files (one per main file, same order and entries) and the branches that must agree entry by entry
    with the main tree's: match={friend branch: main branch}, checked on every chunk that reads the friend."""

    def __init__(self, files, match=None):
        self.files, self.match = files, dict(match or {})


class Dataset:
    """files + tree + chunking + the branch schema. Owns the graph cache of every frame built on it."""

    lenient = False  # several samples (SamplesDataset): a branch missing from a sample's files fails only when read

    def tree_for(self, path):
        """the tree name of the file `path`."""
        return self.tree

    def constants_for(self, path):
        """{column: value} given as constants for the rows of `path` (a sample's), or None."""
        return None

    def __init__(self, files, tree="Events", step=200_000, entry_start=0, entry_stop=None, threads=0, friends=None,
                 rdf_compatible=False, remote_strategy=0, retries=3, retry_delay=10.0, timeout=30.0,
                 uproot_options=None):
        if remote_strategy not in STRATEGIES:
            raise ValueError(f"remote_strategy: 0 / 'raise', 1 / 'skip' or 2 / 'retry', not {remote_strategy!r}")
        self.remote_strategy = STRATEGIES[remote_strategy]
        self.retries, self.retry_delay = int(retries), float(retry_delay)
        self._set_options(timeout, uproot_options)
        self._ranges = {}  # file -> [(start, stop or None)]: only these entries (files given as failed-chunk records)
        if _is_records(files):
            files, tree, friends = self._from_records(files, tree, friends)
        if isinstance(files, str):
            files, tree = _split_tree(files, tree)
        self.files, self.replicas = expand_replicas(files)  # replicas: {file: [urls]}, files with several only
        self.replica_index = {}  # file -> the replica in use (this process)
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
            ffiles, freplicas = expand_replicas(ffiles)
            self.replicas.update(freplicas)
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
                f, t = self._open_tree(path, self.tree)
                friend_trees = {name: self._open_tree(paths[path], name) for name, paths in self.friends.items()}
                schema = [(t, None)] + [(ft, name) for name, (_, ft) in friend_trees.items()]
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
        for ff, _ in friend_trees.values():
            ff.close()
        self.branches[ENTRY] = False
        self.dtypes[ENTRY] = np.dtype(np.int64)
        for name, match in self.friend_match.items():
            for fb, mb in match.items():
                if self.source_tree.get(fb) != name or mb not in self.branches or mb in self.source_tree:
                    raise ValueError(f"friend tree {name}: match {fb!r} -> {mb!r} needs a branch of the friend and one of "
                                     f"the main tree")
        self.collections, self.counters = self._collections()

    def _set_options(self, timeout, uproot_options):
        self.timeout = None if timeout is None else float(timeout)
        self.uproot_options = dict(uproot_options or {})
        if "timeout" in self.uproot_options:
            raise ValueError("uproot_options: give the timeout once, as read_root(timeout=...) / read_samples(timeout=...)")

    def _open_kwargs(self):
        kw = dict(self.uproot_options)
        if self.timeout is not None:  # None: uproot's own default (no key: older uproot hands it to fsspec as is)
            kw["timeout"] = self.timeout
        return kw

    def note_timeout(self, err):
        """the hint for a timed-out request, once, on the exception (its notes are part of the error records)."""
        if self.timeout is None or not _timeout_like(err) or not hasattr(err, "add_note"):
            return
        if any(n.startswith("akstream: timeout=") for n in getattr(err, "__notes__", ())):
            return
        err.add_note(f"akstream: timeout={self.timeout:g} s per request (an open, a stat, one vector read): raise it "
                     f"with read_root(timeout=...) / read_samples(timeout=...) for a slow link, timeout=None for none")

    def _open_tree(self, path, tree, entries=None):
        """(uproot file, TTree `tree`) of the file `path`: its replica in use first, then the others (an open that fails
        moves on to the next one at once; the one that opens stays in use). `entries`: the entries the file has (counted
        on a replica before): another number raises ReplicaMismatch."""
        urls = self.replicas.get(path, [path])
        first = self.replica_index.get(path, 0)
        errors = []
        for i in range(len(urls)):
            k = (first + i) % len(urls)
            try:
                f = uproot.open(urls[k], **self._open_kwargs())
                t = f[tree]
            except Exception as err:  # noqa: BLE001 (any open error: the next replica)
                self.note_timeout(err)
                errors.append((urls[k], err))
                continue
            if entries is not None and t.num_entries != entries:
                f.close()
                raise ReplicaMismatch(f"replica {urls[k]} of {path} has {t.num_entries} entries, {entries} were counted "
                                      f"before: the replicas of a file must be identical")
            self.replica_index[path] = k
            return f, t
        err = errors[-1][1]
        if len(errors) > 1 and hasattr(err, "add_note"):
            err.add_note("akstream: every replica failed: " + "; ".join(f"{u}: {error_text(e)[:200]}"
                                                                      for u, e in errors[:-1]))
        raise err

    def next_replica(self, path):
        """after a failed read of `path`: the next replica from now on; False when it has no other."""
        urls = self.replicas.get(path)
        if urls is None:
            return False
        self.replica_index[path] = (self.replica_index.get(path, 0) + 1) % len(urls)
        return True

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
            base, sep, variation = name.partition("__")  # X_<field>__<v>: the collection X__<v> of a varied file
            if "_" not in base:
                continue
            prefix, field = base.split("_", 1)
            if sep and variation:
                prefix = f"{prefix}__{variation}"
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
                f, t = self._open_tree(path, self.tree)
                with f:
                    n = self._entries[path] = t.num_entries
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
        """a failed chunk, as read_root accepts it back (a file with replicas: "replicas" lists them, and so does a
        friend's entry)."""
        out = {"file": path, "start": int(start), "stop": None if stop is None else int(stop), "error": error_text(err),
               "tree": self.tree,
               "friends": {name: list(self.replicas.get(paths[path], ())) or paths[path]
                           for name, paths in self.friends.items()},
               "match": {name: dict(m) for name, m in self.friend_match.items()}}
        if path in self.replicas:
            out["replicas"] = list(self.replicas[path])
        return out

    def _from_records(self, records, tree, friends):
        """files, tree and friends of a list of failed-chunk records (the ranges go to self._ranges)."""
        files, ffiles, match = [], {}, {}
        for r in records:
            if r["file"] not in self._ranges:
                files.append(list(r["replicas"]) if r.get("replicas") else r["file"])
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


class _FailedRead(Exception):
    """a read error of the main tree (friend None) or of a friend tree, inside Reader._read."""

    def __init__(self, err, friend):
        super().__init__(err)
        self.err, self.friend = err, friend


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
                if tree is None and friend is None:
                    f, tree = ds._open_tree(path, ds.tree_for(path), ds._entries.get(path))
                elif tree is None:
                    f, tree = ds._open_tree(ds.friends[friend][path], friend)
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

    def forget(self, path, friend=all):
        """drop the cached handles of a file (only of its friend tree `friend`, None: the main tree) after a read error
        (a retry opens it again); not closed here: the read-ahead thread may still hold it."""
        with self._lock:
            for key in [k for k in self._open if k[0] == path and (friend is all or k[1] == friend)]:
                self._open.pop(key)
            if friend in (all, None) and self.dataset._open is not None and self.dataset._open[1] == path:
                self.dataset._open = None

    def add_branch(self, branch):
        with self._lock:
            self.lazy_reads.add(branch)
            self.prefetch.add(branch)

    def read(self, path, start, stop):
        """the prefetched branches of one chunk: {branch: numpy or awkward}. A read that fails on a file with replicas
        is repeated at once on its next one, until each was tried."""
        ds = self.dataset
        moved = {}
        while True:
            try:
                return self._read(path, start, stop)
            except _FailedRead as failed:
                err, friend = failed.err, failed.friend
                ds.note_timeout(err)
                name = path if friend is None else ds.friends[friend][path]
                if moved.get(name, 0) + 1 >= len(ds.replicas.get(name, [name])) or not ds.next_replica(name):
                    raise err from None
                moved[name] = moved.get(name, 0) + 1
                self.forget(path, friend)

    def _read(self, path, start, stop):
        ds = self.dataset
        tree = self.open(path)
        with self._lock:
            wanted = sorted(self.prefetch)
        out = {}
        if ENTRY in wanted:
            out[ENTRY] = np.arange(start, stop, dtype=np.int64)
        consts = ds.constants_for(path) or {}
        by_tree = {}
        for b in wanted:
            if b in consts:
                out[b] = np.full(stop - start, consts[b], dtype=ds.dtypes[b])
            elif b != ENTRY:
                by_tree.setdefault(ds.source_tree.get(b), []).append(b)
        kw = {}
        if self._executor is not None:
            kw = dict(decompression_executor=self._executor, interpretation_executor=self._executor)
        for friend, names in by_tree.items():
            t = tree if friend is None else self.open(path, friend)
            present = set(t.keys())
            missing = [b for b in names if b not in present]
            if missing and ds.lenient:  # another sample's branch: an error only if this chunk needs it (Chunk.get)
                names = [b for b in names if b in present]
            elif missing:
                raise KeyError(f"{path}: branches not in the tree {friend or ds.tree}: {missing[:10]}")
            flat = [b for b in names if not ds.branches.get(b, False)]
            jagged = [b for b in names if ds.branches.get(b, False)]
            try:
                if flat:
                    out.update(t.arrays(flat, entry_start=start, entry_stop=stop, library="np", how=dict, **kw))
                if jagged:
                    out.update(t.arrays(jagged, entry_start=start, entry_stop=stop, library="ak", how=dict, **kw))
            except Exception as err:  # noqa: BLE001 (any read error: Reader.read tries the next replica)
                raise _FailedRead(err, friend) from err
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
            consts = ds.constants_for(self.path) or {}
            if branch == ENTRY:
                a = np.arange(self.start, self.stop, dtype=np.int64)
            elif branch in consts:
                a = np.full(self.n, consts[branch], dtype=ds.dtypes[branch])
            else:
                friend = ds.source_tree.get(branch)
                t = self.tree if friend is None else self.reader.open(self.path, friend)
                if ds.lenient and branch not in t.keys():
                    raise KeyError(f"branch {branch!r} is not in the files of sample {ds.sample_of(self.path)!r}: give "
                                   f"it as a constant of that sample in read_samples ({{{branch!r}: value}}): it is "
                                   f"read on every chunk that needs it, a selection does not avoid the read")
                a = t[branch].array(entry_start=self.start, entry_stop=self.stop, library="ak" if jagged else "np")
            self.arrays[branch] = a
            self.reader.add_branch(branch)
            friend = ds.source_tree.get(branch)
            if friend is not None:
                self.check_friend(friend)
        return a


# ---------------------------------------------------------------- several samples (aks.read_samples)
SAMPLE = "sample"  # the column of the sample index (df.sample == "DY")
_SAMPLE_KEYS = ("files", "tree", "step")


class SamplesDataset(Dataset):
    """several samples, each a set of ROOT files with its own tree / step and constants (columns with one value per
    sample: xs, is_data, ...); a chunk never mixes samples. The schema is the union of the samples' (a branch missing in
    a sample is an error when that sample's chunks need it, even if a selection then drops their rows: a constant of
    the same name stands in for it)."""

    lenient = True

    def __init__(self, samples, tree="Events", step=200_000, threads=0, rdf_compatible=False, remote_strategy=0,
                 retries=3, retry_delay=10.0, timeout=30.0, uproot_options=None):
        if not isinstance(samples, dict) or not samples:
            raise ValueError("read_samples: a non-empty dict {name: {'files': ..., constants...}}")
        self.sample_names = [str(n) for n in samples]
        self.children, self.sample_constants, self._sample_of = {}, {}, {}
        for name, spec in samples.items():
            name = str(name)
            if not isinstance(spec, dict) or "files" not in spec:
                raise ValueError(f"read_samples: sample {name!r} needs a dict with 'files' (and optional tree, step, "
                                 f"constants)")
            consts = {k: v for k, v in spec.items() if k not in _SAMPLE_KEYS}
            for k, v in consts.items():
                if k in ("friends", "entry_start", "entry_stop", "remote_strategy"):
                    raise ValueError(f"read_samples: {k!r} per sample is not supported (yet)")
                if isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating)):
                    continue
                raise TypeError(f"read_samples: constant {k!r} of sample {name!r} is {type(v).__name__}: numbers and "
                                f"booleans only")
            child = Dataset(spec["files"], spec.get("tree", tree), spec.get("step", step), threads=threads,
                            rdf_compatible=rdf_compatible, remote_strategy=remote_strategy, retries=retries,
                            retry_delay=retry_delay, timeout=timeout, uproot_options=uproot_options)
            for p in child.files:
                if p in self._sample_of:
                    raise ValueError(f"read_samples: {p} is in samples {self._sample_of[p]!r} and {name!r}")
                self._sample_of[p] = name
            self.children[name], self.sample_constants[name] = child, consts
        self.branches, self.dtypes, self.present = {}, {}, {}
        for name, child in self.children.items():
            self.present[name] = set(child.branches)
            for b, jagged in child.branches.items():
                if b in self.branches and self.branches[b] != jagged:
                    raise ValueError(f"read_samples: branch {b} is a list in one sample and a number in another")
                self.branches.setdefault(b, jagged)
                self.dtypes.setdefault(b, child.dtypes.get(b))
        if SAMPLE in self.branches:
            raise ValueError(f"read_samples: the files have a branch {SAMPLE!r}, the sample column of akstream")
        for k in dict.fromkeys(k for c in self.sample_constants.values() for k in c):
            if self.branches.get(k):
                raise ValueError(f"read_samples: constant {k!r}: a list branch of that name exists")
            if k in self.branches:  # stands in for a branch of the other samples: its type
                dt = self.dtypes[k]
            else:
                dt = np.result_type(*[np.asarray(c[k]) for c in self.sample_constants.values() if k in c])
            self.branches[k], self.dtypes[k] = False, np.dtype(dt)
        self.branches[SAMPLE], self.dtypes[SAMPLE] = False, np.dtype(np.int32)
        first = next(iter(self.children.values()))
        self.files = [p for c in self.children.values() for p in c.files]
        self.tree = tree
        self.rdf_compatible = rdf_compatible
        self.remote_strategy, self.retries, self.retry_delay = first.remote_strategy, first.retries, first.retry_delay
        self._set_options(timeout, uproot_options)
        self.replicas = {p: urls for c in self.children.values() for p, urls in c.replicas.items()}
        self.replica_index = {}
        self._ranges = {}
        self.friends, self.friend_match, self.source_tree = {}, {}, {}
        self.step = int(step)
        self.entry_start, self.entry_stop = 0, None
        self.threads = threads
        self.cache, self.keep, self.booked = {}, [], []
        self.root = Space(self)
        self._open = None
        self._entries = {}
        self.collections, self.counters = self._collections()

    @property
    def open_failures(self):
        return [dict(r, sample=n) for n, c in self.children.items() for r in c.open_failures]

    def sample_of(self, path):
        return self._sample_of[path]

    def tree_for(self, path):
        return self.children[self._sample_of[path]].tree

    def constants_for(self, path):
        name = self._sample_of[path]
        out = dict(self.sample_constants[name])
        out[SAMPLE] = self.sample_names.index(name)
        return out

    def chunks(self):
        return [c for child in self.children.values() for c in child.chunks()]

    def _num_entries(self, path):
        return self.children[self._sample_of[path]]._num_entries(path)

    def open_tree(self, path):
        return self.children[self._sample_of[path]].open_tree(path)

    def record(self, path, start, stop, err):
        return dict(self.children[self._sample_of[path]].record(path, start, stop, err), sample=self._sample_of[path])


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


# ---------------------------------------------------------------- parquet files (aks.read_parquet)
def _pyarrow():
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError as err:
        raise ImportError("parquet files need pyarrow: pip install akstream[parquet] (or pip install pyarrow)") from err
    return pyarrow


def _parquet_column(t):
    """(jagged, numpy dtype) of a flat-ntuple parquet type (a number, a boolean, or a list of them), else None."""
    pa = _pyarrow()

    def number(x):
        if pa.types.is_boolean(x):
            return np.dtype(bool)
        if pa.types.is_integer(x) or pa.types.is_floating(x):
            return np.dtype(x.to_pandas_dtype())
        return None
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        dt = number(t.value_type)
        return None if dt is None else (True, dt)
    dt = number(t)
    return None if dt is None else (False, dt)


class ParquetDataset(Dataset):
    """flat-ntuple parquet files (one row per entry; columns of numbers, booleans, or lists of them, as NanoAOD's
    nJet, Jet_pt, ...), read in chunks of `step` rows; only the columns the plan needs are read."""

    def __init__(self, files, step=200_000, entry_start=0, entry_stop=None, threads=0, rdf_compatible=False):
        import warnings
        pq = _pyarrow().parquet
        self.files = expand(files)
        self.tree = None
        self.rdf_compatible = rdf_compatible
        self.remote_strategy, self.retries, self.retry_delay = 0, 0, 0.0
        self._ranges, self.open_failures = {}, []
        self.friends, self.friend_match, self.source_tree = {}, {}, {}
        self.step = int(step)
        self.entry_start, self.entry_stop = entry_start, entry_stop
        self.threads = threads
        self.cache, self.keep, self.booked = {}, [], []
        self.root = Space(self)
        self._open = None
        self._entries = {}
        schema = pq.read_schema(self.files[0])
        self.branches, self.dtypes, skipped = {}, {}, []
        for field in schema:
            kind = _parquet_column(field.type)
            if kind is None:
                skipped.append(f"{field.name} ({field.type})")
                continue
            self.branches[field.name], self.dtypes[field.name] = kind
        if skipped:
            warnings.warn(f"akstream.read_parquet: {len(skipped)} columns are not flat-ntuple columns (numbers, booleans "
                          f"or lists of them) and are not readable: {', '.join(skipped[:10])}", RuntimeWarning,
                          stacklevel=3)
        if ENTRY in self.branches:
            raise ValueError(f"{self.files[0]}: a column is named {ENTRY}, the entry-number column of akstream")
        self.branches[ENTRY] = False
        self.dtypes[ENTRY] = np.dtype(np.int64)
        self.collections, self.counters = self._collections()

    def _num_entries(self, path):
        n = self._entries[path] = _pyarrow().parquet.ParquetFile(path).metadata.num_rows
        return n

    def chunks(self):
        for path in self.files:
            if path not in self._entries:
                self._num_entries(path)
        return super().chunks()

    def record(self, path, start, stop, err):
        return {"file": path, "start": int(start), "stop": None if stop is None else int(stop), "error": error_text(err)}

    def reader(self, prefetch):
        return ParquetReader(self, prefetch)

    def open_tree(self, path):
        return None


class ParquetReader:
    """per-worker reader of parquet files: reads the needed columns of the row groups a chunk covers and keeps the
    decoded row groups until the chunks have moved past them (chunks need not align with row groups). Safe for one
    background read-ahead thread next to the main thread (one lock)."""

    def __init__(self, dataset, prefetch):
        self.dataset = dataset
        self.prefetch = set(prefetch)
        self.lazy_reads = set()
        self._lock = threading.Lock()
        self._files = {}  # path -> (ParquetFile, row-group first rows), the two most recent
        self._groups = {}  # (path, row group) -> {column: pyarrow array}

    def _file(self, path):
        hit = self._files.get(path)
        if hit is None:
            pf = _pyarrow().parquet.ParquetFile(path)
            sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
            hit = self._files[path] = (pf, np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)]))
            while len(self._files) > 2:
                old = next(iter(self._files))
                self._files.pop(old)
                for key in [k for k in self._groups if k[0] == old]:
                    del self._groups[key]
        return hit

    def columns(self, path, start, stop, names):
        """{name: numpy or awkward} of the rows [start, stop) of `path`."""
        pa = _pyarrow()
        with self._lock:
            pf, first = self._file(path)
            schema = pf.schema_arrow
            missing = [n for n in names if schema.get_field_index(n) < 0]
            if missing:
                raise KeyError(f"{path}: columns not in the file: {missing[:10]}")
            lo, hi = max(int(np.searchsorted(first, start, "right")) - 1, 0), int(np.searchsorted(first, stop, "left"))
            pieces = {n: [] for n in names}
            for g in range(lo, hi):
                cached = self._groups.setdefault((path, g), {})
                todo = [n for n in names if n not in cached]
                if todo:
                    table = pf.read_row_group(g, columns=todo, use_threads=bool(self.dataset.threads))
                    for n in todo:
                        cached[n] = table.column(n).combine_chunks()
                a, b = max(start, first[g]) - first[g], min(stop, first[g + 1]) - first[g]
                for n in names:
                    pieces[n].append(cached[n].slice(a, b - a))
            for key in [k for k in self._groups if k[0] == path and first[k[1] + 1] <= start]:
                del self._groups[key]  # row groups the chunks have moved past
        out = {}
        for n in names:
            p = pieces[n]
            arr = p[0] if len(p) == 1 else pa.concat_arrays(p)
            out[n] = _from_arrow(arr, self.dataset.branches[n])
        return out

    def forget(self, path):
        with self._lock:
            self._files.pop(path, None)
            for key in [k for k in self._groups if k[0] == path]:
                del self._groups[key]

    def add_branch(self, branch):
        with self._lock:
            self.lazy_reads.add(branch)
            self.prefetch.add(branch)

    def read(self, path, start, stop):
        with self._lock:
            wanted = sorted(self.prefetch)
        out = self.columns(path, start, stop, [b for b in wanted if b != ENTRY])
        if ENTRY in wanted:
            out[ENTRY] = np.arange(start, stop, dtype=np.int64)
        return ParquetChunk(self, path, start, stop, out)


def _from_arrow(arr, jagged):
    """numpy (a number column) or an awkward list array (a list column) of a pyarrow array; with None (nulls) the
    awkward conversion, an option type."""
    import awkward as ak
    from .jagged import wrap
    if not jagged:
        if arr.null_count == 0:
            return arr.to_numpy(zero_copy_only=False)
        return ak.from_arrow(arr, generate_bitmasks=True)
    offsets = np.asarray(arr.offsets, dtype=np.int64)
    values = arr.values.slice(int(offsets[0]), int(offsets[-1] - offsets[0])) if len(offsets) else arr.values.slice(0, 0)
    if arr.null_count == 0 and values.null_count == 0:
        return wrap(offsets - offsets[0], values.to_numpy(zero_copy_only=False))
    return ak.from_arrow(arr, generate_bitmasks=True)


class ParquetChunk(Chunk):
    def __init__(self, reader, path, start, stop, arrays):
        super().__init__(reader, path, None, start, stop, arrays)

    def get(self, branch):
        a = self.arrays.get(branch)
        if a is None:
            ds = self.reader.dataset
            if branch not in ds.branches:
                raise KeyError(f"no column {branch!r} in the parquet files")
            if branch == ENTRY:
                a = np.arange(self.start, self.stop, dtype=np.int64)
            else:
                a = self.reader.columns(self.path, self.start, self.stop, [branch])[branch]
                self.reader.add_branch(branch)
            self.arrays[branch] = a
        return a
