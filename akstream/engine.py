"""The event loop: one pass over the chunks of a dataset fills every requested result.

Plan (once per compute):
  - the nodes every result needs, and the branches to read in bulk (fields of materialised records are read lazily);
  - the variations each varied result needs, and for each variation the nodes downstream of the varied ones.
Per chunk: read, fill nominal, then for each variation build a VariationContext (deltas) and fill the varied results.

compute(..., validate=k) also evaluates every variation by brute force (all affected nodes recomputed on all rows)
on the first k chunks and checks that the delta evaluation gives the same histograms.
"""
import time
from collections import defaultdict

import operator

import numpy as np

from . import fused
from .execute import ChunkContext, VariationContext
from .graph import Call, Node, Scalar, Source, Take, Zip, walk
from .io import ENTRY, SAMPLE, ReadAhead, error_text

last_stats = {}
last_failed = []  # the chunks the last compute could not read (records: read_root(last_failed) processes them)


class Plan:
    def __init__(self, results):
        datasets = {id(r.dataset): r.dataset for r in results}
        if len(datasets) != 1:
            raise ValueError("compute: every result must come from the same read_root dataset")
        self.dataset = next(iter(datasets.values()))
        self.results = list(results)
        nominal = [n for r in results for n in r.all_nodes()]
        order = walk(nominal)
        # variations: names per result, and the alternative nodes they need
        self.variations = []
        self.varied_results = defaultdict(list)
        alt_nodes = []
        varied_by = defaultdict(set)  # variation -> ids of the nodes it replaces
        all_names = []
        for n in order:
            for v, alt in n.variations.items():
                varied_by[v].add(n.id)
                if v not in all_names:
                    all_names.append(v)
        children = defaultdict(list)
        for n in order:
            for c in n.inputs():
                if c is not None:
                    children[c.id].append(n)
        self.affected = {}
        by_id = {n.id: n for n in order}
        for v in all_names:
            seen = set(varied_by[v])
            stack = list(varied_by[v])
            while stack:
                i = stack.pop()
                for ch in children.get(i, ()):
                    if ch.id not in seen:
                        seen.add(ch.id)
                        stack.append(ch.id)
            self.affected[v] = seen
        for r in results:
            if r.variations is None:
                continue
            ids = {n.id for n in walk(r.all_nodes())}
            names = [v for v in all_names if self.affected[v] & ids] if r.variations == "auto" else list(r.variations)
            unknown = [v for v in names if v not in self.affected]
            if unknown:
                raise KeyError(f"unknown variations {unknown}; registered: {all_names}")
            r.variation_names = names
            for v in names:
                self.varied_results[v].append(r)
                if v not in self.variations:
                    self.variations.append(v)
        for v in self.variations:
            for i in varied_by[v]:
                alt_nodes.append(by_id[i].variations[v])
        self.subst = {v: {i: by_id[i].variations[v] for i in varied_by[v]} for v in self.variations}
        self.by_id = by_id
        self.fused = {}
        for v in self.variations:
            # rows written (to_root) are taken where the varied inputs change, not compared column by column: a row
            # can only change there (or by entering / leaving a selection, whose masks are compared), and rewriting
            # an unchanged row is exact
            compared = [r for r in self.varied_results[v] if not hasattr(r, "record_variation")]
            result_nodes = {n.id for r in compared for n in r.nodes()}
            by_trigger = len(compared) < len(self.varied_results[v])
            self.fused[v] = fused.plan_variation(v, self.affected[v], self.subst[v], order, children, result_nodes,
                                                 by_trigger)
        for r in self.results:  # written results: their file layout (which variation changes what)
            if hasattr(r, "plan_layout"):
                r.plan_layout(self)
        # several samples: a variation runs on the samples whose files have what its alternatives read
        self.sample_variations = None
        ds = self.dataset
        if getattr(ds, "sample_names", None):
            self.sample_variations = {}
            for name in ds.sample_names:
                have = ds.present[name] | set(ds.sample_constants[name]) | {SAMPLE, ENTRY}
                self.sample_variations[name] = {v for v in self.variations
                                                if _bulk_sources(list(self.subst[v].values())) <= have}
        # variations that replace the same branches (and fill the same results) run their phase 2 stacked together
        self.batches, by_key = [], {}
        for v in self.variations:
            subst = self.subst[v]
            if subst and all(isinstance(by_id[i], Source) for i in subst):
                k = (frozenset(subst), tuple(id(r) for r in self.varied_results[v]))
                if k in by_key:
                    by_key[k].append(v)
                    continue
                by_key[k] = [v]
                self.batches.append(by_key[k])
            else:
                self.batches.append([v])
        # histograms with the same weight are corrected together (shared changed rows, gathers and bins)
        self.groups = {}
        for v in self.variations:
            groups, by_key = [], {}
            for r in self.varied_results[v]:
                if hasattr(r, "fill_variation_shared") and r.axes:
                    # same columns, weight and binning (any selection): one gather + binning for all of them
                    k = r.weight.id if r.weight is not None else None
                    if k not in by_key:
                        by_key[k] = []
                        groups.append(by_key[k])
                    by_key[k].append(r)
                else:
                    groups.append(r)
            self.groups[v] = groups
        self.prefetch = sorted(_bulk_sources(nominal + alt_nodes))
        # the nominal pass frees an intermediate after its last consumer, unless a variation reads its nominal value
        # (a value freed by mistake is only recomputed when asked for)
        consumers = defaultdict(int)
        pinned = set()  # never freed
        for n in order:
            for c in set(x.id for x in n.inputs() if x is not None):
                consumers[c] += 1
            if isinstance(n, Zip):
                pinned.update(f.id for f in n.fields.values())  # touched lazily by record code
        for r in results:
            for n in set(x.id for x in r.nodes()):
                consumers[n] += 1
        if self.variations:
            pinned |= _read_by_variations(self, order, results)
        for i in pinned:
            consumers[i] += 1 << 30
        self.consumers = dict(consumers)


def _read_by_variations(plan, order, results):
    """ids of the nodes whose nominal value a variation pass may read: every node some variation does not recompute,
    the compared nodes (and their arguments), selection masks, results, the varied nodes and their alternatives,
    the inputs of triggers, and the branches."""
    recomputed = set.intersection(*(set(plan.fused[v].value_affected) for v in plan.variations))
    out = {n.id for n in order if n.id not in recomputed or isinstance(n, Source) or n.variations}
    for n in order:
        for s in n.space.ancestors():
            if s.mask is not None:
                out.add(s.mask.id)
        if getattr(n, "trigger", None):
            out.update(t.id for t in n.trigger)
    for v in plan.variations:
        for c in plan.fused[v].compare:
            out.add(c.id)
            out.update(x.id for x in c.inputs() if x is not None)
        for i, alt in plan.subst[v].items():
            out.update((i, alt.id))
    for r in results:
        out.update(n.id for n in r.nodes())
    return out


def _bulk_sources(roots):
    """branches of every Source reachable without going through the fields of a record (Zip) node: record fields are
    read lazily when the record's code touches them."""
    seen, stack, out = set(), list(roots), set()
    while stack:
        n = stack.pop()
        if n is None or n.id in seen:
            continue
        seen.add(n.id)
        if isinstance(n, Source):
            out.add(n.branch)
        if isinstance(n, Zip):
            if n.counts is not None:
                stack.append(n.counts)
            continue
        stack.extend(n.inputs())
    return out


def _per_sample(plan, r):
    """with several samples: a result filled per sample (its value {sample: value}), else one for all of them."""
    return plan.sample_variations is not None and getattr(r, "per_sample", True)


def _new_state(r, part, sample=None):
    st = r.init_state()
    if isinstance(st, dict):
        if "part" in st:
            st["part"] = part  # file-writing results: one output file per task
        if "sample" in st:
            st["sample"] = sample
    return st


def _run_chunks(plan, chunks, opts, part=None):
    shared = {id(r): _new_state(r, part) for r in plan.results if not _per_sample(plan, r)}
    per = {}  # sample -> {id(result): state} of the per-sample results
    ds = plan.dataset

    def states_for(path):
        if plan.sample_variations is None:
            return shared
        name = ds.sample_of(path)
        st = per.get(name)
        if st is None:
            st = per[name] = dict(shared)
            st.update({id(r): _new_state(r, part, name) for r in plan.results if _per_sample(plan, r)})
        return st

    reader = plan.dataset.reader(plan.prefetch)
    stats = defaultdict(float)
    ds = plan.dataset
    strategy = ds.remote_strategy
    failed = []  # ((path, start, stop), error)
    validated = [0]
    for (path, start, stop), get in ReadAhead(reader, chunks, opts.get("read_ahead", "auto")):
        t0 = time.perf_counter()
        try:
            chunk = get()  # with read-ahead: the time waiting for the background read
        except Exception as err:  # noqa: BLE001 (any read error: xrootd, deserialization, a vanished file)
            if strategy == 0:
                raise
            reader.forget(path)
            failed.append(((path, start, stop), err))
            _warn_failed(path, start, stop, err, "retried at the end" if strategy == 2 else "skipped")
            continue
        _process_chunk(plan, chunk, states_for(path), opts, stats, validated, t0)
    for attempt in range(ds.retries if strategy == 2 else 0):
        if not failed:
            break
        time.sleep(ds.retry_delay)
        still = []
        for (path, start, stop), _ in failed:
            t0 = time.perf_counter()
            try:
                chunk = reader.read(path, start, stop)
            except Exception as err:  # noqa: BLE001
                reader.forget(path)
                still.append(((path, start, stop), err))
                continue
            stats["retried_ok"] += 1
            _process_chunk(plan, chunk, states_for(path), opts, stats, validated, t0)
        failed = still
    for (path, start, stop), err in failed:
        if strategy == 2:
            _warn_failed(path, start, stop, err, f"skipped after {ds.retries} retries")
    states = {}
    for r in plan.results:
        if _per_sample(plan, r):
            states[id(r)] = {"__samples__": {name: st[id(r)] for name, st in per.items()}}
            for st in states[id(r)]["__samples__"].values():
                if hasattr(r, "end_task"):
                    r.end_task(st)
        else:
            states[id(r)] = shared[id(r)]
            if hasattr(r, "end_task"):
                r.end_task(shared[id(r)])
    stats["lazy_branches"] = sorted(reader.lazy_reads)
    stats["failed"] = [ds.record(path, start, stop, err) for (path, start, stop), err in failed]
    return states, dict(stats)


def _warn_failed(path, start, stop, err, what):
    import warnings
    warnings.warn(f"akstream: entries [{start}, {stop}) of {path} not read ({error_text(err)}): {what}", RuntimeWarning,
                  stacklevel=3)


def _process_chunk(plan, chunk, states, opts, stats, validated, t0):
    """nominal and variations of one chunk that was read."""
    start, stop = chunk.start, chunk.stop
    t1 = time.perf_counter()
    ctx = ChunkContext(chunk, consumers=plan.consumers)
    for r in plan.results:
        r.fill(states[id(r)], ctx)
        ctx.release(r.nodes())
    t2 = time.perf_counter()
    stats["read"] += t1 - t0
    stats["nominal"] += t2 - t1
    stats["entries"] += stop - start
    stats["chunks"] += 1
    active = None if plan.sample_variations is None else \
        plan.sample_variations[chunk.reader.dataset.sample_of(chunk.path)]
    for batch in plan.batches:
        if active is not None:
            batch = [v for v in batch if v in active]
            if not batch:
                continue
        tv = time.perf_counter()
        _run_batch(plan, ctx, batch, states, opts, stats)
        stats["variations"] += time.perf_counter() - tv
        if validated[0] < opts["validate"]:
            _validate_batch(plan, chunk, ctx, batch, opts)
    if validated[0] < opts["validate"]:
        validated[0] += 1
    for r in plan.results:
        r.end_chunk(states[id(r)])


def _run_batch(plan, ctx, batch, states, opts, stats):
    if opts["engine"] == "fused" and not opts["exact"] and len(batch) > 1:
        fused.run_batch(plan, ctx, batch, states, opts, stats)
        return
    for v in batch:
        _run_variation(plan, ctx, v, states, opts, stats)


def _run_variation(plan, ctx, v, states, opts, stats):
    """one variation of one chunk: the fused sub-chunk pass (default), or the per-node delta engine (exact mode,
    engine="delta")."""
    if opts["engine"] == "fused" and not opts["exact"]:
        fused.run_variation(plan, ctx, v, states, opts, stats)
        return
    vctx = VariationContext(ctx, v, plan.affected[v], opts["dense_fraction"], opts["cutoff"] == "all", opts["exact"])
    for group in plan.groups[v]:
        if isinstance(group, list):
            group[0].fill_variation_shared(group, states, vctx)
        else:
            group.fill_variation(states[id(group)], vctx)
    stats["recomputed_rows"] += vctx.stats["recomputed_rows"]
    stats["dense_nodes"] += vctx.stats["dense_nodes"]
    stats["sparse_nodes"] += vctx.stats["sparse_nodes"]


def _validate_batch(plan, chunk, ctx, batch, opts):
    """brute force: every node recomputed with each variation's substitutions; each varied histogram filled from it
    must equal nominal chunk + this chunk's correction from the engine in use (batched or not)."""
    hists = list({id(r): r for v in batch for r in plan.varied_results[v] if hasattr(r, "_acc")}.values())
    states = {}
    for r in hists:
        st = r.init_state()
        st["chunk"] = r._fill_with(lambda node: ctx.value_at(node, r.space), ctx.size(r.space))
        states[id(r)] = st
    _run_batch(plan, ctx, batch, states, dict(opts, exact=False), defaultdict(float))
    for v in batch:
        _check_variation(plan, chunk, ctx, v, opts, [r for r in hists if r in plan.varied_results[v]], states)


def _check_variation(plan, chunk, ctx, v, opts, hists, states):
    brute = ChunkContext(chunk, subst=plan.subst[v])
    for r in hists:
        expected = r._fill_with(lambda node: brute.value_at(node, r.space), brute.size(r.space))
        state = states[id(r)]
        got = state["chunk"].sumw + (state["corr"][v].sumw if v in state["corr"] else 0)
        got_count = state["chunk"].count + (state["corr"][v].count if v in state["corr"] else 0)
        scale = max(1.0, float(np.abs(expected.sumw).max()) if len(expected.sumw) else 1.0)
        if not np.array_equal(got_count, expected.count) or not np.allclose(got, expected.sumw, rtol=1e-9,
                                                                             atol=1e-9 * scale):
            bad = np.flatnonzero(~np.isclose(got, expected.sumw, rtol=1e-9, atol=1e-9 * scale) |
                                 (got_count != expected.count))
            raise AssertionError(f"variation {v} ({opts['engine']} engine) differs from brute force in "
                                 f"{type(r).__name__} {getattr(r, 'name', '')} bins {bad[:10].tolist()}: "
                                 f"got {got[bad[:5]].tolist()} expected {expected.sumw[bad[:5]].tolist()}")


# ---------------------------------------------------------------- parallel
_WORKER_PLAN = None


def _init_worker(payload):
    """worker process start (spawn): the plan arrives pickled with cloudpickle (lambdas and closures included)."""
    global _WORKER_PLAN
    import cloudpickle
    _WORKER_PLAN = cloudpickle.loads(payload)


def _worker(args):
    chunks, opts, part = args
    plan = _WORKER_PLAN
    states, stats = _run_chunks(plan, chunks, opts, part)
    return {i: _picklable(states[id(r)]) for i, r in enumerate(plan.results)}, stats


def _picklable(state):
    if isinstance(state, dict) and "__samples__" in state:
        return {"__samples__": {k: _picklable(v) for k, v in state["__samples__"].items()}}
    if isinstance(state, dict):
        return {k: v for k, v in state.items() if k not in ("file", "tree")}
    return state


def _merge(r, a, b):
    if isinstance(a, dict) and "__samples__" in a:
        for name, st in b["__samples__"].items():
            mine = a["__samples__"].get(name)
            a["__samples__"][name] = st if mine is None else r.merge(mine, st)
        return a
    return r.merge(a, b)


def _finalize(plan, r, state):
    """the value of a result; with several samples, per sample ({sample: value}, every sample even without rows) and
    with only the variations that run on each sample."""
    if not (isinstance(state, dict) and "__samples__" in state):
        return r.finalize(state) if not hasattr(r, "_deferred") else r.finalize(state, sample=None)
    out = {}
    for name in plan.dataset.sample_names:
        st = state["__samples__"].get(name)
        if st is None:  # no rows of this sample in the run
            st = _new_state(r, None, name)
            if hasattr(r, "end_task"):
                r.end_task(st)
        v = r.finalize(st, sample=name) if hasattr(r, "_deferred") else r.finalize(st)
        if r.variations is not None and isinstance(v, dict) and "nominal" in v:
            active = plan.sample_variations[name]
            v = {k: x for k, x in v.items() if k == "nominal" or k in active}
        out[name] = v
    return out


# ---------------------------------------------------------------- sums inside expressions
_LINEAR = (operator.mul, operator.truediv, operator.neg)


def _uses(results):
    """node id -> [(result, role)]: how each result reads its nodes (a histogram weight is the only linear role)."""
    from .results import Collect, GroupSum, Hist, ToRoot
    uses = defaultdict(list)
    for r in results:
        if isinstance(r, Hist):
            for c in r.columns:
                uses[c.id].append((r, "a histogram axis"))
            if r.weight is not None:
                uses[r.weight.id].append((r, "weight"))
            continue
        role = ("a written column (to_root / to_parquet)" if isinstance(r, ToRoot) else
                "collect / peek / head" if isinstance(r, Collect) else
                "a group sum" if isinstance(r, GroupSum) else type(r).__name__)
        for n in r.nodes():
            uses[n.id].append((r, role))
    return uses


def _linear_targets(scalar, children, uses, masks):
    """([(histogram, power)], None) when the sum only scales histogram / sum weights (value ~ sum ** power), else
    (None, why not)."""
    below = set()
    stack = [scalar]
    while stack:  # everything computed from the sum
        x = stack.pop()
        for c in children.get(x.id, ()):
            if c.id not in below:
                below.add(c.id)
                stack.append(c)
    out, seen, stack = [], set(), [(scalar, 1)]
    while stack:
        x, p = stack.pop()
        if x.id in seen:
            return None, "an expression that uses it twice"
        seen.add(x.id)
        if x.id in masks:
            return None, "a selection"
        for r, role in uses.get(x.id, ()):
            if role != "weight":
                return None, role
            out.append((r, p))
        for c in children.get(x.id, ()):
            if isinstance(c, Take):
                if c.parent is not x:
                    return None, "a selection"
                stack.append((c, p))
                continue
            if isinstance(c, Call) and c.fn in _LINEAR and not c.kwargs:
                pos = [i for i, a in enumerate(c.args) if a is x]
                others = [a for i, a in enumerate(c.args) if i not in pos]
                if len(pos) == 1 and not any(isinstance(a, Node) and a.id in below for a in others):
                    stack.append((c, -p if c.fn is operator.truediv and pos[0] == 1 else p))
                    continue
            return None, f"{getattr(c, 'name', type(c).__name__)}(...)"
    return out, None


def _resolve_scalars(results, opts):
    """the sums used inside expressions: computed in this pass and applied to the histograms at the end when they
    only scale histogram / sum weights, else computed now in a pass of their own (with a warning). Returns the results
    to run (the deferred sums added)."""
    nodes = [n for r in results for n in r.all_nodes()]
    order = walk(nodes)
    alts = [a for n in order for a in n.variations.values()]
    order = walk(nodes + alts)
    scalars = [n for n in order if isinstance(n, Scalar)]
    for r in results:
        if hasattr(r, "_deferred"):
            r._deferred = []
    if not scalars:
        return results
    children = defaultdict(list)
    for n in order:
        for c in n.inputs():
            if c is not None:
                children[c.id].append(n)
    masks = {s.mask.id for n in order for s in n.space.ancestors() if s.mask is not None}
    uses = _uses(results)
    before, extra = [], []
    for sc in scalars:
        if sc.result._done:
            sc.mode = "value"
            continue
        targets, why = _linear_targets(sc, children, uses, masks)
        if why is None:
            sc.mode = "deferred"
            for h, p in targets:
                h._deferred.append((sc.result, p))
            if sc.result not in results and sc.result not in extra:
                extra.append(sc.result)
        else:
            before.append((sc, why))
    if before:
        import warnings
        for sc, why in before:
            warnings.warn(f"akstream: a {sc.label()} used in an expression reaches {why}: it needs its own pass over "
                          f"the files before this one (only multiplying histogram / sum weights is applied at the end, "
                          f"in the same pass)", RuntimeWarning, stacklevel=4)
        todo = list(dict.fromkeys(sc.result for sc, _ in before))
        compute(*todo, **opts)
        for sc, _ in before:
            sc.mode = "value"
        results = [r for r in results if not r._done]
    return results + extra


def compute(*results, workers=1, validate=0, exact=False, dense_fraction=0.25, cutoff="auto", max_entries=None,
            tasks_per_worker=4, engine="fused", batch_rows=None, read_ahead="auto"):
    """run one pass for every result and return their values (also stored in result.value).

    workers         processes (spawned; the graph is sent with cloudpickle); chunks are split in contiguous tasks
    validate        brute-force check of every variation on the first `validate` chunks
    exact           varied histograms refilled on every row (bitwise equal to a full recomputation)
    dense_fraction  above this fraction of changed rows a node is recomputed on every row
    cutoff          "auto": compare with nominal before expensive nodes and after collection selections; "all": after
                    every node
    max_entries     stop after this many entries (head / quick tests)
    engine          "fused": each variation on one sub-chunk of the rows that change (default); "delta": per-node
                    deltas (always used with exact=True)
    read_ahead      chunks read in a background thread ahead of the one being processed: "auto" (one, when the process
                    may use at least two cores), an int, or 0 (sequential)

    Chunks that could not be read (read_root(remote_strategy=1 or 2)) are listed in aks.last_failed after the run.
    batch_rows      fused: variations replacing the same branches run their sub-chunks stacked, up to this many rows
                    per pass (default half the chunk size: same memory as unstacked); 1 disables the stacking
    """
    flat = []
    for r in results:
        flat.extend(r if isinstance(r, (list, tuple)) else [r])
    requested = list(flat)
    flat = _resolve_scalars(flat, dict(workers=workers, validate=validate, exact=exact, dense_fraction=dense_fraction,
                                       cutoff=cutoff, max_entries=max_entries, tasks_per_worker=tasks_per_worker,
                                       engine=engine, batch_rows=batch_rows, read_ahead=read_ahead))
    if not flat:  # everything computed by the passes of the sums
        return [r._value for r in requested]
    plan = Plan(flat)
    chunks = plan.dataset.chunks()
    if max_entries is not None:
        kept, total = [], 0
        for path, s, e in chunks:
            if total >= max_entries:
                break
            e = min(e, s + max_entries - total)
            kept.append((path, s, e))
            total += e - s
        chunks = kept
    if engine not in ("fused", "delta"):
        raise ValueError("engine: 'fused' or 'delta'")
    opts = dict(validate=validate, exact=exact, dense_fraction=dense_fraction, cutoff=cutoff, engine=engine,
                batch_rows=batch_rows, read_ahead=read_ahead)
    t0 = time.perf_counter()
    if workers <= 1 or len(chunks) <= 1:
        states, stats = _run_chunks(plan, chunks, opts)
        states = [states[id(r)] for r in plan.results]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        if len(chunks) < 2 * workers:  # too few chunks to keep every worker busy: split them
            pieces = -(-2 * workers // len(chunks))
            split = []
            for path, a, b in chunks:
                edges = np.linspace(a, b, pieces + 1).astype(int)
                split.extend((path, x, y) for x, y in zip(edges[:-1], edges[1:]) if y > x)
            chunks = split
        ntasks = min(len(chunks), workers * tasks_per_worker)
        bounds = np.linspace(0, len(chunks), ntasks + 1).astype(int)
        tasks = [(chunks[a:b], opts, k) for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])) if b > a]
        import cloudpickle
        payload = cloudpickle.dumps(plan)
        # spawn, not fork: forking a process that already runs threads (BLAS, onnxruntime, uproot) can deadlock
        with ProcessPoolExecutor(min(workers, len(tasks)), mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker, initargs=(payload,)) as pool:
            outs = list(pool.map(_worker, tasks))
        states, stats = None, defaultdict(float)
        stats["failed"] = []
        for part_states, part_stats in outs:
            if states is None:
                states = [part_states[i] for i in range(len(plan.results))]
            else:
                states = [_merge(r, a, part_states[i]) for i, (r, a) in enumerate(zip(plan.results, states))]
            for k, v in part_stats.items():
                if isinstance(v, (int, float)):
                    stats[k] += v
            stats["failed"].extend(part_stats.get("failed", []))
        stats = dict(stats)
    stats["wall"] = time.perf_counter() - t0
    failed = list(plan.dataset.open_failures) + stats.pop("failed", [])
    if failed and workers > 1 and len(chunks) > 1:  # the workers' warnings stay in their processes
        import warnings
        warnings.warn(f"akstream: {len(failed)} chunks not read: {[(r['file'], r['start'], r['stop']) for r in failed]}"
                      f" (records in aks.last_failed)", RuntimeWarning, stacklevel=2)
    last_failed[:] = failed
    stats["failed_chunks"] = len(failed)
    last_stats.clear()
    last_stats.update(stats)
    deferred = {id(res) for r in plan.results for res, _ in getattr(r, "_deferred", ())}
    by_result = dict(zip((id(r) for r in plan.results), states))
    for r in sorted(plan.results, key=lambda r: id(r) not in deferred):  # the sums others are scaled by first
        r._set(_finalize(plan, r, by_result[id(r)]))
    return [r._value for r in requested if r._done]


def varied(result, variations="auto"):
    """ask for the systematic variations of a result: its value becomes {"nominal": ..., "<variation>": ...}."""
    result.variations = variations
    return result
