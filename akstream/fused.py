"""The fused variation pass: a variation evaluated on one sub-chunk of rows, as a hand-written loop would do it.

Per variation the plan splits the affected nodes into
    prefix      affected nodes not downstream of an expensive node (cheap: masks, selections, simple maths)
    downstream  the expensive nodes and everything after them
and marks the comparison points: prefix nodes read by the downstream part or by a result, the affected selection
masks of the prefix, and varied nodes of the downstream part.

Phase 1 evaluates the prefix on the rows where the varied inputs differ from nominal (the whole chunk, without copies,
when that is most of it) and compares the comparison points with nominal: the rows that change, R.
Phase 2 evaluates every needed node on the sub-chunk R as a plain memoised pass (no per-node row bookkeeping):
unaffected nodes are the nominal values on R, phase-1 values are reused, the rest is recomputed.
Histograms are corrected by -fill(nominal on R) + fill(varied on R) (or, when R is every row, the variation's fill
minus the nominal chunk fill).

Every expression is row-local, so the rows outside R keep their nominal values exactly: the result equals a full
recomputation (up to the float summation order of the histogram corrections).
"""
import operator
import time

import numpy as np

from . import rows as R
from .execute import _MISSING, ChunkContext, _deps, _masks
from .graph import Call, Node, Source, Take

DENSE_TRIGGER = 0.3  # above this fraction of rows with varied inputs, phase 1 runs on the whole chunk (no copies)
DENSE_CHANGED = 0.9  # above this fraction of changed rows, phase 2 runs on the whole chunk (no copies)


class FusedInfo:
    """the static split of one variation."""

    def __init__(self, compare, space_affected, value_affected, by_trigger=False):
        self.compare = compare  # nodes compared with nominal in phase 1
        self.by_trigger = by_trigger  # the rows where the varied inputs change are changed rows too (written results)
        self.space_affected = space_affected  # ids of the spaces whose rows the variation can change
        # ids of the nodes whose value on a row can change (the others are affected only because their selection gains
        # or loses rows: a row that stays keeps its nominal value)
        self.value_affected = value_affected


def plan_variation(variation, affected, subst, order, children, result_nodes, by_trigger=False):
    by_id = {n.id: n for n in order}
    downstream = set()
    stack = [i for i in affected if getattr(by_id.get(i), "expensive", False)]
    while stack:
        i = stack.pop()
        if i in downstream:
            continue
        downstream.add(i)
        stack.extend(c.id for c in children.get(i, ()) if c.id in affected)
    prefix = set(affected) - downstream
    compare = {}
    triggered = {}
    for i in downstream:
        n = by_id.get(i)
        if getattr(n, "trigger", None):
            triggered[i] = n
            for t in n.trigger:
                if t.id in prefix:
                    compare[t.id] = t
    for i in prefix:
        n = by_id.get(i)
        if n is None:
            continue
        consumers = [c for c in children.get(i, ()) if c.id in downstream]
        # an argument of a triggered node is not compared (its triggers are), unless something else reads it
        consumers = [c for c in consumers if c.id not in triggered or n in c.trigger]
        if i in result_nodes or consumers:
            compare[i] = n
    spaces = {}
    for n in order:
        for s in n.space.ancestors():
            spaces[s.id] = s
    space_affected = set()
    for s in spaces.values():
        if any(a.mask is not None and a.mask.id in affected for a in s.ancestors()):
            space_affected.add(s.id)
        if s.mask is not None and s.mask.id in prefix:
            compare[s.mask.id] = s.mask
    for i in subst:  # a varied node of the downstream part: compared on its own
        if i in downstream and i in by_id:
            compare[i] = by_id[i]
    # x[mask] with x untouched by the variation changes only where the mask changes: compare the mask instead (it is
    # computed anyway; the masked column is then only built in phase 2, on the changed rows)
    for i, n in list(compare.items()):
        if isinstance(n, Call) and n.fn is operator.getitem and len(n.args) == 2 and isinstance(n.args[0], Node) \
                and isinstance(n.args[1], Node) and n.args[0].id not in affected and n.args[1].id in affected \
                and n.args[1].space is n.space and i not in result_nodes and i not in subst:
            del compare[i]
            compare[n.args[1].id] = n.args[1]
    value_affected = set(subst)
    stack = list(subst)
    while stack:
        i = stack.pop()
        for ch in children.get(i, ()):
            if ch.id in value_affected:
                continue
            if isinstance(ch, Take) and ch.parent.id != i:  # only the selection mask changed: membership, not value
                continue
            value_affected.add(ch.id)
            stack.append(ch.id)
    return FusedInfo(list(compare.values()), space_affected, value_affected, by_trigger)


class SubContext(ChunkContext):
    """the values of one variation on a subset of the chunk's rows (rows: sorted root ids, None = every row)."""

    def __init__(self, base, rows, subst, affected, space_affected, seed=None, value_affected=None, parent=None):
        super().__init__(base.chunk)
        self.base, self.rows = base, rows
        self.subst, self.affected, self.space_affected = subst, affected, space_affected
        self.value_affected = value_affected
        self.parent = parent  # a pass on rows added to a selection: value-affected nodes come from the parent pass
        self._added = {}
        self._parent_pos = {}
        self.seed = seed
        self.m = base.n if rows is None else len(rows)
        self._seed_pos = {}

    # ------------------------------------------------------------ spaces (the root space is `rows`)
    def index(self, space):
        if space.is_root:
            return self.rows
        return super().index(space)

    def size(self, space):
        return self.m if space.is_root else len(self.local_index(space))

    def positions(self, space, ids):
        idx = self.index(space)
        if idx is None:
            return ids
        if len(ids) < 64:
            return np.searchsorted(idx, ids)
        return self.inverse(space)[ids]

    def root_ids(self, space):
        idx = self.index(space)
        return np.arange(self.base.n) if idx is None else idx

    # ------------------------------------------------------------ values
    def value(self, node):
        v = self.values.get(node.id, _MISSING)
        if v is not _MISSING:
            return v
        values = self.values
        stack = [node]
        while stack:  # dependencies first, without deep recursion (see ChunkContext.value)
            n = stack[-1]
            if n.id in values:
                stack.pop()
                continue
            missing = [d for d in self._sub_deps(n) if d.id not in values]
            if missing:
                stack.extend(missing)
                continue
            stack.pop()
            values[n.id] = self._sub(n)
        return values[node.id]

    def _sub_deps(self, node):
        """what _sub(node) will read from this pass (the same decision tree)."""
        alt = self.subst.get(node.id)
        if alt is not None:
            return [alt]
        masks = _masks(node.space, self._local)
        if self.parent is not None and node.id in self.parent.value_affected:
            return masks
        if self.seed is not None and node.id in self.seed.values:
            return masks
        if node.id not in self.affected and node.space.id not in self.space_affected:
            return masks
        if isinstance(node, Source):
            return []
        if self.value_affected is not None and node.id not in self.value_affected:
            return masks
        return _deps(node, self._local)

    def _sub(self, node):
        alt = self.subst.get(node.id)
        if alt is not None:
            return self.value(alt)
        if self.parent is not None and node.id in self.parent.value_affected:
            pos = self._parent_pos.get(node.space.id)
            if pos is None:
                pos = self._parent_pos[node.space.id] = self.parent.positions(node.space, self.root_ids(node.space))
            return R.take(self.parent.value(node), pos)
        if self.seed is not None:
            v = self.seed.values.get(node.id, _MISSING)
            if v is not _MISSING:
                pos = self._seed_positions(node.space)
                return v if pos is None else R.take(v, pos)
        if node.id not in self.affected and node.space.id not in self.space_affected:
            if self.rows is None:
                return self.base.value(node)
            return R.take(self.base.value(node), self.base.positions(node.space, self.index(node.space)))
        if isinstance(node, Source):
            a = self.chunk.get(node.branch)
            return a if self.rows is None else R.take(a, self.rows)
        if self.value_affected is not None and node.id not in self.value_affected:
            return self._membership_only(node)
        return self._compute(node)

    def free(self):
        """drop every value (and break the parent <-> added-rows pass cycle) as soon as the variation is filled."""
        for mini in self._added.values():
            mini.free()
        self._added.clear()
        self.values.clear()
        self._local.clear()
        self._index.clear()
        self._inverse.clear()
        self._rows_of.clear()
        self.seed = self.parent = None

    def _membership_only(self, node):
        """a node whose value does not depend on the variation, in a selection that gains / loses rows: nominal
        values on the rows that were already selected, computed only on the rows the variation adds."""
        space = node.space
        ids = self.root_ids(space)
        in_nominal = self.base.is_row(space, ids)
        nominal = R.take(self.base.value(node), self.base.positions(space, ids[in_nominal]))
        if in_nominal.all():
            return nominal
        added = ids[~in_nominal]
        mini = self._added.get(space.id)
        if mini is None:  # a pass on the added rows only
            mini = self._added[space.id] = SubContext(self.base, added, self.subst, self.affected, self.space_affected,
                                                      parent=self)
        values = R.take(mini.value(node), mini.positions(space, added))
        return R.combine(nominal, values, ~in_nominal)

    def _seed_positions(self, space):
        """positions of this pass's rows of `space` among the seed pass's rows (None: the same rows)."""
        if space.id in self._seed_pos:
            return self._seed_pos[space.id]
        mine, theirs = self.index(space), self.seed.index(space)
        if mine is None and theirs is None:
            pos = None
        elif theirs is None:
            pos = mine
        else:
            pos = np.searchsorted(theirs, mine if mine is not None else np.arange(self.base.n))
            if len(pos) == len(theirs):
                pos = None
        self._seed_pos[space.id] = pos
        return pos


class BatchContext(SubContext):
    """several variations of the same structure (they replace the same branches) evaluated in one pass: the rows of
    variation k are stacked as composite ids k * N + root id (N = chunk size), so every row-local function runs once
    for all of them. Varied branches and phase-1 values are taken per variation; everything else is shared."""

    def __init__(self, base, rows, substs, seeds, affected, space_affected, value_affected, parent=None):
        super().__init__(base, rows, {}, affected, space_affected, value_affected=value_affected, parent=parent)
        self.N, self.K = base.n, len(substs)
        self.substs, self.seeds = substs, seeds
        self.subst_ids = set(substs[0])
        self._bounds = {}

    # composite ids: positions by binary search (an inverse map would be K x N long)
    def positions(self, space, ids):
        return np.searchsorted(self.root_ids(space), ids)

    def parts(self, space):
        """[(k, slice)] of the rows of `space` that belong to variation k."""
        b = self._bounds.get(space.id)
        if b is None:
            ids = self.root_ids(space)
            b = self._bounds[space.id] = np.searchsorted(ids, np.arange(self.K + 1) * self.N)
        return [(k, slice(b[k], b[k + 1])) for k in range(self.K)]

    def _sub_deps(self, node):
        masks = _masks(node.space, self._local)
        if node.id in self.subst_ids:
            return masks
        if self.parent is not None and node.id in self.parent.value_affected:
            return masks
        if all(node.id in sd.values for sd in self.seeds):
            return masks
        if node.id not in self.affected and node.space.id not in self.space_affected:
            return masks
        if isinstance(node, Source):
            return []
        if self.value_affected is not None and node.id not in self.value_affected:
            return masks
        return _deps(node, self._local)

    def _per_variation(self, node, values_of):
        """concatenate, variation by variation, values_of(k, root ids of variation k's rows of node.space)."""
        ids = self.root_ids(node.space)
        pieces = [values_of(k, ids[sl] - k * self.N) for k, sl in self.parts(node.space)]
        return R.concat(pieces)

    def _sub(self, node):
        if node.id in self.subst_ids:  # a varied branch: each variation's alternative
            return self._per_variation(node, lambda k, ids: R.take(self.base.value(self.substs[k][node.id]),
                                                                   self.base.positions(node.space, ids)))
        if self.parent is not None and node.id in self.parent.value_affected:
            return R.take(self.parent.value(node), self.parent.positions(node.space, self.root_ids(node.space)))
        if all(node.id in sd.values for sd in self.seeds):  # computed in each variation's phase 1
            return self._per_variation(node, lambda k, ids: R.take(
                self.seeds[k].values[node.id], np.searchsorted(self.seeds[k].root_ids(node.space), ids)))
        if node.id not in self.affected and node.space.id not in self.space_affected:
            return R.take(self.base.value(node), self.base.positions(node.space, self.root_ids(node.space) % self.N))
        if isinstance(node, Source):
            return R.take(self.chunk.get(node.branch), self.rows % self.N)
        if self.value_affected is not None and node.id not in self.value_affected:
            return self._membership_only(node)
        return self._compute(node)

    def _membership_only(self, node):
        space = node.space
        ids = self.root_ids(space)
        root = ids % self.N
        in_nominal = self.base.is_row(space, root)
        nominal = R.take(self.base.value(node), self.base.positions(space, root[in_nominal]))
        if in_nominal.all():
            return nominal
        added = ids[~in_nominal]
        mini = self._added.get(space.id)
        if mini is None:
            mini = self._added[space.id] = BatchContext(self.base, added, self.substs, self.seeds, self.affected,
                                                        self.space_affected, None, parent=self)
        values = R.take(mini.value(node), mini.positions(space, added))
        return R.combine(nominal, values, ~in_nominal)

    def free(self):
        super().free()
        self.seeds = []
        self._bounds.clear()


def trigger_rows(ctx, plan, variation, exact=False):
    """root ids of the rows where a varied node differs from nominal (None: most rows, use the whole chunk; exact: the
    ids in any case)."""
    parts = []
    for node_id, alt in plan.subst[variation].items():
        node = plan.by_id[node_id]
        differs = R.row_differs(ctx.value(alt), ctx.value(node))
        ids = np.flatnonzero(differs)
        idx = ctx.index(node.space)
        parts.append(ids if idx is None else idx[ids])
    rows = R.union(*parts)
    if len(rows) > DENSE_TRIGGER * ctx.n and not exact:
        return None
    return rows


def _masked_changes(sub, ctx, node):
    """x[m] (jagged boolean mask m, unselected space): it changes on a row iff m changes there or an element of x
    selected by m changes -- computed on the flat contents, without building x[m]. None if not applicable."""
    if not (isinstance(node, Call) and node.fn is operator.getitem and len(node.args) == 2
            and isinstance(node.args[0], Node) and isinstance(node.args[1], Node)):
        return None
    x, m = node.args
    if x.space is not node.space or m.space is not node.space or node.space.id in sub.space_affected:
        return None
    xv, mv, xn, mn = sub.value(x), sub.value(m), None, None
    if sub.rows is None:
        xn, mn = ctx.value(x), ctx.value(m)
    else:
        pos = ctx.positions(node.space, sub.root_ids(node.space))
        xn, mn = R.take(ctx.value(x), pos), R.take(ctx.value(m), pos)
    out = R.masked_row_differs(xv, xn, mv, mn)
    if out is None:
        return None
    return sub.root_ids(node.space)[out]


def changed_rows(sub, ctx, node):
    """root ids where the phase-1 value of `node` differs from nominal (rows entering or leaving its space count)."""
    fast = _masked_changes(sub, ctx, node)
    if fast is not None:
        return fast
    space = node.space
    ids = sub.root_ids(space)
    values = sub.value(node)
    in_nominal = None if (space.is_root or space.id not in sub.space_affected) else ctx.is_row(space, ids)
    if in_nominal is None or in_nominal.all():
        nominal = ctx.value(node) if (sub.rows is None and space.id not in sub.space_affected) else \
            R.take(ctx.value(node), ctx.positions(space, ids))
        return ids[R.row_differs(values, nominal)]
    keep = np.flatnonzero(in_nominal)
    nominal = R.take(ctx.value(node), ctx.positions(space, ids[keep]))
    differs = R.row_differs(R.take(values, keep), nominal)
    entering = ids[~in_nominal]
    leaving = R.EMPTY
    if sub.rows is not None:  # nominal rows of the space inside the phase-1 rows that are no longer in it
        nominal_ids = ctx.index(space)
        inside = nominal_ids[R.member(nominal_ids, sub.rows)]
        leaving = R.difference(inside, ids)
    else:
        leaving = R.difference(ctx.index(space), ids)
    return R.union(ids[keep][differs], entering, leaving)


def phase1(plan, ctx, variation, stats):
    """(phase-1 pass, changed root ids) of one variation, or None when nothing changes."""
    info = plan.fused[variation]
    t0 = time.perf_counter()
    trigger = trigger_rows(ctx, plan, variation, exact=info.by_trigger)
    rows1 = None if trigger is not None and len(trigger) > DENSE_TRIGGER * ctx.n else trigger
    if trigger is not None and not len(trigger):
        stats["fused_phase1"] += time.perf_counter() - t0
        return None
    p1 = SubContext(ctx, rows1, plan.subst[variation], plan.affected[variation], info.space_affected,
                    value_affected=info.value_affected)
    changed = R.union(*[changed_rows(p1, ctx, n) for n in info.compare], *([trigger] if info.by_trigger else []))
    stats["fused_phase1"] += time.perf_counter() - t0
    if not len(changed):
        p1.free()
        return None
    return p1, changed


def run_variation(plan, ctx, variation, states, opts, stats):
    found = phase1(plan, ctx, variation, stats)
    if found is None:
        return
    p1, changed = found
    _phase2_single(plan, ctx, variation, p1, changed, states, stats)


def _phase2_single(plan, ctx, variation, p1, changed, states, stats):
    info = plan.fused[variation]
    t1 = time.perf_counter()
    rows2 = None if len(changed) > DENSE_CHANGED * ctx.n else changed
    p2 = SubContext(ctx, rows2, plan.subst[variation], plan.affected[variation], info.space_affected, seed=p1,
                    value_affected=info.value_affected)
    stats["fused_rows"] += p2.m
    try:
        fill_corrections(plan.varied_results[variation], states, ctx, p2, changed, variation)
    finally:
        p2.free()
        p1.free()
    stats["fused_phase2"] += time.perf_counter() - t1


def run_batch(plan, ctx, variations, states, opts, stats):
    """variations of the same structure: phase 1 each, phase 2 stacked in batches of at most batch_rows rows."""
    limit = opts.get("batch_rows") or max(ctx.n // 2, 1)
    pending, rows = [], 0
    for v in variations:
        found = phase1(plan, ctx, v, stats)
        if found is None:
            continue
        pending.append((v,) + found)
        rows += len(found[1])
        if rows >= limit:
            _phase2_batch(plan, ctx, pending, states, stats)
            pending, rows = [], 0
    if pending:
        _phase2_batch(plan, ctx, pending, states, stats)


def _phase2_batch(plan, ctx, pending, states, stats):
    if len(pending) == 1:
        v, p1, changed = pending[0]
        _phase2_single(plan, ctx, v, p1, changed, states, stats)
        return
    t1 = time.perf_counter()
    first = pending[0][0]
    info = plan.fused[first]
    n = ctx.n
    rows = np.concatenate([k * n + changed for k, (_, _, changed) in enumerate(pending)])
    p2 = BatchContext(ctx, rows, [plan.subst[v] for v, _, _ in pending], [p1 for _, p1, _ in pending],
                      plan.affected[first], info.space_affected, info.value_affected)
    stats["fused_rows"] += len(rows)
    stats["fused_batches"] += 1
    try:
        fill_corrections_batched(plan.varied_results[first], states, ctx, p2,
                                 [(v, changed) for v, _, changed in pending])
    finally:
        p2.free()
        for _, p1, _ in pending:
            p1.free()
    stats["fused_phase2"] += time.perf_counter() - t1


def fill_corrections_batched(results, states, ctx, sub, parts):
    """the corrections of every variation of a batch: varied values from the stacked pass (variation k's rows),
    nominal values on each variation's changed rows; one bincount per (columns, weight, axes) group and side."""
    for w in [r for r in results if hasattr(r, "record_variation")]:  # file writers: the varied rows themselves
        ids = sub.root_ids(w.space)
        for (k, sl), (v, changed) in zip(sub.parts(w.space), parts):
            rows = ids[sl]
            w.record_variation(states[id(w)], v, changed, rows - k * ctx.n,
                               [R.take(sub.value(n), sub.positions(n.space, rows)) for n in w.columns])
    results = [r for r in results if not hasattr(r, "record_variation")]
    groups = _hist_groups(results)
    old_cache = {}
    for hists in groups.values():
        new_targets, old_targets = [], []
        for h in hists:
            ids = sub.root_ids(h.space)
            for (k, sl), (v, changed) in zip(sub.parts(h.space), parts):
                st = states[id(h)]
                acc = st["corr"].get(v)
                if acc is None:
                    acc = st["corr"][v] = type(st["chunk"])(h.size)
                new_targets.append((h, acc, ids[sl]))
                key = (k, h.space.id)
                old = old_cache.get(key)
                if old is None:
                    old = old_cache[key] = changed if h.space.is_root else changed[ctx.is_row(h.space, changed)]
                old_targets.append((h, acc, old))
        _fill_targets(hists[0], new_targets, lambda n, ids: R.take(sub.value(n), sub.positions(n.space, ids)), 1)
        _fill_targets(hists[0], old_targets, lambda n, ids: R.take(ctx.value(n), ctx.positions(n.space, ids)), -1)


def _hist_groups(results):
    groups = {}
    for h in results:
        if getattr(h, "axes", None) and all(n is not None for n in h.nodes()):
            k = (tuple(n.id for n in h.columns), h.weight.id if h.weight is not None else None,
                 tuple(_axis_key(a) for a in h.axes))
        else:
            k = ("single", id(h))
        groups.setdefault(k, []).append(h)
    return groups


def _fill_targets(first, targets, values_at, sign):
    """add sign x the fill of each (hist, acc, rows) target; the histograms share columns, weight and axes: the values
    and bins are computed once on the union of the rows, then one bincount with the target as a bin offset."""
    targets = [t for t in targets if len(t[2])]
    if not targets:
        return
    union = R.union(*[t[2] for t in targets])
    values = {n.id: values_at(n, union) for n in first.nodes()}
    if not first.axes or not all(isinstance(v, np.ndarray) for v in values.values()):
        for h, acc, rows in targets:  # no axes or jagged columns: one target at a time
            acc.add(h._fill_with(lambda n, rows=rows: values_at(n, rows), len(rows)), sign)
        return
    from .results import _boost_axis, _fill_arrays, accumulate
    size = first.size
    sels = [slice(None) if len(rows) == len(union) else np.searchsorted(union, rows) for _, _, rows in targets]
    cols = [values[c.id] for c in first.columns]
    weight = values[first.weight.id] if first.weight is not None else None
    if all(_boost_axis(a) is not None for a in first.axes):
        # boost-convention binning: boost-histogram bins and accumulates every target in one fill
        count, sumw, sumw2 = accumulate(
            first.axes, [np.concatenate([c[sel] for sel in sels]) for c in cols],
            None if weight is None else np.concatenate([weight[sel] for sel in sels]),
            target=np.repeat(np.arange(len(targets)), [len(rows) for _, _, rows in targets]), ntargets=len(targets))
    else:
        # ROOT-convention binning: the bin index once on the union, the target as an offset, numpy bincount
        index, weight = _fill_arrays(first.axes, cols, weight)
        idx = np.concatenate([index[sel] + t * size for t, sel in enumerate(sels)])
        total = size * len(targets)
        count = np.bincount(idx, minlength=total)
        if weight is None:
            sumw = sumw2 = count.astype(np.float64)
        else:
            w = np.concatenate([weight[sel] for sel in sels])
            sumw = np.bincount(idx, weights=w, minlength=total)
            sumw2 = np.bincount(idx, weights=w * w, minlength=total)
    for t, (h, acc, rows) in enumerate(targets):
        part = slice(t * size, (t + 1) * size)
        if sign > 0:
            acc.sumw += sumw[part]
            acc.sumw2 += sumw2[part]
            acc.count += count[part]
        else:
            acc.sumw -= sumw[part]
            acc.sumw2 -= sumw2[part]
            acc.count -= count[part]


def fill_corrections(results, states, ctx, sub, changed, variation):
    """histogram corrections of one variation from its sub-chunk pass. Histograms with the same columns, weight and
    binning on different selections (a variable in several regions) are filled together: their bins are computed once
    on the union of the rows, then one bincount with the region as an offset of the bin index."""
    for w in [r for r in results if hasattr(r, "record_variation")]:  # file writers: the varied rows themselves
        ids = sub.root_ids(w.space)
        w.record_variation(states[id(w)], variation, changed, ids,
                           [R.take(sub.value(n), sub.positions(n.space, ids)) for n in w.columns],
                           replace_all=sub.rows is None)
    results = [r for r in results if not hasattr(r, "record_variation")]
    groups = {}
    for h in results:
        if getattr(h, "axes", None) and all(n is not None for n in h.nodes()):
            k = (tuple(n.id for n in h.columns), h.weight.id if h.weight is not None else None,
                 tuple(_axis_key(a) for a in h.axes))
        else:
            k = ("single", id(h))
        groups.setdefault(k, []).append(h)
    old_rows = {}
    cache = {}  # shared by the groups: rows per selection, their union and positions, column values on the union

    def old_of(space):
        o = old_rows.get(space.id)
        if o is None:
            o = old_rows[space.id] = changed if space.is_root else changed[ctx.is_row(space, changed)]
        return o

    for hists in groups.values():
        accs = []
        for h in hists:
            st = states[id(h)]
            corr = st["corr"].get(variation)
            if corr is None:
                corr = st["corr"][variation] = type(st["chunk"])(h.size)
            accs.append(corr)
        # varied: every row of each selection in the sub-chunk
        _grouped_fill(hists, accs, "new", lambda sp: sub.root_ids(sp),
                      lambda n, rows: R.take(sub.value(n), sub.positions(n.space, rows)), 1, cache)
        if sub.rows is None:  # every row recomputed: subtract the nominal chunk fill
            for h, acc in zip(hists, accs):
                acc.add(states[id(h)]["chunk"], -1)
            continue
        _grouped_fill(hists, accs, "old", old_of,
                      lambda n, rows: R.take(ctx.value(n), ctx.positions(n.space, rows)), -1, cache)


def _axis_key(a):
    if hasattr(a, "edges"):
        return ("v",) + tuple(a.edges.tolist())
    return ("r", a.bins, a.start, a.stop, getattr(a, "convention", "boost"))


def _grouped_fill(hists, accs, side, rows_of, values_at, sign, cache):
    """fill each histogram with the rows of its selection (rows_of(space): root ids); the group shares columns,
    weight and axes. `cache` keeps, per side and set of selections, the union of the rows and each selection's
    positions in it, and the column values on that union (shared with the other variables)."""
    first = hists[0]
    spaces = tuple(h.space.id for h in hists)
    key = (side, spaces)
    hit = cache.get(key)
    if hit is None:
        rows = [rows_of(h.space) for h in hists]
        union = R.union(*[r for r in rows if len(r)])
        sels = [None if len(r) == len(union) else np.searchsorted(union, r) for r in rows]
        hit = cache[key] = (rows, union, sels)
    rows_per_hist, union, sels = hit
    if not len(union):
        return
    if len(hists) == 1 or not first.axes:
        for h, acc, rows in zip(hists, accs, rows_per_hist):
            if len(rows):
                acc.add(h._fill_with(lambda n, rows=rows: values_at(n, rows), len(rows)), sign)
        return
    from .results import _fill_arrays

    def value(n):
        k = (side, spaces, n.id)
        if k not in cache:
            cache[k] = values_at(n, union)
        return cache[k]

    values = {n.id: value(n) for n in first.nodes()}
    if not all(isinstance(v, np.ndarray) for v in values.values()):  # jagged: one entry per object, fill one by one
        for h, acc, rows in zip(hists, accs, rows_per_hist):
            if len(rows):
                acc.add(h._fill_with(lambda n, rows=rows: values_at(n, rows), len(rows)), sign)
        return
    index, weight = _fill_arrays(first.axes, [values[c.id] for c in first.columns],
                                 values[first.weight.id] if first.weight is not None else None)
    size = first.size
    parts_i, parts_w = [], []
    for k, (rows, sel) in enumerate(zip(rows_per_hist, sels)):
        if not len(rows):
            continue
        parts_i.append((index if sel is None else index[sel]) + k * size)
        if weight is not None:
            parts_w.append(weight if sel is None else weight[sel])
    if not parts_i:
        return
    idx = np.concatenate(parts_i)
    total = size * len(hists)
    count = np.bincount(idx, minlength=total)
    if weight is None:
        sumw = sumw2 = count.astype(np.float64)
    else:
        w = np.concatenate(parts_w)
        sumw = np.bincount(idx, weights=w, minlength=total)
        sumw2 = np.bincount(idx, weights=w * w, minlength=total)
    for k, acc in enumerate(accs):
        part = slice(k * size, (k + 1) * size)
        if sign > 0:
            acc.sumw += sumw[part]
            acc.sumw2 += sumw2[part]
            acc.count += count[part]
        else:
            acc.sumw -= sumw[part]
            acc.sumw2 -= sumw2[part]
            acc.count -= count[part]
