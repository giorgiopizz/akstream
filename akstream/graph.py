"""The lazy graph: row spaces and nodes.

A *space* is a set of rows: the root space of a dataset (every entry of the chunk) or a selection `parent[mask]`. Every
node produces one value per row of its space. Nodes are hash-consed per dataset (identical expressions are one node).

Node kinds
    Source  a branch of the input tree (root space only)
    Call    fn(*args, **kwargs) with node arguments resolved to their values; must be row-local (row i of the output
            depends only on row i of the inputs) -- this is what makes partial recomputation exact
    Take    the value of a parent-space node on the rows of a child space
    Zip     a record (jagged collection) built from field nodes, each field loaded only when touched

Variations are attached to nodes: node.variations[name] = alternative node (same space). Every node downstream of a
varied node is affected by that variation.
"""
import itertools

import numpy as np

_ids = itertools.count()


class Space:
    def __init__(self, dataset, parent=None, mask=None):
        self.id = next(_ids)
        self.dataset = dataset
        self.parent = parent
        self.mask = mask  # Node in parent space, boolean per row

    @property
    def is_root(self):
        return self.parent is None

    def ancestors(self):
        s = self
        while s is not None:
            yield s
            s = s.parent

    def __repr__(self):
        return "Space(root)" if self.is_root else f"Space({self.id} <- {self.parent.id})"


class Node:
    kind = "node"

    def __init__(self, space):
        self.id = next(_ids)
        self.space = space
        self.variations = {}
        self.cutoff = False  # compare with nominal after recomputation and drop the rows that did not change
        self.expensive = False  # cut off its inputs before recomputing it (DNNs, user functions)

    def inputs(self):
        """the nodes this node's value is computed from (including the masks of its space)."""
        return []

    def __repr__(self):
        return f"<{self.kind} {self.id} {self.label()}>"

    def label(self):
        return ""


class Source(Node):
    kind = "source"

    def __init__(self, space, branch, jagged, dtype=None):
        super().__init__(space)
        self.branch, self.jagged, self.dtype = branch, jagged, dtype

    def label(self):
        return self.branch


class Call(Node):
    kind = "call"

    def __init__(self, space, fn, args, kwargs, name=None):
        super().__init__(space)
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.name = name or getattr(fn, "__name__", type(fn).__name__)
        self._inputs = list(_nodes_in((args, kwargs)))
        self.trigger = None  # aks.apply(when=...): the nodes whose change on a row can change this node's value there

    def set_trigger(self, nodes):
        self.trigger = list(nodes)
        self._inputs = list({n.id: n for n in self._inputs + self.trigger}.values())

    def inputs(self):
        return self._inputs

    def label(self):
        return self.name


class Scalar(Node):
    """a sum / count result inside an expression (df.genWeight / df.sum("genWeight")): one number, per sample with
    several samples. The engine (engine._resolve_scalars) either computes it in a pass before this one (mode "value")
    or, when it only scales histogram / sum weights, evaluates it as 1 and applies it to those results at the end
    (mode "deferred")."""
    kind = "scalar"

    def __init__(self, space, result):
        super().__init__(space)
        self.result = result
        self.mode = None

    def label(self):
        return f"{type(self.result).__name__.lower()}"

    def value_for(self, sample):
        if self.mode == "deferred":
            return 1.0
        if not self.result._done:
            raise RuntimeError("a sum used in an expression is not computed yet (the engine computes it first)")
        v = self.result._value
        return v[sample] if isinstance(v, dict) else v


class Take(Node):
    kind = "take"

    def __init__(self, space, parent):
        super().__init__(space)
        self.parent = parent

    def inputs(self):
        return [self.parent, self.space.mask]

    def label(self):
        return f"{self.parent.label()}@{self.space.id}"


class Zip(Node):
    """a record per row (counts is None) or a jagged collection of records (counts = node with the number of objects)."""
    kind = "zip"

    def __init__(self, space, fields, counts):
        super().__init__(space)
        self.fields, self.counts = dict(fields), counts

    def inputs(self):
        return list(self.fields.values()) + ([self.counts] if self.counts is not None else [])

    def label(self):
        return "{" + ",".join(self.fields) + "}"


def lift(node):
    """the node a selection's column is taken from (through Take nodes without variations of their own): consumers in
    a selection can read it at their rows directly, sharing its variation deltas with every other selection."""
    while isinstance(node, Take) and not node.variations:
        node = node.parent
    return node


# ---------------------------------------------------------------- structure helpers
def _nodes_in(obj):
    if isinstance(obj, Node):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _nodes_in(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _nodes_in(x)


def resolve(obj, value):
    """obj with every Node replaced by value(node) (lists, tuples, dicts followed)."""
    if isinstance(obj, Node):
        return value(obj)
    if isinstance(obj, tuple):
        return tuple(resolve(x, value) for x in obj)
    if isinstance(obj, list):
        return [resolve(x, value) for x in obj]
    if isinstance(obj, dict):
        return {k: resolve(x, value) for k, x in obj.items()}
    return obj


def key(obj):
    """hash-consing key of a call argument (falls back to identity for unhashable constants)."""
    if isinstance(obj, Node):
        return ("n", obj.id)
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__,) + tuple(key(x) for x in obj)
    if isinstance(obj, dict):
        return ("d",) + tuple((k, key(v)) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])))
    if isinstance(obj, slice):
        return ("s", obj.start, obj.stop, obj.step)
    if isinstance(obj, np.ndarray):
        return ("o", id(obj))
    try:
        hash(obj)
    except TypeError:
        return ("o", id(obj))
    return ("c", type(obj).__name__, obj)


def walk(roots):
    """every node reachable from `roots` (inputs, Take parents and space masks), in dependency order."""
    order, seen = [], set()
    stack = [(n, False) for n in reversed(list(roots))]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if node.id in seen:
            continue
        seen.add(node.id)
        stack.append((node, True))
        for child in reversed(node.inputs()):
            if child is not None and child.id not in seen:
                stack.append((child, False))
    return order
