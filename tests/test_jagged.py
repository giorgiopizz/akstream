"""jagged.py fast paths against awkward: same values, dtypes and types, for every layout they accept."""
import operator

import awkward as ak
import numpy as np
import pytest

from akstream import jagged


def sample(n=300, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(3, n)
    return ak.unflatten(rng.normal(30, 20, counts.sum()).astype(dtype), counts)


def same(a, b):
    assert a.type == b.type, (str(a.type), str(b.type))
    assert ak.all(ak.num(a) == ak.num(b))
    fa, fb = ak.to_numpy(ak.flatten(a)), ak.to_numpy(ak.flatten(b))
    assert fa.dtype == fb.dtype and np.array_equal(fa, fb, equal_nan=fa.dtype.kind == "f")


LAYOUTS = {"offsets": lambda x: x, "sliced": lambda x: x[10:250], "listarray": lambda x: x[np.arange(len(x))[::-1]]}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fn,extra", [(operator.add, 2.5), (operator.mul, 2), (operator.truediv, 3.0),
                                      (operator.lt, 25), (operator.ge, 30.0), (np.abs, None), (np.sqrt, None),
                                      (operator.neg, None), (np.maximum, 20.0)])
def test_elementwise_matches_awkward(layout, fn, extra):
    x = LAYOUTS[layout](sample())
    args = (x,) if extra is None else (x, extra)
    got = jagged.elementwise(fn, args, {})
    same(got, fn(*args))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_elementwise_two_jagged_and_per_row(layout):
    x = LAYOUTS[layout](sample())
    y = LAYOUTS[layout](sample(seed=0) * 0.5)
    m = x > 20
    same(jagged.elementwise(operator.and_, (m, ~m), {}), m & ~m)
    same(jagged.elementwise(operator.sub, (x, y), {}), x - y)
    per_row = np.linspace(0, 50, len(x)).astype(np.float32)
    same(jagged.elementwise(operator.gt, (x, per_row), {}), x > per_row)
    same(jagged.elementwise(np.where, (m, x, 0.0), {}), ak.where(m, x, 0.0))


def test_elementwise_refuses_other_layouts():
    rec = ak.zip({"a": sample(), "b": sample()})
    assert jagged.elementwise(operator.add, (rec.a, rec.b), {}) is not None  # record fields are plain lists
    nested = ak.Array([[[1.0]], [[2.0, 3.0]]])
    assert jagged.elementwise(operator.add, (nested, 1.0), {}) is None
    opt = ak.Array([[1.0, None], []])
    assert jagged.elementwise(operator.add, (opt, 1.0), {}) is None
    a, b = sample(), sample(seed=1)  # different counts
    assert jagged.elementwise(operator.add, (a, b), {}) is None


@pytest.mark.parametrize("layout", LAYOUTS)
def test_mask_take_concat_combine(layout):
    x = LAYOUTS[layout](sample())
    m = x > 25
    same(jagged.mask(x, m), x[m])
    idx = np.random.default_rng(2).integers(0, len(x), 100)
    same(jagged.take(x, idx), x[idx])
    same(jagged.concat([x[:50], x[50:]]), ak.concatenate([x[:50], x[50:]]))
    from_b = np.random.default_rng(3).random(150) < 0.4
    a, b = x[:int((~from_b).sum())], x[:int(from_b.sum())] * 2
    src = np.empty(150, np.int64)
    src[~from_b] = np.arange(len(a))
    src[from_b] = len(a) + np.arange(len(b))
    same(jagged.combine(a, b, from_b), ak.concatenate([a, b])[src])


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_jagged_index(layout):
    x = LAYOUTS[layout](sample())
    for i in (ak.argsort(x)[:, ::-1], ak.local_index(x), ak.argsort(x)[:, :1], -1 - ak.local_index(x)):
        same(jagged.index(x, i), x[i])
    with pytest.raises(IndexError):
        jagged.index(x, ak.local_index(x) + 1)
