import pytest

import akstream as aks
from akstream.execute import NonLocalError


def test_reduction_across_rows_is_refused(events_file):
    df = aks.read_root(events_file)
    df["bad"] = df.mll - aks.mean(df.mll)  # axis=None: a per-chunk mean would silently be wrong
    with pytest.raises(NonLocalError):
        df.collect(["bad"]).value


def test_sort_of_flat_column_is_refused(events_file):
    df = aks.read_root(events_file)
    df["bad"] = aks.sort(df.mll)  # axis=-1 of a flat column = across rows
    with pytest.raises(NonLocalError):
        df.collect(["bad"]).value


def test_mixing_selections_is_refused(events_file):
    df = aks.read_root(events_file)
    sel = df[df.mll > 100]
    with pytest.raises(ValueError):
        sel["x"] = df.mll * 2
    with pytest.raises(ValueError):
        sel.mll + df.mll


def test_truth_value(events_file):
    df = aks.read_root(events_file)
    with pytest.raises(TypeError):
        if df.mll > 100:
            pass


def test_unknown_numpy_function(events_file):
    df = aks.read_root(events_file)
    import numpy as np
    with pytest.raises(TypeError):
        np.cumsum(df.mll)
