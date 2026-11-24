"""Synthetic NanoAOD-like trees: jets with sparse JES-like variations, two muons, weights."""
import numpy as np
import awkward as ak
import pytest
import uproot


def make_events(n, seed=1):
    rng = np.random.default_rng(seed)
    njet = rng.poisson(2.5, n)
    total = int(njet.sum())
    pt = rng.exponential(30, total) + 15
    eta = rng.uniform(-4.7, 4.7, total)
    phi = rng.uniform(-np.pi, np.pi, total)
    mass = rng.exponential(5, total) + 1
    # a sparse "jer_4"-like variation: only jets with 2.5 < |eta| < 3 move; a dense "jes" one: every jet moves
    sparse = (np.abs(eta) > 2.5) & (np.abs(eta) < 3.0)
    out = {
        "event": np.arange(n, dtype=np.int64),
        "mll": rng.uniform(70, 150, n),
        "mu1_pt": rng.exponential(30, n) + 20,
        "mu2_pt": rng.exponential(20, n) + 15,
        "genWeight": rng.normal(1, 0.3, n),
        "pu": rng.uniform(0.8, 1.2, n),
        "pu_up": rng.uniform(0.8, 1.2, n),
        "nJet": njet.astype(np.int32),
    }
    for name, values in {"pt": pt, "eta": eta, "phi": phi, "mass": mass}.items():
        out[f"Jet_{name}"] = values
    for tag, f in (("up", 1.05), ("down", 0.95)):
        out[f"Jet_pt_jer4_{tag}"] = np.where(sparse, pt * f, pt)
        out[f"Jet_mass_jer4_{tag}"] = np.where(sparse, mass * f, mass)
        out[f"Jet_pt_jes_{tag}"] = pt * (1 + (f - 1) * 0.4)
        out[f"Jet_mass_jes_{tag}"] = mass * (1 + (f - 1) * 0.4)
    return out, njet


def write_tree(path, n, seed=1):
    data, njet = make_events(n, seed)
    branches = {}
    jet = {}
    for k, v in data.items():
        if k.startswith("Jet_"):
            jet[k[4:]] = ak.unflatten(v, njet)
        elif k != "nJet":
            branches[k] = v
    branches["Jet"] = ak.zip(jet)
    with uproot.recreate(path) as f:
        f.mktree("Events", {k: (v.type.content if isinstance(v, ak.Array) else v.dtype) for k, v in branches.items()})
        f["Events"].extend(branches)
    return path


@pytest.fixture(scope="session")
def events_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return str(write_tree(d / "events.root", 20_000))


@pytest.fixture(scope="session")
def two_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("data2")
    return [str(write_tree(d / "a.root", 7_000, 2)), str(write_tree(d / "b.root", 5_000, 3))]


@pytest.fixture(scope="session")
def raw(events_file):
    t = uproot.open(events_file)["Events"]
    return t.arrays(library="ak", how="zip")
