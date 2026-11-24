"""read_root(remote_strategy=...): unreadable chunks / files raise (0), are skipped (1) or retried at the end (2); the
skipped ones come back as records that read_root processes later."""
import json

import numpy as np
import pytest
import uproot

import akstream as aks
import akstream.io as aio


def _write(path, n_baskets=5, per_basket=1000, seed=0):
    rng = np.random.default_rng(seed)
    with uproot.recreate(path) as f:
        f.mktree("Events", {"x": np.float64, "event": np.int64})
        for k in range(n_baskets):  # one basket per extend
            f["Events"].extend({"x": rng.uniform(0, 10, per_basket),
                                "event": np.arange(k * per_basket, (k + 1) * per_basket)})
    return str(path)


def _corrupt_basket(path, branch, k):
    """overwrite the compressed payload of basket k of a branch: reading it fails (as a broken remote read)."""
    t = uproot.open(path)["Events"]
    seek = int(t[branch].member("fBasketSeek")[k])
    with open(path, "r+b") as f:
        f.seek(seek + 80)  # past the TKey header
        f.write(b"\xff" * 64)


@pytest.fixture
def files(tmp_path):
    good = _write(tmp_path / "good.root", seed=1)
    bad = _write(tmp_path / "bad.root", seed=2)
    _corrupt_basket(bad, "x", 2)
    return good, bad


def _hist(paths, **kw):
    df = aks.read_root(paths, step=1000, **kw)
    return df.hist("x", (20, 0, 10)).value.values()


def _full(paths):
    return sum(np.histogram(uproot.open(p)["Events"].arrays(["x"], library="np", entry_start=s, entry_stop=e)["x"],
                            20, (0, 10))[0] for p, s, e in paths)


def _ranges(records):
    return [(r["file"], r["start"], r["stop"]) for r in records]


def _all_but(good, bad, failed):
    """the chunks of good + bad (1000 entries each) not in the failed records."""
    gone = {(r["file"], r["start"]) for r in failed}
    return [(p, s, s + 1000) for p in (good, bad) for s in range(0, 5000, 1000) if (p, s) not in gone]


def _check_failed(failed, bad):
    """the corrupted basket (entries 2000-2999) is reported; uproot may also touch it for the chunk before."""
    got = _ranges(failed)
    assert (bad, 2000, 3000) in got and set(got) <= {(bad, 1000, 2000), (bad, 2000, 3000)}


def test_strategy_raise(files):
    good, bad = files
    with pytest.raises(Exception):
        _hist([good, bad])


def test_strategy_skip_and_process_later(files):
    good, bad = files
    with pytest.warns(RuntimeWarning, match="skipped"):
        h = _hist([good, bad], remote_strategy="skip")
    failed = list(aks.last_failed)
    _check_failed(failed, bad)
    assert aks.last_stats["failed_chunks"] == len(failed)
    np.testing.assert_array_equal(h, _full(_all_but(good, bad, failed)))
    # the records are a valid input: once the file is readable again (here: rewritten), process just those chunks
    _write(bad, seed=2)
    saved = json.loads(json.dumps(failed))  # they survive JSON
    later = _hist(saved)
    np.testing.assert_array_equal(later, _full(_ranges(failed)))
    np.testing.assert_array_equal(h + later, _full([(good, 0, 5000), (bad, 0, 5000)]))


def test_strategy_skip_with_workers(files):
    good, bad = files
    df = aks.read_root([good, bad], step=1000, remote_strategy=1)
    h = df.hist("x", (20, 0, 10))
    with pytest.warns(RuntimeWarning, match="last_failed"):
        aks.compute(h, workers=2)
    _check_failed(aks.last_failed, bad)
    np.testing.assert_array_equal(h.value.values(), _full(_all_but(good, bad, aks.last_failed)))


def test_strategy_retry_recovers_a_transient_failure(files, monkeypatch):
    good, _ = files
    orig, calls = aio.Reader.read, {"n": 0}

    def flaky(self, path, start, stop):
        if start == 3000 and calls["n"] == 0:  # the first attempt at this chunk fails
            calls["n"] += 1
            raise OSError("XRootD error: [ERROR] Operation expired")
        return orig(self, path, start, stop)

    monkeypatch.setattr(aio.Reader, "read", flaky)
    df = aks.read_root(good, step=1000, remote_strategy="retry", retry_delay=0)
    out = df.collect(["event"]).value
    assert aks.last_failed == [] and aks.last_stats["retried_ok"] == 1
    events = np.asarray(out["event"])
    np.testing.assert_array_equal(np.sort(events), np.arange(5000))
    assert list(events[-1000:]) == list(range(3000, 4000))  # the retried chunk comes last


def test_strategy_retry_gives_up(files):
    good, bad = files
    with pytest.warns(RuntimeWarning, match="after 2 retries"):
        h = _hist([good, bad], remote_strategy=2, retries=2, retry_delay=0)
    _check_failed(aks.last_failed, bad)
    np.testing.assert_array_equal(h, _full(_all_but(good, bad, aks.last_failed)))


def test_unopenable_file(files, tmp_path):
    good, _ = files
    broken = str(tmp_path / "broken.root")
    with open(broken, "wb") as f:
        f.write(b"not a ROOT file" * 100)
    with pytest.raises(Exception):
        _hist([broken, good])
    with pytest.warns(RuntimeWarning, match="could not be opened"):
        h = _hist([broken, good], remote_strategy=1)  # the schema comes from the next readable file
    assert [(r["file"], r["start"], r["stop"]) for r in aks.last_failed] == [(broken, 0, None)]
    np.testing.assert_array_equal(h, _full([(good, 0, 5000)]))
    _write(broken, seed=3)  # readable now: the record (stop None) means the whole file
    np.testing.assert_array_equal(_hist(list(aks.last_failed)), _full([(broken, 0, 5000)]))


def test_records_keep_the_friends(tmp_path):
    main, friend = str(tmp_path / "m.root"), str(tmp_path / "f.root")
    _write(main, seed=4)
    with uproot.recreate(friend) as f:
        f.mktree("F", {"y": np.float64, "f_event": np.int64})
        for k in range(5):
            f["F"].extend({"y": np.full(1000, float(k)), "f_event": np.arange(k * 1000, (k + 1) * 1000)})
    _corrupt_basket(main, "x", 1)
    df = aks.read_root(main, step=1000, remote_strategy=1,
                       friends={"F": aks.Friend(friend, match={"f_event": "event"})})
    with pytest.warns(RuntimeWarning):
        df.hist("y", (5, 0, 5), weight="x").value
    rec = aks.last_failed[0]
    assert rec["friends"] == {"F": friend} and rec["match"] == {"F": {"f_event": "event"}}
    _write(main, seed=4)
    later = aks.read_root(list(aks.last_failed), step=1000)
    expected = [0.0] * 5
    for r in aks.last_failed:  # y = k on chunk k of the friend
        expected[r["start"] // 1000] = 1000.0
    assert later.hist("y", (5, 0, 5)).value.values().tolist() == expected
