"""read_root(timeout=...): a server that stops answering raises (and remote_strategy applies) instead of hanging;
uproot_options; replica lists (failover at open and at read, sticky, records, entry-count mismatch)."""
import http.server
import os
import threading
import time

import numpy as np
import pytest
import uproot

import akstream as aks
from test_read_failures import _corrupt_basket, _full, _write


class _Server:
    """a local HTTP server of ROOT files (single-range GETs, as uproot / fsspec ask). stall(name, offset, once): the
    requests reaching byte `offset` of `name` hang (the first one only with once=True); offset 0 also hangs HEAD."""

    def __init__(self):
        self.files, self.stalls, self.release = {}, {}, threading.Event()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _stall(self, name, last):
                s = server.stalls.get(name)
                if s is not None and last >= s[0]:
                    if s[1]:
                        server.stalls.pop(name, None)
                    server.release.wait(30)
                    return True
                return False

            def do_HEAD(self):
                name = self.path.lstrip("/")
                if self._stall(name, 0) or name not in server.files:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(server.files[name])))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self):
                name = self.path.lstrip("/")
                data = server.files.get(name)
                if data is None:
                    self.send_error(404)
                    return
                rng = self.headers.get("Range")
                a, b = 0, len(data) - 1
                if rng is not None:
                    spec = rng.split("=", 1)[1]
                    if "," in spec:  # no multipart: uproot falls back to single ranges
                        self.send_response(200)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    lo, hi = spec.split("-")
                    a, b = int(lo), (int(hi) if hi else len(data) - 1)
                if self._stall(name, b):
                    return
                body = data[a:b + 1]
                self.send_response(206 if rng is not None else 200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def serve(self, path, name=None):
        """the URL of `path` (under `name`: a second URL of the same file, a replica)."""
        name = name or os.path.basename(path)
        self.files[name] = open(path, "rb").read()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/{name}"

    def stall(self, name, offset, once=False):
        self.stalls[os.path.basename(name)] = (offset, once)

    def close(self):
        self.release.set()
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.close()


@pytest.fixture
def good(tmp_path):
    return _write(tmp_path / "good.root", seed=1)


def _basket(path, k):
    return int(uproot.open(path)["Events"]["x"].member("fBasketSeek")[k])


# uproot's own HTTP source (fsspec's needs aiohttp): the timeout applies to each of its requests
HTTP = {"handler": uproot.source.http.HTTPSource}


def _hist(files, **kw):
    kw.setdefault("uproot_options", HTTP)
    return aks.read_root(files, step=1000, **kw).hist("x", (20, 0, 10)).value.values()


FULL = [(0, 5000)]


def test_stalled_read_raises_with_the_hint(server, good):
    url = server.serve(good)
    server.stall(url, _basket(good, 2))
    t0 = time.time()
    with pytest.raises(Exception) as info:
        _hist(url, timeout=1, retries=0)
    assert time.time() - t0 < 20
    assert "akstream: timeout=1 s per request" in " ".join(getattr(info.value, "__notes__", [])) or \
        "timeout" in str(info.value).lower()


def test_stalled_read_skipped_and_recorded(server, good):
    url = server.serve(good)
    server.stall(url, _basket(good, 2))
    with pytest.warns(RuntimeWarning, match="skipped"):
        h = _hist(url, timeout=1, remote_strategy="skip")
    failed = list(aks.last_failed)
    assert (url, 2000, 3000) in [(r["file"], r["start"], r["stop"]) for r in failed]
    assert all("timeout=1 s per request" in r["error"] for r in failed)
    gone = {r["start"] for r in failed}
    np.testing.assert_array_equal(h, _full([(good, s, s + 1000) for s in range(0, 5000, 1000) if s not in gone]))


def test_stalled_read_recovers_with_retry(server, good):
    url = server.serve(good)
    server.stall(url, _basket(good, 2), once=True)  # the server answers again afterwards
    with pytest.warns(RuntimeWarning, match="retried at the end"):
        h = _hist(url, timeout=1, remote_strategy="retry", retry_delay=0)
    assert aks.last_failed == [] and aks.last_stats["retried_ok"] >= 1
    np.testing.assert_array_equal(h, _full([(good, 0, 5000)]))


def test_stalled_open_skipped(server, good, tmp_path):
    dead = server.serve(_write(tmp_path / "dead.root", seed=2))
    server.stall(dead, 0)
    with pytest.warns(RuntimeWarning, match="could not be opened"):
        h = _hist([dead, server.serve(good)], timeout=1, remote_strategy="skip")
    assert [(r["file"], r["stop"]) for r in aks.last_failed] == [(dead, None)]
    np.testing.assert_array_equal(h, _full([(good, 0, 5000)]))


def test_timeout_and_uproot_options(good):
    ds = aks.read_root(good, timeout=None, uproot_options={"num_workers": 2})._dataset
    assert ds._open_kwargs() == {"num_workers": 2}  # None: nothing given, uproot's default
    assert aks.read_root(good)._dataset._open_kwargs() == {"timeout": 30.0}
    with pytest.raises(ValueError, match="timeout once"):
        aks.read_root(good, uproot_options={"timeout": 5})
    with pytest.raises(ValueError, match="timeout once"):
        aks.read_samples({"A": {"files": [good]}}, uproot_options={"timeout": 5})


class MarkingSource(uproot.source.file.MemmapSource):
    """a local-file source that leaves a mark of the process opening it (in $AKS_TEST_MARKS)."""

    def __init__(self, file_path, **options):
        with open(os.path.join(os.environ["AKS_TEST_MARKS"], str(os.getpid())), "a") as f:
            f.write(file_path + "\n")
        super().__init__(file_path, **options)


def test_uproot_options_reach_the_workers(good, tmp_path, monkeypatch):
    marks = tmp_path / "marks"
    marks.mkdir()
    monkeypatch.setenv("AKS_TEST_MARKS", str(marks))
    other = _write(tmp_path / "other.root", seed=5)
    df = aks.read_root([good, other], step=1000, uproot_options={"handler": MarkingSource})
    h = df.hist("x", (20, 0, 10))
    aks.compute(h, workers=2)
    np.testing.assert_array_equal(h.value.values(), _full([(good, 0, 5000), (other, 0, 5000)]))
    assert set(os.listdir(marks)) - {str(os.getpid())}  # opened through the handler in the worker processes


def test_replicas_failover_at_open(server, good, tmp_path):
    dead = server.serve(_write(tmp_path / "dead.root", seed=1))
    server.stall(dead, 0)
    t0 = time.time()
    df = aks.read_root([[dead, server.serve(good)]], step=1000, timeout=1, remote_strategy="retry", retry_delay=100,
                       uproot_options=HTTP)
    h = df.hist("x", (20, 0, 10)).value.values()
    assert time.time() - t0 < 30  # no retry_delay between replicas
    np.testing.assert_array_equal(h, _full([(good, 0, 5000)]))
    assert aks.last_failed == [] and df._dataset.files == [dead] and df._dataset.replica_index == {dead: 1}


def test_replicas_failover_at_read(server, good):
    url = server.serve(good)
    server.stall(url, _basket(good, 2))  # opens, then hangs on one chunk: that chunk and the next ones on the copy
    df = aks.read_root([[url, server.serve(good, "copy.root")]], step=1000, timeout=1, uproot_options=HTTP)
    out = df.collect(["event"]).value
    np.testing.assert_array_equal(np.asarray(out["event"]), np.arange(5000))  # in order: no retry at the end
    assert aks.last_failed == [] and df._dataset.replica_index[url] == 1


def test_replicas_entry_mismatch(server, good, tmp_path):
    url = server.serve(good)
    server.stall(url, _basket(good, 2))
    short = _write(tmp_path / "short.root", n_baskets=4, seed=1)
    with pytest.raises(aks.ReplicaMismatch, match="5000 were counted"):
        _hist([[url, server.serve(short)]], timeout=1, remote_strategy="skip")  # never skipped


def test_replicas_all_fail_records(tmp_path):
    a, b = _write(tmp_path / "a.root", seed=1), _write(tmp_path / "b.root", seed=1)
    _corrupt_basket(a, "x", 2)
    _corrupt_basket(b, "x", 2)
    with pytest.warns(RuntimeWarning, match="skipped"):
        _hist([[a, b]], remote_strategy="skip", uproot_options={})
    failed = list(aks.last_failed)
    rec = [r for r in failed if r["start"] == 2000][0]
    assert rec["file"] == a and rec["replicas"] == [a, b]
    _write(b, seed=1)  # one replica readable again: the records read from it
    later = _hist(failed, uproot_options={})
    np.testing.assert_array_equal(later, _full([(b, r["start"], r["stop"]) for r in failed]))


def test_replicas_in_samples_and_friends(server, good, tmp_path):
    dead = server.serve(_write(tmp_path / "dead.root", seed=1))
    server.stall(dead, 0)
    other = _write(tmp_path / "other.root", seed=3)
    df = aks.read_samples({"A": {"files": [[dead, server.serve(good)]]}, "B": {"files": [server.serve(other)]}},
                          step=1000, timeout=1,
                          uproot_options=HTTP)
    got = df.hist("x", (20, 0, 10)).value
    np.testing.assert_array_equal(got["A"].values(), _full([(good, 0, 5000)]))
    np.testing.assert_array_equal(got["B"].values(), _full([(other, 0, 5000)]))
    friend = str(tmp_path / "f.root")
    with uproot.recreate(friend) as f:
        f.mktree("F", {"y": np.float64})
        f["F"].extend({"y": np.ones(5000)})
    deadf = server.serve(friend)
    server.stall(deadf, 0)
    df = aks.read_root(server.serve(good), step=1000, timeout=1, friends={"F": [[deadf, server.serve(friend, "f2.root")]]},
                       uproot_options=HTTP)
    assert df.sum("y").value == 5000.0
