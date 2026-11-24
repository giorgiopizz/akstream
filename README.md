# akstream

Streaming, pandas-like analysis of ROOT trees with awkward arrays. uproot is the only I/O backend (reading and
writing). Columns are lazy; one pass over the files fills every booked result; only the branches that the results
need are read; systematic variations (RDataFrame `Vary`) are evaluated as **deltas**: under a variation only the rows
whose inputs actually change are recomputed.

Documentation: [docs/api.md](docs/api.md) (the interface, with examples) and [docs/internals.md](docs/internals.md)
(where the code is, how a run goes, the C++ extension).

```python
import akstream as aks

df = aks.read_root("skims/*.root", "Events", step=200_000)

# RDataFrame-style chains (each step returns a new frame) ...
df = (df.define("mu1", aks.p4(df.mu1_pt, df.mu1_eta, df.mu1_phi, df.mu1_mass))
        .define("mu2", aks.p4(df.mu2_pt, df.mu2_eta, df.mu2_phi, df.mu2_mass))
        .define("mll", "(mu1 + mu2).mass")
        .define("w", "genWeight * weight_sf_pu"))
# variations first, so that the selections / orderings below vary too (Vary acts on the column as defined so far)
jet = ["Jet_pt", "Jet_mass", "Jet_good", "Jet_bad"]          # stored variations: Jet_pt_jer_up, ...
for syst in ["jer", "jes_Regrouped_HF"]:
    df.vary(jet, up=[f"{c}_{syst}_up" for c in jet], down=[f"{c}_{syst}_down" for c in jet], name=syst)
df.vary("w", up="genWeight * weight_sf_pu_up", down="genWeight * weight_sf_pu_down", name="pu")

# ... pandas-style assignment, awkward-style collections: the same lazy columns
df["Jet"] = df.Jet[df.Jet.good & ~df.Jet.bad & (df.Jet.pt > 25)]   # every Jet_* field selected
df["Jet"] = df.Jet[aks.argsort(df.Jet.pt)[:, ::-1]]                 # and sorted
df.Jet["pt_raw"] = df.Jet.pt * (1 - df.Jet.rawFactor)              # a new field (= df["Jet", "pt_raw"] = df["Jet_pt_raw"])
df["njet"] = aks.num(df.Jet.pt, axis=1)
df["ht"] = aks.sum(df.Jet.pt, axis=1)

sel = df.filter("110 < mll && mll < 150", name="H window").filter("njet >= 2", name="2 jets")
sel["cos_cs"] = aks.collins_soper(sel.mu1, sel.mu2, sel.mu1_charge)[1]
sel["score"] = aks.onnx("model.onnx", [sel.ht, sel.njet, sel.mll])

hs = sel.hists({"score": (50, 0, 1), "cos_cs": (20, -1, 1), "ht": [0, 100, 200, 400, 1000]}, weight="w", varied=True)
cutflow = sel.report()
hs["score"].value  # {"nominal": hist.Hist, "jer:up": ..., "pu:up": ...}; every result is filled in the same pass
print(cutflow)     # entries after each named filter
```

Functions of a whole chunk (the usual shape of awkward analysis code) plug in as they are:

```python
def build_jets(chunk, btag_wp):            # chunk: {"Jet_pt": awkward array, ...}, one entry per event
    ...
    return {"njet": njet, "mjj": mjj, ...}   # one entry per event

df = df.apply(build_jets, ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_btag"], btag_wp=0.3,
              when=[jets.pt])                # under a variation: called only where the selected jets move
df.hist("mjj", (40, 0, 4000))
```

## Interface

| | |
|---|---|
| `aks.read_root(files, tree, step, entry_start, entry_stop, threads, friends, rdf_compatible)` | a `Frame` (files: path, glob, list or `file.root:tree`); `friends={tree: files}` = TTree friends (same entries, one file per main file), `aks.Friend(files, match={"gj_event": "event"})` also checks the alignment chunk by chunk; `rdf_compatible=True`: evaluate as C++ / RDataFrame does (double literals in expression strings, ROOT GenVector 4-vectors), to reproduce RDF code bit for bit |
| `read_root(..., remote_strategy=0 / 1 / 2, retries, retry_delay)`, `aks.last_failed` | unreadable chunks: raise / skip / retry at the end; the skipped ones as records, `read_root(aks.last_failed)` processes them later |
| `aks.from_arrays(data, step)` | a `Frame` over arrays in memory: a dict of numpy / awkward arrays, an awkward record array, or a pandas DataFrame |
| `df.to_awkward(cols)`, `df.to_numpy(cols)`, `df.to_pandas(cols)` | the selected rows, computed now (the inverse of `from_arrays`) |
| `df.entry` | the entry number in its file (`rdfentry_` of a one-file RDataFrame) |
| `df.x`, `df["x"]` | a branch, a defined column, or a collection (`df.Jet`: the jagged `Jet_*` branches, with `nJet` if the tree has it) |
| `df.Jet["f"] = v`, `df["Jet", "f"] = v`, `df["Jet_f"] = v` | the same assignment: a field is the column `Jet_f`; a scalar or a per-event column is broadcast to the objects, a jagged one is checked to have the collection's counts |
| `df["Jet"] = df.Jet[mask / index]`, `aks.pad_none(df.Jet, 2)`, `jets[:, 0]` | replace the collection (every field, and `nJet`) / field-wise awkward functions / one object per event |
| `df.define("y", value)`, `df.define(y=..., z=...)` | a new frame with more columns: value = lazy column, expression string (C++ or Python syntax: `&&`, `!`, `c ? a : b`, `std::abs`), constant |
| `df["y"] = value` | the same, pandas-style (in place) |
| `df.filter(cond, name=)`, `df[mask]` | a selection (new frame; every parent column available, taken lazily) |
| `df.report(weight=)` | cutflow of the named filters leading to a frame (filled with everything else) |
| `df.eval("...")`, `df[["a", "b"]]`, `df.columns`, `df.branches`, `df.dtypes` | as pandas |
| `coll[mask]`, `coll[index]`, `coll.pt`, `coll[["pt", "eta"]]`, `coll.p4` | collections: masks / sorts / indices touch only the fields used |
| operators, `np.<ufunc>(e)`, `np.where/clip/stack/...`, `aks.<any awkward function>` | lazy |
| `aks.p4(pt, eta, phi, m, **fields)`, `aks.p4(record)`, `coll.p4`, `v1 + v2`, `.mass .pt .eta .phi .px ...`, `aks.collins_soper(l1, l2, q1)`, `aks.delta_r` | 4-vectors (C++, `akstream.vector`); extra fields ride along (`aks.p4(..., pdgId=df.mu1_pdgId)` → `mu1.pdgId`; a sum has none); with `rdf_compatible`: ROOT's `PtEtaPhiMVector` arithmetic (`.rapidity` too) |
| `hep.max_mass_pair(df.Jet)`, `hep.pick(df.Jet.pt, idx, 0)`, ... | the `akstream.hep` kernels on lazy columns (tuples of lazy columns where they return tuples) |
| `aks.checked(value, condition, message)` | `value`, with `ValueError(message)` if `condition` fails on a row |
| `df.apply(fn, columns, outputs=, when=, prefix=, **kw)` | `fn(chunk dict, **kw) -> dict`: a new frame with its outputs as columns |
| `aks.apply(fn, *cols, when=)`, `aks.onnx(path, cols)` | row-local user functions of columns / ONNX models (treated as expensive) |
| `df.hist(x, (n, lo, hi) / edges / axis, weight)`, `df.hist([x, y, ...], axes=[...])`, `df.hists({name: binning}, weight, varied)` | `hist.Hist` (Weight storage), any number of dimensions; axes `aks.Regular`, `aks.Variable`, `aks.IntCategory` (or the hist ones); x a column or expression string; jagged x fills per object |
| `df.sum(col, weight)`, `df.count()`, `df.groupby(key).sum(col) / .count()` | `df.count()` = the number of rows; a sum of weights is `df.sum("weight")` |
| `df.collect(cols)`, `df.head(n)`, `df.peek(expr)`, `df.to_root(path, cols)` | awkward arrays / a TTree written with uproot (collections as `nX` + `X_f`) |
| `aks.varied(df.to_root("out/{tag}/f.root", cols))` | one file per variation in the same pass (`{variation}`, `{group}`, `{tag}`; nominal = "nominal"): per chunk the nominal rows with the rows a variation changes replaced by their recomputation |
| `df.vary(col, up=alt, down=alt, name=)`, `df.vary([col, ...], up=[alt, ...], down=[...], name=)` | variations `name:up`, `name:down` (col = a column, or `("Jet", "pt")`; a list = columns varied together; alternative = branch, expression string, or lazy column; name defaults to the column for one column); a varied branch varies wherever it is read, a varied defined column from there on |
| `aks.varied(result)` | its value becomes `{"nominal": ..., "<variation>": ...}` (every variation that affects it) |
| `aks.compute(*results, workers, validate, exact, read_ahead, ...)` | one pass; `result.value` also triggers it (for every booked result, as RDataFrame) |

Expressions must be **row-local** (row *i* of the output depends only on row *i* of the inputs): elementwise maths and
`axis=1` operations. Reductions across rows (`aks.sum(x)` with `axis=None`, `aks.sort` of a flat column,
`np.cumsum`) are refused at run time with a clear error; totals are results (`df.sum`, `df.hist`).

## How it runs

* **Graph.** Every column is a node (`Source` branch, `Call`, `Take` = a parent column on a selection's rows, `Zip`
  = a record whose fields load on touch). Identical expressions are one node (hash-consing).
* **Reading.** `compute` walks the graph from the results: the branches reached are read per chunk in two bulk uproot
  calls (flat as numpy, jagged as awkward). Fields of a record handed whole to an awkward function are read lazily
  and, from the next chunk on, in bulk.
* **Nominal.** Per chunk, each node is evaluated once; a selection stores its row ids. Flat values are numpy (no
  awkward dispatch on per-event columns). Values are freed after their last consumer, except the ones a variation
  pass reads (nodes a variation does not recompute, compared nodes, masks, results).
* **Variations (default: fused).** Per variation, the affected nodes are split into a cheap prefix (selections, masks)
  and the part downstream of expensive nodes (`aks.apply`, `aks.onnx`). Phase 1 evaluates the prefix where the varied
  inputs differ and compares it with nominal: the rows R that change (for written results, `to_root`, R also holds
  every row whose varied inputs differ: no column-by-column comparison). Phase 2 evaluates everything downstream once, on
  the sub-chunk R, as a plain pass (unaffected columns are the nominal values on R); histograms are corrected by
  -fill(nominal on R) +fill(varied on R). Nodes changed only through selection membership keep their nominal values.
  `aks.apply(fn, *args, when=[exprs])` lets the rows where `exprs` change decide the recomputation of fn (e.g. the
  selected jets, for a function of all the jets).
* **Variations (engine="delta", and exact=True).** For each variation, a node downstream of a varied column is a *delta*: the rows whose value differs
  from nominal and their values. A row-local node is recomputed only on the union of its inputs' changed rows;
  selections track the rows a variation removes or adds (added rows get every column computed). Before an expensive
  node (DNN, `aks.apply`) and after collection selections, recomputed values are compared with nominal and unchanged
  rows are dropped ("early cutoff"): e.g. an event whose only shifted jet fails the jet selection both ways never
  reaches the DNN. Histograms are corrected on the changed rows only (`-nominal +varied`), so a sparse variation costs
  O(changed rows), not O(events). Above `dense_fraction` (default 0.25; 0.95 for expensive nodes) of changed rows a node is recomputed on every
  row instead, with no bookkeeping: variations that move everything (muon scale, weights) cost one plain recompute.
* **Checks.** `compute(validate=k)` evaluates every variation by brute force (all affected nodes on all rows) on the
  first *k* chunks and asserts the delta evaluation fills the same histograms. `exact=True` refills every row of each
  variation (identical to a full recomputation; the default correction differs only by float summation order, ~1e-14).
* **Parallel.** `workers=N` spawns processes (the graph travels with cloudpickle) over contiguous chunk ranges;
  results merge (files: one part per task).

## akstream.hep: fast kernels for jagged collections

Generic operations most analyses need, as single passes over flat contents + offsets (numpy gathers, and a small C++
extension `akstream._hep` for the event loops). Row-local, so they work inside `aks.apply` and under variations, and
bit-identical to the equivalent awkward code (dtype promotion included):

| | replaces |
|---|---|
| `hep.nth(x, k, fill=None)`, `hep.pick(x, index, fill)` | `ak.pad_none(x, k + 1)[:, k]` (None where there is no k-th object; with `fill`: `ak.fill_none(..., fill)`), `ak.firsts(x[local == index])`; `hep.nth(df.Jet, 0)` is the leading jet's record (`j1.pt`, `aks.p4(j1)`) |
| `hep.select_sort(keep, key, *fields)` | `x[keep][ak.argsort(key[keep], ascending=False, stable=True)]` for every field |
| `hep.count(mask)`, `hep.per_event_sum(x, mask)` | `ak.sum(mask, axis=1)`, `ak.sum(ak.where(mask, x, 0), axis=1)` |
| `hep.isolated(eta, phi, refs, min_dr2)`, `hep.between(x, lo, hi)` | Delta R cleaning against per-event directions, per-event bounds |
| `hep.max_mass_pair(pt, eta, phi, mass, ...)` | `ak.argcombinations` + 4-vector sums + `argmax` (VBF tagging) |
| `hep.match_dr(a_eta, a_phi, b_eta, b_phi, max_dr, unique=)`, `hep.clean_dr(...)` | `ak.cartesian` + Delta R + `argmin` (17x faster, no pair arrays) / lepton-jet cleaning |
| `hep.ancestor(pdg_id, mother_idx, targets)` | walking `genPartIdxMother` chains (a loop awkward cannot express) |
| `hep.best_pair(leptons, target=91.1876)` | `ak.argcombinations` + charge / flavour cuts + mass closest to a target |
| `hep.met_shift(...)`, `hep.LumiMask(json)` | MET propagation of object pT changes; golden-JSON luminosity mask |

## akstream.vector: Lorentz vectors

`vector.p4(pt, eta, phi, m)` (per event or jagged per object), sums, `.mass / .pt / .eta / .phi` computed only when
used, `vector.collins_soper(l1, l2, charge1)`, `delta_phi`, `delta_r`; one C++ loop each, the same formulas as the
plain numpy code operation by operation, with the C math library's cos / sin / sinh / asinh / atan2 (as ROOT):
bit-identical to numpy where numpy uses that library too (without AVX-512, where numpy has its own SIMD versions). A `P4` is a dict of its components, so it travels through selections and variations like any column; lazily
it is `aks.p4(...)` / `collection.p4`. The scikit-hep `vector` package was measured first: its numpy, awkward and numba
backends are no faster than hand-written numpy on these operations (the cost is cos / sin / sinh, which nothing avoids)
and do not reproduce the numpy formulas' rounding; the gain here is the Collins-Soper frame (one loop instead of ~60
temporaries) and not computing the eta / phi of sums that are not used.

`rdf_compatible=True` (`vector.RootP4`, and `hep.max_mass_pair(..., rdf_compatible=True)`) reproduces ROOT's
`PtEtaPhiMVector` arithmetic instead (ROOT GenVector is the reference implementation of these formulas): E from
pt cosh(eta), sums converted back to (pt, eta, phi, m), negative masses and phi ranges as GenVector. Bit-identical to
ROOT 6.40 on any CPU (the transcendental functions are the C math library's, not numpy's); ROOT 6.36 evaluates GenVector with fused multiply-adds on CPUs that have them, ~1e-16 relative, which
disappears once the result is stored as float.

**Build.** `pip install akstream` installs a prebuilt wheel where one exists (Linux x86_64 / aarch64, macOS); otherwise
the extension is compiled at install time (a C++17 compiler; pybind11 >= 2.12 is fetched by pip). For a development
install: `pip install -e .[test]`, or `python setup.py build_ext --inplace` with pybind11 >= 2.12 available (NumPy 2
support; e.g. the LCG_110 view ships 2.11: `pip install --target <dir> "pybind11>=2.13"` and put `<dir>` first in
PYTHONPATH for the build). It is compiled
without fused multiply-add (`-ffp-contract=off`) and without the sin + cos -> sincos fusion (glibc's `sincos` rounds
differently) so its floating point matches numpy exactly.
`AKSTREAM_STATIC_LIBSTDCXX=1` links libstdc++ statically (for a build used where the compiler runtime is absent).

## Tests

    pip install -e .[test,onnx]
    pytest                                   # 178 tests: vs plain awkward / numpy on synthetic NanoAOD-like trees

The tests that compare with ROOT itself (GenVector arithmetic) run when PyROOT is importable and are skipped otherwise.

## License

BSD 3-Clause, see [LICENSE](LICENSE). akstream depends on awkward, uproot, numpy, hist, boost-histogram and
cloudpickle (all BSD-3-Clause) and is built with pybind11 (BSD-3-Clause).
