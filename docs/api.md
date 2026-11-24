# akstream API

akstream reads ROOT trees with uproot, chunk by chunk, and evaluates lazy column expressions on them. One pass over the
files fills every booked result; systematic variations are evaluated only on the rows they change. The interface
mixes three styles that all build the same lazy graph:

* **RDataFrame**: `define`, `filter(name=)`, `vary`, `hist`, `report`, C++ expression strings;
* **pandas**: `df["x"] = ...`, `df[mask]`;
* **awkward**: collections (`df.Jet.pt`), `aks.<any awkward function>`, field-wise operations.

```python
import numpy as np
import akstream as aks
from akstream import hep

df = aks.read_root("skims/*.root", "Events", step=500_000)
df.vary(("Jet", "pt"), up="Jet_pt_jes_up", down="Jet_pt_jes_down", name="jes")
df["Jet"] = df.Jet[(df.Jet.pt > 25) & (abs(df.Jet.eta) < 4.7)]
df["Jet"] = df.Jet[aks.argsort(df.Jet.pt)[:, ::-1]]
df["ht"] = aks.sum(df.Jet.pt, axis=1)
_, i1, i2, mjj, detajj = hep.max_mass_pair(df.Jet)
df["mjj"] = mjj
vbf = df.filter("mjj > 400 && ht > 100", name="VBF")
h = aks.varied(vbf.hist("mjj", (40, 0, 4000), weight="genWeight"))
print(h.value)            # {"nominal": Hist, "jes:up": Hist, "jes:down": Hist}
print(vbf.report())       # cutflow
```

Contents: [Reading](#reading) · [Columns](#columns-and-expressions) · [Collections](#collections) ·
[Selections](#selections) · [Variations](#variations) · [Results](#results) · [Running](#running) ·
[User functions](#user-functions) · [4-vectors](#4-vectors-akstreamvector) · [HEP kernels](#hep-kernels-akstreamhep) ·
[RDataFrame compatibility](#reproducing-rdataframe-bit-for-bit) · [Rules](#rules) · [Not supported yet](#not-supported-yet)

---

## Reading

```python
aks.read_root(files, tree="Events", step=200_000, entry_start=0, entry_stop=None, threads=0, friends=None,
              rdf_compatible=False, remote_strategy=0, retries=3, retry_delay=10.0) -> Frame
```

| argument | |
|---|---|
| `files` | a path, a glob, a list of them, or `"file.root:tree"` |
| `step` | entries per chunk: memory scales with it (a few hundred thousand is typical) |
| `entry_start`, `entry_stop` | entry range (applied when there is a single file) |
| `threads` | uproot decompression threads per process |
| `friends` | `{tree name: files}` friend trees (as `TTree::AddFriend`): one file per main file, same order, same entries; their branches become columns. `aks.Friend(files, match={"gj_event": "event"})` also checks, on every chunk, that the friend branch equals the main-tree branch |
| `rdf_compatible` | evaluate as C++ / RDataFrame does (see [below](#reproducing-rdataframe-bit-for-bit)) |
| `remote_strategy`, `retries`, `retry_delay` | what to do with chunks that cannot be read (see [below](#unreadable-chunks-remote-files)) |

Only the branches that the booked results need are read. `df.branches`, `df.dtypes` and `df.collections` describe
the tree; `df.entry` is the entry number in its file (`rdfentry_`).

### Unreadable chunks (remote files)

A chunk can fail to read (an xrootd timeout, a broken basket), or a file to open. `remote_strategy` decides:

| | |
|---|---|
| `0` / `"raise"` (default) | raise at the first failure |
| `1` / `"skip"` | warn, skip the chunk (or the file), go on |
| `2` / `"retry"` | warn, go on with the other chunks, then retry the failed ones: `retries` rounds, `retry_delay` seconds apart (files that cannot be opened are retried at once); still failing: skipped |

After the run, `aks.last_failed` lists the skipped chunks as records `{"file", "start", "stop", "error", "tree",
"friends", "match"}` (`stop` None: the whole file, it could not be opened); `aks.last_stats["failed_chunks"]` counts
them. The records survive JSON, and a list of them is a valid `files` argument: a frame over exactly those chunks,
with the same friends. The results of a run with skipped chunks are those of the chunks that were read, so the two
runs add up:

```python
df = aks.read_root(files, remote_strategy="retry", retry_delay=30)
h = df.hist("mll", (80, 70, 150), weight="genWeight")
aks.compute(h, workers=8)
json.dump(aks.last_failed, open("failed.json", "w"))           # if not empty
...                                                             # later
df2 = aks.read_root(json.load(open("failed.json")))
h2 = df2.hist("mll", (80, 70, 150), weight="genWeight")         # the same booking
total = h.value + h2.value
```

Notes: with `"retry"`, the rows of a retried chunk come after the others in `to_root` / `collect`; a chunk is read
before it is processed, so a failure never leaves a chunk half-filled (the exception: a branch read lazily in the
middle of a chunk, which is read in bulk from the second chunk on, raises); a skipped chunk of data is missing
luminosity, so check `aks.last_failed` before using the result.

**Arrays in memory**: `aks.from_arrays(data, step=200_000, rdf_compatible=False)` is a frame over a dict
`{name: array}` (numpy, or awkward for lists), an awkward record array, or a pandas DataFrame, one entry per row.
Record fields become `name_field` columns, so a table from `df.to_awkward(["mll", "Jet"])` comes back with its `Jet`
collection. Everything works as on a file: selections, variations, histograms, `to_root`, workers.

```python
small = aks.from_arrays(sel.to_awkward(["mll", "ht", "Jet"]))     # continue on a selected table
t = aks.from_arrays(pandas_df)                                      # from pandas (one row per event)
```

## Columns and expressions

A column is lazy: an `Expr`. Nothing is read or computed until a result is requested.

```python
df.mll                        # a branch or a defined column
df["mll"]                     # the same
df["ratio"] = df.mu2_pt / df.mu1_pt               # pandas-style, in place
df2 = df.define("ratio", "mu2_pt / mu1_pt")       # RDataFrame-style, a new frame
df2 = df.define(ratio="mu2_pt / mu1_pt", one=1.0) # several at once
df["w"] = 1.0                                     # a constant column
```

Available on columns:

* Python operators (`+ - * / // % ** & | ^ ~ < <= == != > >=`, `abs`); `and` / `or` / `not` cannot be overloaded:
  use `& | ~` (in expression strings they are translated);
* numpy ufuncs: `np.sqrt(e)`, `np.cos(e)`, ...;
* row-local numpy functions: `np.where`, `np.clip`, `np.select`, `np.isnan`, `np.maximum`, `np.minimum`, `np.stack`
  (`axis=1`), `np.digitize`, `np.interp`, `np.full_like`, ...;
* every awkward function, lazily: `aks.sum(x, axis=1)`, `aks.num`, `aks.argsort`, `aks.local_index`, `aks.pad_none`,
  `aks.fill_none`, `aks.firsts`, `aks.where`, `aks.combinations`, ...;
* `e.astype(np.float32)`, indexing `e[:, 0]`, `e[:, ::-1]`, `e[mask]`, attribute access on records / 4-vectors
  (`e.mass`), `e["field"]` on dict-valued columns.

**Expression strings** (in `define`, `filter`, `eval`, `hist`, weights, `vary`) accept Python or C++ syntax:
`&&`, `||`, `!`, `true` / `false`, `std::abs(x)`, and the conditional `c ? a : b` (nested, inside brackets and
function arguments). Names resolve to columns; `abs`, `min`, `max`, `sqrt`, `where`, `np`, `ak`, `aks` are available;
`df.eval(expr, local={...})` adds names.

```python
df["reg"] = df.eval("mll > 120 ? 1 : (mll > 90 && njet >= 2 ? 2 : 0)")
df.filter("110 < mll < 150 and njet >= 2")    # Python syntax too; chained comparisons work
```

`df.peek(expr, n=1000)` computes an expression now, on the first `n` entries of the input (the selections apply after,
so a selected frame gives fewer rows); `df.head(n, columns)` returns its rows among the first `n` entries.

## Collections

A collection is the set of jagged branches `X_<field>` sharing a prefix `X` (NanoAOD naming: `Jet_pt`, `Jet_eta`,
...), with the counter `nX` when the tree has one. `df.Jet` is an awkward-like record view of those columns:

```python
df.Jet.pt            # the column Jet_pt (also df.Jet["pt"], df["Jet", "pt"], df["Jet_pt"])
df.Jet.fields        # ["pt", "eta", ...]
```

**Fields are columns**: these three are the same assignment, and every one of them is what `df.Jet.pt` returns
afterwards.

```python
df.Jet["pt_raw"] = df.Jet.pt * (1 - df.Jet.rawFactor)
df["Jet", "pt_raw"] = ...
df["Jet_pt_raw"] = ...
```

A scalar, or one value per event, is broadcast to the objects; a jagged value must have the collection's object
counts (checked when it is computed, with an error naming the field):

```python
df.Jet["is_data"] = False
df.Jet["genpt"] = np.float32(-1)       # the scalar's dtype is kept: float32 here
df.Jet["w"] = df.genWeight             # every jet of the event gets the event's value
df.Jet["genpt"] = df.GenJetMatch.pt    # a friend's per-jet branch: counts checked
```

**Replacing the collection** selects / reorders every field at once (and updates `nX`):

```python
df["Jet"] = df.Jet[df.Jet.pt > 25]                    # jagged boolean mask
df["Jet"] = df.Jet[aks.argsort(df.Jet.pt)[:, ::-1]]   # jagged integer index (here: sort by decreasing pt)
df["goodJet"] = df.Jet[df.Jet.good]                   # a new collection goodJet_*
```

Only the fields used downstream are ever computed: a collection with 150 fields costs what its used fields cost.

**Other collection operations**

```python
jets = df.Jet[df.Jet.pt > 30]          # a selected collection (not assigned): a value like any other
jets[["pt", "eta"]]                    # a subset of fields
lead = aks.fill_none(aks.pad_none(df.Jet, 2, clip=True), 0.0)   # field-wise awkward functions
lead[:, 0].pt, lead[:, 1].eta          # one object per event
df.Jet.p4                              # 4-vectors from pt, eta, phi, mass
aks.zip({"pt": a, "eta": b})           # a collection from columns
```

The awkward functions applied field by field to a collection are `pad_none`, `fill_none`, `firsts`, `singletons`,
`drop_none`, `mask`, `is_none`, `values_astype`; any other awkward function receives the zipped record array.

## Selections

```python
z = df.filter("mll > 70 && mll < 110", name="Z window")     # a named filter: a line of the cutflow
z = df[(df.mll > 70) & (df.mll < 110)]                       # the same, pandas-style
```

A selection is a new frame: every column of its parent is available on it (taken on its rows when used). Columns
assigned on a frame after a selection was made from it are not seen by the selection (as a pandas copy). A result
must use the columns of the frame it is booked on; mixing columns of different selections is an error.

## Variations

```python
df.vary("w", up="genWeight * pu_up", down=df.genWeight * df.pu_down, name="pu")   # any tag names, not only up / down
df.vary(["Jet_pt", "Jet_mass"], up=["Jet_pt_jer_up", "Jet_mass_jer_up"],          # columns varied together:
        down=["Jet_pt_jer_down", "Jet_mass_jer_down"], name="jer")                 # one alternative per column
```

* The order is RDataFrame's `Vary(columns, expression, tags, variationName)`: the column(s) first, the name last. A
  variation is called `"<name>:<tag>"` (`"jes:up"`). `name` defaults to the column name for one column (as in
  RDataFrame) and is required for a list. Registering the same `name:tag` on a column again with a different
  alternative is an error.
* `column` is a column name, a collection field `("Jet", "pt")`, or a list of them; an `alternative` is a branch /
  column name, an expression string, or a lazy column (a list of them, in the order of the columns, for a list).
* A varied **branch** varies wherever it is read. A varied **defined column** varies from that point on (it gets a
  node of its own, so an identical expression elsewhere is not affected).
* Everything downstream of a varied column follows: selections, sorts, collection replacements, results. Declare the
  variations of the branches **before** you select or sort the collection: `df["Jet"] = df.Jet[df.Jet.pt > 25]`
  written before `df.vary(("Jet", "pt"), ...)` selects with the nominal pt (as RDataFrame's `Vary` acts on the
  column as defined so far). Varied branches such as `Jet_pt_jes_up` are fields of the collection, so they are
  selected along with it.
* `aks.varied(result)` makes a result's value a dict `{"nominal": ..., "<variation>": ...}` with every variation that
  affects it (or `aks.varied(result, ["jes:up"])`).

Evaluation: per chunk, each variation is recomputed only on the rows where it changes something (see
[internals](internals.md)). A variation that moves every row (a weight, a muon scale) costs about one nominal pass.

## Results

| | value |
|---|---|
| `df.hist(x, bins, range=, weight=, name=)` | `hist.Hist` with Weight storage. `x`: a column or an expression string; a jagged `x` fills one entry per object. `bins`: `(n, lo, hi)`, a list of edges, `aks.Regular(n, lo, hi)`, `aks.Variable(edges)`, `aks.IntCategory([0, 1, 2])`, or a `hist` Regular / Variable / IntCategory axis; or `bins=n, range=(lo, hi)`. An IntCategory axis needs an integer (or bool) column (a float one is refused, as boost-histogram); values that are not a category go to its overflow bin |
| `df.hist([x, y, z], axes=[...], weight=)` | an n-dimensional histogram, one axis per column (each in any of the forms above), e.g. `axes=[(80, 70, 150), [0, 1, 2, 5], aks.IntCategory([0, 1, 2])]`; per-event columns are repeated for each object of a jagged one |
| `df.hists({name: binning or (column, binning)}, weight=, varied=)` | `{name: result}` |
| `df.sum(column, weight=)` | a float: the sum of `column` (weighted), or of `weight` alone: `df.sum("weight")` is the sum of weights |
| `df.count()` | the number of rows (entries passing the selections) |
| `df.groupby(key).sum(column, weight=)`, `.count()` | `{key value: total}` |
| `df.report(weight=)` | cutflow `[(filter name, entries, efficiency vs previous, cumulative efficiency)]`; `print(report)` shows a table |
| `df.collect(columns)` | an awkward record array (or `df[["a", "b"]].collect()`) |
| `df.to_root(path, columns, tree="Events", compression=)` | writes a TTree with uproot; value `{"files": [...], "entries": n}`. Collections are written as `nX` + `X_<field>`; `("Jet", ["pt", "eta"])` writes a collection with only those fields |
| `aks.varied(df.to_root("out/{tag}/f.root", cols))` | one file per variation in the same pass: `{variation}`, `{group}` and `{tag}` in the path (`"jet:jes_up"` -> group `jet`, tag `jes_up`; nominal = `"nominal"`) |

### Conversions

Computed when called (every selected row in memory, as `collect`), the inverse of `aks.from_arrays`:

| | |
|---|---|
| `df.to_awkward(columns)` | an awkward record array; a collection (`"Jet"`) is a field of jagged records |
| `df.to_numpy(columns, dtype=None)` | a 2-d array, rows x columns in the given order (per-row columns only; the common dtype, or `dtype`) |
| `df.to_pandas(columns)` | a pandas DataFrame, one row per row of the frame; with jagged columns, awkward's `ak.to_dataframe` layout (one row per object, `(entry, subentry)` index) |

`columns` defaults to the defined columns. A typical use: fit something on a table, then use it as constants in the
streaming graph:

```python
x = sel.to_numpy(["mjj", "detajj", "ptll"])          # one pass, the selected rows only
mean, std = x.mean(axis=0), x.std(axis=0)            # or sklearn's StandardScaler().fit(x)
for i, c in enumerate(["mjj", "detajj", "ptll"]):
    sel[f"{c}_scaled"] = (sel[c] - mean[i]) / std[i]
```

`aks.Regular(n, lo, hi, convention="boost")` bins as boost-histogram; `convention="root"` as ROOT's `TAxis::FindBin`
(the two can differ by one bin exactly at an edge).

## Running

```python
aks.compute(*results, workers=1, validate=0, exact=False, max_entries=None, read_ahead="auto", ...) -> [values]
result.value      # computes every result booked on the dataset that is not done yet (one pass, as RDataFrame)
```

| option | |
|---|---|
| `workers` | processes (spawned; the graph is sent with cloudpickle); chunks are split into contiguous tasks, results merged; files: one part per task |
| `validate=k` | on the first `k` chunks, also evaluate every variation by brute force and check the histograms agree |
| `exact` | refill every row of each varied histogram (bitwise equal to a full recomputation; the default differs by float summation order only, ~1e-14 relative) |
| `max_entries` | stop after this many entries |
| `read_ahead` | read the next chunk in a background thread: `"auto"` (when at least two cores are available), an int, or 0 |
| `engine` | `"fused"` (default) or `"delta"` (per-node deltas; always used with `exact=True`) |
| `batch_rows`, `dense_fraction`, `cutoff` | tuning of the variation engine (see the docstring) |

`aks.last_stats` holds timings of the last run (read, nominal, variations, rows recomputed).

## User functions

**Chunk functions** (the usual shape of awkward analysis code):

```python
def build_jets(chunk, wp):                     # chunk: {"Jet_pt": array, ...}, one entry per event
    ...
    return {"njet": njet, "mjj": mjj}           # one entry per event, numpy or awkward

df = df.apply(build_jets, ["Jet_pt", "Jet_eta"], wp=0.3)     # a new frame with columns njet, mjj
df = df.apply(build_jets, {"pt": "Jet_pt", "eta": df.Jet.eta}, outputs=["njet", "mjj"], prefix="j_")
```

`outputs` defaults to the keys returned on the first 200 rows (or one name when fn returns an array). Under a
variation, fn is called only on the rows whose inputs change. `when=[columns]`: only the rows where those columns
change trigger a recomputation (e.g. the selected jets, for a function that reads all the jets but uses only the
selected ones); this is a promise about fn, checked by `compute(validate=...)`.

**Column functions**: `aks.apply(fn, *columns, when=None, **kwargs)` -> a lazy column of `fn(*values, **kwargs)`
(an array, or a dict of arrays: `e["name"]`). `aks.onnx(path, inputs, output=0, transform=None)` scores an ONNX model
on stacked float32 inputs. Both are treated as expensive: under a variation their inputs are compared with nominal
first and only the rows that really changed are recomputed.

**Checks**: `aks.checked(value, condition, message)` is `value`, with `ValueError(message)` if `condition` fails on
a row. `aks.checkpoint(e)` asks the engine to compare `e` with nominal under variations (rows that did not change are
not propagated further).

## 4-vectors (`akstream.vector`)

```python
mu1 = aks.p4(df.mu1_pt, df.mu1_eta, df.mu1_phi, df.mu1_mass)    # lazy; df.Jet.p4 for a collection
mu1 = aks.p4(df.mu1_pt, df.mu1_eta, df.mu1_phi, df.mu1_mass, pdgId=df.mu1_pdgId)   # extra fields: mu1.pdgId
ll = mu1 + mu2
ll.mass, ll.pt, ll.eta, ll.phi, ll.px, ll.e                       # computed only when used
phi_cs, cos_cs = aks.collins_soper(mu1, mu2, df.mu1_charge)       # mu1 = the positive lepton where charge > 0
dr = aks.delta_r(j1, j2)
```

On arrays (inside chunk functions): `vector.p4(pt, eta, phi, mass)`, `vector.add`, `vector.collins_soper`,
`vector.delta_phi`, `vector.delta_r`; per event (numpy) or per object (jagged). A `P4` is a dict of its components (px,
py, pz, e; pt, eta, phi, mass when built from them), so it travels through selections and variations like any column.
Extra fields (`pdgId=`, `charge=`, ...) are kept as given (their own dtype; a scalar or a per-event value is repeated
for each object of jagged vectors) and read as `v.pdgId`; a sum of vectors has none, and names a 4-vector already uses
(`pt`, `mass`, `px`, `rapidity`, ...) are refused. The loops are C++ (see
[internals](internals.md#the-c-extension-_hepcpp)) and follow the plain numpy formulas operation by operation.

## HEP kernels (`akstream.hep`)

Single-pass versions of common jagged operations. They take awkward arrays (inside chunk functions) or lazy columns
(then the result is lazy; a tuple of lazy columns where the function returns a tuple).

| function | equivalent awkward code |
|---|---|
| `hep.nth(x, k, fill)` | `ak.fill_none(ak.pad_none(x, k + 1)[:, k], fill)`: the k-th object of each event. `x` a collection or jagged 4-vectors: the k-th 4-vector (`fill` a number for each of pt, η, φ, m, or `aks.p4(0, 0, 0, 0)`), e.g. `mjj = (hep.nth(df.Jet, 0, fill=0.0) + hep.nth(df.Jet, 1, fill=0.0)).mass` |
| `hep.pick(x, index, fill)` | the object at a per-event index (`fill` where the index is < 0 or out of range) |
| `hep.count(mask)` | `ak.sum(mask, axis=1)` |
| `hep.per_event_sum(x, mask=None)` | `ak.sum(x[mask], axis=1)`, in the values' own type |
| `hep.broadcast(per_event, x)` | one value per object of `x` |
| `hep.between(x, lo, hi)` | `lo < x < hi` per object, per-event bounds |
| `hep.isolated(eta, phi, refs, min_dr2)` | a flat boolean per object: ΔR² ≥ min_dr2 to every reference direction `refs = [(eta, phi), ...]` (one value per event each) |
| `hep.delta_phi(phi1, phi2)` | wrapped to (-π, π] |
| `hep.select_sort(keep, key, *fields, descending=True)` | the objects passing `keep`, sorted by `key` (stable), same for every field |
| `hep.max_mass_pair(jets or pt, eta, phi, mass, min_mass=400, min_deta=2.5, min_lead_pt=35, leading_only=False)` | `(found, i1, i2, mjj, detajj)`: the pair with the largest invariant mass passing the cuts (VBF tagging) |
| `hep.match_dr(a_eta, a_phi, b_eta, b_phi, max_dr, b_mask=None, unique=False)` | `(index, dr)` jagged like A: the nearest B object within ΔR < max_dr (-1 if none), or one-to-one greedy matching; trigger / gen / GenJet matching. ~17x faster than `ak.cartesian` + `argmin`, without the nA x nB pairs in memory |
| `hep.clean_dr(eta, phi, ref_eta, ref_phi, min_dr, ref_mask=None)` | a jagged boolean: no reference object within ΔR < min_dr (lepton-jet cleaning) |
| `hep.ancestor(pdg_id, mother_idx, targets, abs_id=True)` | per gen particle, the nearest ancestor whose \|pdgId\| is in `targets` (-1 if none), following the mother indices: `hep.ancestor(gen.pdgId, gen.genPartIdxMother, 23) >= 0` = from a Z |
| `hep.best_pair(leptons or pt, eta, phi, mass, charge, target=91.1876, flavour=None, opposite_charge=True)` | `(found, i1, i2, mass)`: the pair with mass closest to `target` (opposite charges; equal `flavour`, e.g. \|pdgId\|, when given): Z candidates |
| `hep.met_shift(met_pt, met_phi, pt_new, pt_old, phi, mask=None)` | `(pt, phi)`: MET with the objects' pT changes propagated (Type-1-like) |
| `hep.LumiMask(golden_json)(run, lumi)` | certified luminosity sections (a path or a dict) |

The results reproduce the awkward code bit for bit, dtype promotion included.

## Reproducing RDataFrame bit for bit

`read_root(..., rdf_compatible=True)` evaluates as C++ / RDataFrame code does:

* float literals in expression strings are doubles: `"x > 0.02"` and `"0.5 * x"` on a float32 `x` are evaluated in
  double (NumPy's default keeps float32 for a Python float);
* 4-vectors (`aks.p4`, `coll.p4`) and `hep.max_mass_pair` use ROOT's `PtEtaPhiMVector` arithmetic: E from
  pt·cosh η, sums converted back to (pt, η, φ, m), GenVector's negative-mass and φ-range rules, `.rapidity`.

Without it, results differ from RDataFrame only in the last bits. ROOT 6.40 is reproduced exactly, on any CPU: the
4-vectors and `hep` functions call the C math library's cos / sin / sinh / cosh / log / atan2, as ROOT does. numpy
functions in your own expressions (`np.sinh(eta)`, `"cosh(x)"`) are numpy's: on CPUs with AVX-512, numpy uses its own
SIMD versions there, which can differ from C++ in the last bit. ROOT 6.36 (LCG_108) builds GenVector with fused
multiply-adds, ~1e-16 relative, invisible once stored as float.

## Rules

* **Row-local expressions.** Row *i* of a column may depend only on row *i* of its inputs: elementwise maths and
  `axis=1` operations. Reductions across rows (`aks.sum(x)` without `axis=1`, `aks.sort` of a flat column,
  `np.cumsum`) are refused at run time; totals are results (`df.sum`, `df.hist`). This is what makes partial
  recomputation under variations exact.
* A lazy column has no truth value and no length: use `& | ~` and `aks.where`, not `and` / `or` / `if`.
* Indexing follows awkward, also in expression strings: element *i* of every row is `x[:, i]`
  (`"weight * LHEPdfWeight[:, 3]"`); `x[3]` would be row 3 (not row-local: refused).
* User functions (`apply`) must be row-local and deterministic, and return one entry per input row.

## Not supported yet

* **String columns** (planned). `df["x"] = "DY"` makes a constant column of text (pandas semantics; expressions
  belong in `define("x", "...")`), but nothing downstream handles text yet: `hist`, `to_root` and comparisons fail
  on it. The plan: string constants and per-event string columns as numpy `<U` arrays (comparisons such as
  `df.sample == "DY"`), written by `to_root` as awkward strings (uproot writes those to a TTree), a string-category
  axis for `df.hist` (`hist.axis.StrCategory`: a process or region axis; integer categories already work with
  `aks.IntCategory`), string keys in `groupby`, string branches
  read from files; for heavy per-event use, pandas-style categoricals (integer codes + labels).
