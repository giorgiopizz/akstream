# akstream internals: where the code is

About 5,600 lines: 13 Python modules and one C++ file. Reading order for someone new: `frame.py` → `expr.py` →
`graph.py` → `engine.py` → `execute.py` → `fused.py`; the others are leaves.

## Map

| file | lines | what it holds |
|---|---|---|
| **front end** | | |
| `__init__.py` | 240 | `read_root`, `aks.<awkward function>` lazy wrappers (and field-wise ones on collections), `aks.apply`, `aks.onnx`, `aks.p4`, `aks.collins_soper`, `aks.delta_r`, `aks.checked`, `aks.checkpoint`, `aks.zip` |
| `frame.py` | 715 | `Frame`: column lookup, `define` / `filter` / `eval` / `vary` / `apply` / `report` / result booking; the collection view (`FrameCollection`) and field assignment; the C++ → Python expression translation (`&&`, `!`, `? :`, double literals) |
| `expr.py` | 299 | `Expr` (a lazy column: operators, numpy protocols), `Collection` (fields, masks, indices, `p4`), `call()` = node construction with hash-consing |
| **graph and planning** | | |
| `graph.py` | 194 | node kinds (`Source`, `Call`, `Take`, `Zip`), row spaces (a selection = a space with a mask), graph walks |
| `engine.py` | 382 | `compute()`: the plan (affected nodes per variation, batches of variations, what can be freed), the chunk loop, workers, `aks.varied`, validation |
| **evaluation** | | |
| `execute.py` | 649 | `ChunkContext` (nominal values of a chunk, row indices of each space, freeing), the per-node delta engine (`engine="delta"`, `exact=True`), `_apply` (calls the node function, with the jagged fast paths) |
| `fused.py` | 692 | the default variation engine: phase 1 (rows that change), phase 2 (recompute on those rows, several variations stacked), histogram and file corrections |
| `rows.py` | 312 | row-id set operations and row-wise take / concat / combine / compare for numpy, awkward and dict values |
| `jagged.py` | 218 | fast paths for the `var * number` layout: elementwise ops, masks, integer indices, row gathers, on the flat numpy contents (the gathers in C++) |
| **I/O and results** | | |
| `io.py` | 721 | `Dataset` (files, schema, collections, friends), chunked reading with uproot (bulk + lazy branches, read-ahead), friend alignment checks; `ArraysDataset` (`aks.from_arrays`): the same interface over arrays in memory; `ParquetDataset` (`aks.read_parquet`): flat-ntuple parquet files with pyarrow (row groups decoded once) |
| `results.py` | 1047 | histograms (own binning + boost-histogram fills), sums, group sums, cutflow, collect, `to_root` / `to_parquet` (one file per variation) |
| **physics kernels** | | |
| `hep.py` | 256 | `akstream.hep`: jagged kernels (`nth`, `pick`, `count`, `per_event_sum`, `isolated`, `between`, `select_sort`, `max_mass_pair`, ...), lazy-aware |
| `vector.py` | 263 | `akstream.vector`: `P4` (lazy mass / pt / eta / phi), `RootP4` (ROOT GenVector arithmetic), `add`, `collins_soper`, `delta_phi`, `delta_r` |
| `_hep.cpp` | 742 | the C++ extension (below) |

## How a run goes

1. **Building** (no data touched): every column is a node; identical expressions are one node (hash-consing).
   A selection creates a new *space* (the rows passing its mask); a parent column used on a selection is a `Take`.
   `vary` attaches alternatives to nodes: `node.variations["jes:up"] = alternative`.
2. **Planning** (`engine.Plan`): from the booked results, the reachable nodes, the branches to read, for each variation
   the nodes it affects, and which variations can run stacked.
3. **Per chunk, nominal** (`execute.ChunkContext`): the branches are read in bulk; each node is evaluated once, flat
   values as numpy; values are freed after their last consumer unless a variation needs them.
4. **Per chunk, variations** (`fused.py`): *phase 1* evaluates the cheap affected nodes (masks, selections) where the
   varied inputs differ and compares them with nominal: the rows R that change. *Phase 2* evaluates everything
   downstream on the rows R only, as a plain pass (unaffected inputs are the nominal values on R). Histograms are
   corrected by −fill(nominal on R) + fill(varied on R); `to_root` replaces the rows R in the variation's file.
5. **Merging**: results of workers / chunks are merged; `result.value` is set.

The details that make it fast: expensive nodes (`aks.apply`, ONNX) recomputed only on rows whose inputs really
changed; rows that only enter or leave a selection keep their nominal values; variations replacing the same branches
are stacked into one phase-2 pass; jagged operations on the plain `var * number` layout run on flat numpy arrays.

## The C++ extension (`_hep.cpp`)

Ten event loops, the jagged-layout gathers, and the C math library elementwise, bound with pybind11. Every
kernel works on the flat contents of jagged arrays plus their int64 offsets, and repeats the numpy formulation it
replaces operation by operation: the file is compiled without fused multiply-add (`-ffp-contract=off`) and without the
sin + cos → sincos fusion, so results are bit-identical to the numpy code where numpy itself uses the C math library
(without AVX-512; there numpy has its own SIMD sinh / asinh / log / atan2), and to ROOT on any CPU.

| C++ function | used by | what it does |
|---|---|---|
| `segment_sum` (float32, float64, int64) | `hep.per_event_sum`, `hep.count` | sum per event, in object order, in the values' own type (as `ak.sum(axis=1)`) |
| `select_sort` (float32, float64, int64 keys) | `hep.select_sort` | the objects passing a mask, sorted by a key per event (stable insertion sort, descending or ascending): new offsets + the gather order |
| `max_mass_pair` | `hep.max_mass_pair` | per event, the pair i < j with the largest float32 invariant mass among pairs with mass > cut, \|Δη\| > cut, max pT > cut (first pair on ties), optionally only among the two leading objects; the mass either as the plain cartesian sum or as ROOT's `PtEtaPhiMVector` sum (`rdf_compatible`) |
| `match_dr` | `hep.match_dr`, `hep.clean_dr` | per object of A, the nearest object of B within ΔR (strict <, first on ties), or one-to-one greedy matching (pairs in increasing ΔR) |
| `ancestor` | `hep.ancestor` | per gen particle, walk the mother indices to the first ancestor with a target pdgId (loop-safe) |
| `best_pair` | `hep.best_pair` | per event the (opposite-charge, same-flavour) pair with mass closest to a target, masses as in `max_mass_pair` |
| `p4_cartesian` | `vector.P4.from_ptetaphim` | (pt, η, φ, m) → (px, py, pz, E) |
| `p4_mass` | `P4.mass` | m = √\|E² − p²\| with the sign of E² − p² |
| `p4_angles` | `P4.pt / .eta / .phi` | pt = hypot(px, py), η = asinh(pz / pt), φ = atan2(py, px), each computed only when asked for |
| `cos`, `sin`, `sinh`, `cosh`, `log`, `atan2`, `hypot` | `hep.max_mass_pair`, `hep.best_pair`, `hep.met_shift`, `vector.RootP4` | the C math library elementwise (any shape), as ROOT calls it: the same bits on every CPU, which numpy's own SIMD versions (AVX-512) are not |
| `gather_rows`, `take_rows`, `combine_rows`, `mask_rows`, `index_rows`, `repeat`, `segment_any` (any dtype of 1, 2, 4 or 8 bytes) | `jagged` (`x[rows]`, `x[mask]`, `x[index]`, per-row broadcasts), `rows` (variation row comparisons), `hep`, histogram fills | the row gathers of the `var * number` layout in one pass, elements copied by size: the same bits as the numpy gathers (`np.repeat` + `arange` + fancy indexing) they replace, which stay as the fallback for other element sizes |
| `collins_soper` | `vector.collins_soper` | (φ_CS, cos θ_CS): boost to the pair rest frame, the Collins-Soper axes from the beams, the positive lepton's direction in that frame, in one loop instead of ~60 numpy temporaries |

Everything else (elementwise arithmetic, sorts by awkward, histogram binning) is numpy, awkward or boost-histogram.
`RootP4` (GenVector arithmetic) is numpy apart from those C math library calls: its cost is the transcendental
functions, which a loop does not avoid.

## Tests

`tests/` (264 tests, synthetic NanoAOD-like trees written in `conftest.py`): each feature against plain awkward /
numpy on the same data; variations against brute-force recomputation; the ROOT-compatibility tests against ROOT
itself when it is importable (skipped otherwise). `test_collections.py` covers the collection API, `test_api.py` the
chain API, `test_vector.py` 4-vectors and RDataFrame compatibility, `test_jagged.py` the fast paths, `test_conversions.py`
`from_arrays` and the `to_*` conversions, `test_read_failures.py` the remote strategies (on files with a corrupted
basket).
