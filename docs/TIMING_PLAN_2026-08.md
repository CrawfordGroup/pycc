# PyCC timing instrumentation (design plan)

**Status:** Phases 1-3 DONE, merged in PR #248 (2026-08-28). Instrumentation covers the property
facades and the Hessian path; verified on MP2/STO-3G and MP2/cc-pVDZ VCD of 3-chloro-1-butyne with
the spectra unchanged. Phase 4 (triples, CC response, the reference path) not started. Started
2026-08-27, this revision 2026-08-28.

_Two separate facilities (an accumulating profile registry for identifying optimization targets,
and progress output for following a long calculation), plus the call sites to instrument first.
Motivated by an MP2/cc-pVDZ VCD calculation on 3-chloro-1-butyne (4616 s total, 4500 s of it in the
Hessian) whose internal cost split could not be determined from the code as it stands. Companion
artifact: <https://claude.ai/code/artifact/4371db76-a19b-4503-9d3f-53c826a5381a>._

## 1. Why two facilities

They answer different questions and disagree on nearly every design axis, so they are built and
used separately. One mechanism would either spam the log with a line per repeated block, or hide
the aggregate that identifies a target.

| | Profile registry | Progress output |
|---|---|---|
| answers | where did the time go | is it alive, how far along |
| emitted | once, at end of output | continuously, during |
| granularity | fine; nested; repeats aggregate | coarse; one line per outer step |
| a block entered 495x | one row | not instrumented at that depth |
| cost | ~1.6 us per block entry (`getrusage`) | negligible |
| must flush | no | **yes** |

Two facts established while diagnosing the cc-pVDZ run:

* `_hessian_blocks` (`correlatedderivs.py:1082`) contains **zero** `print` calls, so the Hessian
  ran 75 minutes in complete silence.
* stdout is block-buffered when redirected to a file, so progress output needs `flush=True`
  (or `python -u`) to be worth having.

The stage split had to be inferred from the growth of the `DerivStore` HDF5 file on disk, and that
inference was off by ~15%.

## 2. Existing timing facilities

`utils.timing(label, seconds)` renders `"X built in N seconds."` and is called in exactly six
places, all constructor/build sites (`CCwfn`, `MPwfn`, `CIwfn`, `HBAR`, `CC density`).
`utils.converged(name, elapsed)` covers iterative solves. Everything else is ad-hoc `time.time()`.

`properties.py` has **no timing at all**, and neither do `correlatedderivs.py`, `derivatives.py`,
or `DerivStore`. That is the gap.

Note: the `timer.dat` files that appear in run directories are psi4's, not pycc's. PyCC's summary
goes at the end of the run's output; no file artifact.

## 3. Profile registry

### Shape

* One module-level registry. No per-wavefunction instances, no method-qualified labels.
* Nestable context manager for blocks; decorator for whole functions. Both feed the same registry.
* Per label: call count, inclusive time, self time. Labels hierarchical (`hessian/pass1/ao_eri2`)
  so the report renders as a tree.
* On by default. Report prints at the end of the run's output.

### Inclusive and self

A nested table double-counts if it reports one number, because the parent's time already contains
the child's. **Inclusive** is enter-to-exit including children, and answers what a whole stage
costs (the cross-method comparison case). **Self** is inclusive minus the children's inclusive,
sums to the wall time, and is the column to sort when looking for a target.

```
PyCC timing summary   (seconds; self = inclusive minus children)
                                                calls  incl wall  self wall      user      sys  cpu/wall
--------------------------------------------------------------------------------------------------------
MP2 Hessian                                         1      69.07       0.99      1.01     0.07     1.10
  relaxed density (Z-vector)                        1       1.39       1.39      1.39     0.01     1.01
  perturbed wave functions                          1      15.35       0.00      0.00     0.00
    perturbed density                              30      15.31       2.88      3.77     0.16     1.36
      two-electron first derivatives (MO)          10      10.89      10.66     10.86     0.26     1.04
  first-derivative integrals                        1       1.00       0.36      0.36     0.04     1.10
    orbital Lagrangian                             30       0.41       0.41      0.55     0.01     1.37
  second-derivative integral terms                  1      31.04       6.90      7.94     0.26     1.19
    one-electron second derivatives               110       0.44       0.44      0.47     0.03     1.14
    two-electron second derivatives (AO)           55      23.70      23.70    162.22     0.55     6.87
  density-response terms                            1       7.80       6.58      8.03     0.17     1.25
    derivative cache read                         300       1.02       1.02      0.48     0.54     1.00
  two-electron first derivatives (MO)              30      12.01      11.52     11.75     0.11     1.03
--------------------------------------------------------------------------------------------------------
total                                                      70.18               209.07     3.98      3.04
```

Abridged from the MP2/STO-3G VCD run (`~/chem/3-chloro-1-butyne/mp2/sto3g/vcd_wus.out`); the full
table has 45 rows and also covers the APT, AAT, and velocity-gauge APT.

Three clocks are kept because wall time alone cannot say *why* a block is slow. `user` and `sys`
come from `getrusage`, which sums over the process's threads, so in-process OpenMP and threaded BLAS
show up. Reading `cpu/wall` (= (user + sys) / self wall):

| reads as | meaning |
|---|---|
| ~= 8 | threaded, roughly 8 cores busy |
| ~= 1, mostly user | serial and compute-bound |
| ~= 1, mostly sys | kernel-bound (syscalls, page faults on a large allocation) |
| << 1 | blocked, waiting on disk |

The `sys` column is not optional: a block that is entirely kernel time and one that is entirely
serial userland compute both report `cpu/wall` = 1.0, and only `sys` separates them. In the table
above `derivative cache read` is 0.48 user against 0.54 sys, i.e. half syscall traffic, which a
ratio-only report would have shown as ordinary compute.

**Set `OMP_WAIT_POLICY=PASSIVE` and `KMP_BLOCKTIME=0`, before the first import that starts the
OpenMP runtime** (numpy loads its BLAS at import, so this must precede `import numpy`). Workers
busy-wait after a parallel region by default and that spinning CPU is charged to whichever block
runs next: measured here, one allocation read 0.18 s user alone and 3.06 s when it followed a
threaded block. Setting it after numpy is silently ignored, and the symptom is a report where
nearly every row shows the same `cpu/wall`. `report()` warns when the variable is unset, but it
cannot tell whether it took effect.

**What this run showed.** `ao_tei_deriv2` is **already threaded**, at 6.87 of the 8 threads psi4
was given, so process-level parallelism over atom pairs would layer parallelism on parallelism that
already exists. Everything else in the Hessian is serial (1.0 to 1.4), including the `mo_tei_deriv1`
transform at 1.04, which is the second-largest single cost and the more promising target. The two
`two-electron first derivatives (MO)` rows, 10.66 s and 11.52 s self, are the same work done twice
because of the duplicate `DerivStore` (recorded separately as an open issue).

### Accepted limitations

* **Multi-target and repeated-target runs aggregate.** A script computing a CISD Hessian and then
  a CCSD Hessian produces one merged `hessian` row; comparing methods means running them as
  separate calculations. The alternative (per-wavefunction registries, or method-qualified labels
  via `properties._method_label`) buys that one case at the cost of a more complicated facility,
  and is deliberately not built.
* **No aggregation across parallel workers.** The measurement that matters for parallelization is
  wall time of the parallel region vs. the sequential one, taken in the parent. Consequence to keep
  in mind: any block instrumented *inside* what later becomes a worker will report zero calls in
  the parent even though it ran. Revisit when parallelization work begins.

## 4. Progress output

Step time, cumulative stage time, flushed, at the coarse loops only. No projected completion times, since those would
require knowledge of the whole computational chain in the user's script, which the library does not
have. No suppression flag until output volume proves to be a problem.

```
  Hessian perturbed wave functions: 2/30 (C1 y)          step    0.12 s   stage      1.8 s
  Hessian second-derivative integrals: 3/55 (C1-Cl3)     step    0.79 s   stage      3.0 s
  Hessian second-derivative integrals: 21/55 (Cl3-C4)    step    0.85 s   stage     14.7 s
  Hessian density response: 3/10 (Cl3)                   step    0.83 s   stage      2.6 s
```

`step` is that iteration alone; `stage` is cumulative since the stage began. Both are shown because
the cumulative figure says how long you have been in a stage while the per-step figure is what shows
the cost structure: `Cl3-Cl3` costs 2.53 s against 0.30 s for `H9-H10`, the roughly 8x spread
expected from 18 basis functions versus 5. Atoms are labelled by element and 1-based input-geometry
index (`atom_label` in `derivatives.py`), matching `geom.txt` and CFOUR's own `C #1` / `CL#3`.

95 lines across the whole Hessian (30 perturbations, 55 atom pairs, 10 atoms). The triples loop
(560 iterations) and anything at v^3 depth are a different regime and stay out of the progress path.

## 5. Instrumentation sites

Depth is the point; the facade-level blocks exist only to give a whole-property total to sit above
the interior rows. Line numbers current as of 2026-08-27.

| Site | file:line | Isolates |
|---|---|---|
| **Facade: one block per property** | | |
| `hessian`, `apt`, `aat` | `properties.py:364, 383, 423` | whole-property totals |
| `gradient`, `polarizability`, `optical_rotation` | `properties.py:230, 265, 329` | whole-property totals |
| **Derivative integrals** | | |
| `ao_eri2` | `derivatives.py:478` | the `ao_tei_deriv2` call itself; measured at cpu/wall 6.87, i.e. already threaded |
| `eri` / `so_eri` | `derivatives.py:366, 659` | `mo_tei_deriv1` transform, per atom |
| `eri2_mo_component` / `so_eri2_mo_component` | `derivatives.py:499, 795` | the four quarter-transforms |
| `core2` / `overlap2` | `derivatives.py:436, 422` | second-derivative OEI blocks |
| `nuclear_hessian_skeletons` | `derivatives.py:838` | the `'mo'` opt-out route |
| **DerivStore: I/O separated from compute** | | |
| `get_or_compute` | `derivatives.py:134` | split three ways: hit-read, write, compute-on-miss. Pass 2 was estimated at ~250 GB of reads and never verified |
| `_eri_cached` | `derivatives.py:923` | store path vs one-atom LRU |
| **Hessian assembly** | | |
| `_hessian_blocks` | `correlatedderivs.py:1082` | parent block; three children below |
| setup loop | `correlatedderivs.py:~1163` | per perturbation, split into `_relaxed_response` / `full_U` / `d.eri` |
| pass 1 body | `correlatedderivs.py:~1268` | integral build vs contraction, per atom pair |
| pass 2 body | `correlatedderivs.py:~1345` | store reads vs contraction |
| `_offload_assembly_idle_tensors` | `correlatedderivs.py:935` | spill and reload of 2-3 nmo^4 arrays, never timed |
| `_relaxed_response` | `correlatedderivs.py:610` | perturbed solve, 3N of them |
| `_orbital_response` / `_so_orbital_response` | `correlatedderivs.py:320, 380` | Z-vector / relaxed density |
| `_skeleton_lagrangian` | `correlatedderivs.py:1425` | per-X skeleton Lagrangian |
| `_effective_2pdm_ao` | `correlatedderivs.py:990` | AO back-transform of Gam_eff |
| **CPHF and reference** | | |
| `CPHF.full_U` / `CPHF.solve` | `cphf.py:354, 224` | orbital response |
| `_hessian_electronic` / `_hessian_response` | `hfwfn.py:321, 373` | SCF reference blocks |
| `_aat_electronic` | `hfwfn.py:531` | SCF AAT |
| **Correlated properties, triples, response** | | |
| `_correlation_aat` | `mpderiv.py:286`, `cideriv.py:443` | MP2 / CISD AAT |
| `_correlation_dipole_derivatives` | `correlatedderivs.py:796` | APT correlation block, both routes |
| `t_tjl` | `cctriples.py:177` | triples batch vs the two Python loops slated for vectorization |
| `solve_right` | `ccresponse.py:213` | per perturbation and frequency |

## 6. Sequence

1. **Build the two facilities.** New module (`pycc/timing.py`) with the registry, context manager, decorator, and
   report renderer; and the progress helper. No instrumentation yet, so the test suite must be green
   with the module merely importable.

2. **Instrument the Hessian path.** Facade, assembly, derivative integrals, store. This is the path
   with a measured baseline to check against, and the one whose interior split is currently
   guesswork.

3. **Re-run both VCD calculations as a regression test.** MP2/STO-3G and MP2/cc-pVDZ VCD on
   3-chloro-1-butyne (`~/chem/3-chloro-1-butyne/mp2/{sto3g,pvdz}`), both of which have complete
   reference data, so each run checks correctness and instrumentation at once.

   | Check | STO-3G | cc-pVDZ |
   |---|---|---|
   | frequencies vs CFOUR, max abs dev | 0.0001 cm^-1 | 0.0001 cm^-1 |
   | IR vs CFOUR, max abs dev | 0.0013 km/mol | 0.0008 km/mol |
   | Hessian wall time | 58.2 s | 4500.1 s |
   | APT / AAT / VG-APT | 0.2 / 0.2 / 0.2 s | 19.8 / 23.5 / 20.7 s |
   | total | 60.9 s | 4616.4 s |

   **Outcome.** Spectra unchanged at both bases. The Hessian split (cc-pVDZ, 5912 s): perturbed
   wave functions 1207 s, second-derivative integrals 2421 s, density response 1401 s, remainder
   883 s. Three findings:

   * `ao_tei_deriv2` is **already OpenMP-threaded** (cpu/wall 6.87 on 8 threads, 10.59 on 12) and
     is only 13% of the cc-pVDZ Hessian, so it is not the parallelization target it appeared to be.
   * Everything else is serial (cpu/wall 0.6 to 1.4), and the `mo_tei_deriv1` transform is
     **computed twice** (569 s and 599 s self), from the duplicate `DerivStore`.
   * The **O(N^6) setup / O(N^4) integral-build model was wrong.** Both stages scaled by ~78 from
     STO-3G to cc-pVDZ, and pass1/setup is 2.02 against 2.01, i.e. basis-independent. Those
     exponents describe growing the *molecule* (perturbation count ~N, pair count ~N^2), not
     extending the basis at fixed geometry, where every count is fixed. Density response is the
     exception at 180x, and that is a regime change rather than an exponent: `dGam` crosses from
     page-cache-resident (10.7 MB) to real disk (768 MB, so 300 reads = 230 GB).

4. **Extend to APT, AAT, triples, response.** Same pattern outward. Finer granularity wherever
   step 3 shows a row that is large and undifferentiated.

## 7. Deferred

* Worker aggregation for parallel regions; revisit when parallelization work begins.
* Suppression flag; add only if the reports clutter normal output.
* Finer progress granularity inside the triples loops.
* Cross-run comparison of reports; deliberately no file artifact for now.

## 8. Related follow-on work (not part of this plan)

* **`t_tjl` vectorization** (`cctriples.py:210-213`, `231-237`): two pure-Python loops over v^3 and
  the a>=b>=c triangle, ~246M and ~43M interpreted iterations respectively for the cc-pVDZ case,
  against ~10-20 s of actual BLAS work. Both vectorize with loop-invariant arrays (the
  `1 + d_ab + d_ac + d_bc` multiplicity array and the triangular mask) hoisted above the ijk loop.
  Expected to matter more than parallelizing the same loops.
* **Parallelization survey** across the wider triples environment: `t_vikings`,
  `t_vikings_inverted`, `t_vikings_so`, `t_invariant_so`, the `l3_*` lambda-triples family,
  `t3_density` / `dt3_density` / `so_t3_density`, and the CC3 response paths (`_so_cc3_iter`,
  `_so_cc3_build_X3`), many sharing the same ijk-loop shape, so one pattern should be reusable.
  Cost estimates for these get much sharper once this instrumentation reports real numbers.
