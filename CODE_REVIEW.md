# PyCC — Overview & Critique

_Review date: 2026-06-10_

## Remaining cleanup (worklist)

Concrete bugs/hygiene fixes are done and merged to `main` (PR: packaging + GPU CCD,
commit `166df28`). Open items below, roughly highest-leverage first. Detail for each is
in the **Critique** section.

- [x] `ccwfn.py:640` — `torch.zero_like` → `torch.zeros_like` (GPU CCD path)
- [x] `pyproject.toml` — discover subpackages (`pycc.rt`/`pycc.data`/`pycc.tests`)
- [x] `pycc/data/__init__.py` — make `data` a real subpackage
- [x] `ccwfn.py:347` — remove orphaned commented-out line
- [x] `.gitignore` — Psi4 scratch (`psi.*.clean`), `ijk.dat`, `profile.txt`, `.DS_Store`
- [ ] **Shared CC solver base** — extract a `_CCSolver` (or composition) shared by
      `ccwfn` and `lccwfn` to kill the ~800–960-line duplication. _Highest leverage._
- [x] **Contraction-backend abstraction** — replace the ~50×/module
      `contract = self.ec.contract if self.einsums ...` + `.clone()/.copy()` device
      branching with one callable that owns library/device/precision. (Root cause of
      the `zero_like` bug.) _Design drafted below — see "Design note:
      contraction-backend abstraction"; decision is to grow `cc_contract`._
      **Smear #3 (array-namespace helpers) DONE.** Added shared free functions to
      `utils.py` — `zeros_like(a)`, `zeros(shape, like)`, `diag(a)`, `clone(a, device=None)`
      (free, not `cc_contract` methods, because `cctriples`' triples kernels are
      module-level and have no `self.contract`) — and adopted them across the torch-aware
      modules (`cclambda`/`ccdensity` PR #102, `ccwfn`/`cctriples`/`rtcc` PR #103),
      collapsing the `if HAS_TORCH … torch.zeros_like … else np.zeros_like` branches and
      removing every `zeros_like`+`pad` idiom. **Smear #2 (`.clone()/.copy()`) DONE (PR
      #105):** `clone(a, device=None)` collapsed ~166 branched copy sites across
      `cchbar`/`ccwfn`/`cclambda`/`ccdensity`/`cctriples`/`rtcc` + `utils`'s `helper_diis`
      (−215 lines); bare `.copy()` in the numpy-only modules (`ccresponse`/`local`/
      `lccwfn`/`cceom`/`integrators`) was left as-is. **The remaining "irreducible" branches
      are now DONE (PR #107):** `helper_diis.extrapolate` collapses to one block via new
      `real_zeros(shape, like)`/`dot`/`absolute`/`solve` helpers (B/resid use
      `real_zeros` = `like.real.dtype`, staying real even for the complex `ccresponse`
      response amplitudes), and `rtcc`'s `torch.trace`/`np.trace` arms become one
      backend-`contract` `_eref(F)`; same PR added `sqrt`/`reshape`/`concatenate`/`conj`
      and collapsed the convergence-`rms`, `add_error_vector`, `extract_amps`, and
      `autocorrelation` branches (11 branches total). **The `ccdensity` complex
      allocations** (`opdm`/`Dooov`/`Dvvvo`) now use `zeros(shape, like=t1)` so dtype
      tracks the amplitude (real ground-state, complex RT) — which also fixed a dormant
      `np.zeros((no,no,no.nv), …)` typo and a torch-vs-numpy DP dtype mismatch in the CCD
      branches (only the numpy/DP sub-path was ever exercised). The only explicit
      `torch.*` left is genuine one-time construction (`torch.tensor`/`complex*`/`device`/
      `cuda` precision seed-casts in `__init__`). **Smear #1 (library dispatch) DONE.**
      The ~27 per-method `contract = self.contract; if self.einsums: contract =
      self.ec.contract` blocks fold into one `ccwfn.__init__` attribute
      `self._contract = self.ec.contract if self.einsums else self.contract`, used by
      ccwfn's builders and the `cctriples` (T) kernels (−35 lines). **The audit changed
      the plan:** einsums is a *partial* feature (only `ccwfn`/`cctriples` dispatch to
      `self.ec`), so the fold is scoped to a ccwfn-private `_contract` rather than
      `self.contract` — folding into `self.contract` would have silently switched the
      H-bar/Λ/density/EOM/response sub-objects (which inherit it) to einsums. Genuine
      einsums-specific code was preserved: `build_tau`'s transpose formula, the MP2
      sanity check, and `_cc3_t_residual`'s deliberate opt_einsum reset before its final
      `'abc,c->ab'` contraction. Verified locally — einsums *is* installed in `p4env`
      (unlike torch), so `test_038–042` exercise the dispatch (the old "not in CI →
      unverifiable" worry was moot). **This item is now fully done; the only remaining
      backend work is the deeper device-placed object the design note describes (a future
      direction, not a worklist gap).**
- [ ] **Real-valued response amplitudes for imaginary perturbations.** The magnetic-dipole
      (`H.m`) and linear-momentum (`H.p`) integrals are stored pure-imaginary (`* 1.0j`),
      so for those perturbations `ccresponse`'s `X1`/`X2` come out **pure imaginary** —
      verified empirically: real part is *exactly* `0.0`, only the imaginary half is used
      (the electric-dipole `MU` amplitudes are real). They are nonetheless carried as
      `complex128`. Factoring the `i` out and storing them as **real** arrays would halve
      their memory, drop the complex arithmetic, and remove a benign `ComplexWarning` in
      DIIS (`np.dot` of two pure-imaginary vectors returns a complex-dtype scalar whose
      imaginary part is exactly zero, which is then cast into the real DIIS `B` matrix —
      no information lost; the warning is **pre-existing on `main`**, not from the
      backend-helper refactor). A focused `ccresponse` change, independent of the
      contraction-backend work. **Caveat:** the `* 1.0j` integral arrays `H.m`/`H.p`
      themselves **cannot** simply be made real — they are shared with the RT-CC code
      (`rtcc` consumes `self.ccwfn.H.m` for the magnetic-field coupling), which relies on
      their imaginary character. So any "store real" change must factor the `i` out
      *locally inside `ccresponse`*'s amplitude/perturbation handling, leaving the
      `hamiltonian` integral storage untouched.
- [x] **Test fixtures** — added `pycc/tests/conftest.py` (`psi4_environment` +
      `rhf_wfn` factory); converted 32 test modules off the copied psi4-setup block
      (~570 fewer lines). Branch `refactor/test-fixtures`, commit `2cfe825`. A
      per-method `(molecule, basis, ref_energy)` parametrization wasn't added —
      reference energies are method-specific and stay inline in each test. Note:
      `test_040` (einsums CC2) hangs locally with or without this change; einsums
      isn't in CI so those tests skip there.
- [x] **Break up god methods** — DONE. _`cclambda` half (PR #101)._
      `cclambda.solve_lambda` (221→122) and `cclambda.residuals` (169→56) each carried a
      near-verbatim ~120-line CC3 lambda-triples block; extracted to shared helpers
      `_build_cc3_lambda_intermediates` + `_cc3_lambda_triples` (a `real_time` flag is the
      only difference between the two former copies). `ccwfn.t3_density`'s flagged dead
      debug block was **already gone** (removed before this effort; the "912–958" line
      ref was stale) and the method is now a clean 86-line (T)-density kernel — dropped
      from this item. A mirror **t3** dedup in `ccwfn.py` was considered but is a
      **non-issue**: unlike `cclambda`, `ccwfn.solve_cc` already delegates to
      `ccwfn.residuals` (it does not inline its own copy of the CC3 t3 block), so there is
      nothing duplicated to extract. **`ccwfn.r_T2` split DONE (PR #104):** the ~100-line
      CCD/CC2/CCSD/CC3 conditional tree became a thin dispatcher over per-model builders
      `_r_T2_ccd`/`_r_T2_cc2`/`_r_T2_ccsd` (each with its own tensor-expression docstring),
      and a double `self.build_tau` call in the CCSD branch was fixed. (The CC2 builder is
      currently a modified-CCSD form; the user may later rewrite it to a true CC2
      formulation.) Finally, the single-instance CC3 t3 block was pulled out of
      `ccwfn.residuals` (79→52 lines) into `_cc3_t_residual`, parallel to
      `cclambda._cc3_lambda_triples` (readability, not dedup). No remaining god methods
      from the original review; `ccwfn.__init__` (~226 lines of linear setup) is long but
      out of scope here (no deep conditional tree or duplication).
- [x] **Custom exception types** + `.upper()` the string kwargs — DONE. New
      `pycc/exceptions.py` with `PyCCError` base and `InvalidKeywordError(PyCCError,
      ValueError)` (carries `keyword`/`value`/`allowed`, standardized message,
      back-compat via `ValueError`). All 9 input-validation `raise Exception` sites
      converted (7 in `ccwfn.__init__`, 1 in `cceom`, 1 in `local`); the 6 `__main__`
      import guards left as-is. Case-protection added for `local` (`isinstance`-guarded
      `.upper()` so `None` survives) and `local_mos` (the real un-protected string kwarg
      the 3× TODOs had missed). The two boolean TODOs (`it2_opt`/`filter`) were category
      errors — `.upper()` never applied — and were removed rather than "fixed".
      Verified in `p4env`: error path raises with the new message; lowercase
      `local`/`local_mos` accepted and drive localization correctly.
- [x] **GPU-fallback warning** — DONE. New `PyCCWarning(UserWarning)` base in
      `pycc/exceptions.py` (filterable/escalatable). In `ccwfn.__init__`, the
      torch-missing `print` became a `warnings.warn(..., PyCCWarning)` and a *second*
      warning was added for the genuinely-silent case — torch present but
      `torch.cuda.is_available()` False, where `device` stays `'GPU'` while tensors
      silently run on CPU (the elif at the device block; tensor routing unchanged).
      Verified in `p4env` (no torch installed → exercises the torch-missing path:
      one `PyCCWarning`, `device` falls back to `'CPU'`). The CUDA-unavailable elif is
      verified by inspection only — testing it needs torch installed, which we declined
      to add to `p4env`; it's safe because it's only reachable when `HAS_TORCH` is True.
- [x] **CI: CPU-torch test lane** — DONE & MERGED (PR #99). GitHub-hosted
      runners have no GPU/CUDA, but the torch/"GPU" code path falls back to CPU tensors
      when CUDA is absent, so it runs on a standard runner once torch is installed —
      which is what catches torch-API regressions (the `torch.zero_like` class of bug).
      Added a `torch` matrix dimension to `.github/workflows/CI.yaml`: the suite runs
      without torch (covers the `HAS_TORCH=False` CPU paths) and, **Linux-only** (cost),
      with a CPU-only torch wheel (covers the torch path); `fail-fast: false`. Also fixed
      a latent `NameError` in `test_025_contract_gpu.py` (used `torch.complex128` but
      never imported torch — masked because the test was always skipped) via
      `torch = pytest.importorskip("torch")`. **The lane earned its keep on the first run**:
      it surfaced two real defects (the six-module missing-`import torch`, and the stale
      always-skipped `test_025` — both in *Real bugs* below). After those fixes all three
      legs pass green, so the torch path is now actually exercised — including
      `test_025`'s RT-CCSD propagation running on torch-CPU tensors and matching the
      validated reference. **Still open:** real-CUDA numerics need actual GPU hardware
      (self-hosted runner or a paid GPU runner, gated to the nightly `schedule:`) — deferred.
- [x] **Type hints + method docstrings** — DONE (both halves):
      shape/index + CCSD-equation docstrings on the intermediate builders and
      residual/energy methods in `ccwfn.py`, `cchbar.py`, `cclambda.py`
      (commit `11d708e`), and on the `lccwfn.py` local-CC builders/residuals/
      energy (`build_Fae/Fmi/Fme/Wmnij/Zmbij/Wmbej/Wmbje`, `r_T1`, `r_T2`,
      `lcc_energy`) — each documents the per-pair list structure, local virtual
      domains (`dim[ij]`), and `Sij**` overlaps. TYPE-HINT half merged
      (PR #96): new `pycc/_typing.py` with `Tensor = ndarray|torch.Tensor`
      (torch arm guarded so it resolves without PyTorch), `Slice`, and
      `Contract` aliases; `from __future__ import annotations` in every touched
      module; and public-API signatures annotated across all class
      constructors and the user-facing drivers
      (`solve_cc/lcc/lambda/eom`, `compute_energy`, the `ccresponse` drivers,
      `rtcc.f/propagate/dipole/autocorrelation`, `Local.filter_*`, and the
      three public `cctriples` (T) drivers). Cross-class object params use
      `TYPE_CHECKING` forward refs; Psi4 objects are typed `Any`. mypy-in-CI
      was deliberately deferred (Psi4 has no stubs; torch/einsums are optional).

## Overview

**PyCC** is the CrawfordGroup's pure-Python coupled-cluster code — a teaching/research
platform on top of Psi4 (SCF + integrals) and NumPy/opt_einsum for the correlated math.
~12.5k lines across a clean set of method modules.

| Module | Role |
|---|---|
| `ccwfn.py` (960) | T-amplitude solver; the central "god object" |
| `lccwfn.py` (785) | Local-CC T-amplitude solver (PAO/PNO/PNO++) |
| `cchbar.py` / `cclambda.py` / `ccdensity.py` | Similarity-transformed H̄, Λ-amplitudes, one- & two-PDMs |
| `cctriples.py` (625) | (T), CC3, and approximate-triples drivers |
| `cceom.py` / `ccresponse.py` (864) | EOM-CCSD and response/property machinery |
| `local.py` (1031) | Virtual-space localizers (PAO, PNO, PNO++, cPNO++) |
| `hamiltonian.py` | MO integral transform (Fock, ERIs, dipole/angular-momentum) |
| `rt/` | Real-time CC: integrators, lasers, autocorrelation |

Capabilities are broad for a Python code: spin-adapted CCD/CC2/CCSD/CCSD(T)/CC3 energies
and densities, EOM-CCSD, RT-CC with multiple integrators, local-CC variants, GPU (PyTorch)
and single/mixed-precision paths, plus an experimental `einsums` backend. Packaging is
modern (`pyproject.toml`, setuptools-scm), BSD-3, with CI, codecov, 43 test files, and
Sphinx docs. As a research scaffold it's in good shape.

## Critique

### Real bugs

- **Six modules use `torch.*` without importing it** (`utils.py`, `ccdensity.py`,
  `cchbar.py`, `cclambda.py`, `cctriples.py`, `rt/rtcc.py`). Each does
  `from pycc.ccwfn import HAS_TORCH` but never `import torch`, then references `torch`
  inside `if HAS_TORCH and isinstance(x, torch.Tensor):` guards. Latent for the same
  reason as the others: with torch absent the guard is never entered, so the no-torch CI
  never hit it. The moment torch is installed, `helper_diis.__init__` (`utils.py:8`, on
  every solve's DIIS path) raises `NameError: name 'torch' is not defined` → **37 of the
  non-slow tests fail**. **[FIXED 2026-06-11]** — added a guarded `if HAS_TORCH: import
  torch` after the `HAS_TORCH` import in each module. **Found by the new CPU-torch CI
  lane on its first run** (PR #99) — exactly the class of bug that lane exists to catch.
- **`test_025_contract_gpu.py` — stale, never-run test (wrong `rtcc` arg order + stale
  reference).** Because it was always skipped (gpu-marked, no torch in CI), it drifted out
  of sync with the `rtcc` API: it passed `phase` *first* to `collect_amps` and unpacked
  `extract_amps` as `(phase, t1, …)`, but both put `phase` **last** (matching its working
  CPU twin `test_024`). With `phase` bound to `t1`, the `isinstance(t1, torch.Tensor)`
  check fell through to the NumPy branch and `.type()` raised
  `AttributeError: 'numpy.ndarray' object has no attribute 'type'`. Its `mu_z` reference
  (`-0.34894577`) was also stale — the test is the *identical* propagation to `test_024`,
  whose reference was corrected to `-0.0780067603267549` ("removing SCF from original ref")
  while this skipped test never got the update. **[FIXED 2026-06-11]** — corrected the arg
  order and reference; also fixed the **root cause**: the `collect_amps`/`extract_amps`
  docstrings listed `phase` first (and had an `l2, l2` typo) while signature/return put it
  last. Surfaced by the CPU-torch lane (PR #99) once the missing-import bug above was cleared.
- **`ccwfn.py:640` — `torch.zero_like(t1)` is not a function** (should be `torch.zeros_like`).
  On the GPU CCD residual path, latent until someone runs CCD on a Torch tensor → `AttributeError`.
  **[FIXED 2026-06-10]**
- **`pyproject.toml` — `packages = ["pycc"]` omitted the subpackages.** `pycc.rt`, `pycc.data`,
  `pycc.tests` were not declared, so a non-editable `pip install` shipped a broken package
  (no `rtcc`, no `molecules.py`, which 21 test files import). Worked only because everyone uses
  `pip install -e .`. **[FIXED 2026-06-10]** — switched to `[tool.setuptools.packages.find]`
  with `include = ["pycc*"]`, and added `pycc/data/__init__.py` so `data` is a real subpackage.
- **`ccwfn.py:347` — orphaned, de-indented commented line** (`#rms = ec.contract(...)`) at
  column 0 inside a method. Harmless leftover from a half-finished `einsums` refactor. (not fixed)
- **`lccwfn.py` — `np.zeros(dim[ij], dim[ij])` (6 sites, in `build_Wmbje` and `r_T2`)**
  passes the second dimension as the `dtype` arg, not as part of the shape. Latent because
  `dim[ij]` is a `numpy.int64`, so NumPy reads it as `dtype=int64` and silently returns a
  1-D `int64` array — and every one of these is immediately overwritten before use, so it was
  dead code. **[FIXED 2026-06-11]** — wrapped the shapes in tuples. (No runtime change; removes
  the trap if an `=` ever becomes `+=`.)
- **`lccwfn.py` — PAO local-CC diverges to `NaN` (open).** `test_pao_ccd_opt` and
  `test_pao_ccsd_opt` (both `slow`, not in CI) run 100 iterations of `nan` then fail with a
  `TypeError` when the test subtracts the `None` returned by the non-converged `solve_lcc`.
  PNO/PNO++ opt paths converge fine, so this is PAO-specific (possibly the `local_cutoff=2e-2`
  domain construction or a sign/projection error on the PAO path) — a genuine numerical bug,
  NOT the `np.zeros` issue above. Needs investigation.
- **`ccwfn.py:solve_cc` — the torch/GPU convergence branch silently skips the `(T)`
  correction (open).** In the convergence block the `if HAS_TORCH and isinstance(self.t1,
  torch.Tensor):` arm prints/returns the bare `ecc` (CCSD energy), while only the NumPy
  `else` arm has the `if self.model == 'CCSD(T)': et = t3_density()/t_tjl(self); ecc += et`
  step. So **`model='CCSD(T)'` run on a torch tensor returns the CCSD energy with no `(T)`
  correction** — a wrong result, not a crash. Latent because torch/GPU isn't in CI (the
  CPU-torch lane added in PR #99 *would* catch it if a `CCSD(T)` torch test existed). Fix:
  hoist the `(T)` step out of the backend branch so both paths compute it. Surfaced while
  auditing remaining `HAS_TORCH` branches for the backend-helper sweep; left as a separate
  behavior fix (untestable locally — no torch in `p4env`).

### Structural issues (the expensive ones)

- **`ccwfn` / `lccwfn` duplication.** Two ~800–960-line solvers implement the same CC iteration
  with no shared base. Every intermediate fix (Fae/Fmi/Wmbej, DIIS wiring, convergence) must be
  made twice, and they're already drifting. A common `_CCSolver` base (or composition) is the
  single highest-leverage refactor.
- **Backend dispatch smeared across every method.** The
  `contract = self.contract; if self.einsums: contract = self.ec.contract` pattern plus
  `if HAS_TORCH and isinstance(t1, torch.Tensor): .clone() else .copy()` appears ~50+ times per
  module. Should be one contraction-backend abstraction (a callable that already knows
  device/precision/library). This smearing is exactly what produced the `zero_like` bug.
- **God methods.** `cclambda.solve_lambda` (~220 lines) and `residuals` (~170) — both
  cut roughly in half by extracting the shared CC3 lambda-triples block (PR #101);
  `ccwfn.t3_density` (now ~86 lines; the once-flagged commented-out debug block is already
  gone); `ccwfn.residuals`/`r_T2` with deep CCD/CC2/CCSD/CC3 conditional trees (still
  open). Hard to test and review.

### Quality / maintainability

- **No type hints anywhere**; method-level docstrings inconsistent (good class-level numpydoc,
  but most `build_*` intermediates undocumented). For a code where tensor index order and o/v
  slicing is the whole ballgame, even shape-documenting docstrings would pay off.
- **No `conftest.py`.** The same `psi4.set_memory / set_output_file / set_options` block is
  copy-pasted into 40+ files; geometries re-defined despite `data/molecules.py`; reference
  energies are hardcoded magic numbers with `1e-11` tolerances. A few fixtures (`rhf_wfn` factory,
  parameterized `(molecule, basis, ref_energy)`) would shrink the suite and unify tolerances.
  Stray `psi.*.clean` / `*.dat` test-output files should be `.gitignore`d.
- **Error handling is all `raise Exception("...")`** for kwarg validation, with
  `# TODO: case-protect this kwarg` noted 3× in `ccwfn.__init__`. Cheap wins: custom exception
  types and `.upper()` on the string kwargs already validated against uppercase lists.
- **Silent CPU fallback** when GPU is requested but Torch is missing — never tells the caller,
  making "why is this slow / why did precision change" hard to diagnose. A one-line warning helps.

### Design note: contraction-backend abstraction

_Drafted 2026-06-11. Design only — no code yet. Decision: grow the existing
`cc_contract` (utils.py:141) into the backend object rather than introduce a new class._

**The item bundles three distinct smears, not one.** Counts are repo-wide across the
CC modules:

| # | Pattern | Sites | Risk |
|---|---|---|---|
| 1 | `contract = self.contract; if self.einsums: contract = self.ec.contract` (per-method library re-dispatch) | ~50 | low |
| 2 | `if HAS_TORCH and isinstance(x, torch.Tensor): x.clone() else x.copy()` (device copy) | ~201 | medium |
| 3 | `torch.sqrt`/`np.sqrt`, `torch.zeros_like`/`np.zeros_like`, etc. (device reductions/constructors) | ~97 `isinstance(_, torch.Tensor)` guards | medium |

`cc_contract` already owns the CPU/GPU branch for *contraction* (`__call__` moves
operands to `device1` and calls `opt_einsum.contract`). It is the natural home for the
array-namespace helpers the other two smears need. `self.contract = cc_contract(device=...)`
is set once at ccwfn.py:245 and threaded to every sub-object (`self.contract =
self.ccwfn.contract`), so a single object already reaches every call site.

**Target shape.** Grow `cc_contract` into a backend that exposes, alongside `__call__`:

```
backend.clone(x)        # x.clone() if torch else x.copy()
backend.zeros_like(x)   # the would-be home of the zero_like bug — one definition
backend.zeros(shape, like=x) / ones(...)
backend.sqrt(x) / abs(x) / real(x) / diag(x) / dot(a,b) / cat(xs)
```

The histogram of `torch.*` calls fixes the helper set: `zeros_like` (47), `diag` (14),
`zeros` (8), `abs` (5), `sqrt`/`real`/`ones`/`dot`/`cat` (≤4 each). `torch.tensor`/
`torch.complex`/`torch.device`/`torch.cuda` are construction-only in `__init__` and stay.

**Dispatch on the operand, not the device flag.** Each helper keeps the
`isinstance(x, torch.Tensor)` check *internally* rather than branching on
`self.device == 'GPU'`. Reason: in GPU mode some arrays deliberately stay on CPU
(`H.ERI`, `H.L` live on `device0=cpu` while `t1/t2/F` live on `device1`; ccwfn.py:255–262),
so the runtime type is the correct key and the device flag would be wrong. This also keeps
the helpers usable from the mixed CPU/GPU reductions without a special case.

**Precision mostly rides along for free.** SP/DP is applied once at construction by casting
the seed arrays (ccwfn.py:234–274). `clone`/`zeros_like`/`sqrt` inherit dtype from the
operand, so they need no precision logic. The only sites needing an explicit `dtype` are
the bare `zeros`/`ones` constructors (≤10) — give those a `like=` operand to copy dtype/
device from, so precision stays a property of the data, not a branch in every method.

**Smear #1 (library dispatch) — do first, separately.** Fold the choice in once:

```python
self.contract = self.ec.contract if self.einsums else cc_contract(device=self.device)
```

then delete the ~50 per-method `if self.einsums:` reassignments. Cheap and mechanical.
**One caveat that gates this:** `ein` is the *external* `einsums` C++ library
(`import einsums as ein`, ccwfn.py:31); the per-method pattern may exist precisely because
einsums doesn't implement every contraction ("...where implemented", ccwfn.py:279) and some
methods are deliberately pinned to `opt_einsum`. Before collapsing, audit whether any method
uses `self.contract` (opt_einsum) while `self.einsums` is true — i.e. a real per-method
override, not redundancy. The einsums path is **not in CI**, so tests won't catch a
regression here; this needs a manual einsums run or an explicit "einsums unverified" note.

**Migration strategy.** Incremental and reviewable:
1. Add the helpers to `cc_contract` (pure addition, no call sites changed) + unit-test each
   helper on a NumPy and a Torch input.
2. Collapse smear #1 across all modules in one mechanical pass (after the audit above).
3. Sweep smears #2/#3 **one module at a time** (`ccwfn` → `cchbar` → `cclambda` →
   `ccdensity` → `cceom`/`ccresponse`/`cctriples` → `lccwfn`), running the CPU test suite
   after each module. GPU and einsums paths aren't in CI — flag both as manually-verified
   or known-unverified per module.

**Why this is worth it:** smear #3 is exactly what produced the `zero_like` bug — there
were 47 places to mistype `zeros_like`, and one was wrong. Centralizing collapses 47 → 1.
Net deletion is on the order of ~250–300 lines once #1–#3 are done.

### Bottom line

Scientifically impressive and broad. Technical debt is concentrated and predictable:
1. the `ccwfn`/`lccwfn` fork,
2. hand-rolled backend dispatch repeated everywhere (source of the `zero_like` bug),
3. test-setup duplication.

Recommended order: fix the concrete bugs (done), then a shared solver base + a contraction-backend
abstraction as the medium-term cleanup, with a `conftest.py` fixture refactor alongside.
