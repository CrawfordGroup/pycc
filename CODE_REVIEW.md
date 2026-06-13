# PyCC — Overview & Critique

_Review date: 2026-06-10 (last updated 2026-06-13 — fixed/landed items pruned from the Critique)_

## Remaining cleanup (worklist)

Concrete bugs/hygiene fixes are done and merged to `main` (PR: packaging + GPU CCD,
commit `166df28`). Open items below, roughly highest-leverage first. Detail for each is
in the **Critique** section.

- [x] `ccwfn.py:640` — `torch.zero_like` → `torch.zeros_like` (GPU CCD path)
- [x] `pyproject.toml` — discover subpackages (`pycc.rt`/`pycc.data`/`pycc.tests`)
- [x] `pycc/data/__init__.py` — make `data` a real subpackage
- [x] `ccwfn.py:347` — remove orphaned commented-out line
- [x] `.gitignore` — Psi4 scratch (`psi.*.clean`), `ijk.dat`, `profile.txt`, `.DS_Store`
- [~] **Shared CC solver base** — _re-scoped after a direct ccwfn/lccwfn audit
      (2026-06-13)._ The original "~800–960-line duplication → extract a `_CCSolver`"
      framing is **misleading**: the two solvers implement the **same CC equations**
      (the `build_*`/energy docstrings match equation-for-equation) but via **different
      data structures** — `ccwfn` uses full canonical tensors with `contract(...)`,
      `lccwfn` uses per-pair PNO/PAO lists with `QL` projections + `Sij` overlaps and
      explicit `ij`/`m`/`n` loops. So there is **almost no textually-identical code to
      hoist**: `build_Fae` is ~6 lines vs ~45/branch, `cc_energy` ~2 lines vs ~25, and
      every intermediate/`r_T1`/`r_T2`/residual is a genuine reimplementation in
      projected local domains. A full shared-intermediate `_CCSolver` is **not feasible**
      without first building a domain-abstracted tensor layer (canonical = local with
      identity projections) — research-scale, changes numerics, **not recommended**.
      _What is actually shared:_ only the ~30-line **driver skeleton** (`solve_cc`/
      `solve_lcc`: init-energy → iterate → residuals → update → rms → energy →
      convergence-print → DIIS) — a thin template-method base is the *only* viable dedup,
      and is **optional** (for a teaching/research code, two explicit solvers may read
      better than a hook-based base). The equation docstrings are already kept in sync by
      hand. **Decision: keep the two solvers explicit; no base-class extraction.**
      _Related observation (not a planned change):_ `lccwfn` has **no DIIS** — it's
      commented out (`#ldiis = helper_ldiis(...)`). `utils.helper_diis` is not a drop-in:
      it assumes uniform `ndarray` `t1`/`t2` (`.ravel()` + `concatenate`), whereas
      `lccwfn`'s `t1`/`t2` are **lists of per-pair arrays of varying dimension**, so a
      local DIIS would need its own pair-aware error-vector handling. The missing
      convergence acceleration plausibly feeds the open **PAO NaN bug**.
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
- [x] **Real-valued response amplitudes for imaginary perturbations** — DONE (PR #111).
      The magnetic-dipole (`H.m`) and linear-momentum (`H.p`) integrals are stored
      pure-imaginary (`* 1.0j`) for the RT-CC code, so for those perturbations
      `ccresponse`'s `X1`/`X2` came out pure imaginary (real part *exactly* `0.0`) but were
      carried as `complex128`. Factored the `i` out **locally in `ccresponse.__init__`**
      (Option 1 of two): the `M`/`M*`/`P`/`P*` pertbars are built from
      `np.real(-1.0j * pert)` — a real-dtype `A`, not a complex array with zero imaginary
      part — with `-1.0j` applied to the conjugate operators too, so `M* = -M` (`P* = -P`)
      stays distinct. `hamiltonian.py`/`rtcc.py` left untouched (Option 2 — making the
      integrals real and re-imaginary-ing in `rtcc` — was rejected: it spreads `i`-handling
      to both modules and `np.conj` of a real array collapses the `M`/`M*` distinction).
      Results-preserving because `pertbar`/`solve_right` are linear in the perturbation and
      pseudoresponse/polarizability are bilinear (`~ conj(c)*c = |c|^2`, and `|i|^2=|1|^2`),
      so every property value is unchanged; `X1`/`X2` are now real `float64` (half memory,
      and the two DIIS `ComplexWarning`s are gone). _Known limitation:_ a genuinely mixed
      real×imaginary property (optical rotation `<<MU;M>>` via `linresp_asym`) would now
      differ by a factor of `i` — untested and not currently functional (the polarizability
      accumulator is a real array), so nothing working regresses; if such properties are
      added later, carry the stripped phase on `pertbar` and apply it in the contraction.
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
      always-skipped `test_025`) — both fixed in the same effort. After those fixes all three
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
| `ccwfn.py` (1163) | T-amplitude solver; the central "god object" |
| `lccwfn.py` (984) | Local-CC T-amplitude solver (PAO/PNO/PNO++) |
| `cchbar.py` / `cclambda.py` / `ccdensity.py` | Similarity-transformed H̄, Λ-amplitudes, one- & two-PDMs |
| `cctriples.py` (666) | (T), CC3, and approximate-triples drivers |
| `cceom.py` / `ccresponse.py` (873) | EOM-CCSD and response/property machinery |
| `local.py` (1042) | Virtual-space localizers (PAO, PNO, PNO++, cPNO++) |
| `hamiltonian.py` | MO integral transform (Fock, ERIs, dipole/angular-momentum) |
| `rt/` | Real-time CC: integrators, lasers, autocorrelation |

Capabilities are broad for a Python code: spin-adapted CCD/CC2/CCSD/CCSD(T)/CC3 energies
and densities, EOM-CCSD, RT-CC with multiple integrators, local-CC variants, GPU (PyTorch)
and single/mixed-precision paths, plus an experimental `einsums` backend. Packaging is
modern (`pyproject.toml`, setuptools-scm), BSD-3, with CI, codecov, 43 test files, and
Sphinx docs. As a research scaffold it's in good shape.

## Critique

### Real bugs

_Open bugs only. The fixed bugs from the original review (the six-module missing-`import
torch`, the stale `test_025`, `torch.zero_like`, the `pyproject.toml` subpackages, the
`ccwfn.py:347` orphaned comment, and the `lccwfn` `np.zeros(dim,dim)` shape trap) are
recorded with their PRs in the worklist above._

- **`lccwfn.py` — PAO local-CC diverges to `NaN` (open).** `test_pao_ccd_opt` and
  `test_pao_ccsd_opt` (both `slow`, not in CI) run 100 iterations of `nan` then fail with a
  `TypeError` when the test subtracts the `None` returned by the non-converged `solve_lcc`.
  PNO/PNO++ opt paths converge fine, so this is PAO-specific (possibly the `local_cutoff=2e-2`
  domain construction or a sign/projection error on the PAO path) — a genuine numerical bug,
  not the (now-fixed) `np.zeros(dim,dim)` shape trap. Needs investigation.
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

- **`ccwfn` / `lccwfn` parallelism.** Two ~1000–1160-line solvers implement the **same CC
  equations** with no shared base. _But the 2026-06-13 audit showed this is conceptual, not
  textual, duplication_ — `lccwfn` reformulates every intermediate/residual/energy in projected
  per-pair PNO/PAO domains (loops + `QL`/`Sij`), so the bodies are genuine reimplementations,
  not copy-paste, and a shared-intermediate base class isn't feasible without a domain-abstracted
  tensor layer. The real maintenance cost is keeping the *equations* in sync (the docstrings
  already mirror each other) — not deduplicating code. See the re-scoped worklist item above for
  the decision (keep both solvers explicit). _Observation:_ `lccwfn` also has no DIIS
  (commented-out `helper_ldiis`); `utils.helper_diis` is not a drop-in (it assumes uniform
  `ndarray` amplitudes, but local `t1`/`t2` are per-pair lists), and the missing convergence
  acceleration plausibly feeds the PAO NaN bug above.

_The other two structural issues from the original review — **backend dispatch smeared across
every method** and **god methods** — are now resolved (see the `[x]` contraction-backend and
break-up-god-methods worklist items; all three contraction-backend smears and the method
splits landed in PRs #101–#109)._

### Design note: contraction-backend abstraction — IMPLEMENTED

_Drafted 2026-06-11, implemented across PRs #102–#109. Kept here for the design rationale
that still informs future GPU work; the migration plan/histograms have been pruned now that
the work is done._

The abstraction was three distinct "smears": (1) per-method `if self.einsums` library
re-dispatch, (2) `.clone()/.copy()` device copies, (3) `torch.*`/`np.*` array-namespace
calls (`zeros_like`/`zeros`/`diag`/`sqrt`/…). Decisions that held up and should guide the
remaining/ future backend work:

- **Free functions in `utils.py`, not `cc_contract` methods** — `cctriples`' (T) kernels are
  module-level and have no `self.contract`, so `zeros_like`/`zeros`/`real_zeros`/`diag`/
  `clone`/`dot`/`absolute`/`conj`/`solve`/`sqrt`/`reshape`/`concatenate` are free functions.
- **Dispatch on the operand's runtime type** (`isinstance(x, torch.Tensor)`), not on a device
  flag — in GPU mode some arrays deliberately stay on CPU (`H.ERI`/`H.L` on `device0`).
- **Precision rides along with the data** — `like=`/`like.real.dtype` seeds inherit dtype
  (and, for torch, device), so SP/DP is never a per-method branch.
- **Smear #1 is scoped, not global** — einsums is a partial feature (only `ccwfn`/`cctriples`
  dispatch to `self.ec`), so the fold went into a ccwfn-private `self._contract`, *not*
  `self.contract` (which the H̄/Λ/density/EOM/response sub-objects inherit and must keep on
  opt_einsum).
- **What stays explicit:** one-time precision/device construction (`torch.tensor`/`complex*`/
  `device`/`cuda` seed-casts in `__init__`), and the genuinely backend-divergent device-placed
  linear algebra. The natural next step (a *future direction*, not a worklist gap) is a deeper
  backend object that owns device-placed ops so even those can move behind helpers.

### Bottom line

Scientifically impressive and broad. Of the three original debt clusters —
(1) the `ccwfn`/`lccwfn` fork, (2) hand-rolled backend dispatch, (3) test-setup duplication —
**(2) and (3) are done**, and **(1) was re-scoped**: the audit showed it's two implementations
of the same equations (not copy-paste), so the decision is to keep both solvers explicit. The
remaining open work is concrete and bounded: the **PAO NaN bug** and the **CCSD(T)-on-GPU
`(T)` skip** — both in the worklist above. (The real-valued response amplitudes landed in
PR #111.)
