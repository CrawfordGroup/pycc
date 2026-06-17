# PyCC Enhancement Plan — Spin-Orbital Path for UHF/ROHF — 2026-06

_Authored from the June 2026 design discussion. This document is **new
territory**, not a continuation of [`REFACTOR_PLAN_2026-06.md`](REFACTOR_PLAN_2026-06.md):
the refactor reorganized the existing closed-shell RHF code onto a `Wavefunction`
base; this plan adds a second, parallel correlation formulation so PyCC can treat
**open-shell UHF (and, later, ROHF) references** via spin orbitals._

## Goal

Let PyCC compute correlated energies on top of a UHF reference (ROHF to follow)
by adding a spin-orbital path alongside the existing spin-adapted spatial path.
The two coexist under one `Wavefunction` base and are chosen by reference type.

The seed for the spin-orbital equations is `~/src/socc` — a separate, fully
spin-orbital code originally forked from PyCC. It already implements spin-orbital
CCSD, CCSD(T), CC3, Lambda, one-particle density, and response (polarizability,
optical rotation) and handles RHF/UHF/ROHF through one Hamiltonian. We graft its
equation kernels in; we do **not** rewrite PyCC around them.

## Decisions (locked, June 2026)

| Question | Decision | Rationale |
|---|---|---|
| Architecture | **Dispatch on reference** | Keep the spatial spin-adapted RHF path as the default for closed shells; add a parallel spin-orbital path under the shared base. Two equation kernels, one base. |
| v1 scope | **Energies first** | MP2 + CCSD + CCSD(T)/CC3 energies. Lambda/density/response deferred to a later phase. |
| Priority reference | **UHF** | UHF is the validation driver. ROHF rides along where correct but is not on the v1 test hook. |

### Why not a flag, and why not a full unification

PyCC's entire correlation layer is written against the closed-shell spin-adapted
tensor

```
L = 2·<pq|rs> − <pq|sr>          # pycc/hamiltonian.py:44
```

`L` exists only for a closed-shell RHF reference (doubly-occupied spatial
orbitals). There is no `L` for UHF/ROHF; the natural object is the antisymmetrized
`<pq||rs>` over spin orbitals, and every residual, intermediate, and energy
expression collapses to the simpler, un-spin-adapted form. So the spin-orbital
capability is a **second formulation**, not a flag — it shares scaffolding with the
spatial path but not equation code.

We rejected unifying everything on spin orbitals (RHF as the α=β special case): it
would discard the 4–8× spin-adaptation speedup *and* the local-correlation and
GPU paths, which all assume spatial `L`, for no correctness gain. Two kernels on one
base is the right cost/clarity trade-off and matches PyCC's reference-implementation
philosophy. We also rejected wholly separate `SOCCwfn`/`SOMPwfn` classes: they would
fragment the public API and duplicate the iteration/DIIS scaffolding.

## Design

### Dispatch mechanism

`Wavefunction.__init__` resolves an `orbital_basis` attribute:

- RHF with `nalpha() == nbeta()` → `'spatial'` (today's behavior, unchanged default).
- UHF, or RHF with an explicit `orbital_basis='spinorbital'` override → `'spinorbital'`.

The base then builds one of two Hamiltonians and exposes a **uniform attribute
surface** (`o, v, no, nv, F, ERI, eps`, plus property integrals `mu/m/p/Q`):

- `Hamiltonian` (current) — spatial; additionally exposes `L`.
- `SpinOrbitalHamiltonian` (new) — block-diagonal Fock from `Fa()`/`Fb()` transformed
  by `Ca`/`Cb`, and antisymmetrized `ERI = <pq||rs>`; **no `L`**.

`_from_shared_base` is untouched — `MPwfn`/`CCwfn` reuse whichever `H` the base built.

### Equation kernels

Method classes keep their public names (`CCwfn`, `MPwfn`, `CIwfn`) and select a
kernel by `self.orbital_basis`:

- `MPwfn._build_mp2` gains a one-line branch: `t2 = ERI[o,o,v,v]/Dijab`, energy
  `¼·<ij||ab>·t2` instead of the `L` contraction.
- `CCwfn` gains a `_residuals_spinorbital` sibling to today's `residuals()`, porting
  `socc`'s `r_T1`/`r_T2` and the `Fae/Fmi/Fme/Wmnij/Wabef/Wmbej` intermediates.

### Reconciliation points

- **Frozen core.** `socc` subtracts `nfrzc()` from *both* spins; PyCC's base already
  slices a frozen core. Reconcile into the base so both kernels agree on the active
  space.
- **Non-diagonal Fock.** `socc` builds denominators from `np.diag(F)` but keeps the
  full `F[v,v]`/`F[o,o]` in `Fae`/`Fmi` — i.e. it does **not** assume a diagonal Fock,
  exactly PyCC's existing discipline. This is what makes ROHF / non-canonical /
  semicanonical orbitals correct without special-casing.
- **Integral build performance.** `socc/hamiltonian.py` assembles `<pq||rs>` with
  `O(nact⁴)` Python loops over spin-orbital quadruples (2× dimension, no
  permutational symmetry). Acceptable as a reference at small basis, but the port
  should be **vectorized** with spin masks before landing.

### Explicitly out of scope for v1

Lambda, density, response, EOM, real-time, local correlation (PNO/PAO), GPU/precision,
and the CPHF/derivative/HF-property stack stay **spatial-RHF-only**. The spin-orbital
path must raise a clear error if asked for any of them — never silently return a wrong
number. ROHF gets a seam and a guard, not half-tested support.

## Validation ladder

The keystone is step 1: it isolates and proves the Hamiltonian fork before any
open-shell physics is introduced.

1. **SO-RHF == spatial-RHF.** Force `orbital_basis='spinorbital'` on a closed shell;
   MP2 and CCSD energies must match the spatial path to ~1e-12.
2. **UHF MP2** vs Psi4 UHF-MP2.
3. **UHF CCSD** vs Psi4 UHF-CCSD — a doublet (e.g. `OH·`, `CH₃·`) or triplet
   (`O₂`/`NO`), STO-3G then cc-pVDZ.
4. **UHF CCSD(T)/CC3** vs Psi4.

## Phasing (one branch per item, PRs merged in the browser)

| # | Branch | Content | Risk |
|---|---|---|---|
| 1 | `feature/spinorbital-hamiltonian` | base dispatch + `SpinOrbitalHamiltonian` (vectorized) + keystone SO-RHF==RHF test | **carries essentially all the architectural risk** |
| 2 | `feature/spinorbital-mp2` | `MPwfn` SO branch + UHF-MP2 test | low |
| 3 | `feature/spinorbital-ccsd` | `CCwfn` SO CCSD kernel + UHF-CCSD test | medium |
| 4a | `feature/spinorbital-pt-ccsdt` | CCSD(T) SO driver (viking) + UHF-CCSD(T) test | low (transcription) |
| 4b | `feature/spinorbital-cc3` | CC3 SO kernel (iterative triples) + UHF-CC3 test | medium (triples module) |

Step 1 is the real work and the real risk. Steps 2–4 are largely transcription from
`socc` plus PyCC-style validation. Phase 4 is split: CCSD(T) (a non-iterative
post-convergence correction) lands first; CC3 (an iterative-triples solver) follows.

## Status / progress

_Last updated 2026-06-17._

| Phase | Status | Landed |
|---|---|---|
| Design / this document | ✅ done | — |
| 1 — SO Hamiltonian + dispatch | ✅ done | PR #133 |
| 2 — SO MP2 | ✅ done | PR #134 |
| 3 — SO CCSD | ✅ done | PR #135 |
| 4a — SO CCSD(T) | 🟡 in review | branch `feature/spinorbital-pt-ccsdt` |
| 4b — SO CC3 | ⬜ not started | — |

### Phase 1 — what landed

- `SpinOrbitalHamiltonian` (`pycc/hamiltonian.py`): block-diagonal Fock from
  `Fa_subset("AO")`/`Fb_subset("AO")` and antisymmetrized `ERI = <pq||rs>`, built by
  **vectorized spin-masked assignment** (no `O(nact⁴)` Python loops): assemble the
  chemist `(pr|qs)` on the four spin-conserving blocks via `np.ix_`, swap to
  physicist `<pq|rs>`, antisymmetrize. Property integrals (`mu/m/p/Q`) built for
  parity; no `L`.
- `Wavefunction` base: `orbital_basis` kwarg + `_resolve_orbital_basis` (auto:
  closed-shell RHF → spatial, else spin-orbital; override allowed). The spatial setup
  moved into `_init_spatial`; new `_init_spinorbital` builds spin-orbital spaces on the
  ACTIVE subset (ordered α-occ, β-occ, α-vir, β-vir). `L` seeding guarded to the
  spatial path.
- `uhf_wfn` fixture (`conftest.py`) and `test_052_spinorbital_hamiltonian.py`:
  **keystone** SO-RHF MP2 == spatial MP2 == Psi4 RMP2 (all-electron + frozen-core,
  ~1e-12); auto-dispatch check; UHF SO-MP2 == Psi4 UMP2 (~1e-10). Full suite green
  (66 passed, no regressions).

### Phase 2 — what landed

- `MPwfn` (`pycc/mpwfn.py`): `compute_energy` branches on `orbital_basis` --
  spatial `E = t2_ijab L_ijab`, spin-orbital `E = 1/4 <ij||ab> t2_ijab`. `_build_mp2`
  was already basis-agnostic (`t2 = ERI[o,o,v,v]/Dijab` and a Fock-diagonal
  denominator hold in either basis), so `pycc.MPwfn(uhf_wfn)` now works end to end via
  auto-dispatch.
- `test_045_mp2.py`: added all-electron and frozen-core **UMP2** checks on the .OH
  doublet through the real `MPwfn`, vs Psi4 conventional UMP2 (~1e-10), alongside the
  existing RHF cases.

### Phase 3 — what landed

- `CCwfn` (`pycc/ccwfn.py`): a spin-orbital CCSD kernel ported from `socc`, selected
  by `orbital_basis`. `residuals`/`cc_energy` gain an early branch to
  `_residuals_spinorbital`/`_cc_energy_spinorbital`; `solve_cc` guards the (nonexistent)
  `L`. The kernel is a set of `_so_*` builders (`tau/taut`, `Fae/Fmi/Fme`,
  `Wmnij/Wabef/Wmbej`, `r_T1/r_T2`) working directly off the antisymmetrized
  `ERI = <pq||rs>`; the Fock is not assumed diagonal. An `__init__` guard keeps the SO
  path to CCSD-only, CPU-only, no local correlation (clear `NotImplementedError`
  otherwise).
- `test_002_ccsd_energy.py` (alongside the RHF CCSD cases): SO-RHF CCSD == spatial
  CCSD on a closed shell (~1e-10, extends the MP2 keystone to the full kernel); UHF
  CCSD vs Psi4 UCCSD, all-electron and frozen-core (~1e-10; measured agreement ~7e-14).
- Folded in the carried-over step-2 cleanup: UHF/SO MP2 test tolerances tightened
  1e-9 → 1e-10.

### Phase 4a — what landed

- `pycc/cctriples.py`: spin-orbital (T) driver `t_vikings_so` + its T3 batch builder
  `t3c_ijk_so`, ported from `socc` (the occupied-batched "viking" algorithm), working
  off the antisymmetrized `ERI = <pq||rs>`. The spatial `t_tjl`/`t_vikings(_inverted)`
  are RHF-specific and untouched.
- `pycc/ccwfn.py`: the `solve_cc` CCSD(T) block branches to `t_vikings_so` when
  `orbital_basis == 'spinorbital'`; the `__init__` guard now admits `CCSD(T)` (CC3
  still rejected). CCSD iterations are unchanged (the (T) is a post-convergence
  correction).
- `test_005_ccsd_t_energy.py`: SO-RHF CCSD(T) == spatial CCSD(T) on a closed shell;
  UHF CCSD(T) vs Psi4 UCCSD(T), all-electron and frozen-core (~1e-10; measured ~7e-14).

**To resume:** read this doc + `git log`. The spin-orbital equation seed lives in
`~/src/socc` (machine-local, not part of this repo); see `socc/hamiltonian.py` for the
Fock/ERI build and `socc/ccwfn.py` for the residual kernels.
