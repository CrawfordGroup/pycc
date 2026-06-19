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

_Last updated 2026-06-19._

| Phase | Status | Landed |
|---|---|---|
| Design / this document | ✅ done | — |
| 1 — SO Hamiltonian + dispatch | ✅ done | PR #133 |
| 2 — SO MP2 | ✅ done | PR #134 |
| 3 — SO CCSD | ✅ done | PR #135 |
| 4a — SO CCSD(T) | ✅ done | PR #136 |
| 4b — SO CC3 | ✅ done | PR #137 |
| 5 — ROHF (semicanonical) | ✅ done | PR #138 |
| 6 — SO CISD/CID (UHF/ROHF) | ✅ done | PR #139 |
| 7 — SO Lambda (CCSD) | ✅ done | PR #140 |
| 8 — SO density + CC dipole | ✅ done | PR #142 |
| 9a-i — SO symmetric polarizability | ✅ done | PR #143 |
| 9b — SO optical rotation | ✅ done | PR #144 |
| 9a-ii — spatial spin-adapted response | ✅ done | PR #145 |

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

### Phase 4b — what landed

- `pycc/ccwfn.py`: spin-orbital CC3. `_residuals_spinorbital` adds the connected-triples
  contribution (`_so_cc3_t_residual`) to the CCSD residuals when `model == 'CC3'`; the
  T1/T2 equations stay CCSD and T3 is rebuilt each iteration from the five T1-dressed
  CC3 W-intermediates (`_so_build_W{oooo,ovoo,ooov,vovv,vvvo}_CC3`). `__init__` guard now
  admits CC3.
- `pycc/cctriples.py`: `t3c_ijk_so` refactored to take `Wvvvo`/`Wovoo` explicitly (bare
  ERI slices for (T), T1-dressed intermediates for CC3); `t_vikings_so` updated to pass
  the slices. Only the memory-light batched driver is ported -- the `store_triples`/field
  CC3 path (socc `CC3_full`/`permute_triples`) is out of scope.
- `test_031_cc3.py`: SO-RHF CC3 == spatial CC3 on a closed shell; UHF CC3 vs Psi4 UCC3,
  all-electron (6-31G to keep the iterative SO run quick; ~1e-10, measured ~1e-13).
  Frozen-core UCC3 is deliberately not retested -- the SO frozen-core active space is
  covered by the MP2/CCSD/CCSD(T) frozen-core tests, and CC3 is the costliest solve.
- Spin-orbital energies are now **complete for v1**: MP2, CCSD, CCSD(T), CC3.

### Phase 5 — what landed (ROHF via semicanonical orbitals)

Completes the plan's "UHF **and ROHF**" goal. ROHF auto-dispatches to the spin-orbital
path (open shell), and the SO machinery is made correct for a non-canonical reference:

- `pycc/hamiltonian.py`: `SpinOrbitalHamiltonian` semicanonicalizes each spin --
  `_semicanonicalize` diagonalizes the occ-occ and vir-vir blocks of the MO Fock and
  rotates the occ/vir MO column-blocks by the eigenvectors, then all AO->MO transforms
  (Fock, ERI, properties) use the rotated coefficients. Well-defined orbital energies
  (hence non-iterative MP2/(T)/CC3 denominators) for ROHF; a no-op for canonical
  RHF/UHF (blocks already diagonal). The base passes the per-spin occupied counts so the
  Hamiltonian knows the occ/vir split.
- `pycc/mpwfn.py`: MP2 gains the first-order **singles** `t1 = f_ia/Dia` and the
  `f_ia t1_ia` energy term (spin-orbital path) -- nonzero only for a non-canonical
  reference. `CCwfn` starts its spin-orbital `t1` from this MP1 guess.
- `pycc/cctriples.py`: the viking (T) gains the non-canonical occ-vir-Fock term
  `x2[i,j] += f_kc t3` (the bare-Fock analog of CC3's `Fme[k]` term); zero for canonical
  refs, so RHF/UHF (T) are unchanged. (Also folded in a CC3 micro-opt: the `Wvovv`
  antisymmetrization now reuses one contraction via `tmp - tmp.swapaxes(0,1)`.)
- `conftest.py` `rohf_wfn` fixture; `test_054_rohf.py`: ROHF MP2/CCSD/CCSD(T) (cc-pVDZ)
  and CC3 (6-31G) on .OH vs Psi4's semicanonical CCENERGY (all ~1e-13/1e-14).

### Phase 6 — what landed (spin-orbital CISD/CID)

Beyond the original energies-first scope, extends `CIwfn` to UHF/ROHF.

- `pycc/ciwfn.py`: `ci_energy`/`sigma1`/`sigma2` branch on `orbital_basis` to spin-orbital
  siblings -- the linear part of the spin-orbital CCSD residual with bare integrals.
  The doubles sigma carries an extra **non-canonical singles->doubles Fock coupling**
  `<Phi_ij^ab|F_N|Phi_k^c> = P(ij)P(ab) f_jb c1_ia`: zero for canonical refs and for CID,
  absorbed by the T1 transformation in CCSD, but required explicitly in linear CISD for
  ROHF. (Found via CFOUR: ROHF-CID matched but ROHF-CISD was 5.5e-5 off until this term.)
- Validation (`test_051`): closed-shell keystone (SO-RHF == spatial); 2-electron CISD =
  FCI (exact) for UHF/ROHF vs Psi4 FCI; and a **CFOUR** cross-check (UHF/ROHF x CISD/CID,
  .OH cc-pVDZ, ~1e-13). Note: Psi4 has no UHF-CISD, and its DETCI ROHF-CISD is
  spin-adapted -- a different method that disagrees with both PyCC and CFOUR (~3e-4).

### Phase 7 — what landed (spin-orbital Lambda)

The first step toward open-shell **properties** (Lambda -> density -> response).

- `pycc/cchbar.py`: `cchbar` builds the 10 spin-orbital HBAR blocks (no separate `Hovov`;
  an inline `Zovov` feeds `Hvvvo`/`Hovoo`) from the antisymmetrized ERI, via a
  `_build_spinorbital` dispatch + `_so_build_*` methods (ported from socc).
- `pycc/cclambda.py`: `__init__`/`solve_lambda`/`build_Goo`/`build_Gvv`/`pseudoenergy`/
  `r_L1`/`r_L2` branch on `orbital_basis` to spin-orbital siblings. SO guess l1=t1, l2=t2;
  pseudoenergy 1/4 <ij||ab> l2. CCSD/CCD only (CC3 Lambda raises in the SO path).
- Validation (`test_003`): keystone (SO-RHF Lambda == spatial, ~1e-13) and a **CFOUR**
  cross-check of the Lambda pseudoenergy for UHF and ROHF (.OH cc-pVDZ, exact to 12
  digits -- CFOUR `PRINT=2` reports the total Lambda pseudoenergy with the same
  definition).

### Phase 8 — what landed (spin-orbital density + CC dipole)

Turns Lambda into an open-shell **property** -- the first CC dipole for UHF/ROHF.

- `pycc/ccdensity.py`: `build_Doo`/`build_Dvv`/`build_Dov` branch on `orbital_basis` to
  the spin-orbital one-particle density blocks (socc's, no RHF spin factors; `Dvo = l1.T`
  is shared). A new `ccdensity.dipole(t1,t2,l1,l2)` returns the unrelaxed (expectation-
  value) CC dipole `sum_pq mu_pq D_pq` (correlation only; reference/nuclear not included),
  CCSD/CCD. The spin-orbital two-particle density is not implemented (one-particle/dipole
  only -- pass `onlyone=True`; raises otherwise).
- Validation (`test_007`): keystone (SO-RHF CC dipole == spatial, ~1e-12) and a **CFOUR**
  cross-check of the UHF/ROHF CC dipole (.OH cc-pVDZ) via `PROP=FIRST_ORDER`,
  `DIFF_TYPE=UNRELAXED`, `CC_PROG=ECC`, `ABCDTYPE=AOBASIS`, plus `FIXGEOM=ON` (Cartesian
  input) so CFOUR keeps the input frame -- its SCF dipole then matches Psi4's exactly and
  the reference is simply CFOUR's `CCSD - SCF` electronic dipole (~1e-10).

### Phase 9 — response (linear response complete)

Decision: a NEW **symmetric** linear-response framework (socc-style: top-level
`polarizability`/`optrot`, right-hand X amplitudes only at +/-omega, no Y) serving BOTH
bases, eventually superseding the asymmetric `ccresponse` for *linear* response. The
asymmetric machinery (`solve_left`, `in_Y*`/`r_Y*`, `linresp_asym`) is **kept (deprecated
for linear response)** -- the quadratic response function will need the left-hand
contributions. Split: 9a polarizability (9a-i spin-orbital, 9a-ii spatial spin-adapted),
9b optical rotation.

**Phase 9a-i (SO symmetric polarizability):**
- `pycc/ccresponse.py`: `pertbar`, `pseudoresponse`, `r_X1`, `r_X2` branch on
  `orbital_basis` to spin-orbital siblings (ported from socc). New top-level
  `polarizability(omega)` + symmetric `linresp_sym` + `LCX`/`LHX1Y1`/`LHX2Y2`/`LHX1Y2`
  (spin-orbital). The spatial branch of `polarizability` raises for now (9a-ii).
- Validation (`test_055`): SO-RHF dynamic polarizability (isotropic, omega=0.1) ==
  Psi4 CCSD polarizability (~1.8e-12); UHF/ROHF **static** polarizability == in-place
  finite difference of the CCSD energy (~1e-8/1e-9). No code does open-shell CC
  *dynamic* polarizabilities, so static-via-finite-difference is the open-shell check
  (the dynamic kernel is validated for RHF and is identical for UHF/ROHF).

**Phase 9b (SO optical rotation):**
- `pycc/ccresponse.py`: new top-level `optrot(omega)` (length gauge, `<<mu;m>>` two-term
  assembly), reusing the X-solver + `linresp_sym`. Also made `pertbar` construction
  **lazy**: the constructor no longer builds all operators (MU/M/M*/P/P*/Q); `self.pertbar`
  is a `_PertbarCache` that builds each via `_build_pertbar` on first access, so a response
  function pays only for the operators it uses (socc's efficiency, transparent to the
  deprecated asymmetric callers/tests).
- Validation (`test_056`): SO-RHF optical-rotation tensor (chiral H2O2/STO-3G,
  omega=0.077357) == Psi4 CCSD optrot (length gauge) ~6.5e-13 (also confirms the SO
  magnetic-dipole H.m integrals). UHF/ROHF: no open-shell CC optrot reference exists (a
  dynamic magnetic property, no finite-difference), so validated by composition; the tests
  only confirm the open-shell path runs and returns a finite tensor.

**Phase 9a-ii (spatial spin-adapted symmetric response):** completes the spatial branch
of the symmetric framework, so `polarizability()`/`optrot()` now run spin-adapted for
closed-shell RHF (not just spin-orbital). With this, linear response is done in both bases.
- `pycc/ccresponse.py`: fill the spatial stubs of `linresp_sym` and `LCX`/`LHX1Y1`/
  `LHX2Y2`/`LHX1Y2`, each derived term-by-term against its `_*_spinorbital` sibling. The
  ph-ring terms reuse the `_r_T2_ccsd` three-term ring (`build_Wmbej`/`build_Wmbje`-style
  intermediates); the `LHX1Y2` voov ring uses X1-dressed intermediates with Y2 as the
  external doubles. The only basis-specific piece of the `linresp_sym` assembly is the HXY
  direct term, which spin-adapts to `2*L` (= 4<ij|ab> - 2<ij|ba>). Each spatial method
  carries a Notes-block docstring with the spin-adapted expressions (ccwfn.py style).
- Also refactored the spatial `pertbar.Avvoo` factor placement (spin-orbital code
  unchanged): store the un-halved, permutationally symmetric `Avvoo`, with the
  compensating factors moved to the consumers (0.5 in `r_X2`, drop the 2.0 in
  `pseudoresponse`, 0.25 in `linresp_asym`). Net results unchanged.
- Validation (`test_057`/`test_058`): spatial-RHF dynamic polarizability (isotropic,
  omega=0.1) and optical-rotation tensor (chiral H2O2, length gauge) == Psi4 CCSD at
  cc-pVDZ; full 3x3 tensors == the spin-orbital path at STO-3G (spatial-vs-SO kept at
  STO-3G to keep the spin-orbital response cheap). These tests are the only coverage of
  the spatial symmetric path -- the 9a-i/9b tests run RHF in the spin-orbital basis and
  their open-shell references auto-resolve to spin orbitals.

**To resume:** read this doc + `git log`. The spin-orbital equation seed lives in
`~/src/socc` (machine-local, not part of this repo); see `socc/hamiltonian.py` for the
Fock/ERI build and `socc/ccwfn.py` for the residual kernels.
