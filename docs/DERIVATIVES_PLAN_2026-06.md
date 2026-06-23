# PyCC enhancement plan — Post-Hartree-Fock analytic derivatives

_Design of record for the post-HF derivative-property effort. Authored 2026-06-21,
after the spin-orbital enhancement (`docs/ENHANCEMENT_PLAN_2026-06.md`) completed.
New territory; not under that plan._

> **Update (2026-06-21): spin-orbital first.** The MP2 gradient is being built in the
> **spin-orbital** basis, where the orbital-response (Z-vector) Lagrangian applies verbatim
> from the spin-orbital CC gradient formulation (Gauss, Stanton & Bartlett, JCP 95, 2623
> (1991); local notes "CC Gradients with Orbital Response"). The spatial spin-adapted
> version (the original decision 4) is deferred to after the SO path works; the spatial
> `cphf`/density helpers (Phases A-B below) seed it. Phase C (relaxed density) is done in SO.

## Motivation

PyCC has a complete set of **RHF** analytic derivative properties — nuclear gradient
(`HFwfn.gradient`, `test_046`), molecular Hessian (`test_049`), atomic polar tensors
(dipole derivatives, `test_048`), and atomic axial tensors (`test_050`) — built on a
reusable MO-basis derivative-integral provider (`derivatives.py`) and an RHF
coupled-perturbed-HF orbital-response solver (`cphf.py`). The natural next direction is
to extend these **beyond Hartree-Fock**, to the correlated wavefunctions PyCC already
computes (MP2, then CC).

The first target is the **MP2 analytic energy gradient**. It is the smallest correlated
derivative, and it forces into existence the two ingredients every higher correlated
derivative needs: the **relaxed (orbital-response) density** and the **Z-vector** solve.
Once MP2 gradients work, the same scaffolding forks toward CC gradients (swap the
density/Lagrangian) or the MP2 property path (APT, Hessian).

## Decisions

| Question | Decision | Rationale |
|---|---|---|
| First target | **MP2 analytic gradient** | Smallest correlated derivative; builds the relaxed-density + Z-vector machinery the rest of the area reuses. |
| Orbital basis | **Spatial RHF (restricted MO) only** | Mirror the existing RHF derivative infrastructure; UHF/ROHF/spin-orbital deferred. |
| Relaxation | **Fully relaxed gradient** | The standard MP2 gradient (matches `psi4.gradient('mp2')`); unrelaxed density is a build-up byproduct, not the deliverable. |
| Frozen core | **All-electron first** | Keep the keystone clean; add frozen-core handling as a follow-on (Psi4's frozen-core MP2 gradient is the later oracle). |
| Generalization | **MP2-specific now** | No premature abstraction. Build clean MP2 code; lift a shared "Lagrangian -> Z-vector -> density -> gradient" layer only when CC gradients arrive. |
| Orbital response | **Reuse `cphf.py`** | Its orbital Hessian `G` and `solve(B)` are exactly the Z-vector solver; its docstring already anticipates promotion for MP2/CI/CC. |

## Theory (relaxed-density / Z-vector formulation)

The relaxed-density energy gradient (MO basis):

```
dE/dX = sum_pq D_pq h^X_pq  +  1/2 sum_pqrs Gamma_pqrs (pq|rs)^X
        -  sum_pq W_pq S^X_pq  +  dV_NN/dX
```

- **D** — relaxed one-particle density: the SCF (idempotent) part + the MP2 correlation
  part (`D_ij`, `D_ab`) + the orbital-relaxation contribution in the `ov` block.
- **Gamma** — two-particle density: the separable SCF part + the MP2 (`t2`-based)
  correlation part.
- **W** — energy-weighted (Lagrangian) density.
- `h^X`, `(pq|rs)^X`, `S^X`, `dV_NN/dX` — skeleton (fixed-MO-coefficient) derivative
  integrals from `derivatives.py`.

Because the MP2 energy depends on the orbitals, the orbital response to nuclear motion
does **not** collapse the way it does for HF (where it reduces to `-2 eps_i S^X_ii`).
The **Z-vector method** (Handy & Schaefer, 1984) avoids 3*N_atom CPHF solves by solving
the orbital response *once*:

```
1. Build the orbital-gradient Lagrangian  X_ai  from the MP2 correlation density.
2. Solve the Z-vector equation  G z = -X   (G = the CPHF orbital Hessian; cphf.solve).
3. Place z in the ov block of D (orbital relaxation); assemble W from D, Gamma, F.
```

The exact index expressions for `D^(2)`, `Gamma`, `X_ai`, and `W` are pinned during
implementation against Psi4 as the oracle (analytic gradient + finite difference).

References: Pople, Krishnan, Schlegel & Binkley (MP2 gradients, 1979); Handy & Schaefer
(Z-vector / interchange, JCP 81, 5031, 1984); Helgaker & Jorgensen (derivative theory).

## What PyCC already provides

| Need | In tree |
|---|---|
| Skeleton 1e/2e derivative integrals `h^X`, `(pq\|rs)^X`, `S^X`, `dV_NN` | `derivatives.py`: `core`, `eri` / `iter_eri` (per-atom, contract-and-discard), `overlap`, `nuclear_repulsion` |
| Orbital Hessian `G` + Z-vector solve | `cphf.py`: `_build_hessian` + `solve(B)` |
| Gradient-assembly template | `HFwfn.gradient` (`D h^X + Gamma eri^X + W S^X + V_NN`) |
| MP2 amplitudes / orbital energies | `MPwfn`: `t2`, `eps_occ`, `eps_vir` |
| 2-PDM contraction patterns (reference for the CC generalization) | `ccdensity`: `Doooo … Doovv` |

The two-electron derivative integrals are the heavy class (`3*N_atom*nmo**4`);
`derivatives.iter_eri` already yields them one atom at a time so each atom's block is
contracted and discarded — never all materialized at once.

## Phased implementation (branch-per-item)

| Phase | Branch | Content | Validation |
|---|---|---|---|
| A | `feature/mp2-gradient` (cphf promote) | Expose `cphf` orbital Hessian + `solve()` to `MPwfn` (per its docstring). No new physics. | HF derivative tests still pass; a direct `G z = B` round-trip. |
| B | `feature/mp2-density` | MP2 unrelaxed 1-PDM (`D_ij`, `D_ab`) and 2-PDM (`Gamma`). | Reconstruct E(MP2) from `sum D h + 1/2 sum Gamma (pq\|rs)`; optional unrelaxed MP2 dipole vs finite field. |
| C | `feature/mp2-zvector` | Orbital-gradient Lagrangian `X_ai`, Z-vector solve, relaxed `D` (ov block) and energy-weighted `W`. | Relaxed MP2 dipole vs Psi4 / finite difference (a cheaper relaxed-density check than the full gradient). |
| D | `feature/mp2-gradient-assembly` | `MPwfn.gradient()`: contract `D`/`Gamma`/`W` with the skeleton integrals over the full occ+vir MO space (per-atom `iter_eri` loop) + `dV_NN`. | **Keystone:** RHF MP2 gradient (H2O/6-31G) == `psi4.gradient('mp2')` to ~1e-8; finite-difference-of-energy cross-check. |

`MPwfn.gradient()` mirrors `HFwfn.gradient()`; the MP2 density / Lagrangian helpers live
on `MPwfn` (MP2-specific, per the decision above).

## Validation strategy

- **Primary oracle:** Psi4 analytic MP2 gradient (`psi4.gradient('mp2')`), all-electron,
  tight convergence — target ~1e-8.
- **Cross-check:** central finite difference of `psi4.energy('mp2')` (and of PyCC's own
  MP2 energy) w.r.t. nuclear displacement.
- **Keystone molecule:** a small closed shell (H2O/STO-3G or 6-31G) with molecular
  symmetry left on, exercising the symmetry-handled `self.C` path already used by the HF
  derivatives.
- Intermediate phases self-check (energy-from-density in B; relaxed dipole in C) so a
  regression is localized before the full gradient assembly in D.

## Status / progress

_Last updated 2026-06-23._

| Phase | Status | Landed |
|---|---|---|
| Design / this document | 🚧 in review | this branch |
| A/B — spatial CPHF access + MP2 densities | exploratory; **code removed** | this branch |
| C — **spin-orbital** relaxed density (GSB Lagrangian + Z-vector) | ✅ done | this branch |
| D — **spin-orbital** gradient assembly (keystone) | ✅ done | this branch |
| **spin-adapted** (closed-shell RHF) MP2 gradient | ✅ done | merged |
| **frozen-core** spin-adapted MP2 gradient | ✅ done | merged (#158) |
| **full-MO spin-orbital Hamiltonian** + **frozen-core SO MP2 gradient** + triples audit | ✅ done | `feature/spinorbital-fc-gradient` |

Phase D (SO): `MPwfn.gradient()` assembles the MP2 analytic nuclear gradient

    dE/dX = E_SCF gradient (HFwfn) + sum_pq D_pq f^X_pq + sum_pqrs Gamma_pqrs <pq||rs>^X
            + sum_pq I_pq S^X_pq

with the relaxed 1-PDM `D`, cumulant 2-PDM `Gamma = 1/4 t2` (oovv/vvoo), and the
energy-weighted density `I` (`I_ij = I'_ij + sum_ak z_ak(<ai||kj>+<aj||ki>)`,
`I_ab = I'_ab`, `I_ia = I_ai = I'_ia + z_ai eps_i`). New on `MPwfn`: `_so_mp2_cumulant`,
`_so_mp2_lagrangian` (full `I'` matrix), `_so_mp2_zvector`, `_so_energy_weighted_opdm`,
`_so_oei_deriv` / `_so_eri_deriv` (SO skeleton derivative integrals, spin-blocked from the
spatial `mints` MO derivatives in the semicanonical gauge), and `gradient()`.
`SpinOrbitalHamiltonian` now stores its semicanonical `Ca`/`Cb` + `spin`/`spat` so the
derivative integrals use the same MO gauge the densities were built in. `f^X = h^X +
sum_m <pm||qm>^X` is the skeleton Fock derivative. Validated (`test_061`):
`MPwfn.gradient()` == `psi4.gradient('mp2')` to ~1e-14 (H2O/6-31G C1, and H2O/cc-pVDZ C2v
-- polarization functions and A2-irrep MOs).

**Spin-orbital MP2 analytic gradient complete.**

**Spin-adapted (closed-shell RHF) MP2 gradient -- DONE (the original decision 4).**
`MPwfn.gradient()` now has the spatial path inline (dispatching to `_so_gradient` for the
spin-orbital case), with unlabeled spatial density helpers `_mp2_corr_opdm` / `_mp2_cumulant`
/ `_mp2_lagrangian` / `_mp2_relaxed_densities` alongside their `_so_*` siblings (PyCC
convention: spatial unlabeled, spin-orbital `_so_*`). The spin-adaptation carries the spin
sum in `l2 = 2(2 t2 - t2.swap)` and the cumulant `Gamma = 2 t2 - t2.swap`, writes the
two-electron 1-PDM/`W` terms with the spin-adapted `L` (= `H.L`), and assembles with the
full-spatial-MO derivative integrals (`f^X = h^X + sum_m L[p,m,q,m]^X`, no extra prefactor
on the spin-summed densities). The Z-vector reuses the basis-aware `self.cphf.solve`
(closed-shell singlet Hessian via `H.L`). Validated (`test_061`): `MPwfn.gradient()` ==
`psi4.gradient('mp2')` ~1e-14 (6-31G C1, cc-pVDZ C2v), and the keystone -- spin-adapted ==
spin-orbital on a closed shell -- to machine precision (~1e-16).

**Frozen-core spin-adapted MP2 gradient -- DONE.** The spatial path is now frozen-core
aware (the all-electron case is `nfzc=0`, no separate branch). The unrelaxed correlation
density stays on the active blocks, but the **orbital response spans the full occupied
space**, leveraging the full-MO Hamiltonian integrals (`H.ERI`/`H.F` are full `nmo`, not
active-only):

  * **core <-> active-occupied** rotations are non-redundant for frozen-core MP2 (they move
    the frozen/active partition, so the MP2 energy responds) but leave the HF energy
    invariant, so their orbital Hessian is just the orbital-energy difference -- a **direct
    divide** `P_co = (I'_ci - I'_ic)/(eps_c - eps_i)`, not a CPHF solve;
  * the **occupied-virtual Z-vector** runs over the full `ndocc x nv` space (incl.
    core-virtual), with `P_co` coupled into the RHS (`X_ai -= sum_jc[<aj||ic>+<ac||ij>]z_jc`),
    solved with the all-electron `HFwfn(ref).cphf` (whose occupied space is the full `ndocc`);
  * `W = I'(D_r)` -- the Lagrangian at the relaxed density -- supplies the core-active
    energy-weighted-density (`z_kc`) term automatically; `f^X` sums `m` over the full occ.

`_mp2_lagrangian` now takes a full-MO `D` (1-PDM term's column over the full occupied space),
and `_mp2_zvector`/`_energy_weighted_opdm` are consolidated into `_mp2_relaxed_densities`.
Validated (`test_061`, H2O/6-31G C1): relaxed frozen-core dipole vs finite field, and the
gradient vs a **5-point finite difference of PyCC's own frozen-core MP2 energy** to <1e-8
(the ground-truth oracle -- Psi4's *analytic* frozen-core MP2 gradient is itself inconsistent
with its *own energy's* finite difference at ~7e-6, so it is not used as the oracle).

**Frozen-core spin-orbital MP2 gradient + full-MO Hamiltonian consistency -- DONE.** The
spin-orbital Hamiltonian is now built over the **full MO space** (frozen core included),
mirroring the spatial path: `_init_spinorbital` orders the spin orbitals
`[a-core, b-core, a-occ, b-occ, a-vir, b-vir]` with `co`/`o`/`v` slices skipping the core
(`nfzc=0` is byte-identical to before). This gives the spin-orbital gradient its
core-virtual/core-active response integrals; the recipe (core-active divide, full-occ
Z-vector with the `z_jc` coupling, `W = I'(D_r)`) then applies **literally** in the
spin-orbital basis (no spin-adaptation), with the full-occupied orbital Hessian built
inline. Validated (`test_061`): keystone **SO == spatial** frozen-core gradient to ~1e-16,
plus the SO relaxed dipole.

The full-MO spin-orbital Hamiltonian exposed a latent **`occ-starts-at-0`** assumption in
the correlated triples kernels (they indexed the Hamiltonian/perturbation with loop
variables, e.g. `ERI[j,k,v,v]`, `pert[k,v]`, valid only when the active occupied began at
index 0). Audited and fixed across the (T), CC3 T-residual/Λ/response, and the `pertbar`
`[A,T3]` term -- each made relative to the `o`/`v` slices (behavior-preserving for
`nfzc=0`). Validated: frozen-core UCCSD(T) (`test_005`), frozen-core CC3 energy + Λ
keystones vs Psi4 (`test_031`), and frozen-core CC3 polarizability vs a finite field of the
CC3 energy (`test_059`).

Next: the MP2 property path (APT / Hessian) or CC gradients (swap the densities; the
Z-vector + assembly carry over). Frozen-core CC gradients reuse the same full-occ response
machinery, now available on both the spatial and spin-orbital paths.

The original spatial Phases A (`MPwfn.cphf` access to the CPHF Z-vector solver) and B (the
spatial MP2 `Doo`/`Dvv` and `oovv` 2-PDM in the `l2 = 2u` convention) were committed while
scoping the problem, then **removed** once the effort pivoted to spin-orbital -- the SO Phase C
builds its own `_so_orbital_hessian` and `_so_mp2_corr_opdm` (with `l2 = t2`) and never used
them. Their formulas/conventions remain in the commit history (`be02dea`, `b5219e3`) and will
be re-introduced, validated by use, when the spin-adapted gradient is built.

Phase C (SO): `MPwfn` gains `_so_mp2_corr_opdm` (`Doo = -1/2 t_imef t_jmef`,
`Dvv = 1/2 t_mnbe t_mnae`), `_so_orbital_hessian` (`A_{ai,bj} = (eps_a-eps_i)delta +
<ab||ij> + <aj||ib>`), `_so_mp2_orbital_lagrangian` (the GSB `I'_pq` with correlation-only
`D` and cumulant `Gamma_ijab = 1/4 t2` in `oovv`/`vvoo`; `X_ai = I'_ia - I'_ai`), and
`mp2_relaxed_opdm` (`A z = X`, `D_ai = D_ia = -z_ai`). Validated: relaxed correlation
`mu_z` vs finite-field Psi4 to ~1e-11 (H2O/6-31G C1, and H2O/cc-pVDZ C2v -- polarization
functions and A2-irrep MOs).

## After MP2 gradients — the fork

Phases A-C are shared sunk cost. After the Phase-D keystone, decide between:

- **CC gradients** — swap the MP2 density/Lagrangian for the CCSD relaxed density
  (Lambda already exists; `ccdensity` has the 2-PDM); the Z-vector solve and gradient
  assembly carry over. This is where the "lift a shared layer" refactor (deferred above)
  would happen.
- **MP2 property path** — MP2 **APT** (`derivatives.dipole` + the same Z-vector) and MP2
  **Hessian** (second-derivative integrals already in `derivatives.py`: `core2`, `eri2`,
  `overlap2`, plus the nuclear CPHF response in `cphf`).

Beyond spatial RHF: UHF/ROHF and spin-orbital MP2/CC gradients, and frozen-core handling,
are later increments on the same machinery.

## Spin-orbital HFwfn properties (branch `feature/spinorbital-hf-gradient`)

To reach open-shell (UHF/ROHF) derivative properties -- and to supply the SCF gradient
the open-shell correlated gradients need -- the `HFwfn` derivative methods get a
spin-orbital route (dispatch on `orbital_basis`, reusing the now basis-aware
`Derivatives`/`CPHF`).

- **HF gradient -- DONE.** `HFwfn._gradient_spinorbital`: the CPHF-free spin-orbital HF
  gradient `dE/dX = sum_i h^x_ii + 1/2 sum_ij <ij||ij>^x - sum_i eps_i S^x_ii + dV_NN/dX`
  (i, j occupied spin orbitals), using `self.derivatives.so_*`. `gradient()` dispatches.
  Validated (`test_062`): keystone closed-shell RHF-forced-to-SO == spatial RHF gradient
  (6-31G C1 and cc-pVDZ C2v) ~1e-15, and **UHF / ROHF gradients vs Psi4 ~1e-15** -- the
  open-shell HF gradient works for all three references.
- **Polarizability -- DONE (RHF/UHF).** The simplest second-derivative property: a pure
  electric-field CPHF response, no derivative integrals. `CPHF.polarizability` is now
  basis-aware -- the only change from the spatial path is the prefactor (closed-shell
  double occupancy `k=4` vs spin orbitals `k=2`); the dipole RHS (`H.mu[o,v]`) and the
  orbital Hessian were already basis-aware. Validated (`test_063`): keystone RHF-forced-
  to-SO == spatial == Psi4 (~1e-13), and UHF vs Psi4 (~1e-6, set by Psi4's iterative
  UHF-CPHF; the direct SO solve matches finite field to ~2e-9).
- **ROHF CPHF response -- DEFERRED (guarded).** The semicanonical spin-orbital response is
  UHF-like and does *not* reproduce the restricted ROHF response (off ~5e-3 vs finite
  field); matching it needs the reference's ROHF Brillouin / orbital-rotation conventions
  (docc-socc, socc-virt couplings), which are not uniquely defined. `CPHF.solve` raises
  `NotImplementedError` for ROHF (detected via `same_a_b_orbs and not same_a_b_dens`). The
  CPHF-free ROHF HF gradient is unaffected -- it does not solve the response.
- **APT (nuclear dipole derivatives) -- DONE (RHF/UHF).** A mixed field-nuclear second
  derivative: it needed the spin-orbital *nuclear* CPHF RHS builder -- the deferred layer,
  now built as `CPHF._build_nuclear_spinorbital` (the antisymmetrized `<pq||rs>` in place
  of `L`, the spin-orbital skeleton derivative integrals from `Derivatives.so_*`, same
  `B = -Q` structure) -- plus the spin-orbital dipole derivatives (`Derivatives.so_dipole`).
  The assembly (`HFwfn._dipole_derivatives_spinorbital`) halves the closed-shell prefactors
  (singly occupied spin orbitals). Validated (`test_064`): keystone RHF-forced-to-SO ==
  spatial APT (6-31G C1 and cc-pVDZ C2v) ~1e-15, and UHF vs finite-difference of the SCF
  dipole ~6e-10. ROHF raises (the nuclear response goes through `CPHF.solve`).
- **Molecular Hessian -- DONE (RHF/UHF).** `HFwfn._hessian_spinorbital`: second-derivative
  skeleton terms (`Derivatives.so_core2`/`so_eri2`/`so_overlap2`, occupied block) plus the
  spin-orbital nuclear CPHF response/cache, with the closed-shell prefactors halved. Subtle
  point: Psi4's `mo_tei_deriv2(A,B)` two-electron second derivative does not satisfy the
  integral's electron-exchange symmetry `(pq|rs) = (rs|pq)` term by term. For the energy
  trace the same-spin terms absorb the resulting bra<->ket relabel (so RHF and the aa/bb
  blocks are already symmetric), but the UHF **cross-spin Coulomb** term
  `sum_{i in a, j in b}(ii|jj)^{ab}` does not -- it carried the entire spurious antisymmetric
  part. Fixed at the integral level: `Derivatives.so_eri2` symmetrizes the chemist deriv2
  over the bra<->ket swap (building the cross-spin `ab`/`ba` blocks from independent
  `(aa|bb)`/`(bb|aa)` calls), which -- since the geometric derivative of a symmetric integral
  is symmetric -- also restores the atom-pair-swap symmetry the Hessian needs, so the
  assembled Hessian is symmetric with no global `0.5*(H+H.T)`. Validated (`test_065`): keystone
  RHF-forced-to-SO == spatial (6-31G C1, cc-pVDZ C2v) ~1e-15; UHF naturally symmetric (~1e-14)
  and vs `psi4.hessian('scf')` ~3e-12. ROHF raises.
- **AAT -- DONE (RHF; UHF unvalidated).** The spin-orbital magnetic-dipole integral `H.m`
  was already built on `SpinOrbitalHamiltonian`, so `CPHF._m_ov`/`rhs_magnetic`/
  `solve_magnetic` already work for spin orbitals; the only new pieces are
  `Derivatives.so_overlap_half` (spin-blocked nuclear half-derivative overlaps) and the SO
  branch `HFwfn._atomic_axial_tensors_spinorbital` (`I^lam_{a,b} = sum_ia [U^R_ai U^B_ai +
  U^B_ai <phi^R_i|phi_a>]`, prefactor 1 vs the closed-shell 2). Validated (`test_066`):
  keystone SO-RHF == spatial AAT (the DALTON-validated `test_050` path) ~1e-15.
  **Caveat:** there is no prior open-shell UHF AAT implementation anywhere to compare
  against, so the UHF result is only checked to run and return a sane (finite, real,
  nonzero) tensor through the same code path the keystone validates. ROHF raises.

**All five spin-orbital HF analytic properties (gradient, polarizability, APT, Hessian,
AAT) are implemented for RHF/UHF** (ROHF response deferred, guarded). Next: spin-adapt the
correlated (MP2) gradient, frozen core, then CC gradients / the MP2 property path.

Once the SO HF gradient is in, `MPwfn.gradient()` can use an SO `HFwfn` for the SCF term,
lifting the closed-shell-only restriction on the spin-orbital MP2 gradient.
