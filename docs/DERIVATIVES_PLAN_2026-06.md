# PyCC enhancement plan — Post-Hartree-Fock analytic derivatives

_Design of record for the post-HF derivative-property effort. Authored 2026-06-21,
after the spin-orbital enhancement (`docs/ENHANCEMENT_PLAN_2026-06.md`) completed.
New territory; not under that plan._

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

_Last updated 2026-06-21._

| Phase | Status | Landed |
|---|---|---|
| Design / this document | 🚧 in review | this branch |
| A — CPHF promote / Z-vector access | ⬜ todo | — |
| B — MP2 densities (1-PDM + 2-PDM) | ⬜ todo | — |
| C — Lagrangian + Z-vector -> relaxed D, W | ⬜ todo | — |
| D — gradient assembly (keystone) | ⬜ todo | — |

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
