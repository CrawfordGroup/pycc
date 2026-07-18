# PyCC post-Hartree–Fock analytic derivatives — design & status

_Design-of-record and status for the post-HF analytic-derivative-property effort: the MP2
correlation gradient and electric/nuclear second derivatives (polarizability, APT, Hessian) via the
2n+1 route; and the coupled-cluster analytic gradients, relaxed dipoles, and polarizabilities (CCSD,
CCSD(T)) built on the same Z-vector / relaxed-density machinery — all now unified behind the shared
`CorrelatedDerivs` base (§9). (An independent explicit-derivative route cross-checked the 2n+1 suite
during development and was then removed; see §9/Appendix B.) Started 2026-06; this revision 2026-07
(rewritten from the original chronological plan — milestone history is in the Changelog appendix).
Preceding spin-orbital infrastructure: `archive/ENHANCEMENT_PLAN_2026-06.md`. Filename retained for
the docstring/test references that point here._

## 1. Status at a glance

MP2 **correlation** analytic derivatives. Every total property splits as **nuclear + reference (HF)
+ correlation**: the correlation part is computed by the method's `CorrelatedDerivs` driver
(`MPderiv` for MP2; §9), the reference part on `HFwfn` (pure electronic), and the nuclear part is a
closed-form geometry/charge term. The `pycc` property facade
(`pycc.dipole`/`gradient`/`polarizability`/`hessian`/`apt`/`aat`) assembles the three into a
`PropertyComponents` (`.total`/`.nuclear`/`.reference`/`.correlation`), for any wavefunction type
(see the property-facade work). All rows below are implemented for **both spin paths** (spin-orbital
and spin-adapted closed-shell RHF), **all-electron and frozen-core**, unless noted.

| Property (derivative) | order | route / method |
|---|---|---|
| dipole  `dE/dF` | 1st | ✅ relaxed dipole (`relaxed_dipole`) |
| nuclear gradient  `dE/dX` | 1st | ✅ `gradient()` (relaxed-density) |
| polarizability  `d²E/dF²` | 2nd | ✅ `polarizability()` |
| APT  `d²E/dF dX` | 2nd | ✅ `dipole_derivatives(route='2n+1-field'|'2n+1-nuclear')` |
| Hessian  `d²E/dX²` | 2nd | ✅ `hessian()` |
| AAT / VCD  `d²E/dB dX` | 2nd | ✅ `atomic_axial_tensors()` (density/overlap form) |
| velocity-gauge APT  `d²E/dp dX` | 2nd | ✅ `velocity_dipole_derivatives()` |

Everything is the **2n+1 route** — the independent explicit-derivative route (originally a
cross-check) was removed in the refactor (§9, PR #194) once the 2n+1 suite was validated; the sole
remaining `route` arguments select the field-vs-nuclear APT variant. Validation is now the SO ==
spatial keystone plus tight finite difference (§4). The AAT and velocity-gauge APT use the
density/wave-function-overlap formulation (unrelaxed densities, no Z-vector); they are orbital-gauge
invariant, with a numerically stable non-canonical default (`gauge=`).

**Reference layer (`HFwfn`).** Gradient, polarizability, APT, Hessian, and AAT are done for
**RHF and UHF**, in both spatial and spin-orbital bases (ROHF orbital response deferred, guarded —
see Roadmap). These supply the reference contribution to every MP2 total property and anchor the
SO machinery through the RHF-forced-to-SO == spatial keystone.

**Coupled-cluster gradients and dipoles.** Analytic nuclear gradients and relaxed electronic dipoles
through the `CCderiv` driver, reusing the MP2 Z-vector / relaxed-density assembly (`_lagrangian`, the
SCF orbital Hessian) with the CCSD/(T) densities and Λ:

| Method | spatial (closed-shell RHF) | spin-orbital (UHF) | frozen core |
|---|---|---|---|
| CCSD `dE/dX` (gradient) | ✅ | ✅ | ✅ |
| CCSD(T) `dE/dX` (gradient) | ✅ | ✅ | ✅ |
| CCSD, CCSD(T) `dE/dF` (relaxed dipole) | ✅ | ✅ | ✅ |
| CCSD, CCSD(T) `d²E/dF²` (dipole polarizability) | ✅ | ✅ | ✅ |

The relaxed dipole reuses the gradient's relaxed density: a static field leaves the AO basis fixed
(`S^F = ⟨pq\|rs⟩^F = 0`), so `mu = Tr(D_rel · mu_ints)` — the same `D_rel` (correlation density + `Pco`
+ (T) κ̄ + ov Z-vector), built by the shared `CCderiv._relaxed_density`/`_so_relaxed_density`, that the
gradient contracts with the skeleton integrals. `pycc.dipole(CCwfn)` returns the usual
nuclear/reference/correlation `PropertyComponents`. Validated against a tight finite difference of
pycc's own correlation energy — 5-point O(h⁴), the gradient ~1e-12 and the relaxed dipole a finite
field of `(E_CC − E_SCF)` ~1e-12 — **not** psi4's analytic derivatives (§4). Unlike CCSD — which reuses
the −½Sˣ ov-only Z-vector because
CCSD is invariant to occ–occ/virt–virt rotations — **CCSD(T) uses canonical perturbed orbitals for
the oo/vv blocks** (dependent-pair κ̄, even all-electron) — not from necessity but for cost: this is
Lee–Rendell's route (diagonal (T) density only), ~2/3 the work of Scuseria's equally-correct
non-canonical route (off-diagonal density, one extra N^7 set); see §5 and §7. (During development the
efficient Z-vector route was cross-checked against an independent explicit-derivative route to machine
precision for CCSD; that route has since been removed — §9/#194.)

**Coupled-cluster polarizability.** The static dipole polarizability `α = −d²E/dF²`
(`CCderiv.polarizability`, `pycc.polarizability`) — the first CC *second*-derivative property — via
the asymmetric (2n+1) route: differentiate the relaxed-density gradient a second time in a field
(`α_ab = Tr(dD_rel·μ_a) + Tr(D_rel·rot(U^b,μ_a))`). Done for **CCSD *and* CCSD(T), spatial RHF and
spin-orbital UHF, all-electron and frozen core.** It adds the perturbed amplitudes `dt/dF`, perturbed
Λ `dΛ/dF` (staying in `cclambda`'s `r_L`, no `Y1/Y2`), the perturbed HBAR (analytic product rule), and
the perturbed Z-vector — all first-order responses, no second-order CPHF. CCSD is FD-validated (spatial
`1.8e-12`; SO==spatial keystone `~1e-13`; open-shell UHF vs an energy finite field `~3e-10`). **CCSD(T)**
additionally threads the perturbed (T) intermediates `dt3` (analytic `(dN − t3·dD)/D`), the perturbed
(T) Λ sources `dS1/dS2`, and the perturbed oo/vv dependent-pairs `dP` into `dD_rel`; validated against
the energy second derivative and, definitively, against **CFOUR** (`POLAR − POLARSCF`, matched GENBAS
6-31G) to `~5e-11` on both routes, AE and FC. One subtlety cost real debugging (§8): the perturbed (T) Λ
L2 source must carry the *same* `P_ij^ab` symmetrization that the unperturbed `r_L2` applies to `S2`, so
the (T) source is now passed *into* the shared `r_L1`/`r_L2` residual (via their `s1`/`s2` argument)
rather than corrected afterward. See §8.

## 2. Formulation

Two independent routes reach the same numbers; PyCC implements both.

**Explicit-derivative route** (`derivints.pdf`). Fold the CPHF coefficients `U^x` into the *full*
derivatives of the Fock matrix and the antisymmetrized two-electron integrals, then contract with
the **unrelaxed** densities:

    d_x E_corr = Σ_pq γ_pq d_x f_pq + Σ_pqrs Γ_pqrs d_x<pq||rs>

a single unrestricted sum with all prefactors absorbed into `γ`, `Γ` (at zeroth order this
reproduces `E_corr` with both coefficients 1). The functional is stationary in the amplitudes, so
the density-response terms drop and only the *integral* derivatives remain. Second derivatives
(Eq. 15) differentiate again into first + second perturbed integrals; the second ones (Eqs. 17/18/20)
carry the **second-order CPHF `U^{xy}`** and the orthonormality term `ξ^{xy}` (Eq. 18). Conceptually
simplest; the Hessian solves `U^{xy}` for all `3N(3N+1)/2` nuclear pairs — an `O(N²)` count of
second-order solves.

**Relaxed-density / 2n+1 route** (Gauss–Stanton–Bartlett, JCP 95, 2623 (1991)). The nuclear
gradient

    d_x E = Σ_pq γ^rel_pq f^(x)_pq + Σ_pqrs Γ_pqrs <pq||rs>^(x) + Σ_pq I_pq S^(x)_pq       (G)

uses **skeleton** (fixed-MO-coefficient) integral derivatives `^(x)`, the **relaxed** 1-PDM
`γ^rel` (unrelaxed correlation density + the orbital-relaxation Z-vector), the cumulant 2-PDM `Γ`,
and the energy-weighted density `I`. The Z-vector (Handy–Schaefer, JCP 81, 5031 (1984)) folds the
orbital response in once, so (G) needs no per-perturbation CPHF. **Second derivatives differentiate
(G):** by the 2n+1 rule only **first-order** responses appear — the perturbed relaxed density
`d_x γ^rel`, perturbed energy-weighted density `d_x I`, and a perturbed Z-vector `z^x` (same orbital
Hessian as the gradient, perturbed RHS) — never `U^{xy}`. Hence `O(N)` solves. Each second
derivative differentiates the gradient from one side: polarizability (field/field), APT
(`'2n+1-nuclear'` differentiates the relaxed dipole w.r.t. nuclei; `'2n+1-field'` differentiates
the nuclear gradient w.r.t. the field), Hessian (nuclear/nuclear). Derivation of the perturbed
Lagrangian / Z-vector / relaxed density: `docs/mp2_2n1_perturbed.tex`.

## 3. Architecture

Two one-directional stacks meet at `CPHF`: **`Derivatives` ← `CPHF` ← `HFwfn`** (the reference layer)
and **`CPHF` ← `CorrelatedDerivs` ← {`MPderiv`, `CCderiv`, `CIderiv`}** (the correlation drivers; §9).

- **`Derivatives`** — skeleton MO derivative integrals only (no `U`): `core`, `overlap`, `eri`,
  `dipole` and second derivatives `core2`/`overlap2`/`eri2` (each with `so_*` spin-orbital twins).
- **`CPHF`** — orbital response. Owns `U`; consumes the skeleton integrals. Solves `G U = B` with the
  singlet orbital Hessian `_mo_hessian(kind)`. `Perturbation('field'|'nuclear'|'magnetic', comp)`
  descriptors key the caches so multi-property runs never recompute the expensive nuclear ERI
  derivatives. `perturbed_fock`/`perturbed_eri` (the response-dressed `d_x f`, `d_x<>`, with the
  `ncore`/`canonical` gauge options); `full_U` (full `nmo×nmo` rotation with the core↔active canonical
  block); `nuclear_hessian_skeletons` (raw second skeletons, cached per atom pair); the magnetic /
  momentum engines (`magnetic_ints`/`momentum_ints`) for the AAT / VG-APT. (The second-order-CPHF
  layer that the explicit route used — `perturbed_fock2`/`_full_U2`/`ξ`/… — was removed with that
  route in §9/#194.)
- **`CorrelatedDerivs`** (base) — the method-agnostic orbital response + assembly + the public
  correlation-property API (Lagrangian, dependent-pair rotations, `_orbital_response`/relaxed density,
  perturbed relaxed density, `relaxed_dipole`/`gradient`/`polarizability`/`dipole_derivatives`/
  `hessian`). Holds a persistent full-occupied `CPHF` (`_full_occ_cphf`) so the response caches
  survive across property calls and frozen core runs over the full occupied space. See §9.
- **`MPderiv`** (leaf) — MP2 densities (`MPwfn._(so_)mp2_corr_opdm`/`_tpdm`) and their closed-form
  first-order responses (`_perturbed_t2`, `_perturbed_unrelaxed_densities`); plus the method-specific
  AAT (`atomic_axial_tensors`) and velocity-gauge APT (`velocity_dipole_derivatives`), which are not
  in the base. `MPwfn` exposes thin shims that delegate to the driver.
- **`CCderiv`** (leaf) — the CCSD / CCSD(T) densities + Λ from `ccdensity` (and for (T) the
  densities/Λ from `CCwfn.t3_density`), with the iterative perturbed-T/Λ response as its
  `_perturbed_unrelaxed_densities` hook. Everything else (gradient, relaxed dipole, polarizability)
  comes from the base; it overrides only `polarizability` to add the model/(T)-intermediate guards.
- **`CIderiv`** (leaf) — a phase-4 stub (§9): inherits the base, density hooks raise
  `NotImplementedError`; CISD properties currently still run on the transitional `CIwfn` code.

Conventions: spatial methods unlabeled, spin-orbital prefixed `_so_`.

## 4. Validation methodology

- **Oracle = PyCC's own finite difference** (energy / dipole / gradient), *not* Psi4's analytic
  derivatives. Psi4's frozen-core MP2 gradient is inconsistent with its own energy (~7e-6), so it
  is unusable as a reference; PyCC's own-energy FD is the ground truth.
- **Dipole-FD beats energy-FD.** For a second derivative, differencing the analytic *dipole*
  (`α = dμ/dF`, a `1/h` stencil) is ~3 orders tighter (~1e-12) than a second difference of the
  *energy* (`1/h²`, ~1e-9). Use 7-point O(h⁶) stencils.
- **SO == spatial keystone** on a closed shell (~1e-15) — the primary internal consistency check
  (spin-orbital vs spin-adapted must agree exactly).
- **Explicit == 2n+1** (historical) — every second derivative was originally computed both ways and
  agreed to ~machine precision; that cross-check retired the explicit route (§9/#194), leaving 2n+1
  as the sole route with the keystone + FD as the live checks.
- **Sum rules** (FD-free physics checks): acoustic/translational `Σ_A P = 0` (APT), `Σ_B H = 0`
  (Hessian).
- **Gauge-invariant response scalars.** A nuclear/field displacement rotates PyCC's `−½S^x`
  semicanonical gauge relative to the canonical MOs, so raw perturbed amplitudes/densities aren't
  directly FD-comparable; compare `Tr(γ²)`, ||Γ||² instead.
- Geometry Cartesian in **bohr** with `no_com`/`no_reorient`, so a nuclear displacement keeps the
  frame fixed and matches the analytic (bohr) integral derivatives.

## 5. Key subtleties & lessons

- **Frozen-core core↔active relaxed density is a Sylvester equation, not a divide.** The block
  `D^rel_ci = (I'_ci − I'_ic)/(ε_c − ε_i)` is the *canonical* form of `f_cc D − D f_oo = I'_ci −
  I'_ic`. The divide is exact for the unperturbed gradient, but its **derivative** (2n+1
  polarizability etc.) needs the off-diagonal coupling `−∂_x f[co,co] @ D + D @ ∂_x f[o,o]` — a
  field leaves the active-occupied space non-canonical, so the diagonal `∂_x ε` alone is wrong by
  ~7e-7. (Baeck–Watts–Bartlett, JCP 107, 3853 (1997); `mp2_2n1_perturbed.tex` Eq. 12.)
- **The ov `ξ`-seed (explicit frozen-core polarizability).** The second-order ov CPHF reuses the
  first-order Hessian `G`, which maps only the *antisymmetric* ov rotation; but the core↔active oo
  block makes `ξ_ov ≠ 0`, so `U^{ab}_ov` isn't antisymmetric. Seed `U^{ab}_ov = −ξ_ov` before the
  RHS (all-electron `ξ_ov = 0`, a no-op). It hid because first-order Brillouin held for frozen core
  but the validated *dipole* never exercised the `[v,o]` block.
- **Full-Fock `termA` in the perturbed Lagrangian.** The GSB Lagrangian's one-electron term is
  `Σ_q f_uq(D_vq + D_qv)`; its derivative is the full matrix product `∂_x f @ (D + Dᵀ)`, not just
  the diagonal `∂_x ε`. Neutral for MP2's unrelaxed `D` (no ov block) but required for the relaxed
  `D`'s ov/core-active blocks (the 2n+1 APT's `d_F W`). **CC extension (2026-07-09, §8.2):** the CC
  *unrelaxed* `D` **does** have ov/vo blocks (Λ≠T†), so the full-Fock `termA` is required even for the
  unrelaxed-density Lagrangian — i.e. for the perturbed *Z-vector RHS* `dX = ∂(I'_ov − I'_voᵀ)`, not
  only for `d_F W`. Building `dX` by a finite-stencil of the diagonal-ε `_mp2_lagrangian` silently
  drops `df_offdiag @ (D+Dᵀ)` and puts a ~9 % error into the CC polarizability. **Always build the
  perturbed Lagrangian via `MPwfn._perturbed_lagrangian` (full `df`), never a stencil.** Insidious:
  fixed-basis/frozen-MO checks route through the same diagonal-ε Lagrangian and are blind to it; only
  the field-relaxed `perturb_h` oracle (canonical field basis ⇒ `F` diagonal) exposes it.
- **The `rot4` transpose (2n+1 Hessian efficiency).** Hoisting the `U^Y` skeleton rotations off the
  `O(N²)` pair loop onto the densities uses `Σ A·rot(U,B) = Σ rot(Uᵀ,A)·B`. The four-index case
  needs the **transpose** (`rot4(Uᵀ, Γ)`), since `rot4` contracts B's index via U's *first* index;
  `rot4(U, Γ)` was wrong by ~0.35.
- **`mo_tei_deriv2` bra↔ket asymmetry (HF Hessian).** Psi4's two-electron second derivative doesn't
  satisfy `(pq|rs) = (rs|pq)` term-by-term. Same-spin traces absorb it, but the UHF cross-spin
  Coulomb term does not. Fixed at the integral level (`Derivatives.so_eri2` symmetrizes over the
  bra↔ket swap), which also restores atom-pair-swap symmetry → the Hessian is symmetric with no
  global `0.5(H + Hᵀ)`.
- **`occ-starts-at-0` triples audit.** Building the SO Hamiltonian over the *full* MO space (frozen
  core included in the MO list) exposed a latent assumption in the (T)/CC3 triples kernels that the
  active occupied began at index 0 (e.g. `ERI[j,k,v,v]`, `pert[k,v]`). Made relative to the `o`/`v`
  slices (behavior-preserving for `nfzc=0`).
- **(T) uses canonical perturbed orbitals — a cost choice, not a requirement.** The (T) energy is
  invariant to oo/vv rotations through the *full* second-order T₃, so the perturbed-MO gauge is free
  (as for CCSD). pycc follows **Lee–Rendell**: hold the perturbed orbitals canonical, so the oo/vv
  blocks carry dependent-pair rotations `κ̄_ij=(I'_ij−I'_ji)/(ε_i−ε_j)`, `κ̄_ab=(I'_ab−I'_ba)/(ε_a−ε_b)` —
  exactly the frozen-core core↔active `Pco` divide (§ above) generalized to *all* oo/vv pairs (added to
  the relaxed density, coupled into the ov Z-vector RHS via the antisymmetrized ERI) — and the (T)
  density is then needed only on the diagonal. **Scuseria's** non-canonical (−½Sˣ) route is equally
  correct but instead carries the *off-diagonal* (T) density against the off-diagonal perturbed Fock, at
  one extra N^7 set. An early reading — "all-electron ⇒ ΔX=0 ⇒ −½Sˣ suffices with the standard t₃" — was
  wrong: it conflated Lee–Rendell's *degeneracy* threshold with Scuseria's separate formulation. The
  correct picture is a single orbital term `κ̄_pq F^(1)_pq` with κ̄ over *all* pairs (ov = CPHF/Z-vector
  solve; oo/vv = the divides). FD-validated to 1.8e-12 (all-electron) / 1.9e-12 (frozen core). Full
  derivation: `docs/ccsdt_orbital_response.tex`; §7.
- **The off-diagonal (T) `Doo`/`Dvv` is not needed in the canonical gauge.** The `⟨0|L₃[E_ij,T₃]|0⟩`
  oo/vv off-diagonals `t3_density` once built are real density-matrix elements, but they belong to
  Scuseria's *non-canonical* route (contracted there with the off-diagonal perturbed Fock), not to
  Lee–Rendell / Hald et al. In pycc's canonical gauge they are not needed; leaving them in `D_rel`
  corrupts `Tr(D·μ)` while staying invisible to the energy reconstruction (canonical F ⇒ only `diag(D)`
  enters `eone`). So pycc's (T) 1-PDM carries only `{Dov, diag(Doo), diag(Dvv)}`; the oo/vv orbital
  response is the κ̄ above. `diag(Doo)` and `diag(Dvv)` are built together in the ijk loop — the old
  separate abc loop was Lee–Rendell's avoidable extra N^7 set (branch `perf/t3-density-diagonal-doo`).

## 6. Roadmap

- **CC gradients + polarizability — done; shared-layer refactor done.** CCSD *and* CCSD(T) analytic
  gradients, relaxed dipoles, and polarizabilities are implemented for **spatial RHF and spin-orbital
  UHF, all-electron + frozen core**, now all on the unified `CorrelatedDerivs` base (§9, complete —
  the explicit-derivative route and its `_gradient_explicit` cross-check were removed once 2n+1 was
  validated). **Next:** CCSD(T) Hessian and APT built on the unified base (the base assemblies
  already support them, pending the CC perturbed 2-PDM/EW-density wiring + validation); the CISD
  `CIderiv` hooks; and the smaller deferred items in §9 (magnetic-Hessian unification, AAT/VG-APT
  hoist, `make_t3_density`/solver-knobs UX).
- **ROHF orbital response — deferred, guarded.** The semicanonical spin-orbital response is UHF-like
  and does not reproduce the restricted ROHF response; `CPHF.solve` raises for ROHF. The CPHF-free
  ROHF HF gradient is unaffected.
- Out of scope: 2n+2 / higher-order (cubic-response) economies.

## 7. CCSD(T) gradient — spatial RHF + spin-orbital UHF (design & status)

**Status.** Done and FD-validated for **closed-shell RHF (spatial MOs) and UHF (spin-orbital),
all-electron *and* frozen core** (`pycc.gradient(CCwfn(wfn, model='CCSD(T)'))` through
`CCderiv`/`ccdensity`). The CCSD(T) Hessian/APT are deferred (below). The theory below is written for
the spatial path; the spin-orbital path is the same construction with `H.L → <pq||rs>` (see
**Spin-orbital** at the end of this section).

**References.**
- **Paper A** — T. J. Lee & A. P. Rendell, *J. Chem. Phys.* **94**, 6229 (1991): closed-shell
  *spatial* CCSD(T) gradient in the Handy–Schaefer Z-vector / effective-density (Gauss–Stanton–
  Bartlett) formulation — pycc's formulation.
- **Paper B** — Hald, Halkier, Jørgensen, Coriani, Hättig & Helgaker, *J. Chem. Phys.* **118**,
  2985 (2003): variational Lagrangian, canonical orbitals — the frozen-core / canonical-orbital guide.

**Why pycc keeps the perturbed MOs canonical.** The triples solve `⟨μ₃|[F,T₃]+[H,T₂]|HF⟩=0` (B-15) is
non-iterative *only because F is diagonal*, so `[F,T₃]` collapses to `D^abc = f_ii+f_jj+f_kk−f_aa−f_bb−f_cc`
(A-5). Non-canonical F ⇒ triples couple ⇒ iterative. So to keep the cheap non-iterative `t₃ = W/D` build,
pycc holds the perturbed orbitals canonical in the oo/vv blocks and carries an explicit **dependent-pair**
rotation there instead of the −½Sˣ gauge. This is a cost choice (Lee–Rendell), not a necessity: Scuseria
keeps the −½Sˣ gauge and instead pays the off-diagonal (T) density plus one extra N^7 set; because the
(T) energy is oo/vv-invariant through the full T₃, both routes yield the same gradient.

**The orbital response (the crux).** CCSD is invariant to occ–occ / virt–virt rotations, so its
gradient uses the −½Sˣ, ov-only Z-vector. **(T) breaks that invariance**, so the canonical perturbed
orbitals acquire dependent-pair rotations
`κ̄_ij = (I'_ij − I'_ji)/(ε_i − ε_j)`, `κ̄_ab = (I'_ab − I'_ba)/(ε_a − ε_b)`
(the Lagrangian asymmetry, Lee–Rendell A-34). This is **exactly the frozen-core core↔active divide
`Pco = (I'[co,o] − I'[o,co]ᵀ)/(ε_c − ε_i)`** already in pycc — **generalized from core↔active to all
oo (i,j) and vv (a,b) pairs** (numerator-gated `|ΔX|<1e-8` for degeneracies), added to the relaxed
density and coupled into the ov Z-vector RHS through the antisymmetrized ERI. Equivalently (Paper B /
`ccsdt_orbital_response.tex`): the only surviving orbital term is `κ̄_pq F^(1)_pq`, with κ̄ over **all**
pairs (ov = the CPHF/Z-vector solve; oo/vv = these divides). `I'` is the (T)-inclusive Lagrangian, so
(T) enters κ̄ only through `I'`.

**The (T) one-particle density** carries only `{Dov, diag(Doo), diag(Dvv)}` in the canonical gauge
(Paper A Eqs 17–19, Paper B Eq 65). The off-diagonal `Doo`/`Dvv` that `t3_density` originally built
(`⟨0|L₃[E_ij,T₃]|0⟩`) are real density elements but belong to **Scuseria's** non-canonical route, not
Lee–Rendell / Paper B; in pycc's canonical gauge they are not used, and leaving them in `D_rel` corrupts
`Tr(D·μ)` while staying invisible to the energy (canonical F ⇒ only `diag(D)` enters). The oo/vv orbital
response is the κ̄ above. Both diagonals are built together in the ijk loop (branch
`perf/t3-density-diagonal-doo`, `t3_density` + `so_t3_density`); the old separate abc loop was the
extra N^7 set Lee–Rendell avoid (~3.3x spatial / ~2.6x SO speedup on the density build).

**Frozen core — no new machinery.** The occupied dependent pairs split into core↔active (carried by
the existing `Pco`, whose `I'` is (T)-inclusive) and active↔active (the generalized oo κ̄), plus the vv
κ̄; the ov-occupied index of the (T) coupling runs over the full occupied space (`ofull`), reducing to
the active space when `nfzc=0`. `Pco` from the (T)-inclusive `I'` **fully captures the (T) core↔active
response** — FD-confirmed to 1.9e-12, no extra core term needed.

**Implementation.**
- `cctriples.t3_density` (a free function returning `(ET, {intermediates})`; `CCwfn.t3_density` is a
  thin delegate-and-cache wrapper, called from the energy code so T3 is built once) yields the (T)
  contributions: the **diagonal** 1-PDM `diag(Doo)`/`diag(Dvv)` (computed directly — `acd,acd->a` /
  `ikl,ikl->i` — not full blocks then filtered) plus `Dov`; the 2-PDM `Goovv/Gooov/Gvvvo`; and the Λ
  residuals `S1/S2` (Paper A's η/γ), added into Λ₁/Λ₂. Housing the builder in `cctriples` (not
  `CCwfn`) keeps the wavefunction from computing density components and avoids a `ccwfn → ccdensity`
  dependency, while preserving the single T3 build.
- `CCderiv.gradient` — `_dependent_pairs(I'[block], ε)` builds the κ̄ divides (numerator-gated); the
  `model=='CCSD(T)'` branch adds κ̄_oo/κ̄_vv to `Drel` and couples them into the Z-vector RHS (ov index
  over `ofull`). Model-gated, so the CCSD path is untouched. Then the standard GSB assembly runs,
  `E^λ = D h^λ + Γ (pq|rs)^λ + I S^λ + Z·(CPHF RHS)`, identical to the CCSD path.

**Validation** (oracle = FD of pycc's own CCSD(T) correlation energy; **not** psi4 — see §4):
- Gradient vs a 5-point O(h⁴) FD, H2O/6-31G: **1.8e-12 all-electron, 1.9e-12 frozen core**
  (h⁴-convergent — the residual is FD truncation, not a real error). Corrects a prior 2.1e-6.
- The (T) density reconstructs `E_corr`, and `Tr(D·μ)` matches a Fock-perturbation FD to ~1e-13 —
  guarding the diagonal-only density against a re-introduced off-diagonal.
- Limit check: CCSD(T)→CCSD (drop the triples) reproduces the CCSD gradient.
- Tests: `test_083` (gradient, all-electron + frozen core, frozen FD references asserted at 1e-11),
  `test_034` (density + dipole).

**How we got here (superseded readings).** The first pass concluded phase 1 could keep the −½Sˣ
ov-only Z-vector unchanged, deferring the dependent-pair terms to a later frozen-core phase, on the
reading that all-electron ⇒ ΔX_mn=0 ⇒ no dependent-pair contribution. That was **wrong**: it took
Lee–Rendell's `|ΔX_mn|<1e-8` *degeneracy* guard for an all-electron cancellation (that cancellation is
Scuseria's separate formulation, which L–R contrast with theirs). A probe (H2O/6-31G) showed
`gradient('ccsd(t)')` off by 2.1e-6 vs psi4; block-wise Fock-perturbation FD localized it to the
off-diagonal oo/vv (T) density — which proved *extraneous* (not a misplaced term), the real fix being
the canonical dependent-pair orbital response above. Also superseded: "ε (A 12–13) is missing" (it is
present — the energy validates it, and adding it explicitly double-counts). Corroborated by the PI's
own Psi4 `relax_I_RHF` (its `delta_I/delta_f_{IJ,AB}` are the κ̄ divides). See Appendix B.

**Spin-orbital (UHF).** The SO path is the same construction with the spin-adapted `H.L` replaced by
the antisymmetrized `<pq||rs>`. `cctriples.so_t3_density` builds the SO (T) density/Λ (its own T3
kernels `t3{c,d}_{ijk,abc}_so`); `ccdensity`/`cclambda` gate the SO (T) 1-/2-PDM and `S1`/`S2` on
`model=='CCSD(T)'`; and `CCderiv._so_gradient` gains the same `model=='CCSD(T)'` branch — `κ̄_oo`/`κ̄_vv`
from `_dependent_pairs` into `Drel`, coupled into the ov Z-vector RHS through the antisymmetrized ERI.
Frozen core rides the SO `Pco` exactly as the spatial path. **Validation** (§4 oracle): closed-shell
SO==spatial keystone **2.3e-13** (6-31G — STO-3G leaves `‖Pvv‖=0`, the minimal-basis trap), frozen-core
keystone **1.4e-13**, and open-shell NH₂ (²B₁, C2v pinned occ / 6-31G) vs a 5-point O(h⁴) FD of pycc's
own SO CCSD(T) energy **3.7e-12**. Tests in `test_083` (open-shell reference hard-wired). ROHF unsupported
(guarded, §6).

**Deferred:** CCSD(T) Hessian/APT on the unified base (§9). (The explicit-route cross-check and its
`_gradient_explicit` were removed in #194, so the earlier "extend `_gradient_explicit` to the (T)
dependent-pair" item is moot.)

## 8. CC static dipole polarizability — design & status

**Status.** **DONE for CCSD (2026-07-09) and CCSD(T) (2026-07-17)** in `CCderiv.polarizability()` —
**spatial RHF and spin-orbital UHF, both all-electron and frozen core** — FD- and CFOUR-validated
(§8.4). First CC *second*-derivative property; built before the `CorrelatedDerivs` refactor because
it exercises the perturbed-relaxed-density machinery the shared base will own, without the
nuclear-skeleton complexity of Hessians/APTs. The asymmetric route (below) stays entirely within
`cclambda`'s `r_L` (no `Y1/Y2` response apparatus), with the perturbed HBAR obtained by the analytic
product rule of the `build_H*` block builders (the finite-difference stencil that seeded it during
development has been retired).

**Route: the ASYMMETRIC approach** — differentiate the (already-2n+1) relaxed-density gradient a second
time; exactly the `docs/mp2_2n1_perturbed.tex` formulation, extended to CC by a density swap. Chosen
because Stanton & Gauss (*Recent Advances in Coupled-Cluster Methods*, ch. on CCSD/CCSD(T) second
derivatives, pp. 54–64) show the asymmetric route is the **preferred, unambiguous** choice for
**CCSD(T)**: the (T) correction has no eigenvalue / similarity-transform structure (its `JI` operator
is intrinsically the interchange construction), so the *symmetric* form — which would drop perturbed Λ
— does not apply cleanly. For pure-CCSD same-class properties (polarizability, force constants) the
symmetric form would be ~2× cheaper (it solves perturbed T only, for all perturbations), but a single
(T)-capable formulation uses asymmetric.

### 8.1 Formalism (worked through with the PI, 2026-07-09)

The relaxed (Z-vector) first derivative of the CC energy for a general real perturbation `x` is, with
all perturbation-dependent factors the **bare skeleton** integral derivatives (no CPHF / orbital
response — that is folded into `D_rel` and `W`):

    ∂_x E = Σ_pq D_rel_pq f^x_pq  +  Σ_pqrs Γ_pqrs V^x_pqrs  +  Σ_pq W_pq S^x_pq

`D_rel` = relaxed 1-PDM (Λ-response `D` + orbital-relaxation blocks); `Γ` = 2-PDM; `W = I'(D_rel)` =
energy-weighted density (the generalized-Fock Lagrangian `I'` evaluated at the *relaxed* density).

**All orbital relaxation lives in specific blocks.** The Z-vector `z` (`−z_ai`) enters only
`D_rel,ai`/`D_rel,ia` (density) and `W_ai`/`W_ia`/`W_ij` (energy-weighted density), and **not**
`D_rel,ij`/`D_rel,ab`/`W_ab`/`Γ`. (From `I'_pq = −½[ε_p(D_pq+D_qp) + δ_{q∈occ}Σ_rs D_rs(⟨rp|sq⟩_L+…) +
4Σ⟨pr|st⟩Γ_qrst]`: the ε-term routes `z` into the ov/vo output blocks, the `δ_{q∈occ}` two-electron
term into the vo/oo blocks, and the Γ-term carries none.)

**For an electric field** the AO basis is fixed: `f^{F_a} = −μ_a`, `V^{F_a} = 0`, `S^{F_a} = 0`. Hence
`∂_{F_a} E = −D_rel·μ_a` (the relaxed dipole is `D_rel·μ_a`), and the second derivative is a plain
product rule — no `W·S` term (`S^F=0`), no second skeleton derivatives (`μ` is field-independent):

    α[a,b] = −∂²E/∂F_a∂F_b = (∂_{F_b}D_rel)·μ_a  +  D_rel·(∂_{F_b}μ_a),   ∂_{F_b}μ_a = U^bᵀμ_a + μ_a U^b

with `D_rel` the unperturbed relaxed density (`CCderiv._relaxed_density`), `∂_{F_b}D_rel` its field
response, and `U^b` the first-order CPHF orbital response (`CPHF.full_U`). Only *first-order* responses
appear — no second-order CPHF `U^{xy}` (that layer, `perturbed_fock2`/`_full_U2`/`_xi`, existed only
for the explicit route and was removed with it — §9/#194).

Note the CC unrelaxed density is non-Hermitian (`D_ai ≠ D_ia`, since Λ≠T†), so
`D_rel,ai = D_ai − z_ai` and `D_rel,ia = D_ia − z_ai` differ (MP2 has `D_ai=D_ia=0`, so both are just
`−z_ai`). This asymmetry is real but **washes out of α** — both terms contract against symmetric
quantities (`μ`, and `rot = U^bᵀμ+μU^b`), so only the symmetric part of `D_rel`/`∂D_rel` survives
(verified: symmetrizing changes α by 5.6e-17). It *would* matter for a nuclear perturbation (`S^x≠0`).

### 8.2 The diagonal-ε gotcha (the Phase-1 bug — READ THIS BEFORE PORTING)

`MPwfn._mp2_lagrangian` builds its Fock term as `termA = eps[:,None]*(D+D.T)` — **diagonal orbital
energies only** (`eps = diag(H.F)`), the canonical simplification valid at `F=0`. This is correct for
the unperturbed Lagrangian and for `W`. **But its field derivative is *not* obtained by finite-stencil
of `_mp2_lagrangian`**: a stencil only captures `∂ε = diag(df)`, whereas the true perturbed Lagrangian
needs the **full Fock matrix product** `df @ (D+D.T)` (MP2's analytic `_perturbed_lagrangian` has
exactly this). The omitted piece is `df_offdiag @ (D+D.T)`:

- for **MP2** it vanishes — the unrelaxed `D` fed to the Z-vector RHS has **no ov block**, so the
  off-diagonal `df` has nothing to couple to (this is why MP2 gets away with `_perturbed_lagrangian`'s
  comment "diagonal `d_x eps` suffices for the unrelaxed `D`");
- for **CC** it is nonzero — the unrelaxed `D` **has** `D_ov`/`D_vo`, and the orbital-relaxation
  off-diagonal `df` couples to them. Dropping it put a **~9 %** error into the perturbed Z-vector RHS
  `dX`, hence into `∂D_rel,ov`, hence into α.

Insidious because it survived every self-consistent check: a fixed-basis FD of `dX`, and the frozen-MO
oracle, both route through the same diagonal-ε `_mp2_lagrangian`, so they agree with the (wrong)
stencil — circular. Only the `perturb_h` energy/dipole oracle exposes it, because at a *field-relaxed
canonical* reference the Fock **is** diagonal, so `_mp2_lagrangian` is correct there.

**Rule for the port:** build the perturbed Lagrangian via the analytic full-`df` formula — reuse
`MPwfn._perturbed_lagrangian` (`dA = df @ (D+D.T) + ε[:,None](dD+dD.T)`), density-generic, with the CC
`D`/`dD`/`Γ`/`dΓ` swapped in — **never** a stencil of `_mp2_lagrangian`. This is the natural
`CorrelatedDerivs` shared primitive and is correct for both MP2 (no ov) and CC (ov present) with no
special-casing.

### 8.3 Machinery & mechanics (validated)

**`dD_rel(F_b)` is the unperturbed relaxed-density build (§7 / `_relaxed_density`) differentiated
term by term** — the MP2 `_perturbed_relaxed_opdm` structure ported to CC, **but** with the full-`df`
perturbed Lagrangian of §8.2 and the CC-specific ov/vo unrelaxed blocks retained:
- oo/vv correlation blocks ← perturbed correlation 1-/2-PDM `dD, dΓ` (product rule of `ccdensity`'s
  density builders in the perturbed amplitudes `dt` **and** perturbed multipliers `dΛ`);
- ov/vo ← the **unrelaxed** `dD_ov`/`dD_vo` (CC-only; zero for MP2 — do **not** overwrite them, cf. the
  MP2 template which assigns ov = `−zx`) **plus** the perturbed Z-vector `dz` (solve the SAME orbital
  Hessian `G` with RHS `dX − (dG)·z`; `dX = ∂(I'_ov − I'_voᵀ)` from the full-`df` perturbed Lagrangian);
- frozen core co/oc ← perturbed Sylvester `dP_co = (dI'_co − dI'_ocᵀ − df[co,co]P_co + P_co df[o,o])/Δε`
  (verbatim from MP2; the field's active–active `df[o,o]` makes this non-trivial);
- (T) oo/vv ← perturbed dependent-pair `dκ_oo`/`dκ_vv` (Sylvester-style, from the field-perturbed
  (T)-inclusive Lagrangian).

**New CC machinery (the parts MP2 got for free — its amplitudes are closed-form and Λ=t), all
prototyped and FD-validated:**
1. **Perturbed-T solve** `dt/dF` — iterative (the CCSD Jacobian / linearized residual applied to `dt`,
   RHS from the CPHF-folded perturbed integrals `df`/`deri`, so orbital relaxation is carried; reuse
   `helper_diis` + `r/Dia` denominators). Distinct from `ccresponse.solve_right`, whose X vectors are
   this WITHOUT the orbital-response terms.
2. **Perturbed-Λ solve** `dΛ/dF` — iterative, *linear*, staying in `cclambda` (no `Y1/Y2`): the
   Λ-Jacobian action is `r_L(dΛ) − r_L(0)` (since `r_L` is affine in Λ), and the inhomogeneity is
   `r_L` evaluated with the **perturbed HBAR** + `dL` (unperturbed `G`) plus the explicit `dG·H`/`dG·L`
   product-rule halves for the terms bilinear in `G`. Iterate `dΛ += (B_Λ + Jacobian)/D` like
   `solve_lambda`. **Required by the asymmetric route**; MP2 avoided it only because Λ=t.
3. **Perturbed HBAR** `dHBAR` — needed only to build the perturbed-Λ inhomogeneity. Obtained for **free
   and exact** by a 5-point central stencil of `cchbar` at `(F±hδf, ERI±hδe, L±hδL, t±hδt)`: every CCSD
   HBAR block is a polynomial of degree ≤4 in the step, so the stencil is algebraically exact (confirmed
   to roundoff, `h` vs `h/2`) — no hand-differentiation. (An analytic `dHBAR` could replace it later for
   efficiency, validated against this stencil; not needed for correctness.)
4. **Perturbed correlation densities** `dD, dΓ` — product rule of `ccdensity`'s density builders in
   `dt` and `dΛ`; also exact via the same degree-≤4 stencil of `gradient_densities`.
5. **(T):** field-perturbed `cctriples.t3_density` outputs (`S1/S2`, `Doo/Dvv/Dov`, 2-PDM) from
   `dt1, dt2, df, deri`, and the perturbed (T) dependent-pair.

The perturbed Z-vector `dz/dF` and (for CC) the perturbed Λ are **intrinsic to the asymmetric route,
not removable overhead** — they are all *first-order* responses (no `U^{xy}`). A direct test confirmed
this: zeroing `zx` in `MPwfn._perturbed_relaxed_opdm` shifts the MP2 polarizability by ~0.66 (it is
load-bearing). An earlier draft of this section pursued the *symmetric* "no perturbed Λ/Z" economy;
that is valid for pure CCSD but not for the (T)-capable path, and the Stanton–Gauss argument settled it
in favor of asymmetric. **No detour into / rewrite of the MP2 code is needed** — CC extends the existing
(asymmetric) MP2 machinery by a density swap plus the perturbed-Λ solve.

Reused: the perturbed Lagrangian (density-generic — the orbital-response part), the perturbed
Z-vector and `dP_co` Sylvester logic, and the first-order `CPHF.perturbed_fock`/`perturbed_eri`/
`full_U`. All of this now lives on `CorrelatedDerivs` (§9): the shared
`_perturbed_lagrangian`/`_perturbed_relaxed_density`, driven by each leaf's
`_perturbed_unrelaxed_densities` hook.

**Implementation mechanics (worked out + partly validated 2026-07-09).**
- **Perturbed-amplitude RHS via the residual, no hand-derivation.** `⟨μ|e^{-T}He^{T}|0⟩` is *linear in
  H*, so the field-derivative of the CC residual at fixed `t` is just the residual evaluated with the
  perturbed integrals: `B = CCwfn.residuals(df, t1, t2)` with `H.ERI→deri`, `H.L→dL` swapped in
  (`df=CPHF.perturbed_fock`, `deri=CPHF.perturbed_eri`, `dL=2·deri−deri.swap(2,3)`). **Validated:** with
  the *bare* dipole (`df=μ`, `deri=0`) this reproduces `ccresponse`'s independently-built `pertbar`
  (`Avo`, `Avvoo`) to 1.7e-16 — so the orbital-relaxed RHS is `df/deri` from CPHF instead of bare `μ`.
- **Perturbed-T LHS = the CCSD Jacobian = HBAR·(dt), built from `cchbar`** (the `ccresponse.r_X1/r_X2`
  contraction pattern — method-agnostic, *not* ccresponse-specific; copied into the derivative code so
  it stays self-contained). Solve `dt += (B + HBAR·dt)/D` with `helper_diis`, exactly like
  `solve_right` but with the orbital-relaxed `B`. Same structure for `dΛ` (the linear `r_L1/r_L2`
  operator + a field-perturbed inhomogeneity).
### 8.4 Validation (CCSD + CCSD(T) complete — spatial + spin-orbital, all-electron + frozen core)

Prototype checkpoint ladder, all cleared for CCSD/H2O/STO-3G (spatial, all-electron):

| piece | check | residual |
|---|---|---|
| perturbed-T RHS | == `ccresponse` `pertbar` (bare dipole) | 1.7e-16 |
| perturbed-T LHS | == `solve_right` X (bare RHS) | 1.1e-13 |
| `dt/dF` | fixed-basis FD (integrals along `df`/`deri`) | 2e-9 |
| `dHBAR` | 5-pt stencil exact (`h` vs `h/2`) | ~roundoff |
| `B_Λ` inhomogeneity | FD of `r_L` | 2e-12 |
| `dΛ/dF` | FD (re-solve CC+Λ at perturbed integrals) | 3e-9 |
| **α (all 9 elements)** | **5-pt finite field of the relaxed dipole (Λ→1e-14, F=5e-4)** | **1.8e-12** |

The `1.8e-12` is the FD floor (it scales as `O(F⁴)` between F=5e-4 → 1e-3; an earlier `1.3e-9` was purely
the loose `1e-10` Λ convergence inside `CCderiv.relaxed_dipole`, not a residual error). Two independent
oracles — a 5-point finite field of the relaxed dipole and a 5-point second derivative of `E_corr`
(both via `psi4 perturb_h`) — agree to machine precision, so the oracle is unimpeachable. **The
fixed-basis / frozen-MO checks are necessary but not sufficient** for the orbital-response half: they
route through the diagonal-ε `_mp2_lagrangian` and so are blind to the §8.2 gotcha; the field-relaxed
`perturb_h` oracle is the one that catches it.

**Production validation** (`CCderiv.polarizability()`, `test_084`), all clear:

| path | check | residual |
|---|---|---|
| spatial, all-electron | tight finite field of the relaxed dipole | 1.8e-12 |
| spatial, frozen core | tight finite field | 1.0e-11 |
| off-diagonal (planar HOF/cc-pVDZ) | finite field; `α_xy` large, `α_xz=α_yz=0`, total positive-definite | 2.6e-10 |
| spin-orbital == spatial (AE / FC) | RHF-forced-to-SO keystone | 8.8e-14 / 3.1e-14 |
| open-shell UHF (2-B1 NH2) | energy 2nd-derivative FD (`α_zz`); symmetric, positive | 2.7e-10 |
| CCSD(T) spatial == SO (AE / FC) | SO==spatial keystone (water diagonal, HOF off-diagonal) | ~1e-12 |
| CCSD(T) vs **CFOUR** (water/HOF, AE/FC, spatial + SO) | `POLAR − POLARSCF`, matched GENBAS 6-31G (`test_ccsdt_polarizability_cfour`) | ~5e-11 |

**Phasing:** (1) CCSD spatial all-electron **[done]**; (2) + frozen core **[done]**; (3) + spin-orbital
**[done]**; (4) + (T) **[done]** — spatial + SO, all-electron + frozen core, CFOUR-anchored.

**Validation (production).** For **CCSD**, primary oracle = **finite difference of
`CCderiv.relaxed_dipole`** in a field (`÷h`, ~1e-12; Λ tight) plus the **SO==spatial keystone**. For
**CCSD(T)** the reliable oracles are the **energy second-derivative FD** and **CFOUR**
(`POLAR − POLARSCF`, exactly matched GENBAS basis, ~5e-11); the relaxed-dipole FD is **not** reliable for
(T) — its dependent-pair *numerator* gate (unperturbed side) disagrees with the analytic *gap* gate, so a
symmetry-forbidden near-degenerate pair (`num=0` at zero field, `dnum≠0`) sits below the `1e-8` numerator
gate at the stencil and the FD drops its `dnum/gap` contribution (~1e-4 floor; switching the FD path to
gap-gating recovers ~2e-9). Both (T) oracles cover frozen core.
Debugging oracles (CCSD only, phase 1–3): `ccresponse.solve_right(ω=0)` X-vectors cross-check the
*unrelaxed* `dt/dF`, and `ccresponse.linresp/polarizability(0.0)` the *unrelaxed* CCSD polarizability —
they isolate the amplitude response from the orbital-relaxation delta. **`ccresponse` is never a code
dependency** (it omits orbital response and cannot do (T)); oracle only.

## 9. CorrelatedDerivs refactor — design & status (COMPLETE)

**Motivation.** MP2, CC, and CI each carried a *parallel* copy of the same orbital-response machinery
(Lagrangian, dependent-pair rotations, Z-vector, relaxed-density assembly, gradient and 2n+1
second-derivative orchestration), differing only in the correlated densities. This refactor factors
all of it into one shared base, leaving each method to supply only its densities.

**Status: DONE** (merged PRs #192–#203, 2026-07). MP2 and CC run entirely through the shared base;
CI has a registered-when-ready `CIderiv` stub. Two method-specific overlap properties (AAT, VG-APT)
were deliberately left on the leaves (see below).

**Hierarchy.**

    CorrelatedDerivs        # base: orbital response + assembly + public properties (method-agnostic)
    |-- MPderiv             # extracted from MPwfn (closed-form density responses)
    |-- CCderiv             # inherits; iterative T/Lambda density responses
    |-- CIderiv             # phase-4 stub: density hooks raise NotImplementedError (a programmer task)

**Base owns** (depends only on the densities + reference, not on how they were obtained): the
Lagrangian `I'(D, Gamma)` (`_lagrangian`/`_so_`); the dependent-pair rotations `P`/`dP`
(`_dependent_pairs`/`_perturbed_dependent_pairs` — one canonical copy, gated by the orbital gauge);
the unperturbed orbital response (`_orbital_response`/`_so_`, returning an **`OrbitalResponse`**
namedtuple whose byproducts `z`/`mo_hessian`/`Pco`/`Poo`/`Pvv` the perturbed machinery reuses);
relaxed-density assembly (`Drel = D + Pco + Poo + Pvv − z`) and `_relaxed_density`; the full-occupied
CPHF accessor (`_full_occ_cphf`); the perturbed Lagrangian (`_perturbed_lagrangian`/`_so_`); the
perturbed relaxed density (`_perturbed_relaxed_density`/`_so_`, returning a **`PerturbedResponse`**
`(dDrel, dGam, dW)` from one solve); and the **public correlation-property API** —
`relaxed_dipole`, `gradient`, `polarizability`, `dipole_derivatives`, `hessian` (the `pycc.properties`
facade reaches these by name on the driver). Spatial and spin-orbital throughout.

**Leaves (method-specific), supplied by each subclass:** just two hooks — the unrelaxed densities
`_unrelaxed_densities() -> (D, Gamma)` and their first-order response
`_perturbed_unrelaxed_densities(pert, df, deri, dL) -> (dD, dGamma)`. Each subclass fully owns its
amplitude/multiplier solve (MP2 closed-form; CC iterative T + Lambda(+t3); CISD CP-CI) and hands back
only densities, so the base never sees a Jacobian. A leaf **overrides a public property only to add
method-specific behavior** — the sole case is `CCderiv.polarizability`, which adds the CCSD/CCSD(T)
model and (T)-intermediate guards then defers to `super()`. AAT (`atomic_axial_tensors`) and the
velocity-gauge APT (`velocity_dipole_derivatives`) are **not** in the base — they are overlap /
half-derivative formulations, not the `(Drel, Gam)`-contraction pattern — so they stay on the leaves
(MP2 on `MPderiv`; CISD on `CIwfn`); a future hoist is possible (see Roadmap).

**Orbital gauge.** The occ-occ / virt-virt perturbed-MO gauge is a single base property
`perturbed_mo_gauge` (not per-method capability flags): `'canonical'` for CCSD(T) (the oo/vv
dependent-pair route, §7), `'non-canonical'` otherwise (MP2, CCSD, CISD — invariant, pairs vanish).

**Phasing as executed** (one PR per phase; full MP2 + CC suite green at each):

- **1a/1b/1c** (#192/#193/#194): `CorrelatedDerivs` base; extract `MPderiv` from `MPwfn`; **remove the
  explicit-derivative route entirely** (scaffolding, never meant to persist) + the now-dead
  second-order CPHF layer (`perturbed_fock2`/`_full_U2`/`_xi`/`_d2fock`/…). Default route flipped to
  2n+1.
- **2a/2b/2c** (#195/#196/#197): hoist the Lagrangian, the relaxed-density/Z-vector, and the
  gradient/`relaxed_dipole`.
- **3-conv → 3d** (#198–#202): inherit convergence from the wavefunction (no hardwired cutoffs
  except algorithmic gates); hoist the perturbed Lagrangian (3a), rename Z-vector→`_orbital_response`
  + `OrbitalResponse` record (3b), hoist the perturbed relaxed density + `PerturbedResponse` +
  `_perturbed_unrelaxed_densities` hook (3c), and hoist the 2n+1 orchestration
  (polarizability/APT/Hessian) as **public base methods**, leaves overriding only for guards (3d,
  the "Design-2" naming decision). #202 also carried a `cphf.py` cleanup (orbital Hessian
  `hessian`→`_mo_hessian`; `_full_U`→`full_U`, `_d2int_blocks`→`nuclear_hessian_skeletons`;
  descriptive `_build_rhs_nuclear`/`_skeleton_derivatives`/`_dipole_ov`…; dead `rhs_nuclear` removed).
- **4** (#203): `CIderiv` scaffold — inherits `CorrelatedDerivs`, the two density hooks stubbed
  (`NotImplementedError` + interface/roadmap docstrings), exported but **not** registered
  (`register_deriv(CIwfn, CIderiv)` commented) so CISD stays on the transitional `CIwfn` path until a
  programmer implements the hooks. #203 also removed the dead `mp2_relaxed_opdm` accessor and merged
  MP2's `_perturbed_densities` into the `_perturbed_unrelaxed_densities` hook.

**Deferred follow-ups (each its own PR):** the programmer's CISD hook implementation + validation +
retiring the transitional `CIwfn` derivative code; unifying cphf's two magnetic orbital-Hessian
paths; hoisting AAT/VG-APT into the base; a `make_t3_density`/fold-the-solve-in UX change and the
solver-knobs (convergence/DIIS) plumbing; the naming/notation sweep vs
`docs/cc_gradients_orbital_response.tex`.

## Appendix A: condensed changelog (by PR)

Reference layer, then the MP2 derivative effort:

| PR | Milestone |
|---|---|
| #121–#129, #153 | `HFwfn` derivative refactor: gradient, CPHF separation, APT, Hessian, `Derivatives` promotion |
| #154 | spin-orbital HF properties (gradient, polarizability, APT, Hessian, AAT) for RHF/UHF |
| #152 | MP2 analytic gradient — spin-orbital (relaxed density + Z-vector; GSB Lagrangian) |
| #155 | spin-adapted (closed-shell RHF) MP2 gradient |
| #156 | `Derivatives` block-aware API |
| #158 | frozen-core spin-adapted MP2 gradient (core↔active divide, full-occ Z-vector) |
| #159 | full-MO spin-orbital Hamiltonian + frozen-core SO gradient + `occ-starts-at-0` triples audit |
| #160 | `np.einsum` → `self.contract` (device backend) |
| #162 | explicit-derivative engine (`CPHF.perturbed_fock`/`perturbed_eri`; correlation dipole/gradient) |
| #163 | MP2 dipole **polarizability** — explicit (second-order CPHF block: `_xi`, `_d2fock`, `_d2eri`, `_full_U2`) |
| #164 | MP2 **APT** — explicit (skeleton generalization; `−μ^X` mixed skeleton) |
| #165 | MP2 **Hessian** — explicit (full Eq. 18 `ξ`; `_d2int_blocks` per-atom-pair cache) |
| #166 | **2n+1** polarizability (perturbed Z-vector; frozen-core Sylvester `∂P_co`) |
| #167 | **2n+1** APT (nuclear- and field-side routes; perturbed energy-weighted density) |
| #168 | **2n+1** Hessian (the `O(N)`-solve payoff; rotations hoisted onto densities) |
| #169 | **HF velocity-gauge APT** (`HFwfn.velocity_dipole_derivatives`; momentum response) |
| #170 | **MP2 AAT** (VCD) — density/overlap form, gauge-invariant, frozen-core, both spins |
| #171 | **MP2 velocity-gauge APT** — AAT machinery with the linear-momentum operator |
| #172 | **property facade** — `pycc.PropertyComponents` + `pycc.dipole/gradient/polarizability/hessian/apt/aat`; nuclear un-bundled from the HF methods; `MPwfn.total_*` removed |
| #176 | **CCSD gradient** — spatial closed-shell RHF via `CCderiv` (CCSD relaxed density + Z-vector, reusing the MP2 assembly) |
| #177 | **CCSD gradient** — spin-orbital UHF (SO 2-PDM + gradient; all-electron + frozen core) |
| #183 | **CCSD(T) gradient** — spatial RHF, all-electron; + frozen core sourced from psi4 only (`frozen_core` pycc override removed) |
| #184 | **frozen-core CCSD(T) gradient** — oo/vv dependent-pair κ̄ generalized from the frozen-core `Pco`; diagonal-only (T) `Doo`/`Dvv` |
| #185 | **spin-orbital CCSD(T) gradient** — SO (T) density + oo/vv κ̄ in `_so_gradient` (all-electron + frozen core); (T) density builders moved to `cctriples` (no `ccwfn`→`ccdensity` dependency) |
| #186 | **CCSD/CCSD(T) relaxed dipole** — `CCderiv.relaxed_dipole` = `Tr(D_rel·μ)`; shared `_relaxed_density`/`_so_relaxed_density` factored out of the gradients; wires `pycc.dipole(CCwfn)` (both spins, all-electron + frozen core) |
| #187 | **CCSD dipole polarizability** — spatial RHF + spin-orbital UHF, all-electron + frozen core, via `CCderiv` (2n+1); the first CC second-derivative property; analytic perturbed HBAR (finite-difference stencil dropped) |
| #188 | **(T) density speedup** — `diag(Doo)` built in the ijk loop alongside `diag(Dvv)` in `t3_density`/`so_t3_density`; the separate abc loop (Lee–Rendell's avoidable extra N^7 set) removed. Bit-identical E(T)/density/gradient; ~3.3x (spatial) / ~2.6x (SO) |
| #189 | **(T) ov 1-PDM fix** — restored the missing `1/4` in `so_t3_density`'s occ-virt (T) density `Dov` (T2† normalization); regression `test_086` (unrelaxed field-FD dipole — the gradient is blind to the ov density, Handy–Schaefer) |
| #190 | **spin-orbital CCSD(T) polarizability** — SO (T), 2n+1, all-electron + frozen core; canonical perturbed orbitals, perturbed (T) intermediates + oo/vv dependent-pairs; energy-second-derivative-FD anchored (`test_084`/`test_085`) |
| #191 | **spatial CCSD(T) polarizability** — closed-shell RHF, all-electron + frozen core; the perturbed (T) Λ source threaded through `r_L1`/`r_L2` (`s1`/`s2`) with matched P_ij^ab symmetrization; CFOUR-anchored tests (`POLAR − POLARSCF`, matched GENBAS 6-31G); all `_DBG_*` scaffolding stripped |
| #192–#194 | **CorrelatedDerivs refactor 1a/1b/1c** (§9): base class; `MPderiv` extracted from `MPwfn`; **explicit-derivative route + second-order-CPHF layer removed** (default flipped to 2n+1) |
| #195–#197 | **refactor 2a/2b/2c**: hoist the Lagrangian, relaxed-density/Z-vector, and gradient/`relaxed_dipole` into the base |
| #198–#200 | **refactor 3-conv/3a/3b**: inherit convergence from the wavefunction; hoist the perturbed Lagrangian; `_zvector`→`_orbital_response` returning the `OrbitalResponse` record (exposes `Poo`/`Pvv`) |
| #201 | **refactor 3c**: perturbed relaxed density → base (`PerturbedResponse` `(dDrel,dGam,dW)` from one solve) + the `_perturbed_unrelaxed_densities` leaf hook; `_full_occ_cphf` hoisted |
| #202 | **refactor 3d**: 2n+1 orchestration (polarizability/APT/Hessian) → public base methods, leaves override only for guards (Design-2); + `cphf.py` cleanup (`hessian`→`_mo_hessian`, `full_U`, `nuclear_hessian_skeletons`, descriptive `_build_rhs_nuclear`/…, dead `rhs_nuclear` removed) |
| #203 | **refactor 4**: `CIderiv` scaffold/stub (density hooks `NotImplementedError`, exported, unregistered); + dead `mp2_relaxed_opdm` removed and MP2 `_perturbed_densities` merged into `_perturbed_unrelaxed_densities` |

Tests: `test_046`–`test_050` (spatial HF), `test_062`–`test_066` (SO HF), `test_061` (MP2
gradient/relaxed density), `test_067` (polarizability), `test_068` (APT), `test_069` (Hessian),
`test_071` (HF velocity APT), `test_072` (MP2 AAT), `test_073`
(MP2 velocity APT), `test_074` (property facade). (`test_070`, the old explicit-vs-2n+1
polarizability cross-check, was removed with the explicit route in #194.) CISD: `test_079`–`test_082`
(LG-APT, VG-APT, AAT, Hessian). CC gradients: `test_076` (CCSD, spatial),
`test_078` (CCSD, spin-orbital), `test_077` (SO 2-PDM), `test_083` (CCSD(T), spatial + spin-orbital,
all-electron + frozen core), `test_034` (CCSD(T) density/dipole). CC polarizability: `test_084`
(CCSD + CCSD(T), spatial + spin-orbital, all-electron + frozen core, incl. the CFOUR-anchored (T)
cases), `test_085` (invariant [F,T3] (T) energy), `test_086` (SO (T) unrelaxed dipole). The 2n+1
APT/Hessian cross-checks live in `test_068`/`test_069`.

## Appendix B: superseded early decisions

The original 2026-06-21 plan made several calls later reversed as the effort matured — recorded so
the record stays coherent (rationale is in git history):

- **"Spatial RHF only"** → **spin-orbital first**. The GSB orbital-response Lagrangian applies
  verbatim in the spin-orbital basis; the spatial spin-adapted path followed and is validated
  against it (the SO == spatial keystone).
- **"All-electron only first"** → **frozen core throughout** (both spin paths, every property).
- **"Explicit route; 2n+1 deferred"** → **both routes implemented**, the 2n+1 route as the
  efficient alternative *and* an independent cross-check of the explicit suite.
- **"MP2-specific, no abstraction"** → **fully abstracted** (§9, PRs #192–#203): a shared
  `CorrelatedDerivs` base owns the orbital response, assembly, and public property API; `MPderiv`
  and `CCderiv` supply only their densities via two hooks, and a `CIderiv` stub awaits CISD. (The
  interim state — `CCderiv` reusing MP2's Lagrangian/CPHF by delegation — is what this refactor
  removed.)
- **"(T) all-electron reuses −½Sˣ, dependent-pair terms deferred to frozen core"** → **(T) uses
  canonical perturbed orbitals for the oo/vv blocks even all-electron** — the dependent-pair κ̄
  generalized from the frozen-core `Pco`. (Canonical is Lee–Rendell's cost choice, not a hard
  requirement: Scuseria's non-canonical route is equally correct at one extra N^7 set — see §5/§7.)
  The early reading mistook Lee–Rendell's `|ΔX_mn|<1e-8` *degeneracy* guard for an all-electron
  cancellation (that cancellation is Scuseria's separate formulation). See §5 and §7.
