# PyCC post-Hartree–Fock analytic derivatives — design & status

_Design-of-record and status for the post-HF analytic-derivative-property effort: the MP2
correlation gradient, the electric/nuclear second derivatives (polarizability, APT, Hessian),
and their two independent implementations (explicit-derivative and 2n+1). Started 2026-06;
this revision 2026-07 (rewritten from the original chronological plan — milestone history is in
the Changelog appendix). Preceding spin-orbital infrastructure: `archive/ENHANCEMENT_PLAN_2026-06.md`.
Filename retained for the docstring/test references that point here._

## 1. Status at a glance

MP2 **correlation** analytic derivatives. Every total property splits as **nuclear + reference (HF)
+ correlation**: the correlation part lives on `MPwfn`, the reference part on `HFwfn` (pure
electronic), and the nuclear part is a closed-form geometry/charge term. The `pycc` property facade
(`pycc.dipole`/`gradient`/`polarizability`/`hessian`/`apt`/`aat`) assembles the three into a
`PropertyComponents` (`.total`/`.nuclear`/`.reference`/`.correlation`), for any wavefunction type
(see the property-facade work). All rows below are implemented for **both spin paths** (spin-orbital
and spin-adapted closed-shell RHF), **all-electron and frozen-core**, unless noted.

| Property (derivative) | order | explicit route | 2n+1 route |
|---|---|---|---|
| dipole  `dE/dF` | 1st | ✅ `_corr_dipole_explicit` | ✅ = relaxed dipole (`mp2_relaxed_opdm`) |
| nuclear gradient  `dE/dX` | 1st | ✅ `_corr_gradient_explicit` | ✅ `gradient()` (relaxed-density) |
| polarizability  `d²E/dF²` | 2nd | ✅ `polarizability()` | ✅ `polarizability(route='2n+1')` |
| APT  `d²E/dF dX` | 2nd | ✅ `dipole_derivatives()` | ✅ `route='2n+1-nuclear'` / `'2n+1-field'` |
| Hessian  `d²E/dX²` | 2nd | ✅ `hessian()` | ✅ `hessian(route='2n+1')` |
| AAT / VCD  `d²E/dB dX` | 2nd | ✅ `atomic_axial_tensors()` (density/overlap form) | n/a |
| velocity-gauge APT  `d²E/dp dX` | 2nd | ✅ `velocity_dipole_derivatives()` | n/a |

The explicit and 2n+1 suites agree to ~machine precision — each is an independent cross-check of
the other. The default `route` is `'explicit'` throughout. The AAT and velocity-gauge APT use the
density/wave-function-overlap formulation (unrelaxed densities, no Z-vector); they are orbital-gauge
invariant, with a numerically stable non-canonical default (`gauge=`).

**Reference layer (`HFwfn`).** Gradient, polarizability, APT, Hessian, and AAT are done for
**RHF and UHF**, in both spatial and spin-orbital bases (ROHF orbital response deferred, guarded —
see Roadmap). These supply the reference contribution to every MP2 total property and anchor the
SO machinery through the RHF-forced-to-SO == spatial keystone.

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

One-directional layering: **`Derivatives` ← `CPHF` ← `HFwfn` / `MPwfn`**.

- **`Derivatives`** — skeleton MO derivative integrals only (no `U`): `core`, `overlap`, `eri`,
  `dipole` and second derivatives `core2`/`overlap2`/`eri2` (each with `so_*` spin-orbital twins).
- **`CPHF`** — orbital response. Owns `U`; consumes the skeleton integrals.
  `Perturbation('field'|'nuclear'|'magnetic', comp)` descriptors key the caches so multi-property
  runs never recompute the expensive nuclear ERI derivatives. `perturbed_fock`/`perturbed_eri`
  (`d_x f`, `d_x<>`, response-dressed); `perturbed_fock2`/`perturbed_eri2` (second, carrying
  `U^{xy}` and `ξ`); `_full_U`/`_full_U2` (with the `ncore` core↔active canonical block);
  `_d2int_blocks` (raw second skeletons, cached per atom pair).
- **`MPwfn`** — densities (`_(so_)mp2_corr_opdm`, `_(so_)mp2_tpdm`, `_(so_)mp2_lagrangian`); the
  relaxed density + Z-vector, centralized and cached in `_(so_)zvector` (the
  `_(so_)mp2_relaxed_densities` delegate to it); first-order responses (`_perturbed_t2`,
  `_perturbed_densities`, `_(so_)perturbed_relaxed_opdm`, `_(so_)perturbed_lagrangian` — the last
  takes an optional `(D, dD)`: unrelaxed → Z-vector RHS, relaxed → `d_x I`); and the property
  methods with their `route=` options.

Conventions: spatial methods unlabeled, spin-orbital prefixed `_so_`. `MPwfn` holds a persistent
full-occupied `CPHF` (`_full_occ_cphf`) so the response caches survive across property calls, and
so frozen core runs the response over the full occupied space in `MPwfn`'s own MO ordering.

## 4. Validation methodology

- **Oracle = PyCC's own finite difference** (energy / dipole / gradient), *not* Psi4's analytic
  derivatives. Psi4's frozen-core MP2 gradient is inconsistent with its own energy (~7e-6), so it
  is unusable as a reference; PyCC's own-energy FD is the ground truth.
- **Dipole-FD beats energy-FD.** For a second derivative, differencing the analytic *dipole*
  (`α = dμ/dF`, a `1/h` stencil) is ~3 orders tighter (~1e-12) than a second difference of the
  *energy* (`1/h²`, ~1e-9). Use 7-point O(h⁶) stencils.
- **SO == spatial keystone** on a closed shell (~1e-15) — the primary internal consistency check
  (spin-orbital vs spin-adapted must agree exactly).
- **Explicit == 2n+1** — every second derivative is computed both ways and must agree to ~machine
  precision.
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
  the diagonal `∂_x ε`. Neutral for the unrelaxed `D` (no ov block) but required for the relaxed
  `D`'s ov/core-active blocks (the 2n+1 APT's `d_F W`).
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

## 6. Roadmap

- **CC gradients.** Swap the MP2 density/Lagrangian for the CCSD relaxed density (Λ exists;
  `ccdensity` has the 2-PDM); the Z-vector solve and gradient assembly carry over. This is where
  the deferred "shared Lagrangian → Z-vector → density → gradient layer" refactor would land
  (kept MP2-specific so far — no premature abstraction). **CCSD(T) gradient (spatial, all-electron)
  is specced in §7** — the (T) densities/Λ already exist in `t3_density()`; the −½Sˣ orbital
  response is reused unchanged (all-electron ⇒ no canonical dependent-pair terms).
- **ROHF orbital response — deferred, guarded.** The semicanonical spin-orbital response is UHF-like
  and does not reproduce the restricted ROHF response; `CPHF.solve` raises for ROHF. The CPHF-free
  ROHF HF gradient is unaffected.
- Out of scope: 2n+2 / higher-order (cubic-response) economies.

## 7. CCSD(T) gradient — spatial, all-electron (design)

**Scope (phase 1):** closed-shell RHF, spatial MOs, all-electron. Target:
`pycc.gradient(CCwfn(wfn, model='CCSD(T)'))` through the existing `CCderiv`/`ccdensity` path.
Frozen-core and spin-orbital are deferred (below).

**References.**
- **Paper A** — T. J. Lee & A. P. Rendell, *J. Chem. Phys.* **94**, 6229 (1991): closed-shell
  *spatial* CCSD(T) gradient in the Handy–Schaefer Z-vector / effective-density (Gauss–Stanton–
  Bartlett) formulation. **This is pycc's formulation — directly applicable.**
- **Paper B** — Hald, Halkier, Jørgensen, Coriani, Hättig & Helgaker, *J. Chem. Phys.* **118**,
  2985 (2003): variational Lagrangian, integral-density direct, canonical orbitals. Conceptual
  cross-check and the frozen-core / canonical-orbital guide.

**Core physics — why (T) needs canonical MOs.** The triples solve
`⟨μ₃|[F,T₃]+[H,T₂]|HF⟩=0` (B-15) is non-iterative *only because F is diagonal*, so `[F,T₃]`
collapses to the orbital-energy denominator `D^abc = f_ii+f_jj+f_kk−f_aa−f_bb−f_cc` (A-5).
Non-canonical F ⇒ triples couple ⇒ iterative. Differentiating that denominator injects the
orbital-energy-derivative terms `∂f_ii/∂λ`, `∂f_aa/∂λ` (A-6) that the CCSD gradient lacks.

**Resolved theory fork — the perturbed-orbital gauge.** Worry: does (T)'s canonical requirement
force *canonical perturbed orbitals* in place of pycc's −½Sˣ non-canonical choice? **No, for
all-electron:**
- Paper A (p. 6232): *"provided that no core and/or virtual orbitals are excluded … there is no
  contribution to the gradient from dependent pairs because for these orbitals ΔX_mn = 0."* The
  occ–occ / virt–virt ("dependent pair") canonical response — the `ΔX_mn/Δf_mn` terms (A-34) —
  vanishes.
- Paper B: the canonical orbital-rotation multiplier `κ̄_pq(F_pq−δ_pqε_p)` exists to *simplify
  the frozen-core approximation*; by the 2n+1 rule the orbital-energy derivatives aren't needed.
- PI's Psi4 CCSD(T)-gradient code made **no (T)-specific change to the orbital response** and does
  not support frozen core — consistent.

**Decision:** phase 1 keeps the CCSD −½Sˣ, ov-only Z-vector **unchanged**. The oo/vv canonical
dependent-pair machinery (`ΔX_mn/Δf_mn`, gated by the numerator threshold `|ΔX_mn|<1e-8`,
A-p.6232 — never a test on Δf) is **deferred to the frozen-core phase**, where ΔX_mn≠0 for
core↔active pairs.

**Anatomy — the six (T) additions (Paper A).** Beyond the CCSD gradient:

| Intermediate | A-Eq | Destination |
|---|---|---|
| η_iᵃ, γ_kjᶜᵈ | 7, 8 | Λ / Z-vector RHS (couple to ∂T1, ∂T2) |
| **ε_i, ε_a** | 12, 13 | **diagonal** of effective 1-density Q (17–18) — the orbital-energy response |
| χ_jkᵇᶜ, ξ_ibᵃᵈ, κ_ijᵃˡ | 15, 9, 10 | effective 2-density G (20–22) |

Then the standard assembly runs: `E^λ = Q h^λ + G (pq|rs)^λ + orbital response` (A 28–33).

**Existing pycc infrastructure** — `CCwfn.t3_density()` (bottom of `ccwfn.py`; deliberately called
from the energy code so T3 is built once). For `model=='CCSD(T)'` it already yields:
- (T) 1-PDM `Doo/Dov/Dvv`, (T) 2-PDM `Goovv/Gooov/Gvvvo`;
- (T) Λ residuals `S1/S2` (Paper A's η/γ), added into Λ₁/Λ₂ (`cclambda:673/743`);
- validated at **energy** (test_005) and **density/dipole** (test_034).

**Audit findings (2026-07-05).** The probe (H2O/6-31G): pycc `gradient('ccsd(t)')` matches psi4 to
**2.1e-6** (CCSD gradient exact 6e-13; CCSD(T) *energy* exact 1e-13), so a small (T)-gradient-specific
term is off (0.2% of the (T) contribution). Localized as follows — note **pycc's (T) density follows
the Vikings (Hald et al., Paper B) Lagrangian formulation, *not* Lee–Rendell**, so map terms to Paper
B:
- **ε ruled out.** Lee–Rendell's ε (A 12–13) is a *diagonal* term the energy already validates; it is
  present in pycc (adding it explicitly makes the gradient ~60× *worse* — double-counting).
- **test_034 is blind to the off-diagonals.** `compute_energy`'s `eone = contract(F[o,o],Doo) +
  contract(F[v,v],Dvv)` with canonical (diagonal) F sees only `diag(Doo)`/`diag(Dvv)` and drops the
  ov block ("Brillouin condition"). So it validates **diag(D) + Γ** only.
- **Block-wise Fock-perturbation FD** (`Tr(D_blk·P)` vs FD of E_corr, `P` a masked Fock block; the
  CCSD control is self-consistent to 5e-11): the (T) 1-PDM fails to differentiate its own energy
  **only in the off-diagonal occ–occ `Doo` (+4.0e-5) and virt–virt `Dvv` (+3.4e-5)** blocks. **`Dov`
  is consistent to 6e-11** (matching Lee–Rendell Eq. 19: *no* ov (T) term) and **diag(D) to 3e-10**.

**Resolution (2026-07-06) — canonical perturbed orbitals via the existing frozen-core machinery.**
The gap is **not** a density block to correct in place; it is a *formulation/gauge* error. Confirmed
against the PI's own Psi4 CCSD(T)-gradient code (`relax_I_RHF`) and Lee–Rendell §II.B:

- **(T) requires *canonical* perturbed orbitals for the oo/vv blocks** — not pycc's current
  `U^X = −½S^X`. When canonical perturbed orbitals are used, the occ–occ `(i,j)` and virt–virt
  `(a,b)` **dependent pairs** contribute to the orbital-response Lagrangian via L–R's `ΔX/Δf` terms
  (their Eq. 34): `(I'_ij − I'_ji)/(f_ii − f_jj)` and `(I'_ab − I'_ba)/(f_aa − f_bb)`, contracted with
  the antisymmetrized `A`/`C` integrals, plus the `f·(∂I/∂f)` pieces. (This is `relax_I_RHF`'s
  `CCSD_T && dertype==1` block; `delta_I/delta_f_{IJ,AB}` are those `ΔX/Δf` responses.) The
  `|ΔX_mn|<1e-8` threshold there is a **degeneracy** guard, not an all-electron cancellation — my
  earlier "all-electron ⇒ ΔX=0 ⇒ −½S^X suffices" reading was **Scuseria's** separate formulation, not
  L–R's or pycc's.
- **The off-diagonal `Doo`/`Dvv` is extraneous — remove it entirely.** pycc's `t3_density` builds
  `⟨0|L₃[E_ij,T₃]|0⟩` off-diagonals that appear in **neither** paper's (T) 1-PDM: L–R (Eqs 17–19) and
  Vikings (Eq 65) give `Dov` + `diag(Doo)`/`diag(Dvv)` only. They are **not** a misplaced density block
  and **not** the `ΔX` numerator — just wrong (per PI, 2026-07-06). Empirically they were spurious in
  `D` (removing → unrelaxed dipole self-consistent to 5.7e-11) yet crudely load-bearing in the gradient
  (removing → 3.0e-5) **only because pycc carries no all-electron oo/vv `κ̄` yet**; supplying that (next
  bullet) is the real fix, and the off-diagonal plays no part in it.
- **The oo/vv dependent-pair `κ̄` reuses pycc's frozen-core machinery.** `κ̄_ij = (I'_ij − I'_ji)/(f_ii
  − f_jj)`, `κ̄_ab = (I'_ab − I'_ba)/(f_aa − f_bb)` — the **Lagrangian asymmetry**, i.e. exactly the
  frozen-core divide `Pco = (Ip[co,o] − Ip[o,co]ᵀ)/(eps_c − eps_i)` (`MPwfn._so_zvector` ~535 and the
  spatial `CCderiv.gradient`), **generalized from core↔active-occ to all oo `(i,j)` and vv `(a,b)`
  pairs**, numerator-gated `|ΔX|<1e-8`, added to `Drel` and coupled into the Z-vector RHS through the
  antisymmetrized ERI. `I'` is built from the *paper-correct* density (`Dov` + `diag` + `Γ` + (T)-Λ);
  the (T) enters `κ̄` only through `I'`, never through an explicit off-diagonal `Doo`/`Dvv`. This is the
  single surviving orbital term `κ̄_pq F^(1)_pq` of `ccsdt_orbital_response.tex`, with `κ̄` running over
  *all* pairs (ov = CPHF/Z-vector solve; oo/vv = these divides).
- Superseded framings: "ε diagonal is missing" (ε is present); "off-diagonal is a density block to
  fix"; "off-diagonal is the `ΔX` numerator misplaced in `D`"; "−½S^X is fine (Scuseria)". Diagonal `ε`
  and `Dov` (L–R Eq. 19 / Vikings Eq. 65) are correct.

**Fix:**
1. **Remove** the off-diagonal `Doo`/`Dvv` entirely (in `t3_density`); keep `diag(Doo)`/`diag(Dvv)`
   and `Dov`.
2. **Generalize the frozen-core `Pco`** to all oo and vv pairs — the `κ̄` divide above, numerator-gated,
   added to `Drel` and coupled into the Z-vector RHS. No new (T)-specific term beyond the correct
   density feeding `I'`.
3. Keep the ov Z-vector as is.
4. Add a `Tr(D·μ)`-vs-FD (unrelaxed-dipole / block-wise) assertion to the (T)-density test — the
   energy reconstruction (test_034) is blind to the off-diagonal blocks.

This reconciles the density (dipole), the gradient, both papers, and the PI's Psi4 implementation.
Diagnostic scripts (scratchpad): `block_decomp.py`, `unrelaxed_ctrl.py`, `dov_only.py`, `offdiag.py`.

**Plan.**
1. **Audit** the (T) densities/Λ for gradient-readiness: relaxed vs unrelaxed convention; that
   `S1/S2` reach the `CCderiv` Z-vector RHS; and the ε question above.
2. **Wire (T) into `CCderiv`**: assemble the (T) effective Q (ε on the diagonal) and G, add η/γ to
   the Z-vector RHS; reuse the CCSD Z-vector solve + −½Sˣ assembly unchanged.
3. **Assemble** `E^λ = D h^λ + Γ (pq|rs)^λ + P S^λ + Z·(CPHF RHS)` — identical to the CCSD path.

**Validation.**
- Primary oracle: psi4 `gradient('ccsd(t)')` (closed-shell, all-electron) — external analytic.
- Cross-check: FD of pycc's own CCSD(T) energy.
- Limit check: CCSD(T)→CCSD (drop triples) reproduces the CCSD gradient (test_076) to machine ε.
- Start H2O/6-31G and /cc-pVDZ (real virtual space), C1.

**Deferred:** frozen-core (dependent-pair canonical response), spin-orbital (SO==spatial keystone),
CCSD(T) Hessian/APT.

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

Tests: `test_046`–`test_050` (spatial HF), `test_062`–`test_066` (SO HF), `test_061` (MP2
gradient/relaxed density), `test_067` (polarizability), `test_068` (APT), `test_069` (Hessian),
`test_070` (2n+1 polarizability), `test_071` (HF velocity APT), `test_072` (MP2 AAT), `test_073`
(MP2 velocity APT), `test_074` (property facade). The 2n+1 APT/Hessian cross-checks live in
`test_068`/`test_069`.

## Appendix B: superseded early decisions

The original 2026-06-21 plan made several calls later reversed as the effort matured — recorded so
the record stays coherent (rationale is in git history):

- **"Spatial RHF only"** → **spin-orbital first**. The GSB orbital-response Lagrangian applies
  verbatim in the spin-orbital basis; the spatial spin-adapted path followed and is validated
  against it (the SO == spatial keystone).
- **"All-electron only first"** → **frozen core throughout** (both spin paths, every property).
- **"Explicit route; 2n+1 deferred"** → **both routes implemented**, the 2n+1 route as the
  efficient alternative *and* an independent cross-check of the explicit suite.
- **"MP2-specific, no abstraction"** → still MP2-specific; the shared-layer refactor remains
  deferred to the CC-gradient work (see Roadmap).
