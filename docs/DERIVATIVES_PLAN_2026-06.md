# PyCC post-Hartree–Fock analytic derivatives — design & status

_Design-of-record and status for the post-HF analytic-derivative-property effort: the MP2
correlation gradient and electric/nuclear second derivatives (polarizability, APT, Hessian) with
two independent implementations (explicit-derivative and 2n+1); and the coupled-cluster analytic
gradients (CCSD, CCSD(T)) built on the same Z-vector / relaxed-density machinery. Started 2026-06;
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

**Coupled-cluster gradients and dipoles.** Analytic nuclear gradients and relaxed electronic dipoles
through the `CCderiv` driver, reusing the MP2 Z-vector / relaxed-density assembly (`_lagrangian`, the
SCF orbital Hessian) with the CCSD/(T) densities and Λ:

| Method | spatial (closed-shell RHF) | spin-orbital (UHF) | frozen core |
|---|---|---|---|
| CCSD `dE/dX` (gradient) | ✅ | ✅ | ✅ |
| CCSD(T) `dE/dX` (gradient) | ✅ | ✅ | ✅ |
| CCSD, CCSD(T) `dE/dF` (relaxed dipole) | ✅ | ✅ | ✅ |
| CCSD `d²E/dF²` (dipole polarizability) | ✅ | ✅ | ✅ |

The relaxed dipole reuses the gradient's relaxed density: a static field leaves the AO basis fixed
(`S^F = ⟨pq\|rs⟩^F = 0`), so `mu = Tr(D_rel · mu_ints)` — the same `D_rel` (correlation density + `Pco`
+ (T) κ̄ + ov Z-vector), built by the shared `CCderiv._relaxed_density`/`_so_relaxed_density`, that the
gradient contracts with the skeleton integrals. `pycc.dipole(CCwfn)` returns the usual
nuclear/reference/correlation `PropertyComponents`. Validated against a tight finite difference of
pycc's own correlation energy — 5-point O(h⁴), the gradient ~1e-12 and the relaxed dipole a finite
field of `(E_CC − E_SCF)` ~1e-12 — **not** psi4's analytic derivatives (§4). Unlike CCSD — which reuses
the −½Sˣ ov-only Z-vector because
CCSD is invariant to occ–occ/virt–virt rotations — **CCSD(T) needs canonical perturbed orbitals for
the oo/vv blocks** (dependent-pair κ̄, even all-electron); see §5 and §7. The efficient Z-vector route
and the independent explicit-derivative route agree to machine precision for CCSD (the (T) explicit
route is pending — §6).

**Coupled-cluster polarizability.** The CCSD static dipole polarizability `α = −d²E/dF²`
(`CCderiv.polarizability`, `pycc.polarizability`) — the first CC *second*-derivative property — via
the asymmetric (2n+1) route: differentiate the relaxed-density gradient a second time in a field
(`α_ab = Tr(dD_rel·μ_a) + Tr(D_rel·rot(U^b,μ_a))`). Done for **CCSD, spatial RHF and spin-orbital UHF,
all-electron and frozen core**; only (T) is pending. It adds the perturbed amplitudes `dt/dF`,
perturbed Λ `dΛ/dF` (staying in `cclambda`'s `r_L`, no `Y1/Y2`), the perturbed HBAR (exact 5-point
stencil), and the perturbed Z-vector — all first-order responses, no second-order CPHF. FD-validated
(spatial `1.8e-12`; SO==spatial keystone `~1e-13`; open-shell UHF vs an energy finite field `~3e-10`);
see §8.

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
- **`CCderiv`** — the CCSD / CCSD(T) analytic gradient and relaxed dipole. Reuses `MPwfn._lagrangian`
  and the SCF orbital Hessian (through a persistent `HFwfn`/`CPHF`); takes the CC relaxed density + Λ
  from `ccdensity`, and for (T) the densities/Λ from `CCwfn.t3_density`. `_dependent_pairs` builds the
  canonical oo/vv κ̄ divides for (T) (§7). The relaxed density `D_rel` is built once by the shared
  `_relaxed_density()` / `_so_relaxed_density()` (the (T) κ̄ + `Pco` + ov Z-vector), then consumed by
  `gradient()` / `_so_gradient()` (skeleton integrals) and `relaxed_dipole()` (dipole integrals);
  `pycc.dipole(CCwfn)` routes the correlation block to the latter. The independent
  `_gradient_explicit()` cross-check is CCSD only so far.

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
- **(T) needs canonical perturbed orbitals — even all-electron.** Unlike CCSD (invariant to occ–occ /
  virt–virt rotations, so the −½Sˣ ov-only Z-vector suffices), the (T) energy is *not* oo/vv-invariant.
  The canonical perturbed orbitals then acquire dependent-pair rotations `κ̄_ij=(I'_ij−I'_ji)/(ε_i−ε_j)`,
  `κ̄_ab=(I'_ab−I'_ba)/(ε_a−ε_b)` — exactly the frozen-core core↔active `Pco` divide (§ above) generalized
  to *all* oo/vv pairs (added to the relaxed density, coupled into the ov Z-vector RHS via the
  antisymmetrized ERI). An early reading — "all-electron ⇒ ΔX=0 ⇒ −½Sˣ suffices" — was wrong: it
  conflated Lee–Rendell's *degeneracy* threshold with Scuseria's separate formulation. The correct
  picture is a single orbital term `κ̄_pq F^(1)_pq` with κ̄ over *all* pairs (ov = CPHF/Z-vector solve;
  oo/vv = the divides). FD-validated to 1.8e-12 (all-electron) / 1.9e-12 (frozen core). Full derivation:
  `docs/ccsdt_orbital_response.tex`; §7.
- **The off-diagonal (T) `Doo`/`Dvv` is not a density term.** `t3_density` had built `⟨0|L₃[E_ij,T₃]|0⟩`
  oo/vv off-diagonals present in neither Lee–Rendell nor Hald et al.; they corrupt `Tr(D·μ)` yet are
  invisible to the energy reconstruction (canonical F ⇒ only `diag(D)` enters `eone`). The (T) 1-PDM
  carries only `{Dov, diag(Doo), diag(Dvv)}`; the oo/vv orbital response is the κ̄ above, not a density.

## 6. Roadmap

- **CC gradients — done, extending.** CCSD *and* CCSD(T) analytic gradients are implemented for
  **spatial RHF and spin-orbital UHF, all-electron + frozen core**, via `CCderiv`, reusing the MP2
  Z-vector / relaxed-density assembly with the CCSD/(T) densities and Λ (details + validation in §7).
  **Next:** CCSD(T) Hessian/APT; and extending `_gradient_explicit` to carry the (T) dependent-pair
  (so the z-vector==explicit cross-check works for (T), as it does for CCSD). `CCderiv` reuses the MP2
  Lagrangian/CPHF primitives via delegation; the fuller "shared Lagrangian → Z-vector → density →
  gradient" layer refactor is still deferred (no premature abstraction).
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

**Why (T) needs canonical MOs.** The triples solve `⟨μ₃|[F,T₃]+[H,T₂]|HF⟩=0` (B-15) is non-iterative
*only because F is diagonal*, so `[F,T₃]` collapses to `D^abc = f_ii+f_jj+f_kk−f_aa−f_bb−f_cc` (A-5).
Non-canonical F ⇒ triples couple ⇒ iterative. The gradient consequence: the perturbed orbitals must
stay canonical in the oo/vv blocks, so those blocks carry an explicit **dependent-pair** rotation
instead of pycc's −½Sˣ non-canonical gauge.

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

**The (T) one-particle density** carries only `{Dov, diag(Doo), diag(Dvv)}` (Paper A Eqs 17–19,
Paper B Eq 65). The off-diagonal `Doo`/`Dvv` that `t3_density` originally built (`⟨0|L₃[E_ij,T₃]|0⟩`)
appears in **neither** paper and is removed: it corrupts `Tr(D·μ)` (spurious in the density) yet is
invisible to the energy reconstruction (canonical F ⇒ only `diag(D)` enters). The oo/vv orbital
response is the κ̄ above, **not** a density block.

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

**Deferred:** CCSD(T) Hessian/APT; extending `_gradient_explicit` to the (T) dependent-pair.

## 8. CC static dipole polarizability — design & status

**Status.** **DONE for CCSD** (2026-07-09) in `CCderiv.polarizability()` — **spatial RHF and
spin-orbital UHF, both all-electron and frozen core** — FD-validated (§8.4); only **(T)** remains
(Phase 4). First CC *second*-derivative property; built before the `CorrelatedDerivs` refactor because
it exercises the perturbed-relaxed-density machinery the shared base will own, without the
nuclear-skeleton complexity of Hessians/APTs. The asymmetric route (below) stays entirely within
`cclambda`'s `r_L` (no `Y1/Y2` response apparatus), with the perturbed HBAR obtained for free by an
exact stencil (below).

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
response, and `U^b` the first-order CPHF orbital response (`CPHF._full_U`). Only *first-order* responses
appear — no second-order CPHF `U^{xy}` (`perturbed_fock2`/`_full_U2`/`_xi` untouched).

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

Reused verbatim: `MPwfn._perturbed_lagrangian` (density-generic — the orbital-response part), the
perturbed-Z-vector and `dP_co` Sylvester logic, and the first-order `CPHF.perturbed_fock/perturbed_eri`
/`_full_U`. Lives on `CCderiv` for now (`_perturbed_relaxed_density(field)` + `polarizability`),
destined for `CorrelatedDerivs`.

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
### 8.4 Validation (CCSD complete — spatial + spin-orbital, all-electron + frozen core)

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

**Phasing:** (1) CCSD spatial all-electron **[done]**; (2) + frozen core **[done]**; (3) + spin-orbital
**[done]**; (4) + (T) — next.

**Validation (production).** Primary oracle = **finite difference of `CCderiv.relaxed_dipole`** in a
field (`÷h`, ~1e-12; Λ tight) — the only oracle that covers **CCSD(T)** and frozen core; plus the
**SO==spatial keystone**.
Debugging oracles (CCSD only, phase 1–3): `ccresponse.solve_right(ω=0)` X-vectors cross-check the
*unrelaxed* `dt/dF`, and `ccresponse.linresp/polarizability(0.0)` the *unrelaxed* CCSD polarizability —
they isolate the amplitude response from the orbital-relaxation delta. **`ccresponse` is never a code
dependency** (it omits orbital response and cannot do (T)); oracle only.

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

Tests: `test_046`–`test_050` (spatial HF), `test_062`–`test_066` (SO HF), `test_061` (MP2
gradient/relaxed density), `test_067` (polarizability), `test_068` (APT), `test_069` (Hessian),
`test_070` (2n+1 polarizability), `test_071` (HF velocity APT), `test_072` (MP2 AAT), `test_073`
(MP2 velocity APT), `test_074` (property facade). CC gradients: `test_076` (CCSD, spatial),
`test_078` (CCSD, spin-orbital), `test_077` (SO 2-PDM), `test_083` (CCSD(T), spatial + spin-orbital,
all-electron + frozen core), `test_034` (CCSD(T) density/dipole). The 2n+1 APT/Hessian cross-checks live in
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
- **"MP2-specific, no abstraction"** → still largely MP2-specific; `CCderiv` reuses the MP2
  Lagrangian/CPHF primitives by delegation, but the fuller shared-layer refactor remains deferred
  (see Roadmap).
- **"(T) all-electron reuses −½Sˣ, dependent-pair terms deferred to frozen core"** → **(T) needs
  canonical perturbed orbitals for the oo/vv blocks even all-electron** — the dependent-pair κ̄
  generalized from the frozen-core `Pco`. The early reading mistook Lee–Rendell's `|ΔX_mn|<1e-8`
  *degeneracy* guard for an all-electron cancellation (that cancellation is Scuseria's separate
  formulation). See §5 and §7.
