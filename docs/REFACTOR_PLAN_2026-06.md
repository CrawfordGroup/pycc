# PyCC Refactor Plan — 2026-06

_Status: scoping/design (not yet started). Authored from the June 2026 design
discussion. Supersedes the GPU/abstraction "future direction" notes in
[`CODE_REVIEW_2026-06.md`](CODE_REVIEW_2026-06.md)._

## Governing principle

**PyCC is a reference implementation**, meant to help construct and validate
production-level quantum-chemistry codes elsewhere (e.g. Psi4). It is deliberately
*not* a fully optimized code — hence Python. When a design tradeoff pits clarity
against performance or memory, **favor the simple, equation-shaped formulation.**

This tenet already explains existing choices (two explicit CC solvers rather than a
shared base; full canonical tensors driven by `contract`) and it adjudicates the new
decisions below — most visibly the choice of **MO-basis derivative integrals** over the
production AO-basis/relaxed-density approach.

## Goals

1. **Allow more wavefunction types than CC** (HF, MBPT/MP2, CI) despite the package
   name. Introduce a top-level `Wavefunction` base class from which specific methods
   inherit shared infrastructure. A **device/precision manager** is created in the base
   constructor so every method below it has uniform access — this is also the keystone
   that lets GPU support generalize across the package.
2. **Add HF-level MO-basis derivative properties** (gradients, Hessians, atomic polar
   tensors, atomic axial tensors) under an `HFwfn` class, with the underlying
   derivative-integral and CPHF machinery structured so MP2/CI/CC derivative code can
   reuse it later.
3. **Remove the `einsums` backend** (poor performance) while preserving the work via a
   tag, and **keep the local (LPNO/PAO) code in `main`** as a CPU-only, frozen subsystem
   pending its own future rewrite.

## Current state (as of 2026-06)

GPU support is a PyTorch-tensor swap-in behind `opt_einsum`, in three layers:

- **State origin** — `ccwfn.__init__`: the only place device/precision state is created
  (`device∈{CPU,GPU}`, `precision∈{SP,DP}` kwargs). In GPU mode it casts amplitudes/Fock
  to torch complex tensors on the compute device (`device1`) and keeps the big ERI/L on
  the CPU device (`device0`). **The GPU path is always complex128/64**, even for
  ground-state CC, because RT-CC shares the tensors — a 2× memory/flops tax.
- **Contraction dispatcher** — `utils.cc_contract`: `self.contract` moves any
  non-resident operand to `device1`, then calls `opt_einsum`. A second, orthogonal axis
  (`self._contract`) selects the C++ `einsums` backend for `ccwfn`/`cctriples` only.
- **Array-namespace helpers** — `utils.py`: twelve free functions
  (`zeros_like`/`zeros`/`clone`/`diag`/`real_zeros`/`dot`/`absolute`/`conj`/`solve`/
  `sqrt`/`reshape`/`concatenate`) dispatching on `isinstance(x, torch.Tensor)`.

GPU reaches only the **canonical ground-state + RT-CC spine**: `ccwfn`, `cchbar`,
`cclambda`, `ccdensity`, `cctriples`, `rt/rtcc`. **NumPy-only (the generalization gap):**
`lccwfn`, `local`, `cceom`, `ccresponse`, `hamiltonian`, `rt/integrators`, `rt/lasers`.

The remaining device-specific code in the GPU-aware modules is the placement/lifetime
smear: scattered `clone(ERI[...], device=ccwfn.device1)` prefetches paired with manual
`del`. Nothing *owns* where a tensor lives or when it is freed — the chief barrier to
generalizing GPU.

## Target architecture

### Class hierarchy

```
Wavefunction (base)
├── HFwfn         # container over the reference + derivative/CPHF engine
├── MPwfn         # MP2 (and the guess CC reuses)
├── CIwfn
└── CCwfn         # current ccwfn, minus the shared setup that moves up
```

### `Wavefunction` base — owns

- psi4 reference intake, canonical `C`, orbital-space/irrep setup (`o/v`,
  `no/nv/nmo/nfzc`, the MO-by-energy table), `eref`
- the **integral transform** (lifted from `ccwfn.__init__`): MO `F`/`ERI`/`L` plus the
  property integrals already in `hamiltonian.py` — `mu` (dipole), `m` (angular
  momentum), `p` (nabla), `Q` (quadrupole)
- orbital energies (`eps_occ`/`eps_vir`)
- the **device/precision manager**

**Spin-generality seam:** RHF only for now, but the base's *public* surface should expose
orbital spaces and coefficients through attributes/accessors that could later carry a
spin index, with spin-summation factors kept in method code, not the base. Do **not**
retrofit existing CC code — it stays RHF/bare-slice.

### Device/precision manager

A single object, created in the base constructor, that owns:

- the `device`/`precision` flags and the `device0`/`device1` handles
- the `cc_contract` callable
- the **cast policy** (replacing the hardcoded casts in `ccwfn.__init__`), including the
  **real-vs-complex decision** — fix the always-complex-on-GPU waste by choosing dtype
  from a per-wavefunction "needs complex?" flag (RT-CC sets it; ground-state CC does not)
- a **transient-tensor lifetime API** — "stage to compute device → contract → free" —
  replacing the scattered `clone(..., device1)` / `del` idiom. This is what makes GPU
  generalizable and is motivated most cleanly by the lazy derivative-integral provider
  (below).

### Integral layer — two lifetimes, all MO basis

Per the governing principle, derivative code uses **MO-basis derivative integrals** (not
the production AO/relaxed-density back-transformation), so the consumer reads like the
equations. The base integral object distinguishes by lifetime:

- **Eager / persistent:** MO `ERI`, `F`, `L`, property integrals. (Unchanged.)
- **One-electron derivative integrals** (Sˣ, hˣ, dipoleˣ, …), MO basis: small
  (`3·N_atom × nmo²`); the provider **may materialize all perturbations at once**.
- **Two-electron derivative integrals**, MO basis: the only memory-heavy class
  (`3·N_atom × nmo⁴`); the provider **must be lazy/transient** — yield one perturbation's
  MO-transformed derivative ERI (from `MintsHelper.deriv1/deriv2`), the consumer
  contracts, it is freed. Never `3·N_atom` resident.

Orbital relaxation enters separately through CPHF/Z-vector; the provider supplies only the
skeleton MO derivative integrals. The provider lives on the **base** so MP2/CI/CC
derivative code can reach it without reaching into `HFwfn`.

### Method classes

- **`CCwfn`** keeps method-specific state: denominators (`Dia`/`Dijab`, from the base's
  orbital energies), amplitudes, and the MP2 guess. **MO localization +
  `Local`/`lccwfn` construction stay here for now** (may be promoted to the base later if
  CI/MP2 need it — so write the re-localize-and-rebuild-`H` step as a self-contained
  operation over base-level state, not entangled with amplitudes). On the local path the
  base builds canonical-MO `H`; `CCwfn` re-localizes and rebuilds `H` as a CC-specific
  override.
- **`MPwfn`** factors the MP2 guess out of CC; CC reuses it. Cheapest validation of the
  base (a second real consumer alongside `CCwfn`).
- **`HFwfn`** is thin over the base (energy is `eref`) plus a new derivative/CPHF engine
  (envisioned as sub-objects, mirroring CC's `cchbar`/`cclambda`/… pattern). The HF
  *gradient* is CPHF-free (energy-weighted density); CPHF enters at Hessian/APT/AAT.
  CPHF lives under `HFwfn` for now but is written promotably — MP2/CI/CC gradients will
  need it too. **`HFwfn` is also the first greenfield, GPU-native consumer of the device
  manager** — the real stress test of the abstraction on fresh code.

## Phased plan

Each phase is behavior-preserving (the test suite is the contract) until new capability
is added at Phase 4. Run the **full `p4env` non-slow suite** green at every phase
boundary; because the local and einsums tests are not in CI, also run them locally.

| Phase | Work | Risk |
|---|---|---|
| 0 | Annotated tag/release at current HEAD — preserves active LPNO-CC **and** the einsums work | none |
| 1 | **Strip einsums.** Remove the import guard, `self.ec`/`einsums` setup + MP2 print, `build_tau`'s einsums branch, the `cctriples` opt_einsum reset, and `test_038–042`. Collapse `_contract`→`contract` (one backend axis remains). | low (opt-in feature, default off) |
| 2 | **Device/precision manager** object: device/precision flags, `device0/1`, `cc_contract`, cast policy + real-vs-complex fix, and the transient-tensor lifetime API. Migrate `ccwfn`'s hardcoded casts and the `clone(...,device1)`/`del` sites onto it. | medium, behavior-preserving |
| 3 | **Extract the lean `Wavefunction` base** from `ccwfn` (reference/orbital setup, integral transform, orbital energies, manager). `ccwfn`→`CCwfn` inherits. Define the two-lifetime integral layer incl. the lazy 2-e derivative-integral provider *interface* (implementation lands in Phase 4). | medium, mechanical but broad (touches sub-object coupling) |
| 4 | **New methods.** `MPwfn` (factor MP2 out) as a cheap second consumer; then `HFwfn` + derivative/CPHF engine consuming the lazy provider — the GPU-native validation of the manager. | additive |
| 5 | **Local stays under `CCwfn`**, CPU-only, frozen pending rewrite; mark `local.py`/`lccwfn.py` accordingly; keep their tests green. | low |

### Why this order

- Einsums-out (Phase 1) is small, contained, and *removes a backend axis*, simplifying
  the manager before it is built.
- The manager (Phase 2) is the keystone for both Goal 1 and GPU generalization; building
  it standalone de-risks the base extraction.
- The base (Phase 3) is discovered by extraction against two real consumers (`CCwfn`, and
  `MPwfn` in Phase 4), not designed speculatively.

## Decisions log

- **PyCC is a reference implementation** → simple/MO-basis formulations over
  memory/performance optimization (governing principle).
- **MO-basis derivative integrals**, not AO/relaxed-density — accept higher memory for
  validation-friendly, equation-shaped code. But **never materialize all 2-e derivative
  integrals at once** (lazy/transient, per perturbation); 1-e derivative integrals may be
  stored eagerly.
- **Derivative-integral provider on the base**, not in `HFwfn` (so MP2/CI/CC can reach
  it). CPHF under `HFwfn` for now, written promotably.
- **Integral transform moves up** from `ccwfn.__init__` into `Wavefunction.__init__`.
- **MO localization stays in `CCwfn`** for now (promotable later); base builds
  canonical-MO `H`, `CCwfn` rebuilds on the local path.
- **GPU dtype**: fix the always-complex waste — choose real vs complex from a per-wfn
  flag, owned by the manager.
- **Spin**: RHF only now; keep the base API spin-promotable; don't retrofit CC.
- **einsums**: removed (tag to preserve); single contraction backend axis going forward.
- **Local CC**: kept in `main`, CPU-only, frozen pending a future rewrite.

## Open questions

- Exact public surface the sub-objects depend on (the base's documented contract,
  replacing `self.ccwfn.<x>` reach-ins).
- Internal structure of the `HFwfn` derivative engine (which pieces become sub-objects;
  where the Z-vector/CPHF solver sits when it is promoted to shared use).
- Tag/release naming for Phase 0.
