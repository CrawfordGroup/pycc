# PyCC Refactor Plan — 2026-06

_Authored from the June 2026 design discussion. Supersedes the GPU/abstraction
"future direction" notes in [`CODE_REVIEW_2026-06.md`](CODE_REVIEW_2026-06.md)._

## Status / progress

_Last updated 2026-06-16. This section is the cross-machine source of truth for
where the refactor stands — the per-session memory used while authoring it is
machine-local, but this doc and `git log` travel with the repo._

| Phase | Status | Landed |
|---|---|---|
| 0 — tag/release snapshot | ✅ done | tag `v0.1.0` |
| 1 — strip einsums | ✅ done | PR #114 |
| 2 — device/precision manager | ✅ done | PR #115 (`DeviceManager` + `ContractionBackend` in `pycc/device.py`) |
| 2b — GPU ground-state real; RT complex via backend promotion | ✅ done | PR #116 |
| _(bonus)_ tighten RT-CCSD tests (`test_024`/`test_025`) | ✅ done | PR #117 (1e-10, real dynamics, reference verified bit-for-bit against pre-refactor code) |
| 3 — extract `Wavefunction` base | ✅ done | PR #119 (`pycc/wavefunction.py`; `ccwfn(Wavefunction)`; kwargs forwarding) |
| 4a — `MPwfn` | ✅ done | PR #120 (`pycc/mpwfn.py`; CC composes `mp = MPwfn.from_wavefunction(self)`, shared base) |
| 4b — `HFwfn` + derivative/CPHF engine | ✅ done | PRs #121–#128 — see HF-derivative arc below |
| _(refactor)_ full-space-`H` unification (frozen core as slice offset) | ✅ done | PR #123 (fixed `cctriples` absolute indices; `Local` fails fast for `nfzc>0`) |
| _(cleanup)_ conftest hygiene (`datadir` fixture hoist) | ✅ done | PR #124 |
| 5 — local frozen under `CCwfn`, CPU-only | 🟡 partial | `Local` raises for `nfzc>0` (#123); full freeze/marking deferred to the local rewrite |

### HF-derivative arc (Phase 4b) — complete

All MO-basis, validated against an external reference, with the nuclear CPHF response
cached and shared across the Hessian and the APTs/AATs.

| Increment | Landed | Validation |
|---|---|---|
| gradient + `Derivatives` provider | PR #121 (tol tightened #122) | vs `psi4.gradient('scf')`, ~1e-14 |
| CPHF solver + static polarizability | PR #125 | vs `psi4` analytic polarizability, ~4e-13 |
| nuclear CPHF → dipole derivatives / APTs | PR #126 | vs finite-difference SCF dipole, ~1e-8 |
| nuclear Hessian + persistent response cache | PR #127 | vs `psi4.hessian('scf')`, ~1e-12 |
| atomic axial tensors (AATs) via magnetic CPHF | PR #128 | vs DALTON SCF AATs (psi4numpy VCD ref), ~5e-9 |

`HFwfn`/`CPHF`/`Derivatives` now hold every SCF VCD ingredient (Hessian → normal modes,
APTs, AATs). The `Derivatives` and `CPHF` classes were written to depend only on
base-level state, so they are **promotable to the base** for MP2/CI/CC derivative work.

**To resume:** read this doc + `git log`. The structural refactor (Phases 1–4) is done;
the canonical CC spine plus `MPwfn` and `HFwfn` all sit on the `Wavefunction` base. Open
threads, in rough priority:
1. **Promote the derivative engine to the base** — move `Derivatives` (and the promotable
   `CPHF`) off `HFwfn` onto `Wavefunction` so MP2/CI/CC gradient/response code can reach
   them. (Next up — see discussion below / decisions log.)
2. **Assemble a full SCF VCD driver** from the now-complete pieces (frequencies + IR
   intensities + rotatory strengths); psi4numpy's `vcd.py` is the recipe.
3. **Phase 5** — formally mark `local.py`/`lccwfn.py` frozen/CPU-only; or extend the base
   to a `CIwfn`.

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

- **Promoting the derivative engine to the base** (the live question): `Derivatives` and
  `CPHF` already depend only on base-level state (basis/molecule/`C`/`H`/`o`/`v`/Fock
  diagonal), so they can move from `HFwfn` onto `Wavefunction`. To settle: do they become
  always-constructed base attributes or lazily built on first use; how does the 2-e
  derivative provider stay lazy/transient under the device manager; and what does an
  MP2/CI/CC gradient need beyond the HF pieces (a Z-vector / relaxed density on top of the
  same CPHF solver).
- _(resolved)_ ~~Internal structure of the `HFwfn` derivative engine~~ — landed as the
  `Derivatives` provider + `CPHF` solver (orbital Hessian, field/nuclear/magnetic RHS,
  shared nuclear-response cache); `HFwfn` delegates. See Phase 4b arc above.
- _(resolved)_ ~~Tag/release naming for Phase 0~~ — `v0.1.0`.
- Base public surface vs the remaining `self.ccwfn.<x>` reach-ins from CC sub-objects
  (works via inheritance today; a documented base contract is still desirable).
