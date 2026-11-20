# Open-shell CCSD(T) second derivatives: an ill-conditioned canonical perturbed-MO gauge

**Status (2026-11-19): open defect, diagnosed but not fixed.** Written as the starting point for the
work enabling non-canonical perturbed MOs for CCSD(T).

## The symptom

Triplet CH2 (3-B1) / cc-pVDZ, UHF reference, spin-orbital route, all-electron, compared against
CFOUR 2.1 (`REFERENCE=UHF, MULTIPLICITY=3, BASIS=PVDZ, SCF_CONV=13, CC_CONV=12, LINEQ_CONV=12`):

| property | CCSD | CCSD(T) |
|---|---|---|
| gradient vs CFOUR | 4.2e-11 | 4.6e-11 |
| relaxed dipole vs tight FD | 2.8e-12 | 2.9e-12 |
| **polarizability vs CFOUR** | **3.7e-11** | **3.9e-8** |
| **Hessian vs CFOUR** | **1.4e-10** | **5.1e-10** |
| Hessian asymmetry (internal check) | 2.1e-15 | **5.5e-11** |
| symmetry-forbidden `mu_y` (should be 0) | 1.4e-16 | **1.6e-13** |
| perturbed `mu_y` T/Lambda solves | converge, 1.3 s | **stagnate at ~5e-13, 300 iters** |

First derivatives are clean for both methods. Only CCSD(T) **second** derivatives are affected, and
only open shell: closed-shell CCSD(T) is clean across 25 CFOUR-anchored rows in the validation table.

## The mechanism

CH2/cc-pVDZ has an **accidental, cross-irrep near-degeneracy** in the alpha virtual space:

    VIRT 0.674466  A1
    VIRT 0.674501  B2      gap = 3.48e-05      (beta pair: 2.58e-04)

`A1 x B2 = B2`, which is the irrep where every symptom appears.
`CorrelatedDerivs._dependent_pairs` forms the canonical dependent-pair rotation

    P_mn = (I'_mn - I'_nm) / (eps_m - eps_n)

amplifying by `1/gap = 2.9e4` (and `1/gap^2` in `dP`). These rotations exist **only in the canonical
perturbed-MO gauge**, i.e. only for CCSD(T) (`perturbed_mo_gauge` is `'canonical'` for CCSD(T),
`'non-canonical'` otherwise), which is why plain UHF-CCSD is clean on every axis. The gauge reaches
the perturbed amplitude equations as well as the density: `df`/`deri` are canonicalised before the
iterative solve (`ccderiv.py`, `_perturbed_amplitudes` docstring).

## What was ruled out, with evidence

- **The UHF reference, the spin-orbital machinery, the Z-vector, the relaxed-density assembly.**
  UHF-CCSD exercises all of these identically on the same reference and is clean everywhere.
- **Convergence tolerance.** Identical results at `r_conv` 1e-11 and 1e-13.
- **DIIS.** Swept `max_diis` = 2, 4, 6, 8, 12, 16, 24: two solves fail at every setting and
  `alpha_yy` stays 3.3-4.4e-8 from CFOUR. (It does vary by 1.1e-8 across settings, which is itself
  proof the reported number comes from an unconverged solve.)
- **The gate criterion.** Gating `_dependent_pairs` on the numerator (Lee-Rendell) or on the ratio
  instead of the gap changes nothing: 3.7e-8 and 4.3e-8 respectively, still 2 unconverged solves.
  The unperturbed numerator for the near-degenerate pair is 3.9e-18 (roundoff) and a numerator gate
  *would* zero it, but `P^(x) = X^(x)/gap` has a genuinely nonzero perturbed numerator that no gate
  will catch. **This is not a gating bug.**

## The key result for the non-canonical work

Forcing the non-canonical gauge (`d._gauge_override = 'non-canonical'`) on the CCSD(T)
polarizability:

| gauge | alpha_yy vs CFOUR | tensor asymmetry | unconverged solves |
|---|---|---|---|
| canonical | +3.9e-8 | 4.4e-12 | 2 |
| **non-canonical** | -7.3e-3 | **5.7e-16** | **0** |

Without the dependent-pair rotations **the pathology vanishes entirely**: every solve converges and
the tensor is symmetric to machine precision. The value is wrong by 7.3e-3 only because the
non-canonical (T) route is not implemented -- the (T) contributions to `Doo`/`Dvv` are built from
their diagonal blocks alone, which is valid only in the canonical gauge and is the `O(N^7)` saving
that motivates that gauge in the first place (see the `perturbed_mo_gauge` docstring).

**So the target for a correct non-canonical-(T) implementation is: the conditioning of the
non-canonical row (0 unconverged, asymmetry ~1e-16) together with CFOUR's value.** On this system
that is `alpha_yy = 0.04267621340`; the full CFOUR tensor is regenerable with the recipe above.

## Reproducing

Geometry is CFOUR's own full-precision frame (r(CH)=1.0776 A, angle=133.93 deg):

    C  0.000000000000000  0.000000000000000  0.114593744572731
    H  0.000000000000000 -1.874001424240600 -0.682224040442087
    H  0.000000000000000  1.874001424240600 -0.682224040442087     (bohr, C1, no_com/no_reorient)

psi4's cc-pVDZ is CFOUR's PVDZ (SCF energies agree to 2.6e-12), so no GENBAS transcription is
needed. psi4's default UHF here is already the stable solution (`stability_analysis: follow`
returns the same energy), so no occupation pinning or stability following is required.

Note that CH2/cc-pVDZ was chosen as the *well-conditioned* open-shell case by the SCF-stability
criterion, and is -- but it happens to carry this accidental cross-irrep virtual near-degeneracy,
which makes it an unusually severe probe of the dependent-pair divide. Useful for this work;
worth remembering when interpreting its numbers elsewhere.

## Related

- `_dependent_pairs` gap-vs-numerator gate history: PR #229 (2026-08-06), and the
  `test_ccsdt_apt_dependent_pair_gap_gate_ethylene_twist` regression. The two cases are **opposite**
  failure modes -- ethylene-twist is a small-but-real numerator over an ordinary gap (needs the gap
  gate), CH2 is a roundoff numerator over a tiny gap (would need the numerator gate). No single
  criterion handles both, which is further reason the remedy is not a gate.
- `ccderiv.py` now reports perturbed solves that exhaust `maxiter` instead of returning unconverged
  amplitudes silently (`_unconverged`). That guard is what made this visible; keep it armed while
  working on the non-canonical route.
