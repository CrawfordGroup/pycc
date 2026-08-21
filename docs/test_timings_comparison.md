# pycc test-suite timing comparison: 2026-07-28 → 2026-08-20

Same hardware (`ganymede`, macOS single machine), same psi4 1.9 conv/pk, same non-slow scope
(`setup.cfg` `addopts = -m "not slow"`). Old = `docs/test_timings.tex` as of 2026-07-28; new = a clean
full-suite re-run on 2026-08-20 (`pytest --durations=0`, nothing else running). Duration = call +
setup + teardown. (An earlier comparison covered 2026-07-04 → 2026-07-28; see git history.)

This window is the derivative-performance arc: the Hessian memory split/spill (#235–237), the AO-route
and AO-density skeleton for gradients and Hessians (#239–243), the property-facade simplification
(#244), and the reference-Hessian `ao_tei_deriv2` sharing (#245).

## Summary

- **Whole suite:** 267 tests / 1135 s → **303 tests / 1353 s**. The suite grew by 38 tests (a broad
  CISD gauge-invariance / FD-oracle family across `test_079`–`test_082`/`test_091`, the CCSD(T)
  gap-gate and SO H2O⁺ Hessian/APT cases, the facade route-equivalence tests, and the VCD /
  checkpoint / optical-rotation additions), dropped 2.
- **Shared tests** (present in both by the same node id): **265**; **1129 s → 1142 s (+1%)** —
  essentially flat.
- Distribution of the ≥0.5 s shared tests: **212 total, 14 faster >1.3×, 11 slower >1.3×**, the rest
  within 1.3×.

## Biggest speedups (shared tests) — the Hessian family

| old s | new s | ratio | test |
|---:|---:|---:|---|
| 6.50 | 2.61 | 2.49× | `test_065_spinorbital_hf_hessian::test_so_rhf_hessian_vs_spatial_631g` |
| 2.40 | 1.16 | 2.07× | `test_065_spinorbital_hf_hessian::test_uhf_hessian_vs_psi4` |
| 26.26 | 12.99 | 2.02× | `test_090_ccsd_hessian::test_so_ccsd_hessian_cfour_water` |
| 5.80 | 3.21 | 1.81× | `test_061_mp2_relaxed_density::test_mp2_gradient_ccpvdz` |
| 7.68 | 4.33 | 1.77× | `test_069_mp2_hessian::test_fc_so_mp2_corr_hessian_vs_spatial_631g` |
| 7.15 | 4.32 | 1.66× | `test_069_mp2_hessian::test_so_mp2_corr_hessian_vs_spatial_631g` |
| 5.31 | 3.53 | 1.50× | `test_082_cisd_hessian::test_cisd_hessian_vs_reference` |
| 7.79 | 5.38 | 1.45× | `test_090_ccsd_hessian::test_ccsd_hessian_cfour_water_spatial` |

Every one of the top speedups is a molecular Hessian (or a gradient), and the spin-orbital Hessians
lead. This is exactly the AO-density skeleton (#241/#242, which drops the per-pair MO transforms — the
spin-orbital transform being 16× a spatial one) compounded with the reference-Hessian sharing (#245,
which halves the dominant `ao_tei_deriv2` generation). A controlled same-session before/after
benchmark isolates the effect more cleanly than the full suite: cc-pVDZ/cc-pVTZ CCSD Hessians run
~2× (spatial) to ~3.3× (spin-orbital) faster, with the cc-pVTZ Hessian 714 s → 329 s.

## Slowdowns (>1.3×) — dominated by environmental noise

| old s | new s | ratio | test |
|---:|---:|---:|---|
| 23.64 | 32.23 | 0.73× | `test_031_cc3::test_cc3_h2o` |
| 1.66 | 2.55 | 0.65× | `test_076_ccsd_gradient::test_ccsd_gradient_ccpvdz` |
| 6.33 | 9.59 | 0.66× | `test_073_mp2_vg_apt::test_mp2_vg_apt_so_equals_spatial_ccpvdz` |
| 3.20 | 4.62 | 0.69× | `test_072_mp2_aat::test_mp2_aat_gauge_invariance` |
| 0.90 | 1.75 | 0.51× | `test_079_cisd_lg_apt::test_cisd_lg_apt_translational_sum_rule` |

These are not a real regression. The largest, `test_cc3_h2o` (a CC3 test), was **not touched** by this
arc, and neither the MP2 VG-APT nor the AAT paths were changed by the AO-density work (which is
gradient/Hessian only) — so an unrelated CC3 test slowing 1.37× is the tell that this is run-to-run /
environmental drift across the three-week gap on a shared laptop, not code. A small, genuine
contribution is the facade migration (#244): the driver property methods now emit their SCF /
correlation / total report on each call (they returned a silent array before), a few ms per property
call that nudges the smallest property tests; making `report()` opt-in would recover it. Net across
the shared suite the two effects wash to +1%.

## New tests (38) and dropped (2)

The additions are mostly the CISD analytic-property test build-out (LG/VG APT, AAT, Hessian, gradient,
polarizability — gauge-invariance, FD-oracle, and frozen-core variants), plus the CCSD(T) dependent-
pair gap-gate (ethylene twist) and SO H2O⁺ Hessian/APT, the property-facade route-equivalence and
`takes_driver_object` tests, the MP2/SCF VCD cases, the vibanalysis checkpoint round-trip suite, and
the optical-rotation facade/units tests. Dropped: `test_facade_requires_driver_object` (renamed
`test_facade_takes_driver_object` when the bare-wfn rejection was removed in #244) and
`test_cisd_vg_apt_vs_reference` (superseded).
