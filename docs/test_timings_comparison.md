# pycc test-suite timing comparison: 2026-07-04 → 2026-07-28

Same hardware (`ganymede`, macOS single machine), same psi4 1.9 conv/pk, same non-slow scope.
Old = `docs/test_timings.tex` as of 2026-07-04; new = a clean full-suite re-run on 2026-07-28
(`pytest --durations=0`, nothing else running). Duration = call + setup + teardown.

## Summary

- **Whole suite:** 228 tests / 1409 s  →  **267 tests / 1135 s**. The suite grew (the CC APT/Hessian/
  response families `test_084`–`test_094`) yet the total dropped.
- **Shared tests** (present in both by the same node id): **200**; **1009 s → 636 s (−37%)**.
- Distribution of the ≥0.5 s shared tests: **53 faster >1.3×, 103 within 1.3×, 2 slower >1.3×**.
- **28 original tests dropped/renamed**, mostly the `route='explicit'` MP2 tests expunged during the
  deriv-object migration, the whole `test_070_mp2_2n1_polarizability` file removed, and two CC3
  response tests (`test_cc3_polarizability`, `test_cc3_optical_rotation_zz`) reclassified `@slow`.

## Biggest speedups (shared tests)

| old s | new s | ratio | test |
|---:|---:|---:|---|
| 15.28 | 0.70 | 0.05× | `test_074_property_facade::test_facade_route_option` (explicit-route expunge) |
| 40.38 | 2.59 | 0.06× | `test_067::test_fc_so_mp2_corr_polarizability_vs_spatial_ccpvdz` |
| 6.90 | 0.39 | 0.06× | `test_068::test_mp2_corr_apt_dipfd_631g` (now frozen oracle, no live FD) |
| 30.01 | 2.59 | 0.09× | `test_067::test_so_mp2_corr_polarizability_vs_spatial_ccpvdz` |
| 109.89 | 14.84 | 0.14× | `test_068::test_so_mp2_corr_apt_vs_spatial_ccpvdz` |

Drivers: the deriv-object / 2n+1 migration (big algorithmic wins on the MP2 SO-vs-spatial
derivative keystones), the `route='explicit'` expunge, and the freezing of disposable FD oracles
(the MP2 `*_dipfd_*`/`*_gradfd_*` tests no longer execute finite difference).

## Residual slowdowns (>1.3×)

| old s | new s | ratio | test |
|---:|---:|---:|---|
| 3.06 | 4.88 | 1.59× | `test_058_spatial_optrot::test_spatial_rhf_optrot_vs_spinorbital` |
| 1.23 | 1.66 | 1.35× | `test_076_ccsd_gradient::test_ccsd_gradient_ccpvdz` |

Both are small in absolute terms and pass. The SO-optrot one is the only plausibly-real cost (the
old doc predates the ccresponse reformulation, PR #214); the other is within run-to-run noise. An
earlier run showed spurious ~2–3× slowdowns on the low-numbered RT/PNO++ tests — those were
measurement contention (concurrent work during the run), not regressions, and are gone in this
clean run.

## FD-column audit

The old doc marked **11** tests as executing finite difference; the refreshed doc marks **6**. The
MP2 `*_dipfd_*` / `*_gradfd_*` and `test_mp2_nuclear_t2_response` tests were refactored to assert
against frozen constants (their `_findiff`/`_dipfd`/`_gradfd` helpers are now un-called regeneration
recipes), so they no longer execute FD. The genuine live-FD set is:

- `test_060_field_energies::test_ccsd_field_findiff`, `::test_cc3_field_findiff` (finite-field energy)
- `test_059_cc3_response::test_cc3_polarizability_zz_frzc` (finite-field FC CC3)
- `test_061_mp2_relaxed_density::test_fc_mp2_gradient_vs_energy_fd_631g` (energy-FD gradient)
- `test_086_ccsdt_unrelaxed_dipole::test_so_ccsdt_unrelaxed_dipole_vs_findiff` (5-point dipole FD)
- `test_091_cisd_polarizability::test_cisd_polarizability_vs_dipole_fd_full_631g` (7-point field FD)

## Notes

- Single run per data point; sub-~2 s tests carry run-to-run noise around 1.3×.
- The DerivStore disk cache (on by default in the suite now) does **not** slow a single Hessian:
  measured 2–5% *faster* store-ON vs store-OFF on `test_089`/`test_090`. Its wins are peak memory
  and cross-property reuse (gradient→APT→Hessian on one driver), which a lone Hessian test does not
  exercise; the ~90% cost is `mo_tei_deriv2`, untouched.
