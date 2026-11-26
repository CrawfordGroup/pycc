# ARC task: DerivStore breakdown by quantity for the 3-chloro-1-butyne VCD calculation

## Why

A JCP reviewer asked that "the 2.3 TiB DerivStore footprint of the CISD/cc-pVTZ example deserves
a short breakdown by quantity."  We can report the *total* high-water but not the split, because
the runs logged only a running total and the store is deleted at job end.  We need the split.

The answer we want is one or two sentences for a reviewer response, of the form: *of the stored
tensors, X% are perturbed MO two-electron integrals over the 3N+6 perturbations and Y% are
per-atom AO derivative integrals.*  **Proportions are what matter**, not absolute sizes, so this
can be measured in a small basis and quoted as structure.

## The instrumentation (already written and pushed)

Branch `feat/derivstore-accounting` on `CrawfordGroup/pycc`, commit `cd9c7dc`, one commit on top
of `main`.  It adds `DerivStore.report()` in `pycc/derivatives.py`:

- walks the live HDF5 file and sums `nbytes` per quantity
- prints a table (quantity, GiB, % of total, dataset count) and returns the mapping
- runs automatically at `close()` when the environment variable `PYCC_DERIV_STORE_REPORT` is set
- wrapped in try/except so accounting can never kill a calculation

No bookkeeping during the run is needed: `DerivStore._name()` already encodes
`(quantity, perturbation, context)` in each dataset name.

Only three quantities are ever stored, so the table will have at most three rows:

| quantity | what it is | written at |
|---|---|---|
| `deri` | full MO-basis perturbed ERI, one `nmo^4` per perturbation, ctx = (ncore, canonical) | `cphf.py:633` |
| `deri_field` | the electric-field counterpart | `cphf.py:812` |
| `eri1` | AO first-derivative ERIs, a 3-Cartesian stack per atom, ctx = transform/blocks | `derivatives.py:1001` |

One subtlety already handled, worth not undoing: `_name()` sanitises `|` to `_`, so the quantity
is recovered by longest-prefix match against the known names.  A naive split on `_` folds
`deri_field` into `deri`.

## The calculation

Reproduce the existing CCSD/cc-pVDZ VCD run of 3-chloro-1-butyne with the reporting enabled.
The prior run is the reference point:

    data/3cby/cby_ccsd_vcd/ccsd/pvdz/      (job 1268470)
    driver:   data/3cby/cby_ccsd_vcd/scripts/vcd_run.py  (+ vcd_run.sbatch, submit_vcd.sh)
    host:     owl011, 12 cores, local NVMe scratch
    wall:     58 min  (22:00:31 -> 22:58:32, 24 Oct 2026, exit 0)
    store:    72.99 GiB high-water
    pycc:     main@a2e8f10     psi4 env: p4env_owl_hiam
    geometry: data/3cby/cby_ccsd_vcd/ccsd/pvdz/geom.txt   (10 atoms, C4H5Cl)
    basis:    cc-pVDZ, frozen core

So: **~1 hour on 12 cores, ~75 GiB of scratch.**  Use local NVMe scratch as before, not a network
filesystem.  Set `PYCC_DERIV_STORE_REPORT=1`; everything else unchanged.

The only change from the prior run should be the branch and that variable.  If `vcd_run.py` pins a
pycc commit, point it at `cd9c7dc` (or merge the branch into whatever is current -- it touches one
file and adds one method, so it should merge cleanly).

## What to send back

1. The `DerivStore contents by quantity` table printed at close.
2. The high-water line from the job wrapper, to confirm it reproduces ~73 GiB.
3. Wall time and exit status.

## Two things to watch

- **`report()` sums live dataset bytes; the high-water is a max over time.**  These should agree
  here, because the store only grows (the single `del` site is a partial-group rewrite), but if
  they differ materially say so rather than reconciling them -- that would itself be informative.
- **HDF5 does not reclaim space on delete**, so the file on disk can exceed the summed dataset
  bytes.  Compare proportions, not absolute file size.

## Optional, only if it is cheap

If the cc-pVDZ split is obtained easily, the same run at cc-pVTZ (~21 h, 12 cores, ~1.8 TiB --
job 1278292 is the reference) would confirm the proportions hold with basis size.  Not required:
cc-pVDZ at 118 functions and cc-pVTZ at 224 gave 72.99 and 1812.97 GiB, a ratio of 24.8 against
the nmo^4 expectation of 13.0, so the scaling is not clean and is itself worth a remark if the
larger run happens.
