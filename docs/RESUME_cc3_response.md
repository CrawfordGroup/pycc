# Resume notes — spin-orbital CC3 linear response

**Scratch handoff doc — delete before the PR.** Branch:
`feature/spinorbital-cc3-response`. Phase A is committed (`4a05e33`) and pushed.

## Scope & decisions (fixed)
- **Spin-orbital ONLY.** Spatial CC3 response is deferred (needs heavy spin-adaptation
  equation-checking); `solve_right` raises `NotImplementedError` for spatial + CC3.
- **Validate term-by-term vs socc** (`~/src/socc`, importable in the `p4env` conda env).
  Reference setup: `socc/socc/tests/test_008_cc3_polar.py` (H2O/STO-3G CC3, omega, uses
  `store_triples=True`). socc's `linresp` prints each CC3 component (L2CX3, L3CX3,
  L3CX1T3, L3CX2T2, L2HX1Y3, L3HX1Y2, L3HX1Y1T2).
- Builds on the merged SO CC3 Lambda (PR #149): gives l1/l2; l3 rebuilt on the fly via
  `l3_ijk_so`.
- House style: loop-over-(i,j,k)/(a,b,c), `if model=='CC3'` then branch on orbital_basis.

CC3 response has TWO sides: **A = amplitude solver (perturbed X3)** and
**B = response-function terms**.

## Phase A — DONE & validated (committed `4a05e33`)
Perturbed triples X3 coupling in the perturbed-wavefunction solver.
- `pycc/cctriples.py`: `t3c_abc_so` (virtual-batched spin-orbital connected T3; companion
  to the existing `t3c_ijk_so`). Port of socc `t3_abc`.
- `pycc/ccresponse.py`:
  - SO `pertbar.Avvoo` CC3 term: `+= <kc> t3_ijkabc` (loop ijk, `t3c_ijk_so`). Port of
    socc pertbar (CC3 branch, ~line 1230).
  - `_cc3_response_setup_spinorbital(pertbar)`: builds CC3 W's (`_so_build_*_CC3`) +
    once-only `Yoovo`/`Yovvv` (ground-T3·ERI) and `Zovoo`/`Zvvvo` (Aov·t2). Port of socc
    `CC3_noniter` (IJK).
  - `_cc3_iter_spinorbital(pertbar, omega, ints)`: per-iteration `z1`/`z2`. Rebuilds X3
    (IJK loop: [A,T3]_Avv + dressed-t3 + [H,X2]; ABC loop: [A,T3]_Aoo) with the
    **omega-shifted denominator**, folds into r_X1/r_X2. Port of socc `CC3_iter`
    (IJK+ABC, socc ccresponse ~860-946).
  - `solve_right`: CC3 branch (build setup once, add z1/z2 each iter) + the spatial guard.
- **Validation (passed):** SO CC3 perturbed-wfn pseudoresponse (MU_Z, omega=0.1,
  H2O/STO-3G) == socc to **2.1e-13**. CCSD response unaffected (test_027/055/056 pass,
  7 tests).
- Validation script (deleted; reproduce): run socc and pycc in one process off the same
  SCF wfn; socc side `from socc.ccresponse import pertbar as socc_pertbar` →
  `socc_pertbar(sc.H.mu[2], sc)` → `sresp.solve_right(pb, 0.1)` returns `[_,...], pseudo`;
  pycc side `presp.pertbar['MU_Z']` → `solve_right(...)` returns `(X1,X2,pseudo)`. Compare
  the two pseudo scalars. socc ccwfn: `socc.ccwfn(wfn, model='CC3')`,
  `solve_cc(1e-12,1e-12,store_triples=False)`, then cchbar/cclambda/ccresponse.

## Phase B — IN PROGRESS

### Step 1 — DONE & validated (uncommitted as of this note)
`_cc3_build_X3_spinorbital(pertbar, omega, ints)` in `pycc/ccresponse.py`: builds the
full converged perturbed triples `X3[ijkabc]` from converged X1/X2, replicating the IJK
(+ ABC) block construction inside `_cc3_iter_spinorbital` (correct-by-construction — the
same blocks the validated Phase A loop folds into z1/z2). Note: pycc does NOT store
t3/l3/X3 (no `store_triples`); this separate post-convergence builder is the agreed
near-term approach (vs teaching the hot `_cc3_iter` loop to also store).
Validated vs socc stored `self.X3` (store_triples=True), H2O/STO-3G, omega=0.1: pseudo
6e-16, X1 8e-15, X3 norm/`<X3|X3>` ~13 digits, `1/4 X3.<jk||bc>` (singles space) 7e-16.

### CRITICAL: pycc-vs-socc spin-block PHASE convention (affects ALL Phase B validation)
pycc and socc differ by a benign relative **phase convention** in certain spin blocks:
~32/1600 X2 elements and ~864/64000 X3 elements are EXACTLY NEGATED (norms match to ~13
digits). pycc's `pertbar` carries the matching sign, so this cancels out of every
physical full-contraction quantity (pseudoresponse, energies). CONSEQUENCE: validate
Phase B **term-by-term by SCALAR value** against socc's printed component breakdown — the
response terms are all full contractions (physical scalars, convention-invariant).
Do NOT element-wise-compare intermediate l3/t3/X3 tensors against socc; they differ by
phase. (socc has no X3_ijk/X3_abc builders — the doc below speculated wrongly; socc
builds X3 holistically in CC3_iter_full.)

## Phase B — TODO (response-function CC3 terms)
The 7 CC3 terms add to the symmetric response in `_linresp_sym_spinorbital`. socc
reference: `linresp` CC3 block (socc ccresponse **281-412**) and the term methods:
- `LCX_CC3` (socc **544-583**) → L2CX3, L3CX3, L3CX1T3, L3CX2T2.
- `L2HX1Y3_CC3` (socc **586-607**).
- `L3HX1Y2_CC3` (socc **610-641**).
- `L3HX1Y1T2_CC3` (socc **644-677**).

### CRITICAL design note (read before coding Phase B)
socc has two paths: `store_triples=True` (lines 282-297) uses the dedicated term methods
above and **is the complete, validated path** (test_008 uses `store_triples=True`). The
`store_triples=False` on-the-fly block (lines 298-405) appears **INCOMPLETE** — it only
accumulates L2CX3, L3CX3, L2HX1Y3 and leaves L3CX1T3/L3CX2T2/L3HX1Y2/L3HX1Y1T2 at 0.0. So
**do NOT port the on-the-fly linresp block** — port the `store_triples=True` term methods.

Recommended approach:
1. **Build & store the full X3 once** after `solve_right` converges (from converged
   X1/X2), for each perturbation. socc builds full X3 in `CC3_iter_full` (X3 construction
   at socc ~954-989; note `self.X3 = X3` at ~992). Either port that, or use socc's
   `X3_ijk`/`X3_abc` (socc cctriples **325**/**349**) accumulated into a stored array.
   On the tiny validation systems X3 storage is cheap; a no-store on-the-fly version can
   come later. NEW SO builders likely needed: `X3_ijk_so`, `X3_abc_so` in
   `pycc/cctriples.py` (port socc X3_ijk/X3_abc; they take the pertbar + Zvvvo/Zovoo from
   `CC3_linresp_intermediates`, socc **1022-1040**).
2. **Thread X3** through `solve_right` return and `polarizability`/`optrot`: the X tuples
   become `[X1, X2, X3]` (today they are `(X1, X2)`; pycc `solve_right` returns
   `(X1, X2, pseudo)` and the drivers build `[X1.copy(), X2.copy()]`).
3. **Port the 4 term methods** (SO, antisymmetrized) needing l3/t3 (rebuilt per ijk on the
   fly via `l3_ijk_so`/`t3c_ijk_so`) and the stored X3.
4. **Wire into `_linresp_sym_spinorbital`** under `if self.ccwfn.model == 'CC3'`:
   `LCX_CC3(A,X_B)+LCX_CC3(B,X_A)` + `L2HX1Y3(X_A,X_B)+L2HX1Y3(X_B,X_A)` +
   `L3HX1Y2(X_A,X_B)+L3HX1Y2(X_B,X_A)` + `L3HX1Y1T2(X_A,X_B)` (no swap on the last — see
   socc 296). NOTE pycc `_linresp_sym_spinorbital` returns a SCALAR (sums terms); socc
   returns a component array. Add the CC3 terms to the scalar; for term-by-term debugging,
   temporarily print each or compare totals against socc's per-component print.

### Phase B validation
- Full polarizability tensor vs socc (`store_triples=True`), H2O/STO-3G, omega=0.0 and
  0.1; socc test_008 also has a Dalton reference at omega=0.0:
  diag ~ [0.061593757, 7.0661684, 3.0604929].
- Term-by-term vs socc's printed component breakdown (strongest check).
- optrot shares the kernel; validate vs socc optrot too (socc test_007 is CCSD optrot —
  for CC3 optrot generate from socc directly).
- Add `pycc/tests/test_0xx_cc3_response.py` (SO), socc-derived references hardcoded.

## Remaining after Phase B
- Update `docs/ENHANCEMENT_PLAN_2026-06.md` (CC3 response increment note).
- Decide whether to keep store-X3 or add a no-store on-the-fly response path later.
- Spatial CC3 response stays deferred (guarded).
