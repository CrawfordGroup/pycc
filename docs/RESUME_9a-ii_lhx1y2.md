# Resume notes — phase 9a-ii, spatial spin-adapted symmetric response (LHX1Y2)

**Scratch handoff doc — delete before the 9a-ii PR.** Branch:
`feature/spatial-symmetric-response`. Everything here is pushed.

## Where we are

Building the spin-adapted (spatial) symmetric linear-response components in
`pycc/ccresponse.py`, each validated **term-by-term against the spin-orbital
`_*_spinorbital` sibling** on an RHF reference (SO-RHF == spatial-RHF).

Done & committed (spatial branch fills the `raise NotImplementedError` stubs):
- `LCX` (commit `ffe703e`)
- `LHX1Y1` (`f543949`)
- `LHX2Y2` (`7d5acf2`)

**`LHX1Y2`** — DONE & validated (all 8 sub-terms, incl. the voov ring L6_8). Full
term matches `_LHX1Y2_spinorbital` to ~1e-14 with distinct X,Y across all 3 axes;
existing response tests (027/036/055/056) still pass. Then only **`linresp_sym`**
remains.

## Validation methodology (proven, reusable)

Build the SAME RHF system in both bases; for a perturbation solve `X`; compare each
component (and the pseudoresponse, which confirms the amplitudes correspond) between
bases. Reference geometry: H2O, STO-3G, `symmetry c1`, all-electron. Perturbation
`MU_Z` at omega=0.1; for the term value use `X=Y=` that amplitude (X1=its singles,
Y2=its doubles). Final check must use **distinct** X,Y (e.g. mu_a at -w vs +w) since
X != Y in general.

### Testbed skeleton (run from repo root in the p4env conda env)
```python
import psi4, pycc, numpy as np
psi4.core.set_output_file('o.dat', False); psi4.set_memory('2 GB')
psi4.set_options({'scf_type':'pk','reference':'rhf','basis':'STO-3G','freeze_core':'false',
                  'e_convergence':1e-12,'d_convergence':1e-12,'r_convergence':1e-12})
psi4.geometry("\nO\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")
_, wfn = psi4.energy('SCF', return_wfn=True)
def build(basis):
    cc=pycc.CCwfn(wfn,frozen_core=False,orbital_basis=basis); cc.solve_cc(e_conv=1e-11,r_conv=1e-11)
    hbar=pycc.cchbar(cc); lam=pycc.cclambda(cc,hbar); lam.solve_lambda(e_conv=1e-11,r_conv=1e-11)
    dens=pycc.ccdensity(cc,lam,onlyone=True) if basis=='spinorbital' else pycc.ccdensity(cc,lam)
    return pycc.ccresponse(dens)
rsp=build('spatial'); rso=build('spinorbital')
X1s,X2s,_=rsp.solve_right(rsp.pertbar["MU_Z"],0.1)   # spatial X1,X2
# SO reference: rso.LHX1Y2([X1o,None],[None,X2o])
```
(Redirect script stdout to a file and `grep`/`sed` it — a concurrent process keeps
deleting the harness task-output files, and the CC iterations are very verbose.)

## SO reference sub-term values (X=Y=MU_Z, H2O/STO-3G)
`_LHX1Y2_spinorbital` (pycc/ccresponse.py ~line 700). Total = -0.002357277587.
```
L3  Zov.Y2     = -0.000641679832   (l1-part)
L4  Zvve.X1    =  0.000333141063   (l1-part)
L5  Zooe.X1    =  0.000437432973   (l1-part)
L1_7 Zooh.Y2   =  0.009906135095   (l2-part)
L2_9 Zvvh.Y2   = -0.005641438262   (l2-part)
L6_8 Zvoov.Y2  = -0.004343057400   (l2-part)  <-- OPEN (voov ring)
L10 Zoooo.Y2   = -0.002319053545   (l2-part)
L11 Zoovo.X1   = -0.000088757679   (l2-part)
```

## SOLVED spatial LHX1Y2 forms (ready to assemble)

Notation: `c=contract`, `L=H.L[o,o,v,v]`, `Lerr...`; `Hov,Hvovv,Hooov` are hbar blocks;
`HvL = 2*Hvovv - Hvovv.swapaxes(2,3)`, `HoL = 2*Hooov - Hooov.swapaxes(0,1)`;
`Y2L = 2*Y2 - Y2.swapaxes(2,3)`. `X1=X[0]`, `Y2=Y[1]`.

**l1-part** — `polar = 1.0 * c('ia,ia->', l1, tmp1)`:
```
# L3
Zov = c('mnef,me->nf', L, X1)
tmp1  = c('nf,nifa->ia', Zov, Y2L)
# L4
Zvv = -c('mnef,mnaf->ea', L, Y2)
tmp1 += c('ea,ie->ia', Zvv, X1)
# L5
Zoo = -c('mnef,inef->mi', L, Y2)
tmp1 += c('mi,ma->ia', Zoo, X1)
```

**l2-part** — `polar += 2.0 * c('ijab,ijab->', l2, tmp2)`:
```
# L1_7
Zoo = c('me,ie->mi', Hov, X1) + c('mnie,ne->mi', HoL, X1)
tmp2  = -0.5 * c('mi,mjab->ijab', Zoo, Y2)
# L2_9
Zvv = c('me,ma->ea', Hov, X1) - c('amef,mf->ea', HvL, X1)
tmp2 += -0.5 * c('ea,ijeb->ijab', Zvv, Y2)
# L10  (note: Zoooo built WITHOUT the SO 0.5; the outer 0.5 + factor-2 supplies it)
Zoooo = c('mnie,je->mnij', Hooov, X1)
tmp2 += 0.5 * c('mnij,mnab->ijab', Zoooo, Y2)
# L11  (Zoovo WITHOUT the SO 0.5)
Zoovo = c('amef,ijef->ijam', Hvovv, Y2)
tmp2 += -0.5 * c('ijam,mb->ijab', Zoovo, X1)
# L6_8  <-- OPEN (see below)
```
All 7 validated to ~1e-13 vs the SO sub-values above.

## SOLVED: L6_8, the voov ring (diagrams 6,8)

SO (`_LHX1Y2_spinorbital`, lines 723-729):
```
Zvoov  = c('anfe,if->anie', hbar.Hvovv, X1)     # +Hvovv part
Zvoov -= c('mnie,ma->anie', hbar.Hooov, X1)     # -Hooov part
tmp   += c('anie,njeb->ijab', Zvoov, Y2)        # diagrams 6 and 8
```
SO target -0.004343057400 — matched to 4.5e-14.

It's **one ph ring**, the `_r_T2_ccsd` three-term ring (ccwfn lines 777-779) with Y2 as
the external doubles and the **X1-dressed** ph intermediate. The key was that Zvoov uses
the *dressed* Hbar blocks, so ERI index symmetries don't apply — the intermediates must be
built from the native `hbar.Hvovv`/`hbar.Hooov` orderings, matching `build_Wmbej`/
`build_Wmbje`'s t1-part term-by-term (t1->X1, ERI->Hbar). Derivation: cast the SO ring
into the canonical `D[imae] W[mbej]` form (relabel via l2's i<->j,a<->b symmetry); the
transposed Y2 external collapses back to Y2 by its own permutational symmetry.

Final spatial form (in `LHX1Y2`, folded into `tmp2` which carries the outer 2.0; the ring
needs relative weight 0.5 because the SO l2 contraction has factor 1, not 2):
```
Wmbej =  c('jf,bmfe->mbej', X1, Hvovv) - c('nb,nmje->mbej', X1, Hooov)
Wmbje = -c('jf,bmef->mbje', X1, Hvovv) + c('nb,mnje->mbje', X1, Hooov)
ring  = c('imae,mbej->ijab', Y2 - Y2.swapaxes(2,3), Wmbej)
ring += c('imae,mbej->ijab', Y2, Wmbej + Wmbje.swapaxes(2,3))
ring += c('mjae,mbie->ijab', Y2, Wmbje)
tmp2 += 0.5 * ring
```
Resolved both prior open questions: there is no standalone "diagram 6" (the dia6==L11
coincidence was an artifact of the Hvovv-only split); and the missing half WAS the
`-Hooov*X1` dressing, which enters as the second term of each of Wmbej/Wmbje (mirroring
the `-t1*ERI` term in build_Wmbej/build_Wmbje).

## DONE: `linresp_sym` (spatial assembly)

Identical in structure to `_linresp_sym_spinorbital`: LCX/LHX1Y1/LHX2Y2/LHX1Y2 each
dispatch to their own (validated) spatial code. The ONLY basis-specific piece is the HXY
direct term. **Correction to the earlier note:** it spin-adapts to **2*L**, not L. Spin-
summing `<ij||ab> X_A[ia] X_B[jb]` over (sigma_i, sigma_j): the direct `<ij|ab>` survives
all 4 spin combos, the exchange `<ij|ba>` only the 2 with sigma_i==sigma_j, giving
`4<ij|ab> - 2<ij|ba> = 2*L`. (Found via a per-term polar decomposition: HXY was the lone
term off, by exactly 2x.)
```
polar += 2.0 * c('ijab,ia,jb->', self.H.L[o,o,v,v], X_A[0], X_B[0])
```
Validation (H2O/STO-3G, omega=0.1):
- `polarizability(0.1)`: spatial == spinorbital to 1.2e-12; spatial isotropic ==
  Psi4 CCSD to 9.9e-13.
- `optrot(0.1)`: spatial == spinorbital to 8.5e-14 (so the M / M* pertbar path through
  the same kernel also works).
- Existing response tests 027/036/055/056 still pass.

The `Avvoo` TODO did NOT break either property -- the perturbed X2 it yields gives
polarizability matching Psi4 to ~1e-12. Leave the TODO note but it is not blocking 9a-ii.

## Remaining
1. (Optional) cross-validate spatial symmetric vs the existing asymmetric `linresp_asym`,
   then deprecate the asymmetric linear-response path.
2. Separate/non-blocking: the `Avvoo` TODO in the spatial `pertbar`
   (ccresponse.py, `TODO(spin-adapted pertbar)`), flagged by TDC.
3. Flip the plan doc's 9b row to "done / PR #144" and add the 9a-ii rows inside this
   branch's commits before the PR.
