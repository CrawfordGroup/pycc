# Validation plan for the CC AAT implementation

Written 2026-10-10. Equation numbers refer to `cc_aat_derivation.tex`.

## Aim

Validate the implemented CC AAT without finite-difference wave-function
overlaps. The numerical approach used for the MP2 and CID AATs
(Shumberger and Crawford, JCTC 2024, `ct4c00747.pdf`) needs overlaps of excited
determinants in different orbital bases and MO phase tracking, and is expensive.
The tests below replace it with exact-limit comparisons and with identities that
hold exactly in a finite basis.

Tests 1, 2, 4 and 5 rest on standard properties of CC theory. The sum rule in
test 3 was derived in discussion and has not been checked numerically.

**Status, 2026-10-22** (branch `feat/cc-aat`, CCSD, both orbital bases):

| Test | Status |
|---|---|
| 1. Two-electron vs CISD | **PASS**, 2.0e-17 (STO-3G) and 3.2e-14 (6-31G) |
| 2. Non-interacting fragments | **PASS in the limit**, deviation falls as R^-2 to 3.8e-6 relative at 2000 bohr |
| 3. Translational sum rule | **PASS for CC**, 4e-12; independently confirmed on validated CISD, 1.1e-12; MP2 open |
| 4. Orbital-convention invariance | **PASS**, 3.8e-13 (nuclear knob) and 5.4e-13 (magnetic knob) |
| 5. Amplitude block vs response | **PASS in magnitude** to 3e-7 extrapolated; relative sign is a convention |

The spin-orbital route adds a sixth check, pycc's usual closed-shell keystone: the spin-orbital and
spin-adapted paths agree to **1.6e-14** on water/STO-3G, and the spin-orbital path independently
passes both gauge invariances (1.7e-14 magnetic, 4.3e-14 nuclear) and the sum rule (7.5e-14).

Test 4 found and fixed the one implementation error so far: the orbital-orbital
block is a **commutator**, not the `M_R^T M_H - M_H^T M_R` form that the nuclear
side's symmetry suggests. The two half-derivative overlaps have opposite
transpose symmetry. Orthonormality gives `<d_x phi_p|phi_q> = -<phi_p|d_x phi_q>`,
so `M_R = U^R + half_S^T` is antisymmetric (measured 5.9e-15), while for the
magnetic perturbation `C^H = i C U` with real orbitals makes the same condition
read `U_pq = U_qp`, so `M_H` is symmetric once the `i` is stripped (measured
6.7e-16). Hence

    <d_R phi_p|d_H phi_q> - <d_H phi_p|d_R phi_q> = -i ( M_R M_H - M_H M_R )

Before the fix both knobs failed at 3.5e-2 and 5.9e-2.

## The quantity under test

The working equation, Eq. (35):

$$
2 I^\lambda_{\alpha\beta} =
\sum_\mu \left( \lambda^R_\mu t^H_\mu - \lambda^H_\mu t^R_\mu \right)
+ \sum_{pq} \left[ D^R_{pq} \langle \phi_p | \partial_H \phi_q \rangle
                 - D^H_{pq} \langle \phi_p | \partial_R \phi_q \rangle \right]
+ \sum_{pq} D_{pq} \left[ \langle \partial_R \phi_p | \partial_H \phi_q \rangle
                        - \langle \partial_H \phi_p | \partial_R \phi_q \rangle \right]
$$

with $R = R_{\lambda\alpha}$, $H = H_\beta$, and $D^x$ the full
amplitude-derivative one-particle density. The three sums are referred to below
as the amplitude-amplitude, amplitude-orbital and orbital-orbital blocks.

The code works in real arithmetic with the magnetic dipole operator entered as
$\mathrm{Im}(m_\beta)$, so the factor of $i$ is carried analytically.

## Test 1: two-electron systems against the CISD code

For two electrons CCSD is exact, so the CCSD AAT must equal the normalized CISD
AAT element by element, to convergence. CCD must likewise equal CID, because
$T_2^2$ vanishes.

- Systems: H$_2$, HeH$^+$ or H$_3^+$ at a distorted $C_1$ geometry, so that the
  tensors are non-trivial.
- What it tests: the whole pipeline at once (perturbed $T$ and $\Lambda$, all
  three blocks, the orbital terms and the prefactors).
- It also tests the normalization result. The CISD side needs the explicit
  intermediate-normalization correction, Eq. (10) of the CISD manuscript, and
  the CC side must reproduce the same number with no correction.
- It fixes the overall sign convention that comes from writing the operator as
  $\mathrm{Im}(m_\beta)$.
- What it cannot see: any term that vanishes for two electrons, including the
  nonlinear terms in the perturbed amplitude equations.
- New code needed: none.

**Result (PASS).** H$_3^+$ at a distorted $C_1$ geometry (1.700 / 1.430 / 0.170
bohr offsets), all-electron, CCSD and CISD both to 1e-13. Correlation energies
agree to 6.5e-16 (STO-3G) and 5.0e-15 (6-31G), confirming exactness, and the
correlation AATs agree element by element: max deviation **2.0e-17** (STO-3G)
and **3.2e-14** (6-31G), ratio exactly 1. This fixes the overall sign from
entering the operator as Im$(m_\beta)$ and the prefactor, neither of which the
invariance tests can see. Note a diatomic is useless here: HeH$^+$ gives a
correlation AAT of 1e-19 (vanishing by symmetry), so fragment geometries must be
non-linear.

## Test 2: non-interacting two-electron fragments

For two distant two-electron fragments, supermolecule CCSD is still exact and
CISD is not. The CCSD AAT for an atom in fragment A must equal the monomer
value for A, which test 1 ties to monomer CISD.

Why it holds. $T = T_A + T_B$ and $\Lambda = \Lambda_A + \Lambda_B$. For a
displacement of an atom in A:

- $T^R = T^R_A$ only, and likewise $\Lambda^R$ and $U^R$. B's electrons do not
  feel A's nucleus and none of B's basis functions move.
- $T^H = T^H_A + T^H_B$, and likewise $\Lambda^H$ and $U^H$. The field
  perturbation is a sum of one-electron operators on each fragment, and the
  perturbed equations are connected.

Every term in the working equation then contracts something localized on A with
something additive, and the B parts drop out. For example,
$\lambda^R \cdot t^H = \lambda^R_A \cdot (t^H_A + t^H_B) = \lambda^R_A \cdot t^H_A$.

- What it tests: size intensivity of the AAT.
- What it does not test: the antisymmetrization specifically. The CC bra is
  $1 + \Lambda_A + \Lambda_B$, with no $\Lambda_A \Lambda_B$ term, so a single
  overlap has no cross term for the antisymmetrization to remove.
- Practical points:
  - The AAT depends on the gauge origin. Fragment A must sit at the same
    coordinates relative to the origin in the monomer and supermolecule runs.
  - Use non-identical fragments, or non-canonical perturbed orbitals. Two
    identical distant monomers give degenerate, delocalized canonical orbitals.
- New code needed: none.

**Result (PASS in the limit).** Two neutral non-identical fragments, A = water
at a distorted $C_1$ geometry and B = H$_2$, 6-31G, all-electron, `no_com` and
`no_reorient` so the gauge origin is common and fragment A sits at identical
absolute coordinates in both runs. Monomer scale max$|I|$ = 1.59e-2. Fragment-A
deviation against separation:

| R (bohr) | deviation | relative | ratio vs previous | R^-2 |
|---|---|---|---|---|
| 40 | 1.5368e-04 | 9.66e-3 | | |
| 120 | 1.6968e-05 | 1.07e-3 | 9.057 | 9.000 |
| 400 | 1.5229e-06 | 9.58e-5 | 11.14 | 11.11 |
| 1000 | 2.4348e-07 | 1.53e-5 | 6.255 | 6.250 |
| 2000 | 6.0855e-08 | 3.83e-6 | 4.001 | 4.000 |
| 5000 | 9.7349e-09 | 6.12e-7 | 6.252 | 6.250 |
| 10000 | 2.4340e-09 | 1.53e-7 | 4.000 | 4.000 |
| 20000 | 6.0797e-10 | 3.82e-8 | 4.004 | 4.000 |
| 50000 | 9.4065e-11 | 5.91e-9 | 6.249 | 6.250 |
| 100000 | 2.2221e-11 | 1.40e-9 | 4.233 | 4.000 |

Clean R^-2 across 40 to 100000 bohr: a 2500-fold range of separation and seven
decades of decay, with no numerical floor reached. Only the last step departs
from the power law (4.233 against 4.000), consistent with the onset of one: the
magnetic integrals on the distant fragment grow like R while the deviation falls
like R^-2, so precision erodes from both ends. **Control:** the identical setup on the
validated CISD code gives 1.152e-3 at *both* 40 and 120 bohr, identical to four
figures, which is its size-consistency error and confirms the test discriminates.

The R^-2 residual is attributed (hypothesis, not verified) to the common gauge
origin: with the origin on A, the operator on B is
$m_B \approx \tfrac12 (\mathbf R \times \mathbf p_B) + \dots$, whose first term
grows like R, so B's magnetically perturbed density carries an induced dipole
growing like R and produces a field $\sim R/R^3 = R^{-2}$ at A. Charge-independent,
which matches: a cationic fragment A (H$_3^+$) gives the same magnitude and the
same power. Confirm by replacing H$_2$ with He, whose magnetic response is much
smaller; the deviation should drop sharply at fixed R.

**TO DISCUSS WITH THE PI.** The R^-2 explanation above is an untested hypothesis
and the PI has asked to discuss it before it is relied on (2026-10-22). What is
established is the measured power law and the CISD control; the attribution to
the common gauge origin is not.

## Test 3: translational sum rule (finite-basis form)

Displacing every nucleus by the same amount translates the wave function
rigidly, basis functions included. Summing the AAT over atoms then gives

$$
\sum_\lambda I^\lambda_{\alpha\beta}
= \frac{i}{2\hbar} \frac{\partial \langle P_\alpha \rangle}{\partial H_\beta},
$$

where $\langle P_\alpha \rangle$ is the expectation value of the electronic
canonical momentum.

How to evaluate the right-hand side. Treat $\langle P_\alpha \rangle$ as an
unrelaxed expectation value and differentiate it with respect to the field,
using the same field-dependent wave function as in the AAT:

$$
\frac{\partial \langle P_\alpha \rangle}{\partial H_\beta}
= \langle \partial_H \tilde\Psi | P_\alpha | \Psi \rangle
+ \langle \tilde\Psi | P_\alpha | \partial_H \Psi \rangle ,
$$

where $\partial_H$ includes both the amplitude response and the orbital
response $U^H$. The field is the perturbation that relaxes the orbitals.
$P_\alpha$ is never added to the Hamiltonian, so there is no orbital response
to it and no Z-vector. In terms of quantities the AAT code already has:

$$
\sum_\lambda 2 I^\lambda_{\alpha\beta}
= \sum_{pq} D^H_{pq} \langle \phi_p | \nabla_\alpha | \phi_q \rangle
+ \sum_{pq} D_{pq} \left[ \nabla_\alpha , \kappa^H \right]_{pq} ,
$$

with $\nabla_\alpha$ the MO-basis gradient-integral matrix. It needs $D$,
$D^H$, $U^H$ and the gradient integrals, and no nuclear-displacement response.

Intermediate checks that come with it, each a test of an individual perturbed
quantity:

- $\sum_\lambda t^R = 0$
- $\sum_\lambda \lambda^R = 0$
- $\sum_\lambda U^R = 0$, hence $\sum_\lambda D^R = 0$
- $\sum_\lambda \langle \phi_p | \partial_{R_{\lambda\alpha}} \phi_q \rangle
  = -\langle \phi_p | \nabla_\alpha | \phi_q \rangle$

Conditions:

- Every basis function depends on position only through $\mathbf r - \mathbf R_A$
  for the atom it sits on. This is not specific to Gaussians. It fails for
  ghost or floating functions held fixed in space, and for London orbitals.
- Field-independent basis functions and a common gauge origin.

Do not use the dipole-moment form of the sum rule,
$\sum_\lambda I^\lambda_{\alpha\beta} = (i/4\hbar) \sum_\gamma
\epsilon_{\alpha\beta\gamma} \mu^{\rm el}_\gamma$, as the test. Its last step,
$\langle \mathbf p \rangle = -e \langle \mathbf A \rangle$, uses an exact-state
property and is not exact in a finite basis. (This form was quoted from memory
and should be checked against Stephens before it is cited.)

- What it tests: all $3N$ displacement responses, every block, and the
  nonlinear terms in the perturbed amplitude equations. It is the most
  stringent test for molecules with more than two electrons.
- New code needed: the momentum-expectation derivative above.

**Result (PASS for CC).** Distorted $C_1$ water, 6-31G, all-electron, CCSD. The
gradient matrix is $\nabla_\alpha = \mathrm{Re}(-i\,$`H.p`$)$, the same `i`-strip
convention as `H.m`. Every nonzero element agrees to 1e-14 absolute, 4e-12
relative:

| $\alpha\,\beta$ | LHS $= \sum_A 2I$ | RHS |
|---|---|---|
| 0 2 | 4.071383730953e-03 | 4.071383730968e-03 |
| 1 2 | 7.534722378737e-05 | 7.534722377955e-05 |
| 2 0 | 9.415947827673e-03 | 9.415947827705e-03 |
| 2 1 | 2.652791998996e-04 | 2.652791999534e-04 |

The remaining five elements vanish on both sides at 1e-16. This is strong
standalone evidence: the two sides share no machinery. The left needs all $3N$
displacement responses through the four sectors, the right needs only $D$, $D^H$,
$U^H$ and the gradient integrals with no nuclear response at all. It also fixes
the $[\nabla_\alpha, \kappa^H]$ convention, with $\kappa^H = U^H$ and a plus sign.

**Intermediate checks (all PASS)**, same system, every Cartesian:

| quantity | $\max|\sum_A \cdot|$ |
|---|---|
| $\sum_A \partial t_1$, $\partial t_2$ | 2.5e-14, 1.0e-14 |
| $\sum_A \partial\lambda_1$, $\partial\lambda_2$ | 7.7e-14, 2.2e-14 |
| $\sum_A U^R$ | 2.8e-15 |
| $\sum_A D^R$ | 7.7e-14 |
| $\sum_A \langle\phi_p|\partial_R\phi_q\rangle + \nabla_\alpha$ | 2.8e-15 (scale 1.33) |

**The CISD confirmation step works (2026-10-22), once the right density is used.** It is the
step the suggested order puts first, and it succeeds: the rule is verified against a code whose AAT
is externally validated, independently of the code it is testing.

| system | max\|LHS - RHS\| | ratios |
|---|---|---|
| H$_3^+$, 2 electrons | 4.74e-15 | 1, 1 |
| H$_2$O, 10 electrons | 1.12e-12 | 1, 1.00000001, 1, 1 |

**The density for an imaginary perturbation is the ANTISYMMETRIZED one-sided object.** With
$|\partial_H\Psi\rangle = i|X\rangle$ for real $X$ and real orbitals, and the one-sided
$T_{pq} = \langle\Psi|p^\dagger q|X\rangle$,

    <Psi|p+q|d_H Psi> = +i T_pq,    <d_H Psi|p+q|Psi> = -i T_qp

so $D^H_{pq} = i(T_{pq} - T_{qp})$: with the `i` stripped, `T - T^T`. The same algebra for a real
perturbation gives the **symmetrized** `T + T^T`, which is what `_perturbed_cisd_corr_opdm` builds
(it sets `dD[v,o] = dD[o,v].T`) and is correct for the nuclear and field perturbations the
Hessian/APT path feeds it. Verified at two electrons, where CISD == CCSD exactly: against CC's own
$D^H$, `T - T^T` agrees to **1.3e-13**, while the one-sided `T` alone is off by 3.3e-3, the
symmetrized form by 6.7e-3, and `_perturbed_cisd_corr_opdm` by 6.9e-3. Independently, CC's magnetic
$D^H$ is itself exactly antisymmetric, `max|D + D^T|` = 1.3e-14.

In `cideriv.py` the one-sided $T$ is `_build_Dtilde`, which is what the CISD AAT already contracts;
only this RHS needed it.

**An earlier version of this section concluded the rule "does not transfer to CISD" and suspected
intermediate normalization. That was wrong on both counts** -- the RHS had been given the
symmetrized density. Normalization is not implicated anywhere: at two electrons the normalized
unperturbed density agrees with CC's to 1.6e-14 and $U^H$ agrees bit for bit.

**MP2 still fails**, with ratios 0.917 and -2.306 (H$_3^+$) and 1.047, 1.046, -0.021, -0.274
(H$_2$O), using the antisymmetrized `R - R^T`. **TO RESOLVE SOON (PI 2026-10-23)** -- the
sharp diagnostic is the two-electron limit, which is what cracked the CISD case: there MP2 is
not exact, so compare instead against its own apyib-validated AAT and look for the failure in
this RHS rather than in `mpderiv.py`.

Caveat on the MP2 result: the folded-density formulation exposes no `D`/`D^H`
accessor, so both expressions were replicated from `_correlation_aat` and carry
oo/vv blocks only. MP2 has no singles, so the ov/vo blocks that `_build_Dtilde`
carries for CISD should vanish, which argues the reconstruction is complete --
but that is an argument, not a check, and the MP2 AAT is externally validated
against apyib. **Treat the MP2 discrepancy as open and most likely in this RHS,
not in `mpderiv.py`**, on the precedent that the CISD "failure" was exactly that.

## Test 4: invariance to the perturbed-orbital convention

The AAT must not depend on how the occupied-occupied and virtual-virtual blocks
of $U^x$ are chosen. Running with canonical and with non-canonical perturbed
orbitals moves contributions between the amplitude and orbital blocks, so
agreement checks that the two are treated consistently.

- A related check: frozen core against all-electron in a case where the core is
  truly inert.
- What it tests: the whole pipeline, including the nonlinear amplitude terms.
- New code needed: none, if both conventions are available.

**Result (PASS).** Water/STO-3G, CCSD. The two knobs are independent and were
varied separately: the nuclear perturbed-MO convention (`perturbed_mo_gauge`,
via `_gauge_override`) and the magnetic `orbital_gauge`. All four combinations
now give an identical tensor, deviations **3.8e-13** (nuclear) and **5.4e-13**
(magnetic). This test is what exposed the commutator error above; the frozen-core
against all-electron check has not been run.

## Test 5: amplitude block against the existing response code

With orbitals frozen and an electric-dipole operator in place of the nuclear
displacement, the amplitude-amplitude block,
$\lambda^x \cdot t^y - \lambda^y \cdot t^x$, is the zero-frequency slope of the
CC linear response function for the electric and magnetic dipoles. PyCC already
computes that response function for optical rotation. Two response calculations
at $\pm\omega$ for small $\omega$ give the slope numerically.

- What it tests: the perturbed $\Lambda$ equations, including their quadratic
  terms, against trusted code. Amplitude block only.
- Caveat: the equivalence was shown for the sum-over-states form and for the
  amplitude block. Whether the EOM-like and linear-response choices of
  $\lambda^x$ agree in the implementation is exactly what this test will show,
  so a discrepancy here needs interpretation before it is called a bug.
- New code needed: a small driver.

**Result (PASS in magnitude, sign is a convention).** Distorted $C_1$ water,
6-31G, CCSD. LHS is the amplitude block from BARE operators (orbitals frozen,
`zero_eri`/`zero_L` into the perturbed amplitude and Lambda solvers at
$\omega = 0$); RHS is $G'(\omega)/\omega$, since $G'$ is already the odd-in-$\omega$
part so its slope at zero is the same as $[LR(+\omega) - LR(-\omega)]/2\omega$.

| $\omega$ | ratios RHS/LHS, four nonzero elements |
|---|---|
| 0.020 | -1.001654, -0.993444, -1.005468, -1.006134 |
| 0.010 | -1.000413, -0.998367, -1.001362, -1.001528 |
| 0.005 | -1.000103, -0.999592, -1.000340, -1.000382 |

Errors fall by 4 per halving of $\omega$, the expected $O(\omega^2)$ for a central
slope. Richardson on the last two rows gives -1.0000000, -1.0000003, -0.9999993,
-1.0000000: unit magnitude to 3e-7. **So the perturbed $\Lambda$ equations,
including their quadratic terms, agree with the trusted response code, and the
EOM-like and linear-response choices of $\lambda^x$ do coincide in the
implementation.**

The overall sign is read as a convention mismatch, not a defect:
`linear_response(a, b)` evaluates $\mathrm{Tr}(\partial_{b_j} D \, a_i)$, so the
perturbation is $b = m$ and the observable is $a = \mu$, the reverse of the
ordering $\lambda^\mu t^m - \lambda^m t^\mu$ used here, and reversing an
antisymmetric block flips its sign exactly. It does not bear on the AAT's
absolute sign, which test 1 pins against externally validated CISD.

## One-off confirmation, not a regression test

The two-particle term is never coded, because it cancels by the pair-exchange
symmetry $\Gamma_{prqs} = \Gamma_{rpsq}$. That symmetry has been confirmed for
PyCC's spin-orbital CC density (`_so_full_twopdm()`): about $10^{-16}$ for
H$_2$O$_2$/STO-3G and OH/STO-3G. The same density is not bra-ket symmetric
($5\times10^{-4}$ and $5\times10^{-5}$), and the spatial `Doovv` block alone
violates pair exchange at $10^{-2}$, so the spin-orbital or fully assembled
density is the one to use if the term is ever evaluated explicitly.

## Dropped

- "The tensor is purely imaginary": not checkable in real arithmetic.
- "The $T^R T^H$ contributions cancel": those terms never appear in the code.

## Coverage

| Test | Whole pipeline | Nonlinear amplitude terms | New code | Status |
|---|---|---|---|---|
| 1. Two-electron vs CISD | yes | no | none | PASS 3.2e-14 |
| 2. Non-interacting fragments | yes | no | none | PASS in the limit, R^-2 |
| 3. Translational sum rule | yes | yes | momentum-expectation derivative | PASS 4e-12, CISD-confirmed |
| 4. Orbital-convention invariance | yes | yes | none, if both conventions exist | PASS 5.4e-13 |
| 5. Amplitude block vs response | amplitude block only | yes | small driver | PASS 3e-7 |

## Suggested order

1. Check the sum rule of test 3 with the existing CISD code.
2. Test 1, then test 2, on the CC code.
3. The intermediate zero sums of test 3, then the full sum rule.
4. Tests 4 and 5.

Tests 1 and 2 pin down the formula and its normalization exactly but cannot see
terms that vanish for two electrons. Tests 3 to 5 cover those. If all five
pass, a finite-difference overlap calculation should add nothing; keep it as a
last resort for a specific discrepancy.
