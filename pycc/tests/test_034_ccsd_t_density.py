"""
Test CCSD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import numpy as np
from ..data.molecules import *
from ..cctriples import t_vikings, t_vikings_inverted, t_tjl

def test_ccsd_t_h2o(rhf_wfn):
    """H2O cc-pVDZ"""
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    geom = """
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
"""
    wfn = rhf_wfn(geom, "STO-3G", freeze_core="false")

    etot = psi4.energy('CCSD(T)')
    ecc_psi4 = psi4.variable('CCSD(T) CORRELATION ENERGY')

    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True)
    ecc = cc.solve_cc(e_conv, r_conv, maxiter, max_diis=0)
    assert (abs(ecc_psi4 - ecc) < 1e-11)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis=0)
    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc_density = ccdensity.compute_energy()
    eone = ccdensity.eone
    etwo = ccdensity.etwo

    lambda_psi4 = -0.069084521221746
    eone_psi4 = 0.104463374777302
    etwo_psi4 = -0.175243393781829
    assert (abs(lambda_psi4 - lcc) < 1e-11)
    assert (abs(eone_psi4 - eone) < 1e-11)
    assert (abs(etwo_psi4 - etwo) < 1e-11)

    psi4.core.clean()

    wfn = rhf_wfn(geom, "cc-pVDZ", freeze_core="false")

    etot = psi4.energy('CCSD(T)')
    ecc_psi4 = psi4.variable('CCSD(T) CORRELATION ENERGY')

    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True)
    ecc = cc.solve_cc(e_conv, r_conv, maxiter)
    assert (abs(ecc_psi4 - ecc) < 1e-11)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc_density = ccdensity.compute_energy()
    eone = ccdensity.eone
    etwo = ccdensity.etwo

    lambda_psi4 = -0.227199866607450
    eone_psi4 = 0.251210862963227
    etwo_psi4 = -0.479006477929931
    assert (abs(lambda_psi4 - lcc) < 1e-11)
    assert (abs(eone_psi4 - eone) < 1e-11)
    assert (abs(etwo_psi4 - etwo) < 1e-11)


# Frozen Fock-perturbation FD oracle for the (T) unrelaxed correlation dipole mu_z (H2O/6-31G,
# frame locked; closed-shell, platform-reproducible).  Validated (once) against a 5-point FD of
# E_corr[CCSD(T)] w.r.t. V = -F*mu_z (fixed orbitals), agreeing to 1.9e-13 (see _findiff_t_dipole_z,
# the regeneration recipe).
T_DIPOLE_Z_REF = 0.08826237289720

_T_DIPOLE_GEOM = """
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
no_com
no_reorient
"""


def _findiff_t_dipole_z(wfn, h=1e-4):
    """Regeneration recipe for T_DIPOLE_Z_REF (not run in the tests): 5-point central finite
    difference of the CCSD(T) correlation energy w.r.t. a static Fock perturbation V = -s*mu_z
    (fixed orbitals)."""
    def E(s):
        c = pycc.ccwfn(wfn, model='ccsd(t)')
        c.H.F = np.asarray(c.H.F) - s * np.asarray(c.H.mu[2])
        return c.solve_cc(1e-12, 1e-12, 100)
    return -(-E(2 * h) + 8 * E(h) - 8 * E(-h) + E(-2 * h)) / (12 * h)


def test_ccsd_t_dipole(rhf_wfn):
    """The analytic (T) unrelaxed correlation dipole mu_z matches the finite-difference-validated
    frozen reference (H2O/6-31G) -- the guard that the (T) 1-PDM carries only the physical blocks
    {Dov, diag(Doo), diag(Dvv)}.  The off-diagonal Doo/Dvv appears in neither Lee-Rendell nor Hald
    et al.; it would corrupt Tr(D.mu) yet stay invisible to the eone reconstruction above (the
    canonical Fock is diagonal, so only diag(D) enters).  6-31G, not STO-3G: STO-3G has too few
    same-symmetry virtuals for mu_z to probe an off-diagonal Dvv."""
    wfn = rhf_wfn(_T_DIPOLE_GEOM, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True)
    cc.solve_cc(1e-12, 1e-12, 100)
    cclambda = pycc.cclambda(cc, pycc.cchbar(cc))
    cclambda.solve_lambda(1e-12, 1e-12)
    ccdensity = pycc.ccdensity(cc, cclambda)
    ccdensity.compute_energy()
    mu_analytic = ccdensity.dipole(cc.t1, cc.t2, cclambda.l1, cclambda.l2)[2]
    # frozen ref is the analytic dipole itself (matched the Fock-perturbation FD to 1.9e-13); the
    # deterministic recomputation reproduces it to ~3e-15, so 1e-11 is a tight, robust guard.
    assert abs(mu_analytic - T_DIPOLE_Z_REF) < 1e-11, mu_analytic

    psi4.core.clean()
