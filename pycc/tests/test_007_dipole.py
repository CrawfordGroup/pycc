"""
Test CCSD electric and magnetic dipole on H2 dimer (spatial RHF), and the
spin-orbital (UHF/ROHF) CC dipole from ccdensity.dipole
(docs/archive/ENHANCEMENT_PLAN_2026-06.md).
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import numpy as np
from ..data.molecules import *

# OH doublet at 1.83 bohr (verbatim for both codes). CFOUR unrelaxed CCSD-minus-SCF
# electronic dipole (z, cc-pVDZ, all-electron), generated with CFOUR PROP=FIRST_ORDER,
# DIFF_TYPE=UNRELAXED, CC_PROG=ECC, ABCDTYPE=AOBASIS, and FIXGEOM=ON (Cartesian input)
# so CFOUR keeps the input frame -- its SCF dipole then matches Psi4's exactly, and the
# value below is simply CFOUR's (CCSD - SCF) electronic dipole.
OH_BOHR = """
0 2
O 0.0 0.0 0.0
H 0.0 0.0 1.83
units bohr
no_com
no_reorient
symmetry c1
"""
CFOUR_UHF_DIPOLE_Z = -0.0434209710
CFOUR_ROHF_DIPOLE_Z = -0.0443023197


def _cc_dipole(wfn, **cckwargs):
    cc = pycc.CCwfn(wfn, **cckwargs)
    cc.solve_cc(e_conv=1e-11, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-11, r_conv=1e-11)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    return np.array(dens.dipole(cc.t1, cc.t2, lam.l1, lam.l2), dtype=float)


def test_so_dipole_equals_spatial_rhf(rhf_wfn):
    """Spin-orbital CC dipole (forced) reproduces the spatial CC dipole on a closed
    shell, isolating the spin-orbital density kernel."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    d_spatial = _cc_dipole(wfn)
    d_so = _cc_dipole(wfn, orbital_basis="spinorbital")
    assert np.max(np.abs(d_so - d_spatial)) < 1e-10


def test_uhf_dipole_vs_cfour(uhf_wfn):
    """Open-shell .OH cc-pVDZ: spin-orbital UHF CC dipole vs CFOUR (unrelaxed)."""
    wfn = uhf_wfn(OH_BOHR, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    d = _cc_dipole(wfn)
    assert abs(d[0]) < 1e-10 and abs(d[1]) < 1e-10
    assert abs(d[2] - CFOUR_UHF_DIPOLE_Z) < 1e-8


def test_rohf_dipole_vs_cfour(rohf_wfn):
    """Open-shell .OH cc-pVDZ: spin-orbital ROHF CC dipole vs CFOUR (unrelaxed,
    semicanonical)."""
    wfn = rohf_wfn(OH_BOHR, "cc-pVDZ", freeze_core="false",
                   e_convergence=1e-12, d_convergence=1e-12)
    d = _cc_dipole(wfn)
    assert abs(d[0]) < 1e-10 and abs(d[1]) < 1e-10
    assert abs(d[2] - CFOUR_ROHF_DIPOLE_Z) < 1e-8


def test_dipole_h2_2_cc_pvdz(rhf_wfn):
    """H4 cc-pVDZ"""
    e_conv = 1e-13
    r_conv = 1e-13

    wfn = rhf_wfn("(H2)_2", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)
    cc = pycc.ccwfn(wfn)
    ecc = cc.solve_cc(e_conv, r_conv)

    hbar = pycc.cchbar(cc)

    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)

    ccdensity = pycc.ccdensity(cc, cclambda)

    # no laser
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, None, magnetic = True)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, ecc).astype('complex128')
    t1, t2, l1, l2, phase = rtcc.extract_amps(y0)

    ref = np.array([0, 0, -0.0007395036977002]) # computed by removing SCF from original ref

    mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2)

    assert (abs(ref[0] - mu_x) < 1E-10)
    assert (abs(ref[1] - mu_y) < 1E-10)
    assert (abs(ref[2] - mu_z) < 1E-10)

    ref = [0, 0, -2.3037968376087573E-5]
    m_x, m_y, m_z = rtcc.dipole(t1, t2, l1, l2, magnetic = True)

    assert (abs(ref[0]*1.0j - m_x) < 1E-10)
    assert (abs(ref[1]*1.0j - m_y) < 1E-10)
    assert (abs(ref[2]*1.0j - m_z) < 1E-10)
