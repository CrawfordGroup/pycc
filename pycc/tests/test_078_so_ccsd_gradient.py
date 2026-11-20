"""
Spin-orbital (UHF) CCSD analytic nuclear gradient -- pycc.gradient(ccwfn) / CCderiv._so_gradient.

The spin-orbital gradient mirrors the spatial one (Z-vector route) with the antisymmetrized
generalized-Fock Lagrangian, an inline orbital Hessian, and the spin-orbital derivative integrals.
Validated by: the SO == spatial keystone (a closed-shell RHF driven through the spin-orbital path
must reproduce the spatial closed-shell gradient), psi4's analytic UHF-CCSD gradient, and -- for
frozen core, which psi4's CC gradients do not support -- the spatial frozen-core gradient.
"""

import numpy as np
import psi4
import pycc
from .conftest import record


E_CONV = R_CONV = 1e-11


def _so_ccwfn(geom, reference="uhf", freeze_core="false", basis="STO-3G"):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'reference': reference,
                      'freeze_core': freeze_core, 'e_convergence': 1e-12, 'd_convergence': 1e-12,
                      'r_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc = pycc.ccwfn(wfn, orbital_basis='spinorbital')
    cc.solve_cc(E_CONV, R_CONV, 300)
    return cc, wfn


# closed-shell water, fixed frame -- for the SO == spatial keystone and the frozen-core check
H2O = "O 0.0 0.0 0.118\nH 0.0 0.758 -0.472\nH 0.0 -0.758 -0.472\nunits angstrom\nsymmetry c1\nno_com\nno_reorient"
# NH2 radical (2-B1, bent), fixed frame -- the open-shell UHF reference.  Deliberately NOT a
# linear/degenerate radical like OH (2-Pi): OH's degenerate pi orbitals leave the UHF solution (and
# which pi holds the unpaired electron) numerically non-reproducible across platforms/BLAS, and the
# near-degenerate orbital rotation makes the CPHF/Z-vector Hessian near-singular.  NH2's SOMO is
# non-degenerate, so the reference is unique and the orbital response is well-conditioned.
NH2 = "0 2\nN\nH 1 1.02\nH 1 1.02 2 103.0\nsymmetry c1\nno_com\nno_reorient"
# OH radical (2-Pi), run in its computational point group -- NO "symmetry c1", so psi4 uses C2v.
# This is the *degenerate* open-shell case: the two pi orbitals split into different irreps (b1/b2),
# which keeps the generalized Fock block-diagonal by irrep and the Z-vector right-hand side
# orthogonal to the near-singular b1<->b2 rotation.  Symmetry therefore makes the reference
# well-defined and the gradient platform-reproducible even though the orbital Hessian is
# near-singular.  (In C1 the SCF could break the degeneracy arbitrarily; C2v does not.)  6-31G, not
# STO-3G: STO-3G/OH leaves a virtual symmetry block empty, which crashes psi4's C2v CC gradient.
OH_C2V = "0 2\nO\nH 1 0.97"


def test_so_ccsd_gradient_keystone_equals_spatial():
    """SO == spatial: a closed-shell RHF driven through the spin-orbital path reproduces the spatial
    closed-shell CCSD correlation gradient (the strongest structural check)."""
    psi4.core.clean(); psi4.core.clean_options()
    psi4.geometry(H2O)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'freeze_core': 'false',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc_sp = pycc.ccwfn(wfn); cc_sp.solve_cc(E_CONV, R_CONV, 200)
    cc_so = pycc.ccwfn(wfn, orbital_basis='spinorbital'); cc_so.solve_cc(E_CONV, R_CONV, 200)
    g_sp = pycc.CCderiv(cc_sp).gradient().correlation
    g_so = pycc.CCderiv(cc_so).gradient().correlation
    assert np.max(np.abs(g_so - g_sp)) < 1e-10, (g_so, g_sp)


def test_so_ccsd_gradient_route_equivalence():
    """The AO-density skeleton route ('aod', the default) and the per-atom MO route ('mo') give
    bit-identical spin-orbital gradients.  Guards the spin-blocked back-transform in
    _effective_2pdm_ao_so and exercises the 'mo' opt-out branch of _so_gradient (open-shell UHF)."""
    cc, _ = _so_ccwfn(NH2, reference="uhf")
    d = pycc.CCderiv(cc)
    d._skel_eri_route = 'mo'
    g_mo = np.asarray(d.gradient().correlation)
    d._skel_eri_route = 'aod'
    g_aod = np.asarray(d.gradient().correlation)
    assert np.max(np.abs(g_aod - g_mo)) < 1e-11, np.max(np.abs(g_aod - g_mo))


def test_so_ccsd_gradient_vs_psi4_uhf():
    """The total spin-orbital (UHF) CCSD gradient reproduces psi4's analytic UHF-CCSD gradient.
    This is the open-shell validation: an external check against a converged, non-degenerate UHF
    reference (NH2)."""
    cc, _ = _so_ccwfn(NH2, reference="uhf")
    r = pycc.gradient(pycc.CCderiv(cc))
    assert np.max(np.abs(r.total - (r.nuclear + r.reference + r.correlation))) < 1e-12

    psi4.core.clean_options()
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'reference': 'uhf',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12, 'r_convergence': 1e-12})
    g_psi4 = np.asarray(psi4.gradient('ccsd'))
    assert np.max(np.abs(np.asarray(r.total) - g_psi4)) < 1e-8, (r.total, g_psi4)


def test_so_ccsd_gradient_vs_psi4_uhf_oh_c2v():
    """Open-shell UHF-CCSD gradient for an *orbitally degenerate* reference: OH (2-Pi) run in the
    computational point group C2v, vs psi4's analytic UHF-CCSD gradient (6-31G).

    The degenerate pi orbitals fall in different irreps (b1/b2), so symmetry keeps the Z-vector
    right-hand side orthogonal to the near-singular b1<->b2 orbital rotation -- the reference is
    well-defined and the gradient is platform-reproducible even though the orbital Hessian is
    near-singular (this is the case that is *not* reproducible when forced into C1)."""
    cc, wfn = _so_ccwfn(OH_C2V, reference="uhf", basis="6-31G")
    assert wfn.molecule().point_group().symbol() == "c2v"        # actually running in C2v
    r = pycc.gradient(pycc.CCderiv(cc))

    psi4.core.clean_options()
    psi4.set_options({'basis': '6-31G', 'scf_type': 'pk', 'reference': 'uhf',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12, 'r_convergence': 1e-12})
    g_psi4 = np.asarray(psi4.gradient('ccsd'))
    assert np.max(np.abs(np.asarray(r.total) - g_psi4)) < 1e-8, (r.total, g_psi4)


def test_so_ccsd_gradient_frozen_core():
    """Frozen-core spin-orbital CCSD gradient.  psi4 has no frozen-core CC gradient, so validate
    against the (psi4-validated) spatial frozen-core gradient -- the SO == spatial keystone with a
    frozen core (the core spans 2*nfzc spin-orbitals)."""
    psi4.core.clean(); psi4.core.clean_options()
    psi4.geometry(H2O)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'freeze_core': 'true',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc_sp = pycc.ccwfn(wfn); cc_sp.solve_cc(E_CONV, R_CONV, 200)
    cc_so = pycc.ccwfn(wfn, orbital_basis='spinorbital'); cc_so.solve_cc(E_CONV, R_CONV, 200)
    assert cc_so.nfzc > 0
    deriv = pycc.CCderiv(cc_so)
    g_so = deriv.gradient().correlation
    assert np.max(np.abs(g_so - pycc.CCderiv(cc_sp).gradient().correlation)) < 1e-9  # SO == spatial (frozen core)


def test_so_ccsd_gradient_rohf_raises():
    """ROHF is not supported (the semicanonical response does not reproduce the restricted ROHF
    response); the spin-orbital CCSD gradient raises rather than return a wrong number."""
    import pytest
    cc, _ = _so_ccwfn(NH2, reference="rohf")
    with pytest.raises(NotImplementedError):
        pycc.CCderiv(cc).gradient().correlation


# ---- CH2 (3-B1) / cc-pVDZ: the well-conditioned open-shell anchor -------------------------------
# Triplet CH2 is the suite's clean genuinely spin-polarized case.  Unlike the doublets (NH2, H2O2+),
# psi4's default UHF here converges straight to the stable solution -- `stability_analysis: follow`
# returns the same energy -- so no occupation pinning and no stability following are needed, and the
# response equations are not near-singular (CFOUR's own tensor is exactly symmetric here, whereas for
# H2O2+ it is only symmetric to 1.4e-9).  PyCC matches the oracle at CFOUR's ten-decimal print floor,
# the same level the closed-shell cases reach: there is no general open-shell penalty.
#
# Oracle: xcfour 2.1 (CALC=CCSD, REFERENCE=UHF, MULTIPLICITY=3, BASIS=PVDZ, SCF_CONV=13, CC_CONV=12,
# LINEQ_CONV=12).  Geometry is CFOUR's own full-precision MOL-file frame (r(CH)=1.0776 A,
# angle=133.93 deg); psi4's cc-pVDZ is CFOUR's PVDZ, so no GENBAS transcription is needed.
CH2_CFOUR_FRAME = """
0 3
C 0.000000000000000  0.000000000000000  0.114593744572731
H 0.000000000000000 -1.874001424240600 -0.682224040442087
H 0.000000000000000  1.874001424240600 -0.682224040442087
symmetry c1
units bohr
no_com
no_reorient
"""


def _ch2_so_ccwfn():
    """Converged spin-orbital UHF-CCSD wavefunction for 3-B1 CH2 / cc-pVDZ in CFOUR's frame."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(CH2_CFOUR_FRAME)
    psi4.set_options({'basis': 'cc-pVDZ', 'scf_type': 'pk', 'reference': 'uhf',
                      'e_convergence': 1e-13, 'd_convergence': 1e-13, 'maxiter': 300})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc = pycc.ccwfn(wfn, orbital_basis='spinorbital')
    cc.solve_cc(1e-13, 1e-13, 300)
    return cc


CFOUR_GRAD_CH2_PVDZ = np.array(
    [[ 0.0000000000e+00,  0.0000000000e+00, -5.4761065000e-03],
     [ 0.0000000000e+00,  9.2608332000e-03,  2.7380532000e-03],
     [ 0.0000000000e+00, -9.2608332000e-03,  2.7380532000e-03]])

def test_so_ccsd_gradient_cfour_ch2_triplet():
    """Open-shell UHF-CCSD correlation gradient for 3-B1 CH2 / cc-pVDZ vs the CFOUR GRD oracle
    (GRD(CCSD) - GRD(SCF)).  The suite's only CFOUR-anchored CC gradient and its only genuinely
    spin-polarized gradient validated against an external code; matches to 4.2e-11, at CFOUR's
    ten-decimal print floor, in ~8 s."""
    g = np.asarray(pycc.gradient(pycc.CCderiv(_ch2_so_ccwfn())).correlation)
    _dev = float(np.max(np.abs(g - CFOUR_GRAD_CH2_PVDZ)))
    record('Gradient', 'CCSD', 'all-electron', 'cfour', _dev,
           molecule='CH2', note='spin-orbital', ao_basis='cc-pVDZ',
           limited_by='oracle-print')
    assert np.max(np.abs(g - CFOUR_GRAD_CH2_PVDZ)) < 1e-9, g
    psi4.core.clean()
