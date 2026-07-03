"""
CCSD analytic nuclear gradient -- pycc.gradient(ccwfn) via the CCderiv driver (explicit-derivative
route).  The spatial closed-shell path, all-electron.

Validated against psi4's analytic CCSD gradient, with the density-adapter energy-reconstruction
check (the two-particle density in the gradient convention reproduces the CCSD correlation energy)
and the PropertyComponents decomposition.
"""

import psi4
import pycc
import numpy as np


# H2O, fixed frame (no reorientation) so pycc and psi4 gradients share the molecular axes.
H2O = """
O  0.000000  0.000000  0.118000
H  0.000000  0.758000 -0.472000
H  0.000000 -0.758000 -0.472000
symmetry c1
units angstrom
no_com
no_reorient
"""


def _ccwfn(basis="STO-3G", freeze_core="false"):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(H2O)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc = pycc.ccwfn(wfn, frozen_core=(freeze_core == "true"))
    cc.solve_cc(1e-12, 1e-12, 200)
    return cc


def test_ccsd_gradient_vs_psi4():
    """pycc.gradient(ccwfn).total reproduces psi4's analytic CCSD gradient (same frame)."""
    cc = _ccwfn()
    r = pycc.gradient(cc)
    # PropertyComponents decomposition is exact
    assert np.max(np.abs(r.total - (r.nuclear + r.reference + r.correlation))) < 1e-12

    psi4.core.clean_options()
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'e_convergence': 1e-12,
                      'd_convergence': 1e-12, 'r_convergence': 1e-12})
    g_psi4 = np.asarray(psi4.gradient('ccsd'))
    assert np.max(np.abs(np.asarray(r.total) - g_psi4)) < 1e-8, (r.total, g_psi4)


def test_ccsd_gradient_density_energy():
    """The gradient-convention densities (CCderiv adapter) reproduce the CCSD correlation energy:
    E_corr = contract(D, F) + contract(Gamma, ERI) (no prefactor on the two-particle term)."""
    cc = _ccwfn()
    deriv = pycc.CCderiv(cc)
    dens = deriv._density()
    ecc = dens.compute_energy()                       # CCSD correlation energy from the densities
    D, G = dens.gradient_densities()
    F = np.asarray(cc.H.F)
    ERI = np.asarray(cc.H.ERI)
    E = cc.contract('pq,pq->', D, F) + cc.contract('pqrs,pqrs->', G, ERI)
    assert abs(E - ecc) < 1e-10, (E, ecc)


def test_ccsd_gradient_correlation_nonzero():
    """The CCSD correlation gradient is a real, nonzero contribution on top of the SCF reference."""
    cc = _ccwfn()
    r = pycc.gradient(cc)
    assert np.max(np.abs(r.correlation)) > 1e-3
    # the total differs from the bare SCF gradient by the correlation contribution
    assert np.max(np.abs(r.total - (r.nuclear + r.reference))) > 1e-3


def test_ccsd_gradient_zvector_equals_explicit():
    """The efficient Z-vector route (the default gradient()) agrees with the independent
    explicit-derivative route to machine precision -- they are two formulations of the same
    correlation gradient (one Z-vector solve vs 3*natom CPHF solves)."""
    cc = _ccwfn()
    deriv = pycc.CCderiv(cc)
    g_zvector = deriv.gradient()
    g_explicit = deriv._gradient_explicit()
    assert np.max(np.abs(g_zvector - g_explicit)) < 1e-9, (g_zvector, g_explicit)


def test_ccsd_gradient_frozen_core():
    """Frozen-core CCSD gradient: the Z-vector and explicit routes agree (the core<->active P_co
    response is handled), and the frozen-core gradient differs from the all-electron one. psi4 has
    no frozen-core CC gradient, so this is cross-validated by the two independent routes (each also
    checked against a finite difference of the frozen-core CCSD energy)."""
    cc_fc = _ccwfn(freeze_core="true")
    assert cc_fc.nfzc > 0
    deriv = pycc.CCderiv(cc_fc)
    g_zvector = deriv.gradient()
    g_explicit = deriv._gradient_explicit()
    assert np.max(np.abs(g_zvector - g_explicit)) < 1e-9, (g_zvector, g_explicit)
    # frozen core actually changes the correlation gradient vs all-electron (small for the deep
    # O 1s in STO-3G, but nonzero)
    g_ae = pycc.CCderiv(_ccwfn(freeze_core="false")).gradient()
    assert np.max(np.abs(g_zvector - g_ae)) > 1e-5
