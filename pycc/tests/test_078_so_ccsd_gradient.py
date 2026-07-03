"""
Spin-orbital (UHF) CCSD analytic nuclear gradient -- pycc.gradient(ccwfn) / CCderiv._so_gradient.

The spin-orbital gradient mirrors the spatial one (Z-vector route) with the antisymmetrized
generalized-Fock Lagrangian, an inline orbital Hessian, and the spin-orbital derivative integrals.
Validated by: the SO == spatial keystone (a closed-shell RHF driven through the spin-orbital path
must reproduce the spatial closed-shell gradient), psi4's analytic UHF-CCSD gradient, the
independent explicit-derivative route (Z-vector == explicit), and -- for frozen core, which psi4's
CC gradients do not support -- the spatial frozen-core gradient.
"""

import numpy as np
import psi4
import pycc


E_CONV = R_CONV = 1e-11


def _so_ccwfn(geom, reference="uhf", freeze_core="false"):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'reference': reference,
                      'freeze_core': freeze_core, 'e_convergence': 1e-12, 'd_convergence': 1e-12,
                      'r_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc = pycc.ccwfn(wfn, orbital_basis='spinorbital', frozen_core=(freeze_core == "true"))
    cc.solve_cc(E_CONV, R_CONV, 300)
    return cc, wfn


# closed-shell water, fixed frame -- for the SO == spatial keystone and the frozen-core check
H2O = "O 0.0 0.0 0.118\nH 0.0 0.758 -0.472\nH 0.0 -0.758 -0.472\nunits angstrom\nsymmetry c1\nno_com\nno_reorient"
# OH radical, fixed frame -- for the UHF-vs-psi4 comparison
OH = "0 2\nO\nH 1 0.97\nsymmetry c1\nno_com\nno_reorient"


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
    g_sp = pycc.CCderiv(cc_sp).gradient()
    g_so = pycc.CCderiv(cc_so).gradient()
    assert np.max(np.abs(g_so - g_sp)) < 1e-10, (g_so, g_sp)


def test_so_ccsd_gradient_vs_psi4_uhf():
    """The total spin-orbital (UHF) CCSD gradient reproduces psi4's analytic UHF-CCSD gradient."""
    cc, _ = _so_ccwfn(OH, reference="uhf")
    r = pycc.gradient(cc)
    assert np.max(np.abs(r.total - (r.nuclear + r.reference + r.correlation))) < 1e-12

    psi4.core.clean_options()
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'reference': 'uhf',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12, 'r_convergence': 1e-12})
    g_psi4 = np.asarray(psi4.gradient('ccsd'))
    assert np.max(np.abs(np.asarray(r.total) - g_psi4)) < 1e-8, (r.total, g_psi4)


def test_so_ccsd_gradient_zvector_equals_explicit():
    """The Z-vector route (default) and the independent explicit-derivative route agree to machine
    precision -- for a closed shell and for an open-shell UHF reference."""
    psi4.core.clean(); psi4.core.clean_options()
    psi4.geometry(H2O)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'freeze_core': 'false',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc = pycc.ccwfn(wfn, orbital_basis='spinorbital'); cc.solve_cc(E_CONV, R_CONV, 200)
    deriv = pycc.CCderiv(cc)
    assert np.max(np.abs(deriv.gradient() - deriv._gradient_explicit())) < 1e-9

    cc_u, _ = _so_ccwfn(OH, reference="uhf")
    deriv_u = pycc.CCderiv(cc_u)
    assert np.max(np.abs(deriv_u.gradient() - deriv_u._gradient_explicit())) < 1e-9


def test_so_ccsd_gradient_frozen_core():
    """Frozen-core spin-orbital CCSD gradient.  psi4 has no frozen-core CC gradient, so validate
    against the (psi4-validated) spatial frozen-core gradient -- the SO == spatial keystone with a
    frozen core (the core spans 2*nfzc spin-orbitals) -- and the independent explicit route."""
    psi4.core.clean(); psi4.core.clean_options()
    psi4.geometry(H2O)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'freeze_core': 'true',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc_sp = pycc.ccwfn(wfn, frozen_core=True); cc_sp.solve_cc(E_CONV, R_CONV, 200)
    cc_so = pycc.ccwfn(wfn, orbital_basis='spinorbital', frozen_core=True); cc_so.solve_cc(E_CONV, R_CONV, 200)
    assert cc_so.nfzc > 0
    deriv = pycc.CCderiv(cc_so)
    g_so = deriv.gradient()
    assert np.max(np.abs(g_so - deriv._gradient_explicit())) < 1e-9      # zvector == explicit
    assert np.max(np.abs(g_so - pycc.CCderiv(cc_sp).gradient())) < 1e-9  # SO == spatial (frozen core)


def test_so_ccsd_gradient_rohf_raises():
    """ROHF is not supported (the semicanonical response does not reproduce the restricted ROHF
    response); the spin-orbital CCSD gradient raises rather than return a wrong number."""
    import pytest
    cc, _ = _so_ccwfn(OH, reference="rohf")
    with pytest.raises(NotImplementedError):
        pycc.CCderiv(cc).gradient()
