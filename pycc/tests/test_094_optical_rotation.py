"""Facade optical rotation (pycc.optical_rotation) and the specific rotation.

Two routes to the CCSD optical-rotation (optical-activity) tensor G'(omega) for chiral H2O2, which
must give identical results:
  * pycc.optical_rotation(CCderiv(cc), omega) -- the facade over CCderiv.optical_rotation (the
    orbital-unrelaxed density / 2n+1 route);
  * ccresponse.optrot(omega)                  -- the independent symmetric linear-response code.

The report/facade also converts the trace of G' to the specific rotation [alpha] in
deg/(dm (g/mL)) via the Rosenfeld expression (pycc.properties._specific_rotation); that conversion
factor was cross-checked against an independent Mathematica implementation.

CCSD, spatial (closed-shell RHF), 6-31G, omega ~ 589 nm (sodium D line).  This complements the
method-level cross-check in test_093.
"""

import psi4
import pycc
import numpy as np
import pytest

from pycc.properties import _specific_rotation


# Chiral H2O2 (Dalton geometry), as in test_093 / test_058.
H2O2 = """
O   1.3133596569   0.0000000000  -0.0932359644
O  -1.3133596569  -0.0000000000  -0.0932359644
H   1.6917745981   0.7334825768   1.4797224976
H  -1.6917745981  -0.7334825768   1.4797224976
units bohr
symmetry c1
no_reorient
no_com
"""
OMEGA = 0.077357          # ~589 nm (sodium D line)

# CCSD/6-31G frozen-core, spatial route, at OMEGA.  G'(omega) is a pseudotensor (not symmetric).
G_REF = np.array([[-1.681837431953e-02,  3.461783788385e-01,  0.0],
                  [ 3.046945632411e-02,  2.025642104123e-01,  0.0],
                  [ 0.0,                  0.0,                -1.895273738325e-01]])
TRACE_REF = -0.003781537740       # Tr(G'), a.u.
ALPHA_REF = 18.540192             # specific rotation, deg/(dm (g/mL))


def _h2o2_ccsd():
    """Converged CCSD/6-31G (frozen core) H2O2 wavefunction."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(H2O2)
    psi4.set_options({'basis': '6-31G', 'scf_type': 'pk', 'freeze_core': 'true',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    return wfn, cc


def test_optical_rotation_facade():
    """pycc.optical_rotation (facade -> CCderiv.optical_rotation) reproduces the reference G'; the
    unrelaxed response carries no HF term (reference block zero); the reported specific rotation
    matches the frozen value; and a static (omega = 0) call is rejected."""
    wfn, cc = _h2o2_ccsd()
    d = pycc.CCderiv(cc)
    pc = pycc.optical_rotation(d, OMEGA)
    G = np.asarray(pc.correlation)
    assert np.max(np.abs(G - G_REF)) < 1e-11, G
    assert np.allclose(np.asarray(pc.reference), 0.0)              # unrelaxed: no Hartree-Fock term
    assert abs(np.trace(G) - TRACE_REF) < 1e-11
    alpha = _specific_rotation(float(np.trace(G)), OMEGA, wfn.molecule())
    assert abs(alpha - ALPHA_REF) < 1e-4, alpha
    with pytest.raises(ValueError):
        pycc.optical_rotation(d, 0.0)                             # no static optical rotation
    psi4.core.clean()


def test_optical_rotation_ccresponse():
    """ccresponse.optrot reproduces the same reference G' -- the symmetric linear-response route
    and the CCderiv density route are identical."""
    wfn, cc = _h2o2_ccsd()
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar)
    lam.solve_lambda(1e-12, 1e-12, 100)
    dens = pycc.ccdensity(cc, lam)
    G = np.asarray(pycc.ccresponse(dens).optrot(OMEGA))
    assert np.max(np.abs(G - G_REF)) < 1e-11, G
    psi4.core.clean()
