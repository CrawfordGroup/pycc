"""Orbital-unrelaxed (response) CCSD optical rotation via the density reformulation --
CCderiv.optical_rotation(omega) -- cross-checked against the independent symmetric
linear-response code, ccresponse.optrot(omega).

Optical rotation is the odd-in-omega response of the electric dipole to the magnetic
dipole, G'(omega) = <<mu; m>>_omega.  Because the magnetic dipole is anti-Hermitian, a
single density evaluation is not the response function (unlike the electric polarizability):
the density route gives G'(omega) = 1/2[Tr(d_m D(omega).mu) - Tr(d_m D(-omega).mu)], the
odd-in-omega combination, which equals the symmetric ccresponse.optrot = 0.5(S1 - S2).  The
anti-Hermitian source also requires the perturbed-amplitude singles source to be taken from
the [v,o] block (A_ai), not the [o,v] block (the ground-state f_ia convention, valid only
for a symmetric perturbation); see CCderiv._perturbed_amplitudes.

CCSD, omega != 0; spatial (closed-shell RHF) and spin-orbital.  Chiral H2O2 gives a
physically nonzero (nonvanishing-trace) rotation.  See docs/ccresponse_reformulation_plan.md."""

import psi4
import pycc
import numpy as np
import pytest


# Chiral H2O2 (from Dalton), the standard optical-rotation test molecule (cf. test_058).
H2O2 = """
O   1.3133596569   0.0000000000  -0.0932359644
O  -1.3133596569  -0.0000000000  -0.0932359644
H   1.6917745981   0.7334825768   1.4797224976
H  -1.6917745981  -0.7334825768   1.4797224976
units bohr
"""
OMEGA = 0.077357          # ~589 nm (sodium D line), as in test_058


def _ccresponse_optrot(wfn, route, omega=OMEGA):
    """Optical-rotation tensor at frequency omega from the independent symmetric linear-response
    code -- the oracle for the reformulation."""
    cc = pycc.ccwfn(wfn, orbital_basis=route)
    cc.solve_cc(1e-12, 1e-12, 100)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(1e-12, 1e-12, 100)
    dens = pycc.ccdensity(cc, lam, onlyone=(route == 'spinorbital'))
    return np.asarray(pycc.ccresponse(dens).optrot(omega))


def test_optical_rotation_spatial_vs_ccresponse(rhf_wfn):
    """Spatial (closed-shell RHF) unrelaxed optical rotation (all 9 elements) matches the symmetric
    linear-response oracle; the isotropic rotation (trace) is nonzero (chiral); optical_rotation(0)
    is rejected (no static optical rotation)."""
    wfn = rhf_wfn(H2O2, "6-31G", freeze_core="true")
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    d = pycc.CCderiv(cc)
    G = np.asarray(d.optical_rotation(OMEGA))
    ref = _ccresponse_optrot(wfn, 'spatial')
    assert np.max(np.abs(G - ref)) < 1e-10, G
    assert abs(np.trace(G)) > 1e-4                            # physically nonzero (chiral)
    with pytest.raises(ValueError):
        d.optical_rotation(0.0)                              # no static optical rotation
    psi4.core.clean()


def test_so_optical_rotation_vs_ccresponse(rhf_wfn):
    """SO == spatial keystone: the spin-orbital unrelaxed optical rotation of an RHF reference forced
    into the spin-orbital basis matches the spin-orbital symmetric-response oracle -- carrying the
    check (and the anti-Hermitian [v,o] source) onto the SO machinery."""
    wfn = rhf_wfn(H2O2, "STO-3G", freeze_core="true")
    cc = pycc.ccwfn(wfn, orbital_basis='spinorbital')
    cc.solve_cc(1e-12, 1e-12, 100)
    G = np.asarray(pycc.CCderiv(cc).optical_rotation(OMEGA))
    ref = _ccresponse_optrot(wfn, 'spinorbital')
    assert np.max(np.abs(G - ref)) < 1e-10, G
    psi4.core.clean()
