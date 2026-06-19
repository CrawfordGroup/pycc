"""
Spin-adapted (spatial) CCSD dipole polarizability via the symmetric linear-response
function (right-hand perturbed amplitudes only); docs/ENHANCEMENT_PLAN_2026-06.md.

This pins the spatial spin-adapted symmetric path (linresp_sym -> LCX / LHX1Y1 /
LHX2Y2 / LHX1Y2), which the spin-orbital tests in test_055 do NOT exercise: the
open-shell UHF/ROHF references there auto-resolve to the spin-orbital basis, and the
RHF case is run explicitly in the spin-orbital basis. Validation:
  * spatial-RHF dynamic polarizability (omega != 0, isotropic) vs Psi4's CCSD
    polarizability, cc-pVDZ.
  * spatial-RHF full 3x3 tensor vs the spin-orbital path (SO-RHF == spatial-RHF),
    STO-3G -- pins every component (incl. off-diagonals / distinct X_A, X_B axes),
    which the isotropic-vs-Psi4 check alone does not. (STO-3G to keep the spin-orbital
    response cheap; the spatial path itself is validated vs Psi4 at cc-pVDZ above.)
"""

import psi4
import pycc
import numpy as np


def _polar(wfn, omega, **cckwargs):
    cc = pycc.CCwfn(wfn, **cckwargs)
    cc.solve_cc(e_conv=1e-11, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-11, r_conv=1e-11)
    onlyone = cckwargs.get("orbital_basis") == "spinorbital"
    dens = pycc.ccdensity(cc, lam, onlyone=onlyone)
    return pycc.ccresponse(dens).polarizability(omega)


def test_spatial_rhf_polar_vs_psi4(rhf_wfn):
    """Spatial-RHF dynamic CCSD polarizability (isotropic, omega=0.1) vs Psi4."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12, r_convergence=1e-12)
    iso = np.trace(_polar(wfn, 0.1, frozen_core=False, orbital_basis="spatial")) / 3.0

    psi4.set_options({'omega': [0.1, 'au'], 'gauge': 'length'})
    psi4.properties('ccsd', properties=['polarizability'])
    ref = next(psi4.variable(k) for k in psi4.core.variables()
               if 'DIPOLE POLARIZABILITY' in k.upper())
    assert abs(iso - ref) < 1e-9


def test_spatial_rhf_polar_vs_spinorbital(rhf_wfn):
    """Spatial-RHF full polarizability tensor (omega=0.1) == spin-orbital path."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12, r_convergence=1e-12)
    spatial = _polar(wfn, 0.1, frozen_core=False, orbital_basis="spatial")
    so = _polar(wfn, 0.1, frozen_core=False, orbital_basis="spinorbital")
    assert np.max(np.abs(spatial - so)) < 1e-9
