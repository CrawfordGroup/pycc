"""
Spin-adapted (spatial) CCSD optical rotation (length gauge) via the symmetric
linear-response function (docs/archive/ENHANCEMENT_PLAN_2026-06.md).

This pins the spatial spin-adapted symmetric path through optrot (linresp_sym ->
LCX / LHX1Y1 / LHX2Y2 / LHX1Y2, with the magnetic-dipole pertbars M / M*), which the
spin-orbital tests in test_056 do NOT exercise (RHF is run explicitly in the
spin-orbital basis; open-shell auto-resolves to spin orbitals). Validation:
  * spatial-RHF optical-rotation tensor (chiral H2O2) vs Psi4's CCSD optical
    rotation, cc-pVDZ.
  * spatial-RHF full 3x3 tensor vs the spin-orbital path (SO-RHF == spatial-RHF),
    STO-3G (to keep the spin-orbital response cheap; the spatial path itself is
    validated vs Psi4 at cc-pVDZ above).
"""

import psi4
import pycc
import numpy as np

# Chiral H2O2 (from Dalton), for the optical-rotation reference. No "symmetry c1":
# Psi4 detects C2 and produces symmetry-adapted MOs.
H2O2 = """
O   1.3133596569   0.0000000000  -0.0932359644
O  -1.3133596569  -0.0000000000  -0.0932359644
H   1.6917745981   0.7334825768   1.4797224976
H  -1.6917745981  -0.7334825768   1.4797224976
units bohr
"""

# Psi4 CCSD optical-rotation tensor (length gauge, omega=0.077357, H2O2/cc-pVDZ,
# frozen core), parsed from Psi4's "CCSD Optical Rotation Tensor (Length Gauge)" block.
PSI4_OPTROT = np.array([[-0.012765691587815,  0.266267947958189, 0.0],
                        [ 0.064272290616858,  0.155734081606001, 0.0],
                        [ 0.0, 0.0, -0.127276678468211]])


def _optrot(wfn, omega, **cckwargs):
    cc = pycc.CCwfn(wfn, **cckwargs)
    cc.solve_cc(e_conv=1e-12, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-12, r_conv=1e-11)
    onlyone = cckwargs.get("orbital_basis") == "spinorbital"
    dens = pycc.ccdensity(cc, lam, onlyone=onlyone)
    return pycc.ccresponse(dens).optrot(omega)


def test_spatial_rhf_optrot_vs_psi4(rhf_wfn):
    """Spatial-RHF CCSD optical-rotation tensor (H2O2/cc-pVDZ) vs Psi4 (length gauge)."""
    wfn = rhf_wfn(H2O2, "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12, r_convergence=1e-12)
    optrot = _optrot(wfn, 0.077357, orbital_basis="spatial")
    assert np.max(np.abs(optrot - PSI4_OPTROT)) < 1e-10


def test_spatial_rhf_optrot_vs_spinorbital(rhf_wfn):
    """Spatial-RHF full optical-rotation tensor == spin-orbital path."""
    wfn = rhf_wfn(H2O2, "STO-3G", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12, r_convergence=1e-12)
    spatial = _optrot(wfn, 0.077357, orbital_basis="spatial")
    so = _optrot(wfn, 0.077357, orbital_basis="spinorbital")
    assert np.max(np.abs(spatial - so)) < 1e-10
