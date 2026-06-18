"""
Spin-orbital CCSD optical rotation (length gauge) via the symmetric linear-response
function (docs/ENHANCEMENT_PLAN_2026-06.md).

Validation:
  * SO-RHF optical-rotation tensor (chiral H2O2) vs Psi4's CCSD optical rotation --
    the symmetric kernel reproduces the spin-adapted result (and confirms the
    spin-orbital magnetic-dipole integrals H.m, which optrot is the first to use).
  * UHF/ROHF: no code computes open-shell CC optical rotation (a dynamic magnetic
    property, with no finite-difference analog), so these are validated by
    composition -- the kernel is validated for RHF above and is identical for
    UHF/ROHF, on already-validated orbitals/integrals. The tests here only confirm
    the open-shell path runs and returns a finite, real tensor.
"""

import psi4
import pycc
import numpy as np

# Chiral H2O2 (from Dalton), for the optical-rotation reference. No "symmetry c1":
# Psi4 detects C2 and produces symmetry-adapted MOs (transforming as the C2 irreps).
# The spin-orbital result must be identical to the C1 case (and matches Psi4 to ~1e-13),
# confirming that the occ/vir ordering of Ca_subset("AO","ACTIVE") holds under symmetry.
H2O2 = """
O   1.3133596569   0.0000000000  -0.0932359644
O  -1.3133596569  -0.0000000000  -0.0932359644
H   1.6917745981   0.7334825768   1.4797224976
H  -1.6917745981  -0.7334825768   1.4797224976
units bohr
"""

# Open-shell doublet (hydroxyl radical) for the open-shell smoke checks.
OH = """
0 2
O 0.0 0.0 0.0
H 0.0 0.0 1.83
units bohr
no_com
no_reorient
symmetry c1
"""

# Psi4 CCSD optical-rotation tensor (length gauge, omega=0.077357, H2O2/STO-3G,
# frozen core) -- the same reference socc uses.
PSI4_OPTROT = np.array([[-0.012969077546209,  0.238413335857954, 0.0],
                        [-0.039665773503841,  0.124406421486684, 0.0],
                        [ 0.0, 0.0, -0.112853202255656]])


def _optrot(wfn, omega, **cckwargs):
    cc = pycc.CCwfn(wfn, **cckwargs)
    cc.solve_cc(e_conv=1e-12, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-12, r_conv=1e-11)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    return pycc.ccresponse(dens).optrot(omega)


def test_so_rhf_optrot_vs_psi4(rhf_wfn):
    """SO-RHF CCSD optical-rotation tensor (H2O2/STO-3G) vs Psi4 (length gauge)."""
    wfn = rhf_wfn(H2O2, "STO-3G", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12, r_convergence=1e-12)
    optrot = _optrot(wfn, 0.077357, orbital_basis="spinorbital")
    assert np.max(np.abs(optrot - PSI4_OPTROT)) < 1e-10


def test_uhf_optrot_runs(uhf_wfn):
    """Open-shell UHF optical rotation: runs and returns a finite, real tensor (no
    external open-shell reference exists)."""
    wfn = uhf_wfn(OH, "STO-3G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    optrot = _optrot(wfn, 0.077357, frozen_core=False)
    assert optrot.shape == (3, 3)
    assert np.all(np.isfinite(optrot))


def test_rohf_optrot_runs(rohf_wfn):
    """Open-shell ROHF optical rotation: runs and returns a finite, real tensor."""
    wfn = rohf_wfn(OH, "STO-3G", freeze_core="false",
                   e_convergence=1e-12, d_convergence=1e-12)
    optrot = _optrot(wfn, 0.077357, frozen_core=False)
    assert optrot.shape == (3, 3)
    assert np.all(np.isfinite(optrot))
