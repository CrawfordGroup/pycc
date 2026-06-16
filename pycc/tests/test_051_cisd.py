"""
Test the CISD/CID correlation energy (CIwfn).

CISD is validated against Psi4's DETCI (the exact CISD). The default 'cisd' driver is
NOT used: it routes to fnocc, which applies frozen natural orbitals / truncation and so
does not reproduce the exact CISD energy. DETCI has no doubles-only mode, so CID is
validated against CFOUR reference values (the same H2O geometry as magpy's
test_004_CID, reproduced exactly below so the comparison is geometry-for-geometry).
"""

import psi4
import pycc

# Exact H2O geometry behind the CFOUR CID references (bohr); identical to magpy's.
H2O = """
O  0.000000000000  -0.143225816552   0.000000000000
H  1.638036840407   1.136548822547  -0.000000000000
H -1.638036840407   1.136548822547  -0.000000000000
units bohr
no_com
no_reorient
symmetry c1
"""


def test_cisd_h2o(rhf_wfn):
    """All-electron CISD vs Psi4 DETCI (exact CISD)."""
    wfn = rhf_wfn(H2O, "cc-pVDZ", freeze_core="false", qc_module="detci",
                  e_convergence=1e-12, d_convergence=1e-12)
    eci = pycc.CIwfn(wfn, frozen_core=False).solve_ci(e_conv=1e-11, r_conv=1e-11)
    eref = psi4.energy('detci', ci_type='cisd') - wfn.energy()
    assert abs(eci - eref) < 1e-10


def test_cisd_h2o_frozen_core(rhf_wfn):
    """Frozen-core CISD vs Psi4 DETCI."""
    wfn = rhf_wfn(H2O, "cc-pVDZ", freeze_core="true", qc_module="detci",
                  e_convergence=1e-12, d_convergence=1e-12)
    eci = pycc.CIwfn(wfn).solve_ci(e_conv=1e-11, r_conv=1e-11)
    eref = psi4.energy('detci', ci_type='cisd') - wfn.energy()
    assert abs(eci - eref) < 1e-10


def test_cid_h2o(rhf_wfn):
    """All-electron CID (doubles-only model seam) vs CFOUR (cc-pVDZ H2O)."""
    wfn = rhf_wfn(H2O, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    ecid = pycc.CIwfn(wfn, frozen_core=False, model="CID").solve_ci(e_conv=1e-12, r_conv=1e-12)
    assert abs(ecid - (-0.21279410950205)) < 1e-11


def test_cid_h2o_frozen_core(rhf_wfn):
    """Frozen-core CID vs CFOUR (cc-pVDZ H2O)."""
    wfn = rhf_wfn(H2O, "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)
    ecid = pycc.CIwfn(wfn, model="CID").solve_ci(e_conv=1e-12, r_conv=1e-12)
    assert abs(ecid - (-0.21098966441656)) < 1e-11
