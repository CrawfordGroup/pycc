"""
Test the CISD/CID correlation energy (CIwfn).

The references are exact: CISD against Psi4 DETCI, CID against CFOUR (the same H2O
geometry as magpy's test_004_CID, reproduced exactly below). Both are hardcoded rather
than recomputed at runtime: the default 'cisd' driver routes to fnocc (frozen natural
orbitals / truncation, ~4e-5 off exact), and a live DETCI call is flaky on CI -- on the
larger all-electron CISD, DETCI's Davidson intermittently hits its iteration cap. The
DETCI values below were computed locally and match PyCC to ~2e-13; with the tightly
converged RHF (1e-12) the comparison is essentially code/version-independent, so the
1e-10 tolerance is comfortable (the CID tests, hardcoded CFOUR values, pass at 1e-11).
"""

import pycc

# Exact H2O geometry behind the references (bohr); identical to magpy's test_004_CID.
H2O = """
O  0.000000000000  -0.143225816552   0.000000000000
H  1.638036840407   1.136548822547  -0.000000000000
H -1.638036840407   1.136548822547  -0.000000000000
units bohr
no_com
no_reorient
symmetry c1
"""

# Exact CISD correlation energies (Psi4 DETCI) for the geometry above.
CISD_REF = -0.213962927361777        # all-electron
CISD_REF_FC = -0.212162092109267     # frozen-core
# Exact CID correlation energies (CFOUR) for the geometry above.
CID_REF = -0.21279410950205          # all-electron
CID_REF_FC = -0.21098966441656       # frozen-core


def test_cisd_h2o(rhf_wfn):
    """All-electron CISD vs Psi4 DETCI (exact CISD)."""
    wfn = rhf_wfn(H2O, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    eci = pycc.CIwfn(wfn, frozen_core=False).solve_ci(e_conv=1e-11, r_conv=1e-11)
    assert abs(eci - CISD_REF) < 1e-10


def test_cisd_h2o_frozen_core(rhf_wfn):
    """Frozen-core CISD vs Psi4 DETCI."""
    wfn = rhf_wfn(H2O, "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)
    eci = pycc.CIwfn(wfn).solve_ci(e_conv=1e-11, r_conv=1e-11)
    assert abs(eci - CISD_REF_FC) < 1e-10


def test_cid_h2o(rhf_wfn):
    """All-electron CID (doubles-only model seam) vs CFOUR (cc-pVDZ H2O)."""
    wfn = rhf_wfn(H2O, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    ecid = pycc.CIwfn(wfn, frozen_core=False, model="CID").solve_ci(e_conv=1e-12, r_conv=1e-12)
    assert abs(ecid - CID_REF) < 1e-11


def test_cid_h2o_frozen_core(rhf_wfn):
    """Frozen-core CID vs CFOUR (cc-pVDZ H2O)."""
    wfn = rhf_wfn(H2O, "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)
    ecid = pycc.CIwfn(wfn, model="CID").solve_ci(e_conv=1e-12, r_conv=1e-12)
    assert abs(ecid - CID_REF_FC) < 1e-11
