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


# --- Spin-orbital CISD/CID (UHF/ROHF) -------------------------------------------
# These are spin-orbital (determinant) CISD/CID. The external reference is CFOUR, which
# computes the same spin-orbital CISD/CID (matching to ~1e-13 for UHF and ROHF, CISD and
# CID). Psi4 cannot validate this: it has no UHF-CISD, and its DETCI ROHF-CISD is
# spin-adapted (CSF) -- a genuinely different method that disagrees with both PyCC and
# CFOUR for open shells. The kernel is additionally validated by the closed-shell
# keystone (SO-RHF == spatial) and the 2-electron identity CISD = FCI (exact,
# orbital-invariant) vs Psi4 FCI.

# Triplet H2 at 1.4 bohr: 2 electrons, so CISD = FCI exactly.
H2_TRIPLET = """
0 3
H 0.0 0.0 0.0
H 0.0 0.0 1.4
units bohr
no_com
no_reorient
symmetry c1
"""
# Exact FCI total energy for H2_TRIPLET / 6-31G (Psi4 FCI; = CISD for 2 electrons).
FCI_TOTAL_H2T = -0.757409321179


def test_so_cisd_cid_equals_spatial_rhf(rhf_wfn):
    """Spin-orbital CISD and CID (forced) reproduce the spin-adapted spatial values on a
    closed shell, isolating the spin-orbital CI kernel."""
    wfn = rhf_wfn(H2O, "6-31G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    for model in ("CISD", "CID"):
        e_spatial = pycc.CIwfn(wfn, frozen_core=False, model=model).solve_ci(
            e_conv=1e-11, r_conv=1e-11)
        so = pycc.CIwfn(wfn, frozen_core=False, model=model, orbital_basis="spinorbital")
        assert so.orbital_basis == "spinorbital"
        e_so = so.solve_ci(e_conv=1e-11, r_conv=1e-11)
        assert abs(e_so - e_spatial) < 1e-10


def test_uhf_cisd_equals_fci(uhf_wfn):
    """2-electron triplet: UHF-CISD = FCI (exact, orbital-invariant oracle)."""
    wfn = uhf_wfn(H2_TRIPLET, "6-31G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    ecisd = pycc.CIwfn(wfn, frozen_core=False).solve_ci(e_conv=1e-11, r_conv=1e-11)
    assert abs((wfn.energy() + ecisd) - FCI_TOTAL_H2T) < 1e-10


def test_rohf_cisd_equals_fci(rohf_wfn):
    """2-electron triplet: ROHF-CISD = FCI (exact); validates the ROHF code path."""
    wfn = rohf_wfn(H2_TRIPLET, "6-31G", freeze_core="false",
                   e_convergence=1e-12, d_convergence=1e-12)
    ecisd = pycc.CIwfn(wfn, frozen_core=False).solve_ci(e_conv=1e-11, r_conv=1e-11)
    assert abs((wfn.energy() + ecisd) - FCI_TOTAL_H2T) < 1e-10


# OH doublet at 1.83 bohr (verbatim for both codes); CFOUR spin-orbital CISD/CID
# correlation energies, cc-pVDZ, all-electron.
OH_BOHR = """
0 2
O 0.0 0.0 0.0
H 0.0 0.0 1.83
units bohr
no_com
no_reorient
symmetry c1
"""
CFOUR_UHF_CISD = -0.161059480423
CFOUR_UHF_CID = -0.160472710690
CFOUR_ROHF_CISD = -0.164616502232
CFOUR_ROHF_CID = -0.161769562501


def test_uhf_cisd_cid_vs_cfour(uhf_wfn):
    """Multi-electron UHF CISD and CID (.OH cc-pVDZ) vs CFOUR's spin-orbital CISD/CID."""
    wfn = uhf_wfn(OH_BOHR, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    ecisd = pycc.CIwfn(wfn, frozen_core=False, model="CISD").solve_ci(e_conv=1e-11, r_conv=1e-11)
    ecid = pycc.CIwfn(wfn, frozen_core=False, model="CID").solve_ci(e_conv=1e-11, r_conv=1e-11)
    assert abs(ecisd - CFOUR_UHF_CISD) < 1e-10
    assert abs(ecid - CFOUR_UHF_CID) < 1e-10


def test_rohf_cisd_cid_vs_cfour(rohf_wfn):
    """Multi-electron ROHF CISD and CID (.OH cc-pVDZ) vs CFOUR's spin-orbital CISD/CID.
    Exercises the non-canonical singles->doubles Fock coupling that distinguishes
    spin-orbital CISD from the spin-adapted (DETCI) variant."""
    wfn = rohf_wfn(OH_BOHR, "cc-pVDZ", freeze_core="false",
                   e_convergence=1e-12, d_convergence=1e-12)
    ecisd = pycc.CIwfn(wfn, frozen_core=False, model="CISD").solve_ci(e_conv=1e-11, r_conv=1e-11)
    ecid = pycc.CIwfn(wfn, frozen_core=False, model="CID").solve_ci(e_conv=1e-11, r_conv=1e-11)
    assert abs(ecisd - CFOUR_ROHF_CISD) < 1e-10
    assert abs(ecid - CFOUR_ROHF_CID) < 1e-10
