import psi4
import pycc
import pytest
import numpy as np
from ..data.molecules import moldict

def test_pao_H8(rhf_wfn):
    """PAO-CCSD Test"""
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    max_diis = 8

    geom = """
        H 0.000000 0.000000 0.000000
        H 0.750000 0.000000 0.000000
        H 0.000000 1.500000 0.000000
        H 0.375000 1.500000 -0.649520
        H 0.000000 3.000000 0.000000
        H -0.375000 3.000000 -0.649520
        H 0.000000 4.500000 -0.000000
        H -0.750000 4.500000 -0.000000
        symmetry c1
        noreorient
        nocom
        """
    wfn = rhf_wfn(geom, "DZ", freeze_core="False", guess="core", diis=8)
    ccsd = pycc.ccwfn(wfn, local='PAO', local_cutoff=2e-2)

    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter, max_diis)

    # (H2)_4 REFERENCE:
    psi3_ref = -0.108914240219735

    assert (abs(psi3_ref - eccsd) < 1e-7)

def test_pao_h2o(rhf_wfn):
    """PAO-CCSD Test 2"""
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7
    max_diis = 8

    wfn = rhf_wfn("H2O_Teach", "6-31g", geom_extra="\nnoreorient\nnocom\nsymmetry c1",
                  freeze_core="False", guess="core", diis=8)
    ccsd = pycc.ccwfn(wfn, local='PAO', local_cutoff=2e-2)

    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter, max_diis)

    # NOTE: the following reference was generated with a custom build of Psi3
    # which removed PAOs based on their norm EVEN IF freeze_core = false
    psi3_ref = -0.149361947815815
    assert (abs(psi3_ref - eccsd) < 1e-7)

@pytest.mark.skip(reason="Local CC does not support a frozen core; its QL/domain "
                         "transforms assume the full occupied space. Frozen-core "
                         "local CC is deferred to the planned local rewrite, and "
                         "Local.__init__ now raises PyCCError for nfzc>0.")
def test_pao_h2o_frzc(rhf_wfn):
    """PAO-CCSD Frozen Core Test"""
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7
    max_diis = 8

    wfn = rhf_wfn("H2O_Teach", "6-31g", geom_extra="\nnoreorient\nnocom\nsymmetry c1",
                  freeze_core="True", guess="core", diis=8)
    ccsd = pycc.ccwfn(wfn, local='PAO', local_cutoff=2e-2)

    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter, max_diis)

    psi3_ref = -0.148485522656349

    assert (abs(psi3_ref - eccsd) < 1e-7)
