import psi4
import pycc
import numpy as np
from ..data.molecules import moldict

def test_pao_H4():
    """PAO-CCSD Test"""
    # Psi4 Setup
    psi4.set_memory('1 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '6-31g',
                      'scf_type': 'pk',
                      'guess': 'core',
                      'mp2_type': 'conv',
                      'freeze_core': 'False',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 8})
    mol = psi4.geometry("""
        H         0.000000000000     0.000000000000     0.000000000000
        H         0.000000000000     0.000000000000     1.417294491434
        H         2.834588982868     0.000000000000     1.417294491434
        H         2.834588982868     1.227413034226     0.708647245717
        units au
        noreorient
        nocom
        """)
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7
    max_diis = 8
    
    ccsd = pycc.ccwfn(rhf_wfn, local='PAO', local_cutoff=2e-2)
    
    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter, max_diis)
    
    # (H2)_2 REFERENCE:
    psi3_ref = -0.052863089856486

    assert (abs(psi3_ref - eccsd) < 1e-7)

def test_pao_h2o():
    """PAO-CCSD Test 2"""
    # Psi4 Setup
    psi4.set_memory('1 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '6-31g',
                      'scf_type': 'pk',
                      'guess': 'core',
                      'mp2_type': 'conv',
                      'freeze_core': 'False',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 8})
    mol = psi4.geometry(moldict["H2O_Teach"] + "\nnoreorient\nnocom")
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7
    max_diis = 8
    
    ccsd = pycc.ccwfn(rhf_wfn, local='PAO', local_cutoff=2e-2)
    
    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter, max_diis)
    
    # NOTE: the following reference was generated with a custom build of Psi3
    # which removed PAOs based on their norm EVEN IF freeze_core = false
    psi3_ref = -0.149361947815815
    assert (abs(psi3_ref - eccsd) < 1e-7)

def test_pao_h2o_frzc():
    """PAO-CCSD Frozen Core Test"""
    # Psi4 Setup
    psi4.set_memory('1 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '6-31g',
                      'scf_type': 'pk',
                      'guess': 'core',
                      'mp2_type': 'conv',
                      'freeze_core': 'True',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 8})
    mol = psi4.geometry(moldict["H2O_Teach"] + "\nnoreorient\nnocom")
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7
    max_diis = 8
    
    ccsd = pycc.ccwfn(rhf_wfn, local='PAO', local_cutoff=2e-2)
    
    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter, max_diis)
    
    psi3_ref = -0.148485522656349

    assert (abs(psi3_ref - eccsd) < 1e-7)
