"""
Test basic LPNO-CCSD energy and Lambda code
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
import numpy as np

def test_pipek_mezey_CO_sto3g():
    """CO Pipek-Mezey Test"""
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'STO-3G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry("""
        units au
        O 0.0 0.0 2.132
        C 0.0 0.0 0.0
        """)

    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    print("Canonical Occupied MOs:")
    C_occ = rhf_wfn.Ca_subset("AO", "ACTIVE_OCC")
    print(np.asarray(C_occ))

    pm_O_1s_canon = 0.9941
    pm_C_1s_canon = 0.9936

    Local = psi4.core.Localizer.build("PIPEK_MEZEY", rhf_wfn.basisset(), C_occ)
    Local.localize()
    print("Pipek-Mezey Localized Occupied MOs:")
    print(np.asarray(Local.L))

    # From Table II of J. Pipek and P.G. Mezey, J. Chem. Phys. 90, 4916-4926 (1989).
    pm_O_1s_local = 1.0250
    pm_C_1s_local = 1.0189

    assert((np.abs(np.asarray(C_occ)[0,0]) - pm_O_1s_canon) < 1e-4)
    assert((np.abs(np.asarray(C_occ)[5,1]) - pm_C_1s_canon) < 1e-4)
    assert((np.abs(np.asarray(Local.L)[0,0]) - pm_O_1s_local) < 1e-4)
    assert((np.abs(np.asarray(Local.L)[5,1]) - pm_C_1s_local) < 1e-4)

def test_pipek_mezey_CO_631gss():
    np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

    """CO Pipek-Mezey Test"""
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '6-31G**',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry("""
        units au
        O 0.0 0.0 2.132
        C 0.0 0.0 0.0
        """)

    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    print("Canonical Occupied MOs:")
    C_occ = rhf_wfn.Ca_subset("AO", "ACTIVE_OCC")
    print(np.asarray(C_occ))

    pm_O_1s_canon = 0.9947
    pm_C_1s_canon = 0.9960

    Local = psi4.core.Localizer.build("PIPEK_MEZEY", rhf_wfn.basisset(), C_occ)
    Local.localize()
    print("Pipek-Mezey Localized Occupied MOs:")
    print(np.asarray(Local.L))

    # From Table II of J. Pipek and P.G. Mezey, J. Chem. Phys. 90, 4916-4926 (1989).
    pm_O_1s_local = 1.0187
    pm_C_1s_local = 1.0161

    assert((np.abs(np.asarray(C_occ)[0,0]) - pm_O_1s_canon) < 1e-4)
    assert((np.abs(np.asarray(C_occ)[5,1]) - pm_C_1s_canon) < 1e-4)
    assert((np.abs(np.asarray(Local.L)[0,0]) - pm_O_1s_local) < 1e-4)
    assert((np.abs(np.asarray(Local.L)[5,1]) - pm_C_1s_local) < 1e-4)

    assert(False)
