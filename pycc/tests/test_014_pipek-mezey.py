"""
Test basic LPNO-CCSD energy and Lambda code
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
import numpy as np


def test_lpno_ccsd():
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

    Local = psi4.core.Localizer.build("PIPEK_MEZEY", rhf_wfn.basisset(), C_occ)
    Local.localize()
    print("Pipek-Mezey Localized Occupied MOs:")
    print(np.asarray(Local.L))

    assert(False)  # To force printing in GHA
