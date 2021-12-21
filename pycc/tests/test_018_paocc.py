import psi4
import pycc
import numpy as np
import pickle as pk
from pycc.data.molecules import *

"""PAO-CCSD Test"""
# Psi4 Setup
psi4.set_memory('2 GB')
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
# explicit units in au so matching w/ psi3 is easier
#    H         0.000000000000     0.000000000000     0.000000000000
#    H         0.000000000000     0.000000000000     1.417294491434
#    H         2.834588982868     0.000000000000     1.417294491434
#    H         2.834588982868     1.227413034226     0.708647245717
# units au
mol = psi4.geometry("""
    C  26.580000  26.520000  27.119999
    O  26.580000  26.520000  28.520000
    C  27.840000  26.520000  27.840000
    C  26.500000  27.700001  26.289999
    H  25.559999  27.700001  25.739998
    H  26.559999  28.590000  26.910000
    H  27.330000  27.700001  25.580002
    H  26.070002  25.629999  26.760002
    H  28.410000  25.629999  27.579998
    H  28.410000  27.410000  27.579998
    noreorient
    nocom
    """)
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
print("RHF_E: {}".format(rhf_e))

maxiter = 75
e_conv = 1e-7
r_conv = 1e-7
max_diis = 8

#ccsd = pycc.ccwfn(rhf_wfn)
ccsd = pycc.ccwfn(rhf_wfn, local='PAO', local_cutoff=2e-2)

eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter, max_diis)
print("PAO energy    : {}".format(eccsd))

# (H2)_2 REFERENCES:
#print("6-31g psi3 PAO: -0.052863089856486")
#print("6-31g ref     : -0.052863084256471564")

# METOX REFERENCES:
#print("6-31g psi3 PAO: -0.426259408543049")
#print("6-31g ref     : -0.43714625109972294")
# FRZC
print("6-31g psi3 PAO: -0.421904536004824")
#print("6-31g psi3 PAO: -0.421226226923507") # this is WITH cutting-by-norm!
print("6-31g ref     : -0.4324193006718509")

