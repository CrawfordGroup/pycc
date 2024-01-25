# -*- coding: utf-8 -*-
"""
A simple python script to calculate CCSD hyperpolarizability in length using coupled cluster linear response theory.

References: 
- Equations and algorithms from [Koch:1991:3333] and [Crawford:xxxx]
"""

__authors__ = "Monika Kodrycka"
__credits__ = [
    "Ashutosh Kumar", "Daniel G. A. Smith", "Lori A. Burns", "T. D. Crawford"
]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-02-20"

import os.path
import sys
import numpy as np
np.set_printoptions(precision=15, linewidth=200, suppress=True)
# Import all the coupled cluster utilities
sys.path.insert(0, '')
from helper_ccenergy import *
from helper_cchbar import *
from helper_cclambda import *
from helper_ccpert import *
import pycc
import psi4
from .psi4 import constants as pc

psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# can only handle C1 symmetry

mol = psi4.geometry("""
F    0.000000000000     0.000000000000    0.000000000000
H    0.000000000000     0.000000000000   -1.732800000000
units bohr
no_reorient
symmetry c1
""")

#O
#H 1 1.8084679
#H 1 1.8084679 2 104.5
#units bohr
#symmetry c1
#no_reorient

# setting up SCF options
psi4.set_options({
    'basis': 'aug-cc-pvdz',
    'scf_type': 'PK',
    'd_convergence': 1e-12,
    'e_convergence': 1e-12,
    'r_convergence' : 1e-12,
})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

print('RHF Final Energy                          % 16.10f\n' % rhf_e)

# Calculate Ground State CCSD energy
ccsd = HelperCCEnergy(mol, rhf_e, rhf_wfn, memory=2)
ccsd.compute_energy(e_conv=1e-12, r_conv=1e-12)

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

# Now that we have T1 and T2 amplitudes, we can construct
# the pieces of the similarity transformed hamiltonian (Hbar).
cchbar = HelperCCHbar(ccsd)

# Calculate Lambda amplitudes using Hbar
cclambda = HelperCCLambda(ccsd, cchbar)
cclambda.compute_lambda(r_conv=1e-12)

# frequency of calculation
omega1 = 0.0656
omega_sum = -(omega1) #+omega2)

#already defined in ccresponse
cart = ['X', 'Y', 'Z']

#already defined in ccresponsee
Mu = {}

ccpert = {}
ccpert_om1 = {}
ccpert_om_sum = {}

ccpert_om1_2nd = {}
ccpert_om_sum_2nd = {}

polar_AB = {}

# Obtain AO Dipole Matrices From Mints
dipole_array = ccsd.mints.ao_dipole()

for i in range(0, 3):
    string = "MU_" + cart[i]

    # Transform dipole integrals from AO to MO basis
    Mu[string] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                           np.asarray(dipole_array[i]))

    # Initializing the perturbation classs corresponding to dipole perturabtion at the given omega
    # First set
    ccpert_om_sum[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
                                  omega_sum)

    ccpert_om1[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
                                  omega1)

#    #Second set
#    ccpert_om_sum_2nd[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
#                                  -(-omega1-omega2))
#
#    ccpert_om1_2nd[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
#                                  -omega1)
#
#    ccpert_om2_2nd[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
#                                  -omega2)

    # Solve X and Y amplitudes corresponding to dipole perturabtion at the given omega
    print('\nsolving right hand perturbed amplitudes for omega_sum%s\n' % string)
    ccpert_om_sum[string].solve('right', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om_sum[string].solve('left', r_conv=1e-12)

    print('\nsolving right hand perturbed amplitudes for omega1 %s\n' % string)
    ccpert_om1[string].solve('right', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om1[string].solve('left', r_conv=1e-12)

#    # Solve a second set of equations for Solve X and Y amplitudes corresponding to dipole perturabtion at the given omega 
#    print('\nsolving right hand perturbed amplitudes for -omega_sum %s\n' % string)
#    ccpert_om_sum_2nd[string].solve('right', r_conv=1e-12)
#
#    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
#    ccpert_om_sum_2nd[string].solve('left', r_conv=1e-12)
#
#    print('\nsolving left hand perturbed amplitudes for -omega1 %s\n' % string)
#    ccpert_om1_2nd[string].solve('right', r_conv=1e-12)
#
#    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
#    ccpert_om1_2nd[string].solve('left', r_conv=1e-12)
#
#    print('\nsolving right hand perturbed amplitudes for -omega2  %s\n' % string)
#    ccpert_om2_2nd[string].solve('right', r_conv=1e-12)
#
#    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
#    ccpert_om2_2nd[string].solve('left', r_conv=1e-12)



# Ex. Beta_xyz = <<mu_x;mu_y;nu_z>>, where mu_x = x and mu_y = y, nu_z = z

print("\nComputing <<Mu;Mu> tensor @ %.4f nm" %(omega1))

polar_AB_1st = np.zeros((3,3))
polar_AB_2nd = np.zeros((3,3))
polar_AB = np.zeros((3,3))
for a in range(0, 3):
    str_a = "MU_" + cart[a]
    for b in range(0, 3):
        str_b = "MU_" + cart[b]
        polar_AB_1st[a,b] =  HelperCCLinresp(cclambda, ccpert_om_sum[str_a],
                             ccpert_om1[str_b]).linresp()
        #polar_AB_2nd[a,b] =  HelperCCLinresp(cclambda, ccpert_om_sum_2nd[str_a],
        #                     ccpert_om1_2nd[str_b]).linresp()
        #polar_AB[a,b] = (polar_AB_1st[a,b] + polar_AB_2nd[a,b])/2
        #hyper_AB_2nd[a,b,c] =  HelperCCQuadraticResp(ccsd, cchbar, cclambda, ccpert_om_sum_2nd[str_a],
        #                      ccpert_om1_2nd[str_b],ccpert_om2_2nd[str_c]).quadraticresp()
        #hyper_AB[a,b,c] = (hyper_AB_1st[a,b,c] + hyper_AB_2nd[a,b,c] )/2


print("\nPolarizability:")
print("\Alpha_xx = %10.12lf" %(polar_AB_1st[0,0]))
print("\Alpha_yy = %10.12lf" %(polar_AB_1st[1,1]))
print("\Alpha_zz = %10.12lf" %(polar_AB_1st[2,2]))

