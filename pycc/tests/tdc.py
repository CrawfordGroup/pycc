"""
Test CCSD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import sys
sys.path.insert(0, '../data')
import molecules as mol
import numpy as np

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)


# Psi4 Setup
psi4.set_memory('2 GiB')
psi4.core.set_output_file('output.dat', False)
memory = 2
psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'r_convergence': 1e-13,
                  'diis': 1})
mol = psi4.geometry(mol.moldict["(H2O)_2"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

maxiter = 75
e_conv = 1e-13
r_conv = 1e-13
cc = pycc.ccenergy(rhf_wfn)
ecc = cc.solve_ccsd(e_conv, r_conv, maxiter)

hbar = pycc.cchbar(cc)

maxiter = 75
max_diis = 8
cclambda = pycc.cclambda(cc, hbar)
lecc = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis)

ccdensity = pycc.ccdensity(cc, cclambda)
ecc_test = ccdensity.compute_energy()

print("ECC from density       = %20.15f" % ccdensity.ecc)
print("ECC from wave function = %20.15f" % cc.ecc)

# compute the dipole moment
onepdm = ccdensity.onepdm()
# add SCF component
for i in range(cc.no):
    onepdm[i,i] += 2.0
#print(onepdm)
mints = psi4.core.MintsHelper(cc.ref.basisset())
dipole_ints = mints.ao_dipole()
C = np.asarray(cc.ref.Ca_subset("AO", "ACTIVE"))
cart = ["X", "Y", "Z"]
for i in range(0,3):
    mu = C.T @ np.asarray(dipole_ints[i]) @ C
    dip = onepdm.flatten().dot(mu.flatten())
    print("MU-%s = %20.12f" % (cart[i], dip))

# All the ugly steps to get Psi4 to produce a CCSD dipole moment in PSI_API mode
psi4.core.set_global_option('WFN', 'CCSD')
epsi4, ccwfn = psi4.energy('CCSD', return_wfn=True)
psi4.core.cchbar(ccwfn)
psi4.core.set_global_option('DERTYPE', 'NONE')
psi4.core.set_global_option('ONEPDM', 'TRUE')
psi4.core.cclambda(ccwfn)
psi4.core.ccdensity(ccwfn)
oe = psi4.core.OEProp(ccwfn)
oe.add('DIPOLE')
oe.set_title('CCSD')
oe.compute()
