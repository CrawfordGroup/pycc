# Test case for computing dipole moment with simulation code" 
import psi4 
import pycc
import numpy as np
#import h5py
# from ccwfn import ccwfn
# from cchbar import cchbar
# from cclambda import cclambda
# from ccdensity import ccdensity
import sys
sys.path.append ("/Users/jattakumi/pycc/pycc")
from data.molecules import *
from opt_einsum import contract
#from hamiltonian_AO import Hamiltonian_AO
#from hfwfn import hfwfn
import os

hf = """
F 0.0000000 0.0000000  -0.087290008493927
H 0.0000000 0.0000000 1.645494724632280
units bohr
no_reorient
symmetry c1
"""

## local code 
# intialize variables from ccwfn, cchabr, cclambda, hamiltonian (MO mu), etc. 
# Psi4 Setup
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12,
                  'diis': 1
})

mol = psi4.geometry(hf)

PNO_cutoff = 0
basis = 'PNO'
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
ccsd_local = pycc.ccwfn(rhf_wfn, model= 'CCSD', local=basis, local_cutoff = PNO_cutoff, filter=False)

e_conv = 1e-8
r_conv = 1e-8
maxiter = 200

eccsd = ccsd_local.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
hbar_local = pycc.cchbar(ccsd_local)
cclambda_local = pycc.cclambda(ccsd_local, hbar_local)
lccsd_local = cclambda_local.solve_llambda(e_conv, r_conv, maxiter)
ccdensity_local = pycc.ccdensity(ccsd_local, cclambda_local)
#ecc_density = ccdensity_local.compute_energy()
t1 = ccdensity_local.ccwfn.t1
t2 = ccdensity_local.ccwfn.t2
l1 = ccdensity_local.cclambda.l1
l2 = ccdensity_local.cclambda.l2

#opdm = ccdensity_local.compute_onepdm(t1,t2,l1,l2) # withref = True)
lDoo = ccdensity_local.Doo
lDvv = ccdensity_local.Dvv
lDov = ccdensity_local.Dov

#lDoo, oo_energy = ccdensity_local.build_lDoo(t1, t2, l1, l2)
#lDvv, vv_energy = ccdensity_local.build_lDvv(lDvv, t1, t2, l1, l2)
#lDov, ov_energy = ccdensity_local.build_lDov(lDov, t1, t2, l1, l2)
#lDvo, vo_energy = ccdensity_local.build_lDvo(t2, l2)

#hard coding local dipole moment contract('pq,pq->', opdm, mu)
mu = ccsd_local.H.mu
o = ccsd_local.o
v = ccsd_local.v
no = ccsd_local.no
nv = ccsd_local.nv
lQ = ccsd_local.Local.Q
lL = ccsd_local.Local.L
dim = ccsd_local.Local.dim

# convert MO mu to PNO mu
lmu_oo = mu[2][o,o]
lmu_vv = []
lmu_ov = []
lmu_vo = []

# no need to convert lmoo to PNO since not possible for occupied-occupied space
for i in range(no):
    ii = i * no + i
    lmu_ov.append(mu[2][i,v] @ (lQ[ii] @ lL[ii]))
    lmu_vo.append((lQ[ii] @ lL[ii]).T @ mu[2][v,i])
    for j in range(no):
        ij = i * no + j
        lmu_vv.append((lQ[ij] @ lL[ij]).T @ mu[2][v, v] @ (lQ[ij] @ lL[ij]))

# similar format to simulation code block 
# simulation code 

mu_z_oo = 0 
mu_z_ov = 0 
mu_z_vv = 0
mu_z_vo = 0
for i in range(no):
    ii = i*no + i
    mu_z_ov += contract('a,a->', lDov[i], lmu_ov[i])
    mu_z_vo += contract('a,a->', l1[i], lmu_vo[i])
    for j in range(no):
        ij = i * no + j
        mu_z_vv += contract('ab, ab->', lDvv[ij], lmu_vv[ij]) #lDvv[no + a,no+ b] * lmu_vv[no + a,no + b] 
        mu_z_oo += lDoo[i,j] * lmu_oo[i,j]
print("ov", mu_z_ov)
print("vo", mu_z_vo)
print("vv", mu_z_vv)
print("oo", mu_z_oo)
#for i in range(no):
#    for j in range(no):
#        #print("i,j", i,j)
#        mu_z_oo += lDoo[i, j] * lmu_oo[i, j]
#
#print("added ij", mu_z_oo)
#
#for i in range(no):
#    for a in range(nv):
#        mu_z_ov += lDov[i,no + a] * lmu_ov[i, no + a]
#        #mu_z_vo += lDvo[no +a , i] * lmu_vo[no + a, i]
#
#print("added ia", mu_z_ov)
#print("added ai", mu_z_vo)
#
#for a in range(nv):
#    for b in range(nv):
#        mu_z_vv += lDvv[no + a,no+ b] * lmu_vv[no + a,no + b]
#
#print("added ab", mu_z_vv)
print("total", mu_z_oo + mu_z_ov + mu_z_vv + mu_z_vo) #+ dpz_scf)
