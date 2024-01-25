import psi4
import sys 
sys.path.append('/Users/jattakumi/pycc/pycc') 
from data.molecules import * 
 
geom_inp ="""
F 0.0000000 0.0000000  -0.087290008493927
H 0.0000000 0.0000000 1.645494724632280
units bohr
no_reorient
symmetry c1
"""

psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', True)
psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12,
                  'diis': 1,
                  #'perturb_h': True,
                  #'perturb_with':'Dipole',
                  #'perturb_dipole': [0.0,0.0,0.0001]
})

geom = psi4.geometry(geom_inp)
rhf_e, rhf_wfn = psi4.energy('scf', return_wfn=True)
psi4.properties('CCSD',properties=['dipole'])

dpz_tot = psi4.core.variable('CCSD DIPOLE')

#dpz_tot contains contributions from SCF and CCSD as well as nuclear, need to remove SCF contribution and nuclear contribution
dpz_scf = psi4.core.variable('SCF DIPOLE')
mu_n = geom.nuclear_dipole()
print('dpz_tot', dpz_tot - mu_n[2])
elec_muz_scf = dpz_scf[2] - mu_n[2]
elec_muz_tot = dpz_tot[2] - mu_n[2]
elec_muz_cc = elec_muz_tot - elec_muz_scf
print('electric contribution only for CC', elec_muz_tot - elec_muz_scf) 
print(elec_muz_cc + elec_muz_scf + mu_n[2])
