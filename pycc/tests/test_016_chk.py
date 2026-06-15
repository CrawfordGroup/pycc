"""
Test checkpointing for rt propagation
Pull checkpoint file for 0-5.1au, then finish propagation to 10au
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pickle as pk
import numpy as np
from pycc.rt.integrators import rk2
from pycc.rt.lasers import gaussian_laser
from pycc.data.molecules import *

def test_chk(datadir):
    # run cc
    psi4.set_memory('600MB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8,
                      'r_convergence': 1e-8,
                      'diis': 8})
    mol = psi4.geometry(moldict["H2"])

    # pull ref wfn, psi4 is picky about strings 
    rhf_dir = str(datadir.join(f"ref_wfn.npy"))
    rhf_wfn = psi4.core.Wavefunction.from_file(rhf_dir)

    e_conv = 1e-8
    r_conv = 1e-8
    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    ccdensity = pycc.ccdensity(cc, cclambda)

    # narrow Gaussian pulse
    F_str = 0.001
    sigma = 0.01
    center = 0.05
    V = gaussian_laser(F_str, 0, sigma, center=center)

    # RTCC setup
    phase = 0
    h = 0.1
    tf = 10
    rtcc = pycc.rtcc(cc,cclambda,ccdensity,V,magnetic=True,kick='z')

    # pull chk files for 0-5.1au
    chk_file = datadir.join(f"chk_5.pk")
    with open(chk_file,'rb') as cf:
        chk = pk.load(cf)

    # propagate to 10au
    ODE = rk2(h)
    y0 = chk['y']
    y0 = np.append(y0, phase)
    ti = chk['time']
    ofile = datadir.join(f"output.pk")
    tfile = datadir.join(f"t_out.pk")
    ret, ret_t = rtcc.propagate(ODE, y0, tf, ti=ti, ref=False, chk=True, tchk=1,
            ofile=ofile, tfile=tfile, k=2)

    # reference is "full" propagation (0-10au)
    refp_file = datadir.join(f"output_full.pk")
    with open(refp_file,'rb') as pf:
        ref_p = pk.load(pf)
    reft_file = datadir.join(f"t_out_full.pk")
    with open(reft_file,'rb') as ampf:
        ref_t = pk.load(ampf)
        
    # check properties
    pchk = ['ecc','mu_x','mu_y','mu_z','m_x','m_y','m_z']
    for k in ref_p.keys():
        for p in pchk:
            assert np.allclose(ret[k][p],ref_p[k][p])

    # check amplitudes
    tchk = ['t1','t2','l1','l2']
    for k in ref_t.keys():
        for t in tchk:
            assert np.allclose(ret_t[k][t],ref_t[k][t])
