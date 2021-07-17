import numpy as np
from opt_einsum import contract
import scipy.linalg

# Need to generate the transformation matrices for the virtual-MO spaces for LPNO
# Build MP2 amplitudes via iteration -- need Hamiltonian
# Loop over i,j pairs:
#   Build onepdm
#   Diagonalize onepdm 
#   Eliminate VNOs based on input cutoff

class Local(object):
    
    def __init__(self, wfn, o, v, H, cutoff):

        # Compute MP2 amplitudes in non-canonical MO basis
        eps_occ = np.diag(H.F)[o]
        eps_vir = np.diag(H.F)[v]
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # initial guess amplitudes
        t2 = H.ERI[o,o,v,v]/Dijab

        emp2 = contract('ijab,ijab->', t2, H.L[o,o,v,v])
        print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E" % (0, emp2, -emp2))

        e_conv=1e-7
        r_conv=1e-7
        maxiter=100
        ediff = emp2
        rmsd = 0.0
        niter = 0

        while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
            elast = emp2

            r2 = 0.5 * H.ERI[o,o,v,v].copy()
            r2 += contract('ijae,be->ijab', t2, H.F[v,v]) 
            r2 -= contract('imab,mj->ijab', t2, H.F[o,o])
            r2 = r2 + r2.swapaxes(0,1).swapaxes(2,3)
            
            t2 += r2/Dijab

            rmsd = np.sqrt(contract('ijab,ijab->', r2/Dijab, r2/Dijab))

            emp2 = contract('ijab,ijab->', t2, H.L[o,o,v,v])
            ediff = emp2 - elast

            print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E  rmsd = % .5E" % (niter, emp2, ediff, rmsd))

        no = o[1] - o[0]
        for ij in range(no*no):

