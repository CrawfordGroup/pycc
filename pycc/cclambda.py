"""                                                                                                            
cclambda.py: Lambda-amplitude Solver
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import numpy as np
import time
from opt_einsum import contract
from .utils import helper_diis
from .lambda_eqs import r_L1, r_L2, build_Goo, build_Gvv, pseudoenergy


class cclambda(object):
    """
    An RHF-CCSD wave function and energy object.

    Attributes
    ----------
    ccwfn : PyCC ccenergy object
        the coupled cluster T amplitudes and supporting data structures
    hbar : PyCC cchbar object
        the coupled cluster similarity-transformed Hamiltonian
    l1 : NumPy array
        L1 amplitudes
    l2 : NumPy array
        L2 amplitudes

    Methods
    -------
    solve_lambda()
        Solves the CCSD Lambda amplitude equations
    """
    def __init__(self, ccwfn, hbar):
        """
        Parameters
        ----------
        ccwfn : PyCC ccenergy object
            the coupled cluster T amplitudes and supporting data structures
        hbar : PyCC cchbar object
            the coupled cluster similarity-transformed Hamiltonian

        Returns
        -------
        None
        """

        self.ccwfn = ccwfn
        self.hbar = hbar

        self.l1 = 2.0 * self.ccwfn.t1
        self.l2 = 2.0 * (2.0 * self.ccwfn.t2 - self.ccwfn.t2.swapaxes(2, 3))

    def solve_lambda(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1):
        """
        Parameters
        ----------
        e_conv : float
            convergence condition for correlation energy (default if 1e-7)
        r_conv : float
            convergence condition for wave function rmsd (default if 1e-7)
        maxiter : int
            maximum allowed number of iterations of the CC equations (default is 100)
        max_diis : int
            maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
        start_diis : int
            earliest iteration to start DIIS extrapolations (default is 1)

        Returns
        -------
        lecc : float
            CCSD pseudoenergy

        """
        lambda_tstart = time.time()

        o = self.ccwfn.o
        v = self.ccwfn.v
        t2 = self.ccwfn.t2
        l1 = self.l1
        l2 = self.l2
        Dia = self.ccwfn.Dia
        Dijab = self.ccwfn.Dijab
        ERI = self.ccwfn.ERI
        L = self.ccwfn.L

        Hov = self.hbar.Hov
        Hvv = self.hbar.Hvv
        Hoo = self.hbar.Hoo
        Hoooo = self.hbar.Hoooo
        Hvvvv = self.hbar.Hvvvv
        Hvovv = self.hbar.Hvovv
        Hooov = self.hbar.Hooov
        Hovvo = self.hbar.Hovvo
        Hovov = self.hbar.Hovov
        Hvvvo = self.hbar.Hvvvo
        Hovoo = self.hbar.Hovoo

        lecc = pseudoenergy(o, v, ERI, l2)

        print("\nLCCSD Iter %3d: LCCSD PseudoE = %.15f  dE = % .5E" % (0, lecc, -lecc))

        diis = helper_diis(l1, l2, max_diis)

        for niter in range(1,maxiter+1):

            lecc_last = lecc

            l1 = self.l1
            l2 = self.l2

            Goo = build_Goo(t2, l2)
            Gvv =  build_Gvv(t2, l2)
            r1 = r_L1(o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)
            r2 = r_L2(o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)

#print("niter = %d; r1 norm = %20.15f;  r2 norm = %20.15f" %(niter, np.linalg.norm(r1), np.linalg.norm(r2)))
            self.l1 += r1/Dia
            self.l2 += r2/Dijab

            rms = contract('ia,ia->', r1/Dia, r1/Dia)
            rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
            rms = np.sqrt(rms)

            lecc = pseudoenergy(o, v, ERI, self.l2)
            ediff = lecc - lecc_last
            print("LCCSD Iter %3d: LCCSD PseudoE = %.15f  dE = % .5E  rms = % .5E" % (niter, lecc, ediff, rms))

            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nLambda-CCSD has converged in %.3f seconds.\n" % (time.time() - lambda_tstart))
                return lecc

            diis.add_error_vector(self.l1, self.l2)
            if niter >= start_diis:
                self.l1, self.l2 = diis.extrapolate(self.l1, self.l2)
