"""
cclambda.py: Lambda-amplitude Solver
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import numpy as np
import time
from opt_einsum import contract
from .utils import helper_diis


class cclambda(object):
    """
    An RHF-CC wave function and energy object.

    Attributes
    ----------
    ccwfn : PyCC ccwfn object
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
        Solves the CC Lambda amplitude equations
    residuals()
        Computes the L1 and L2 residuals for a given set of amplitudes and Fock operator
    """
    def __init__(self, ccwfn, hbar):
        """
        Parameters
        ----------
        ccwfn : PyCC ccwfn object
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
            CC pseudoenergy

        """
        lambda_tstart = time.time()

        o = self.ccwfn.o
        v = self.ccwfn.v
        t2 = self.ccwfn.t2
        l1 = self.l1
        l2 = self.l2
        Dia = self.ccwfn.Dia
        Dijab = self.ccwfn.Dijab
        ERI = self.ccwfn.H.ERI
        L = self.ccwfn.H.L

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

        lecc = self.pseudoenergy(o, v, ERI, l2)

        print("\nLCC Iter %3d: LCC PseudoE = %.15f  dE = % .5E" % (0, lecc, -lecc))

        diis = helper_diis(l1, l2, max_diis)

        for niter in range(1, maxiter+1):

            lecc_last = lecc

            l1 = self.l1
            l2 = self.l2

            Goo = self.build_Goo(t2, l2)
            Gvv = self.build_Gvv(t2, l2)
            r1 = self.r_L1(o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)
            r2 = self.r_L2(o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)

            if self.ccwfn.local is not None:
                inc1, inc2 = self.ccwfn.Local.filter_amps(r1, r2)
                self.l1 += inc1
                self.l2 += inc2
                rms = contract('ia,ia->', inc1, inc1)
                rms += contract('ijab,ijab->', inc2, inc2)
                rms = np.sqrt(rms)
            else:
                self.l1 += r1/Dia
                self.l2 += r2/Dijab
                rms = contract('ia,ia->', r1/Dia, r1/Dia)
                rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                rms = np.sqrt(rms)

            lecc = self.pseudoenergy(o, v, ERI, self.l2)
            ediff = lecc - lecc_last
            print("LCC Iter %3d: LCC PseudoE = %.15f  dE = % .5E  rms = % .5E" % (niter, lecc, ediff, rms))

            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nLambda-CC has converged in %.3f seconds.\n" % (time.time() - lambda_tstart))
                return lecc

            diis.add_error_vector(self.l1, self.l2)
            if niter >= start_diis:
                self.l1, self.l2 = diis.extrapolate(self.l1, self.l2)

    def residuals(self, F, t1, t2, l1, l2):
        """
        Parameters
        ----------
        F : NumPy array
            current Fock matrix (useful when adding one-electron fields)
        t1, t2: NumPy arrays
            current T1 and T2 amplitudes
        l1, l2: NumPy arrays
            current L1 and L2 amplitudes

        Returns
        -------
        r1, r2: L1 and L2 residuals: r_mu = <0|(1+L) [HBAR, tau_mu]|0>
        """

        o = self.ccwfn.o
        v = self.ccwfn.v
        ERI = self.ccwfn.H.ERI
        L = self.ccwfn.H.L
        hbar = self.hbar

        Hov = hbar.build_Hov(o, v, F, L, t1)
        Hvv = hbar.build_Hvv(o, v, F, L, t1, t2)
        Hoo = hbar.build_Hoo(o, v, F, L, t1, t2)
        Hoooo = hbar.build_Hoooo(o, v, ERI, t1, t2)
        Hvvvv = hbar.build_Hvvvv(o, v, ERI, t1, t2)
        Hvovv = hbar.build_Hvovv(o, v, ERI, t1)
        Hooov = hbar.build_Hooov(o, v, ERI, t1)
        Hovvo = hbar.build_Hovvo(o, v, ERI, L, t1, t2)
        Hovov = hbar.build_Hovov(o, v, ERI, t1, t2)
        Hvvvo = hbar.build_Hvvvo(o, v, ERI, L, Hov, Hvvvv, t1, t2)
        Hovoo = hbar.build_Hovoo(o, v, ERI, L, Hov, Hoooo, t1, t2)

        Goo = self.build_Goo(t2, l2)
        Gvv = self.build_Gvv(t2, l2)
        r1 = self.r_L1(o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)
        r2 = self.r_L2(o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)

        return r1, r2

    def build_Goo(self, t2, l2):
        return contract('mjab,ijab->mi', t2, l2)


    def build_Gvv(self, t2, l2):
        return -1.0 * contract('ijeb,ijab->ae', t2, l2)


    def r_L1(self, o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
        if self.ccwfn.model == 'CCD':
            r_l1 = np.zeros_like(l1)
        else:
            r_l1 = 2.0 * Hov.copy()
            r_l1 = r_l1 + contract('ie,ea->ia', l1, Hvv)
            r_l1 = r_l1 - contract('ma,im->ia', l1, Hoo)
            r_l1 = r_l1 + contract('me,ieam->ia', l1, (2.0 * Hovvo - Hovov.swapaxes(2,3)))
            r_l1 = r_l1 + contract('imef,efam->ia', l2, Hvvvo)
            r_l1 = r_l1 - contract('mnae,iemn->ia', l2, Hovoo)
            if self.ccwfn.model != 'CC2':
                r_l1 = r_l1 - 2.0 * contract('ef,eifa->ia', Gvv, Hvovv)
                r_l1 = r_l1 + contract('ef,eiaf->ia', Gvv, Hvovv)
                r_l1 = r_l1 - 2.0 * contract('mn,mina->ia', Goo, Hooov)
                r_l1 = r_l1 + contract('mn,imna->ia', Goo, Hooov)
        return r_l1


    def r_L2(self, o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
        if self.ccwfn.model == 'CCD':
            r_l2 = L[o,o,v,v].copy()
            r_l2 = r_l2 + contract('ijeb,ea->ijab', l2, Hvv)
            r_l2 = r_l2 - contract('mjab,im->ijab', l2, Hoo)
            r_l2 = r_l2 + 0.5 * contract('mnab,ijmn->ijab', l2, Hoooo)
            r_l2 = r_l2 + 0.5 * contract('ijef,efab->ijab', l2, Hvvvv)
            r_l2 = r_l2 + contract('mjeb,ieam->ijab', l2, (2.0 * Hovvo - Hovov.swapaxes(2,3)))
            r_l2 = r_l2 - contract('mibe,jema->ijab', l2, Hovov)
            r_l2 = r_l2 - contract('mieb,jeam->ijab', l2, Hovvo)
            r_l2 = r_l2 + contract('ae,ijeb->ijab', Gvv, L[o,o,v,v])
            r_l2 = r_l2 - contract('mi,mjab->ijab', Goo, L[o,o,v,v])
        else:
            r_l2 = L[o,o,v,v].copy()
            r_l2 = r_l2 + 2.0 * contract('ia,jb->ijab', l1, Hov)
            r_l2 = r_l2 - contract('ja,ib->ijab', l1, Hov)
            r_l2 = r_l2 + 2.0 * contract('ie,ejab->ijab', l1, Hvovv)
            r_l2 = r_l2 - contract('ie,ejba->ijab', l1, Hvovv)
            r_l2 = r_l2 - 2.0 * contract('mb,jima->ijab', l1, Hooov)
            r_l2 = r_l2 + contract('mb,ijma->ijab', l1, Hooov)
            if self.ccwfn.model == 'CC2':
                r_l2 = r_l2 + contract('ijeb,ea->ijab', l2, (self.ccwfn.H.F[v,v] - contract('me,ma->ae', self.ccwfn.H.F[o,v], self.ccwfn.t1)))
                r_l2 = r_l2 - contract('mjab,im->ijab', l2, (self.ccwfn.H.F[o,o] + contract('ie,me->mi', self.ccwfn.t1, self.ccwfn.H.F[o,v])))
            else:
                r_l2 = r_l2 + contract('ijeb,ea->ijab', l2, Hvv)
                r_l2 = r_l2 - contract('mjab,im->ijab', l2, Hoo)
                r_l2 = r_l2 + 0.5 * contract('mnab,ijmn->ijab', l2, Hoooo)
                r_l2 = r_l2 + 0.5 * contract('ijef,efab->ijab', l2, Hvvvv)
                r_l2 = r_l2 + contract('mjeb,ieam->ijab', l2, (2.0 * Hovvo - Hovov.swapaxes(2,3)))
                r_l2 = r_l2 - contract('mibe,jema->ijab', l2, Hovov)
                r_l2 = r_l2 - contract('mieb,jeam->ijab', l2, Hovvo)
                r_l2 = r_l2 + contract('ae,ijeb->ijab', Gvv, L[o,o,v,v])
                r_l2 = r_l2 - contract('mi,mjab->ijab', Goo, L[o,o,v,v])

        r_l2 = r_l2 + r_l2.swapaxes(0,1).swapaxes(2,3)
        return r_l2


    def pseudoenergy(self, o, v, ERI, l2):
        return 0.5 * contract('ijab,ijab->',ERI[o,o,v,v], l2)
