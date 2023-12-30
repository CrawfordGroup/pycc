"""
cceom.py: Computes excited-state CC wave functions and energies
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import time
import numpy as np


class cceom(object):
    """
    An Equation-of-Motion Coupled Cluster Object.

    """

    def __init__(self, cchbar):
        """
        Parameters
        ----------
        cchbar : PyCC cchbar object

        Returns
        -------
        """

        self.cchbar = cchbar

    def solve_eom(self, nstates=1, e_conv=1e-5, r_conv=1e-5, maxiter=100):
        """

        """

        time_init = time.time()

        hbar = self.cchbar
        o = hbar.o
        v = hbar.v
        no = hbar.no
        nv = hbar.nv
        H = hbar.ccwfn.H
        contract = hbar.contract

        L = nstates * 3 # size of guess space (varies)
        maxL = L * 10 # max size of subspace (fixed)

        # Initialize guess vectors
        _, C1 = self.cis(L, o, v, H, contract)
        C2 = np.zeros((L, no, no, nv, nv))

        # Compute sigma vectors
        s1 = np.zeros_like(C1)
        s2 = np.zeros_like(C2)
        for state in range(L):
            s1[state] = self.s1(hbar, C1[state], C2[state])
            s2[state] = self.s2(hbar, C1[state], C2[state])

        # Build and diagonalize subspace Hamiltonian
        G = np.zeros((L,L))

        # Build and orthonormalize correction vectors

        # 

        print("\nCCEOM converged in %.3f seconds." % (time.time() - time_init))



    def cis(self, nstates, o, v, H, contract):
        """
        Compute CIS excited states as guess vectors
        """
        no = o.stop - o.start
        nv = v.stop - v.start
        F = H.F
        L = H.L

        CIS_H = L[v,o,o,v].swapaxes(0,1).swapaxes(0,2).copy()
        CIS_H += contract('ab,ij->iajb', F[v,v], np.eye(no))
        CIS_H -= contract('ij,ab->iajb', F[o,o], np.eye(nv))

        eps, c = np.linalg.eigh(np.reshape(CIS_H, (no*nv,no*nv)))

        c = np.reshape(c.T[slice(0,nstates),:], (nstates, no, nv)).copy()

        return eps, c


    def s1(self, hbar, C1, C2):

        contract = hbar.contract

        s1 = contract('ie,ae->ia', C1, hbar.Hvv)
        s1 -= contract('mi,ma->ia', hbar.Hoo, C1)
        s1 += 2.0 * contract('maei,me->ia', hbar.Hovvo, C1)
        s1 -= contract('maie,me->ia', hbar.Hovov, C1)
        s1 += 2.0 * contract('miea,me->ia', C2, hbar.Hov)
        s1 -= contract('imea,me->ia', C2, hbar.Hov)
        s1 += 2.0 * contract('imef,amef->ia', C2, hbar.Hvovv)
        s1 -= contract('imef,amfe->ia', C2, hbar.Hvovv)
        s1 -= 2.0 * contract('mnie,mnae->ia', hbar.Hooov, C2)
        s1 += contract('nmie,mnae->ia', hbar.Hooov, C2)

        return s1


    def s2(self, hbar, C1, C2):

        contract = hbar.contract
        L = hbar.ccwfn.H.L
        t2 = hbar.ccwfn.t2
        o = hbar.ccwfn.o
        v = hbar.ccwfn.v

        Zvv = 2.0 * contract('amef,mf->ae', hbar.Hvovv, C1)
        Zvv -= contract('amfe,mf->ae', hbar.Hvovv, C1)
        Zvv -= contract('nmaf,nmef->ae', C2, L[o,o,v,v])

        Zoo = -2.0 * contract('mnie,ne->mi', hbar.Hooov, C1)
        Zoo += contract('nmie,ne->mi', hbar.Hooov, C1)
        Zoo -= contract('mnef,inef->mi', L[o,o,v,v], C2)

        s2 = contract('ie,abej->ijab', C1, hbar.Hvvvo)
        s2 -= contract('mbij,ma->ijab', hbar.Hovoo, C1)
        s2 += contract('ijeb,ae->ijab', t2, Zvv)
        s2 += contract('mi,mjab->ijab', Zoo, t2)
        s2 += contract('ijeb,ae->ijab', C2, hbar.Hvv)
        s2 -= contract('mi,mjab->ijab', hbar.Hoo, C2)
        s2 += 0.5 * contract('mnij,mnab->ijab', hbar.Hoooo, C2)
        s2 += 0.5 * contract('ijef,abef->ijab', C2, hbar.Hvvvv)
        s2 -= contract('imeb,maje->ijab', C2, hbar.Hovov)
        s2 -= contract('imea,mbej->ijab', C2, hbar.Hovvo)
        s2 += 2.0 * contract('miea,mbej->ijab', C2, hbar.Hovvo)
        s2 -= contract('miea,mbje->ijab', C2, hbar.Hovov)

        return s2 + s2.swapaxes(0,1).swapaxes(2,3)
