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

        M = nstates * 2 # size of guess space (varies)
        maxM = M * 10 # max size of subspace (fixed)

        # Initialize guess vectors
        E, C1 = self.guess(M, o, v, H, contract)
        C2 = np.zeros((M, no, no, nv, nv))
        E = E[:nstates]
        print("EOM Iter %3d: M = %3d" % (0, M))
        print("             E/state                   dE                 norm")
        for state in range(nstates):
            print("%20.12f                  ---                  ---" % (E[state]))

        # Build preconditioner (energy denominator)
        hbar_occ = np.diag(hbar.Hoo)
        hbar_vir = np.diag(hbar.Hvv)
        Dia = hbar_occ.reshape(-1,1) - hbar_vir
        Dijab = (hbar_occ.reshape(-1,1,1,1) + hbar_occ.reshape(-1,1,1) - 
                hbar_vir.reshape(-1,1) - hbar_vir)
        D = np.hstack((Dia.flatten(), Dijab.flatten()))

        maxiter = 20
        for niter in range(maxiter):
            E_old = E
    
            # Compute sigma vectors
            s1 = np.zeros_like(C1)
            s2 = np.zeros_like(C2)
            for state in range(M):
                s1[state] = self.s1(hbar, C1[state], C2[state])
                s2[state] = self.s2(hbar, C1[state], C2[state])

            # Build and diagonalize subspace Hamiltonian
            S = np.hstack((np.reshape(s1, (M, no*nv)), np.reshape(s2, (M, no*no*nv*nv))))
            C = np.hstack((np.reshape(C1, (M, no*nv)), np.reshape(C2, (M, no*no*nv*nv))))
            G = C @ S.T
            l, a = np.linalg.eigh(G)
            E = l[:nstates]

            # Build correction vectors
            # r --> (M, no*nv + no*no*nv*nv)
            # a --> (M, M)
            # S --> (M, no*nv + no*no*nv*nv)
            # C --> (M, no*nv + no*no*nv*nv)
            # l --> (M) 
            r = a @ (S - np.diag(l) @ C)
            delta = r/np.subtract.outer(l,D) # element-by-element division

            # Add new vectors to guess space and orthonormalize
            Q, _ = np.linalg.qr(np.concatenate((C, delta[:nstates])).T)
            C = Q.T.copy()
            M = C.shape[0]

            # Print status and check convergence and print status
            dE = E - E_old
            r_norm = np.linalg.norm(r[:nstates], axis=1)
            print("EOM Iter %3d: M = %3d" % (niter, M))
            print("             E/state                   dE                 norm")
            for state in range(nstates):
                print("%20.12f %20.12f %20.12f" % (E[state], dE[state], r_norm[state]))



            # Re-shape guess vectors for next iteration
            C1 = np.reshape(C[:,:no*nv], (M,no,nv)).copy()
            C2 = np.reshape(C[:,no*nv:], (M,no,no,nv,nv)).copy()

        print("\nCCEOM converged in %.3f seconds." % (time.time() - time_init))



    def guess(self, M, o, v, H, contract, method='CCS'):
        """
        Compute CIS excited states as guess vectors
        """
        no = o.stop - o.start
        nv = v.stop - v.start
        F = H.F
        L = H.L

        # Build CIS matrix and diagonalize
        CIS_H = L[v,o,o,v].swapaxes(0,1).swapaxes(0,2).copy()
        CIS_H += contract('ab,ij->iajb', F[v,v], np.eye(no))
        CIS_H -= contract('ij,ab->iajb', F[o,o], np.eye(nv))
        eps, c = np.linalg.eigh(np.reshape(CIS_H, (no*nv,no*nv)))

        # Build list of guess vectors (C1 tensors)
        guesses = np.reshape(c.T[slice(0,M),:], (M, no, nv)).copy()

        return eps[:M], guesses


    def s1(self, hbar, C1, C2):

        contract = hbar.contract

        s1 = contract('ie,ae->ia', C1, hbar.Hvv)
        s1 -= contract('mi,ma->ia', hbar.Hoo, C1)
        s1 += contract('maei,me->ia', hbar.Hovvo, C1) * 2.0
        s1 -= contract('maie,me->ia', hbar.Hovov, C1)
        s1 += contract('miea,me->ia', C2, hbar.Hov) * 2.0
        s1 -= contract('imea,me->ia', C2, hbar.Hov)
        s1 += contract('imef,amef->ia', C2, hbar.Hvovv) * 2.0
        s1 -= contract('imef,amfe->ia', C2, hbar.Hvovv)
        s1 -= contract('mnie,mnae->ia', hbar.Hooov, C2) * 2.0
        s1 += contract('nmie,mnae->ia', hbar.Hooov, C2)

        return s1


    def s2(self, hbar, C1, C2):

        contract = hbar.contract
        L = hbar.ccwfn.H.L
        t2 = hbar.ccwfn.t2
        o = hbar.ccwfn.o
        v = hbar.ccwfn.v

        Zvv = contract('amef,mf->ae', hbar.Hvovv, C1) * 2.0
        Zvv -= contract('amfe,mf->ae', hbar.Hvovv, C1)
        Zvv -= contract('nmaf,nmef->ae', C2, L[o,o,v,v])

        Zoo = contract('mnie,ne->mi', hbar.Hooov, C1) * -2.0
        Zoo += contract('nmie,ne->mi', hbar.Hooov, C1)
        Zoo -= contract('mnef,inef->mi', L[o,o,v,v], C2)

        s2 = contract('ie,abej->ijab', C1, hbar.Hvvvo)
        s2 -= contract('mbij,ma->ijab', hbar.Hovoo, C1)
        s2 += contract('ijeb,ae->ijab', t2, Zvv)
        s2 += contract('mi,mjab->ijab', Zoo, t2)
        s2 += contract('ijeb,ae->ijab', C2, hbar.Hvv)
        s2 -= contract('mi,mjab->ijab', hbar.Hoo, C2)
        s2 += contract('mnij,mnab->ijab', hbar.Hoooo, C2) * 0.5
        s2 += contract('ijef,abef->ijab', C2, hbar.Hvvvv) * 0.5
        s2 -= contract('imeb,maje->ijab', C2, hbar.Hovov)
        s2 -= contract('imea,mbej->ijab', C2, hbar.Hovvo)
        s2 += contract('miea,mbej->ijab', C2, hbar.Hovvo) * 2.0
        s2 -= contract('miea,mbje->ijab', C2, hbar.Hovov)

        return s2 + s2.swapaxes(0,1).swapaxes(2,3)
