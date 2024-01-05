"""
cceom.py: Computes excited-state CC wave functions and energies
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import time
import numpy as np

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)


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

    def solve_eom(self, N=1, e_conv=1e-5, r_conv=1e-5, maxiter=100, guess='UNIT'):
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

        M = N * 2 # size of guess space (varies)
        maxM = M * 10 # max size of subspace (fixed)

        # Build preconditioner (energy denominator)
        hbar_occ = np.diag(hbar.Hoo)
        hbar_vir = np.diag(hbar.Hvv)
        Dia = hbar_occ.reshape(-1,1) - hbar_vir
        Dijab = (hbar_occ.reshape(-1,1,1,1) + hbar_occ.reshape(-1,1,1) - 
                hbar_vir.reshape(-1,1) - hbar_vir)
        D = np.hstack((Dia.flatten(), Dijab.flatten()))

        # Initialize guess vectors
        valid_guesses = ['UNIT', 'CIS', 'HBAR_SS']
        if guess not in valid_guesses:
            raise Exception("%s is not a valid choice of initial guess vectors." % (guess))
        _, C1 = self.guess(M, o, v, hbar, D, contract, guess)
        # Store guess vectors as rows of a matrix
        C = np.hstack((np.reshape(C1, (M, no*nv)), np.zeros((M, no*no*nv*nv))))
        print("Guess vectors obtained from %s." % (guess))

        # Array for excitation energies
        E = np.zeros((N))

        maxiter = 3
        for niter in range(1,maxiter+1):
            E_old = E.copy()

            # Orthonormalize current guess vectors
            Q, _ = np.linalg.qr(C.T)
            C = Q.T.copy()
            M = C.shape[0]

            print("EOM Iter %3d: M = %3d" % (niter, M))

            # Extract guess vectors for sigma calculation
            C1 = np.reshape(C[:,:no*nv], (M,no,nv)).copy()
            C2 = np.reshape(C[:,no*nv:], (M,no,no,nv,nv)).copy()

            # Compute sigma vectors
            s1 = np.zeros_like(C1)
            s2 = np.zeros_like(C2)
            for state in range(M):
                s1[state] = self.s1(hbar, C1[state], C2[state])
                s2[state] = self.s2(hbar, C1[state], C2[state])

            # Build and diagonalize subspace Hamiltonian
            S = np.hstack((np.reshape(s1, (M, no*nv)), np.reshape(s2, (M, no*no*nv*nv))))
            G = C @ S.T
#print(G)
            l, a = np.linalg.eig(G)
            print(l)
            idx = l.argsort()[:N]
            l = l[idx]
            a = a[:,idx]
            E = l[:N]

            # Build correction vectors
            # r --> (N, no*nv + no*no*nv*nv)
            # a --> (M, N)
            # S --> (M, no*nv + no*no*nv*nv)
            # C --> (M, no*nv + no*no*nv*nv)
            # l --> (N) 
            r = a.T @ (S - np.diag(l) @ C)
            print(r[N:].T)
            delta = r/np.subtract.outer(l,D) # element-by-element division

            # Add new vectors to guess space and orthonormalize
            C = np.concatenate((C, delta[:N]))
            M = C.shape[0]

            # Print status and check convergence and print status
            dE = E - E_old
            print(r[:N])
            r_norm = np.linalg.norm(r[:N], axis=1)
            print("             E/state                   dE                 norm")
            for state in range(N):
                print("%20.12f %20.12f %20.12f" % (E[state], dE[state], r_norm[state]))


        print("\nCCEOM converged in %.3f seconds." % (time.time() - time_init))



    def guess(self, M, o, v, hbar, D, contract, method):
        """
        Compute guess vectors for EOM-CC Davidson algorithm
        """
        no = o.stop - o.start
        nv = v.stop - v.start

        # Use unit vectors corresponding to smallest H_ii - H_aa values
        if method == 'UNIT':
#idx = D[:no*nv].argsort()[::-1][:M]
            idx = D[:no*nv].argsort()[:M]
            c = np.eye(no*nv)[:,idx]
#eps = np.sort(D[:no*nv])[::-1]
            eps = np.sort(D[:no*nv])
        # Use CIS eigenvectors
        elif method == 'CIS':
            F = hbar.ccwfn.H.F
            L = hbar.ccwfn.H.L
            H = L[v,o,o,v].swapaxes(0,1).swapaxes(0,2).copy()
            H += contract('ab,ij->iajb', F[v,v], np.eye(no))
            H -= contract('ij,ab->iajb', F[o,o], np.eye(nv))
            eps, c = np.linalg.eigh(np.reshape(H, (no*nv,no*nv)))
        # Use eigenvectors of singles-singles block of hbar (mimics Psi4)
        elif method == 'HBARSS':
            H = (2.0 * hbar.Hovvo.swapaxes(1,2).swapaxes(2,3) - hbar.Hovov.swapaxes(1,3)).copy()
            H += contract('ab,ij->iajb', hbar.Hvv, np.eye(no))
            H -= contract('ij,ab->iajb', hbar.Hoo, np.eye(nv))
            eps, c = np.linalg.eig(np.reshape(H, (no*nv,no*nv)))
            idx = eps.argsort()
            eps = eps[idx]
            c = c[:,idx]

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
