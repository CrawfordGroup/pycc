"""
cceom.py: Computes excited-state CC wave functions and energies
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import time
import psi4
import numpy as np


class cceom(object):
    """
    An Equation-of-Motion Coupled Cluster Object.

    Attributes
    ----------
    cchbar : PyCC cchbar object
    D : NumPy array
        orbital energy difference array (only needed for unit-vector guesses)

    Methods
    -------
    solve_eom()
        Solves the right-hand EOM-CC eigenvalue problem using the Davidson algorithm

    guess()
        Generate initial guesses to eigenvalue problem using various single-excitation methods
    s1()
        Build the singles components of the sigma = HBAR * C vector
    s2()
        Build the doubles components of the sigma = HBAR * C vector
    """

    def __init__(self, cchbar):
        """
        Parameters
        ----------
        cchbar : PyCC cchbar object

        Returns
        -------
        None
        """
        self.hbar = cchbar

        # Build preconditioner (energy denominator)
        hbar_occ = np.diag(cchbar.Hoo)
        hbar_vir = np.diag(cchbar.Hvv)
        Dia = hbar_occ.reshape(-1,1) - hbar_vir
        Dijab = (hbar_occ.reshape(-1,1,1,1) + hbar_occ.reshape(-1,1,1) - 
                hbar_vir.reshape(-1,1) - hbar_vir)
        self.D = np.hstack((Dia.flatten(), Dijab.flatten()))

    def solve_eom(self, N=1, e_conv=1e-5, r_conv=1e-5, maxiter=100, guess='HBAR_SS'):
        """
        Solves the right-hand EOM-CC eigenvalue problem using the Davidson algorithm

        Parameters
        ----------
        N : int
            number of EOM-CC excited states to compute
        e_conv : float
            convergence condition for excitation energies (default 1e-6)
        r_conv : float
            convergence condition for RMSD on excitation vectors (default 1e-6)
        maxiter : int
            maximum allowed number of iterations in the Davidson algorithm (default is 100)
        guess : str
            method to use for computing guess vectors

        Returns
        -------
        None
        """

        time_init = time.time()

        hbar = self.hbar
        o = hbar.o
        v = hbar.v
        no = hbar.no
        nv = hbar.nv
        H = hbar.ccwfn.H
        contract = hbar.contract
        D = self.D

        s1_len = no*nv
        s2_len = no*no*nv*nv

        M = N * 2 # initial size of guess space
        sigma_done = 0 # number of sigma vectors already computed
        maxM = N * 10 # max size of subspace

        # Initialize guess vectors
        valid_guesses = ['UNIT', 'CIS', 'HBAR_SS']
        guess = guess.upper()
        if guess not in valid_guesses:
            raise Exception("%s is not a valid choice of initial guess vectors." % (guess))
        _, C1 = self.guess(M, guess)
        # Store guess vectors as rows of a matrix
        C = np.hstack((np.reshape(C1, (M, s1_len)), np.zeros((M, s2_len))))
        print("Guess vectors obtained from %s." % (guess))

        # Initialize sigma vector storage
        sigma_len = s1_len + s2_len
        S = np.empty((0,sigma_len), float)      

        # Array for excitation energies
        E = np.zeros((N))

        converged = False
        for niter in range(1,maxiter+1):
            E_old = E

            # Orthonormalize current guess vectors
            Q, _ = np.linalg.qr(C.T)
            phase = np.diag((C @ Q)[:M])
            phase = np.append(phase, np.ones(Q.shape[1]-M))
            Q = phase * Q
            C = Q.T.copy()
            M = C.shape[0]

            print("EOM Iter %3d: M = %3d" % (niter, M))

            # Extract guess vectors for sigma calculation
            nvecs = M - sigma_done
            C1 = np.reshape(C[sigma_done:M,:s1_len], (nvecs,no,nv))
            C2 = np.reshape(C[sigma_done:M,s1_len:], (nvecs,no,no,nv,nv))

            # Compute sigma vectors
            s1 = np.zeros_like(C1)
            s2 = np.zeros_like(C2)
            for state in range(nvecs):
                s1[state] = self.s1(hbar, C1[state], C2[state])
                s2[state] = self.s2(hbar, C1[state], C2[state])
            sigma_done = M

            # Build and diagonalize subspace Hamiltonian
            S = np.vstack((S, np.hstack((np.reshape(s1, (nvecs, s1_len)), np.reshape(s2, (nvecs, s2_len))))))
            G = C @ S.T
            E, a = np.linalg.eig(G)

            # Sort eigenvalues and corresponding eigenvectors into ascending order
            idx = E.argsort()[:N]
            E = E[idx]; a = a[:,idx]

            # Build correction vectors
            r = a.T @ S - np.diag(E) @ a.T @ C
            r_norm = np.linalg.norm(r, axis=1)
            delta = r/np.subtract.outer(E,D) # element-by-element division

            # Print status and check convergence and print status
            dE = E - E_old
            print("             E/state                   dE                 norm")
            for state in range(N):
                print("%20.12f %20.12f %20.12f" % (E[state], dE[state], r_norm[state]))

            if (np.abs(np.linalg.norm(dE)) <= e_conv):
                converged = True
                break

            if M >= maxM:
                # Collapse to N vectors if subspace is too large
                print("\nMaximum allowed subspace dimension (%d) reached. Collapsing to N roots." % (maxM))
                C = a.T @ C
                M = N
                E = E_old
                sigma_done = 0
                S = np.empty((0,sigma_len), float)      
            else:
                # Add new vectors to guess space
                C = np.concatenate((C, delta[:N]))

        if converged:
            print("\nCCEOM converged in %.3f seconds." % (time.time() - time_init))
            print("\nState     E_h           eV")
            print("-----  ------------  ------------")
            eVconv = psi4.qcel.constants.get("hartree energy in ev")
            for state in range(N):
                print("  %3d  %12.10f  %12.10f" %(state, E[state], E[state]*eVconv))

            return E, C


    def guess(self, M, method):
        """
        Compute single-excitation guess vectors for EOM-CC Davidson algorithm

        Parameters
        ----------
        M : int
            number of guesses to generate
        method : str
            choice of method to generate guesses

        Returns
        -------
        eps : NumPy array
            eigenvalues/energies associated with guess vectors
        guesses : NumPy array
            guess vectors (as rows of matrix)
        """

        hbar = self.hbar
        o = hbar.o
        v = hbar.v
        no = hbar.no
        nv = hbar.nv
        contract = hbar.contract
        D = self.D

        # Use unit vectors corresponding to smallest (not most negative) H_ii - H_aa values
        if method == 'UNIT':
            idx = D[:no*nv].argsort()[::-1][:M]
            c = np.eye(no*nv)[:,idx]
            eps = np.sort(D[:no*nv])[::-1]
        # Use CIS eigenvectors
        elif method == 'CIS':
            F = hbar.ccwfn.H.F
            L = hbar.ccwfn.H.L
            H = L[v,o,o,v].swapaxes(0,1).swapaxes(0,2).copy()
            H += contract('ab,ij->iajb', F[v,v], np.eye(no))
            H -= contract('ij,ab->iajb', F[o,o], np.eye(nv))
            eps, c = np.linalg.eigh(np.reshape(H, (no*nv,no*nv)))
        # Use eigenvectors of singles-singles block of hbar (mimics Psi4)
        elif method == 'HBAR_SS':
            H = (2.0 * hbar.Hovvo.swapaxes(1,2).swapaxes(2,3) - hbar.Hovov.swapaxes(1,3)).copy()
            H += contract('ab,ij->iajb', hbar.Hvv, np.eye(no))
            H -= contract('ij,ab->iajb', hbar.Hoo, np.eye(nv))
            eps, c = np.linalg.eig(np.reshape(H, (no*nv,no*nv)))
            idx = eps.argsort()
            eps = eps[idx]; c = c[:,idx]

        # Build list of guess vectors (C1 tensors)
        guesses = np.reshape(c.T[slice(0,M),:], (M, no, nv)).copy()

        return eps[:M], guesses


    def s1(self, hbar, C1, C2):
        """
        Build the singles components of the sigma = HBAR * C vector

        Parameters
        ----------
        hbar : PyCC cchbar object
        C1, C2 : NumPy arrays
            the singles and doubles vectors for the current guess

        Returns
        -------
        s1 : NumPy array
            the singles components of sigma
        """
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

        return s1.copy()


    def s2(self, hbar, C1, C2):
        """
        Build the doubles components of the sigma = HBAR * C vector

        Parameters
        ----------
        hbar : PyCC cchbar object
        C1, C2 : NumPy arrays
            the singles and doubles vectors for the current guess

        Returns
        -------
        s2 : NumPy array
            the doubles components of sigma
        """
        contract = hbar.contract
        L = hbar.ccwfn.H.L
        t2 = hbar.ccwfn.t2
        o = hbar.o
        v = hbar.v

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

        return (s2 + s2.swapaxes(0,1).swapaxes(2,3)).copy()
