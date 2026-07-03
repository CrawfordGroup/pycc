from __future__ import annotations

import time
from typing import TYPE_CHECKING

import psi4
import pycc
from pycc.data.molecules import *
import numpy as np
from pycc.exceptions import InvalidKeywordError

if TYPE_CHECKING:
    from pycc.ccwfn import CCwfn
    from pycc.cchbar import cchbar

class cceom(object):
    """
    An Equation-of-Motion Coupled Cluster Object.

    Attributes
    ----------
    cchbar : PyCC cchbar object
    ccwfn : PyCC ccwfn object
    D : NumPy array
        orbital energy difference array (only needed for unit-vector guesses)

    Methods
    -------
    solve_eom()
        Solves the right and left-hand EOM-CC eigenvalue problem using the Davidson algorithm

    guess()
        Generate initial guesses to eigenvalue problem using various single-excitation methods
    s_r1()
        Build the singles components of the sigma = HBAR * C vector
    s_r2()
        Build the doubles components of the sigma = HBAR * C vector
    s_l1()
        Build the singles components of the sigma = C * HBAR vector
    s_l2()
        Build the doubles components of the sigma = C * HBAR vector
    """

    def __init__(self, ccwfn: "CCwfn", cchbar: "cchbar") -> None:
        """
        Parameters
        ----------
        cchbar : PyCC cchbar object

        Returns
        -------
        None
        """
        self.hbar = cchbar
        self.ccwfn = ccwfn
        self.contract = self.ccwfn.contract

        # Build preconditioner (energy denominator)
        hbar_occ = np.diag(cchbar.Hoo)
        hbar_vir = np.diag(cchbar.Hvv)

        Dia = hbar_occ.reshape(-1,1) - hbar_vir
        Dijab = (hbar_occ.reshape(-1,1,1,1) + hbar_occ.reshape(-1,1,1) - 
                hbar_vir.reshape(-1,1) - hbar_vir)
        self.D = np.hstack((Dia.flatten(), Dijab.flatten()))

        # Get the initial guess for l1 and l2.  The spin-orbital seed is l1 = t1, l2 = t2
        # (the 2*t1 / 2(2*t2 - t2.T) spin-adaptation factors apply only to the spatial path).
        if self.ccwfn.orbital_basis == 'spinorbital':
            self.l1 = self.ccwfn.t1.copy()
            self.l2 = self.ccwfn.t2.copy()
        else:
            self.l1 = 2.0 * self.ccwfn.t1
            self.l2 = 2.0 * (2.0 * self.ccwfn.t2 - self.ccwfn.t2.swapaxes(2, 3))

    def solve_eom(self, N: int = 1, e_conv: float = 1e-5, r_conv: float = 1e-5, maxiter: int = 100, guess: str = 'HBAR_SS', eom_type: str = 'RIGHT', root_floor: float = 1e-3):
        """
        Solves the left and right-hand EOM-CC eigenvalue problem with a block Davidson-Liu
        iteration.  The expansion subspace ``V`` and its sigma vectors ``W = A V`` are kept in
        strict lockstep (new preconditioned residual corrections are orthonormalized against ``V``
        only, never by re-orthonormalizing the whole subspace), all N roots are expanded as a block
        with per-root convergence locking, and the subspace is soft-restarted from the current Ritz
        vectors when it overflows.  This resolves near-degenerate roots -- e.g. the spin-orbital
        triplets and the near-degenerate excited pairs of open-shell (UHF/ROHF) references

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
        eom_type : str
            type of the eienvalue problem to solve, left or right
        root_floor : float
            lower bound (hartree) on the real part of an accepted excitation energy; Ritz values
            below it (the spurious near-zero manifold) are skipped during root selection. Relevant
            mainly to the spin-orbital path, whose redundant doubles produce many ~0 eigenvalues.

        Returns
        -------
        None
        """

        time_init = time.time()

        no = self.hbar.no
        nv = self.hbar.nv
        contract = self.contract
        D = self.D

        s1_len = no*nv
        s2_len = no*no*nv*nv
        sigma_len = s1_len + s2_len

        # Block Davidson-Liu subspace: V holds the orthonormal expansion vectors (rows) and W the
        # matching sigma vectors W = A V (A = HBAR for RIGHT, its transpose for LEFT, whichever the
        # sigma builder applies), kept in strict lockstep -- new corrections are orthonormalized
        # against V only, so V and W never drift out of sync (unlike a full re-orthonormalization
        # of the whole subspace, which silently rotates near-degenerate vectors away from their
        # stored sigma vectors).  Cap the subspace, then soft-restart from the current Ritz vectors.
        max_space = min(sigma_len, max(N * 10, N + 40))

        eom_type = eom_type.upper()

        # Initialize guess vectors (a small block, a few more than N)
        valid_guesses = ['UNIT', 'CIS', 'HBAR_SS']
        guess = guess.upper()
        if guess not in valid_guesses:
            raise InvalidKeywordError('guess', guess, valid_guesses)
        nguess = 2 * N
        _, C1 = self.guess(nguess, guess)
        V = np.hstack((np.reshape(C1, (nguess, s1_len)), np.zeros((nguess, s2_len))))
        V = self._orthonormalize(V)
        W = np.empty((0, sigma_len), float)     # sigma vectors, in lockstep with V
        print("Guess vectors obtained from %s." % (guess))

        E = np.zeros((N))
        a_sel = None
        converged = False
        for niter in range(1, maxiter+1):
            E_old = E

            # Bring W up to date: compute sigma only for the newly added V rows
            if W.shape[0] < V.shape[0]:
                W = np.vstack((W, self._eom_sigma(V[W.shape[0]:], eom_type)))
            M = V.shape[0]
            print("EOM Iter %3d: M = %3d" % (niter, M))

            # Subspace matrix G = V W^T (right eigenvectors give the RIGHT EOM vectors, or the LEFT
            # ones when W was built with the left sigma -- G is the transpose of the old S C^T form,
            # so the spectrum is identical and the right eigenvectors here match the old left ones).
            G = V @ W.T
            theta, a = np.linalg.eig(G)

            # Select the N target roots: real, positive, above root_floor (skip the spurious
            # near-zero manifold of redundant non-antisymmetric doubles and any complex Ritz
            # values), padding with the next-smallest if fewer than N are exposed yet.
            order = theta.real.argsort()
            physical = [i for i in order if abs(theta[i].imag) < 1e-6 and theta[i].real > root_floor]
            if len(physical) < N:
                physical += [i for i in order if i not in physical]
            idx = np.array(physical[:N])
            E = theta[idx].real
            a_sel = a[:, idx]

            # Ritz vectors X = a^T V and their images A X = a^T W (real; the operator is real)
            X = (a_sel.T @ V).real
            AX = (a_sel.T @ W).real
            R = AX - E[:, None] * X                  # residuals
            r_norm = np.linalg.norm(R, axis=1)

            dE = E - E_old
            print("             E/state                   dE                 norm")
            for state in range(N):
                print("%20.12f %20.12f %20.12f" % (E[state], dE[state], r_norm[state]))

            if np.max(r_norm) <= r_conv:
                converged = True
                break

            # Preconditioned corrections for the unconverged roots only (locking)
            additions = [R[k] / (E[k] - D) for k in range(N) if r_norm[k] > r_conv]

            # Soft restart if the subspace would overflow: collapse to the current Ritz vectors and
            # rebuild their sigma vectors (no convergence progress is lost -- the Ritz vectors carry
            # it), then continue expanding from there.
            if M + len(additions) > max_space:
                print("Subspace limit (%d) reached; restarting from the current Ritz vectors." % max_space)
                V = self._orthonormalize(X)
                W = np.empty((0, sigma_len), float)
                continue

            # Orthonormalize each correction against V (twice, for stability) and against the
            # previously accepted corrections; add it unless it is (numerically) linearly
            # dependent on the current subspace.  The dependence test is *relative* (how much of
            # the vector survives projection) -- an absolute size cutoff would wrongly discard the
            # small-but-valid corrections that appear near convergence and stall the residual.
            for delta in additions:
                d = delta.real.copy()
                nrm0 = np.linalg.norm(d)
                if nrm0 < 1e-14:
                    continue
                for _ in range(2):
                    d = d - V.T @ (V @ d)
                nrm = np.linalg.norm(d)
                if nrm / nrm0 > 1e-6:
                    V = np.vstack((V, d / nrm))

            # If every correction collapsed (all linearly dependent), restart to avoid a stall.
            if V.shape[0] == M:
                V = self._orthonormalize(X)
                W = np.empty((0, sigma_len), float)

        C = V
        a = a_sel
        if converged:
            print("\nCCEOM converged in %.3f seconds." % (time.time() - time_init))
            print("\nState     E_h           eV")
            print("-----  ------------  ------------")
            eVconv = psi4.qcel.constants.get("hartree energy in ev")
            for state in range(N):
                print("  %3d  %12.10f  %12.10f" %(state, E[state].real, E[state].real*eVconv))

            # Build converged eigenvectors (R vectors) from Krylov basis
            R_full = a.T @ C  # shape (N, s1_len + s2_len)
            s1_len = no * nv
            s2_len = no * no * nv * nv
            norm_eigvec = []

            so = (self.ccwfn.orbital_basis == 'spinorbital')
            for itm in range(len(R_full)):
                r1 = R_full[itm, :s1_len]
                r2 = R_full[itm, s1_len:]
                r2 = np.reshape(r2, (no,no,nv,nv))
                if so:
                    # spin-orbital metric <R|R> = r1.r1 + 1/4 r2.r2 (antisymmetrized amplitudes)
                    norm = 1/np.sqrt(np.sum(r1**2) + 0.25*contract('ijab,ijab->', r2, r2))
                else:
                    norm = 1/np.sqrt(2*np.sum(r1**2) + contract('ijab,ijab->', (2*r2-r2.swapaxes(2,3)), r2))
                norm_eigvec.append(R_full[itm]*norm)

            return E, norm_eigvec


    def _orthonormalize(self, rows):
        """Return an orthonormal row basis for the row space of ``rows`` (M, sigma_len), dropping
        linearly dependent rows.  Used to condition the initial guess block and the restart Ritz
        vectors."""
        Q, Rm = np.linalg.qr(np.asarray(rows).real.T)
        keep = np.abs(np.diag(Rm)) > 1e-8
        return Q.T[keep].copy()

    def _eom_sigma(self, block, eom_type):
        """Apply the EOM sigma operator to each row of ``block`` (k, sigma_len), returning the
        sigma vectors as rows (k, sigma_len).  RIGHT uses ``s_r1``/``s_r2`` (sigma = HBAR * C);
        LEFT uses ``s_l1``/``s_l2`` (sigma = C * HBAR), with the per-vector G intermediates."""
        hbar = self.hbar
        no, nv = hbar.no, hbar.nv
        s1_len = no * nv
        t2 = self.ccwfn.t2
        out = np.empty((block.shape[0], block.shape[1]), float)
        for k in range(block.shape[0]):
            C1 = block[k, :s1_len].reshape(no, nv)
            C2 = block[k, s1_len:].reshape(no, no, nv, nv)
            if eom_type == 'RIGHT':
                s1 = self.s_r1(hbar, C1, C2)
                s2 = self.s_r2(hbar, C1, C2)
            else:
                Goo = self.build_Goo(t2, C2)
                Gvv = self.build_Gvv(t2, C2)
                s1 = self.s_l1(hbar, C1, C2, Goo, Gvv)
                s2 = self.s_l2(hbar, C1, C2, Goo, Gvv)
            out[k, :s1_len] = np.asarray(s1).ravel()
            out[k, s1_len:] = np.asarray(s2).ravel()
        return out

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
            if hbar.ccwfn.orbital_basis == 'spinorbital':
                # spin-orbital CIS matrix H_{ia,jb} = f_ab d_ij - f_ij d_ab + <aj||ib>,
                # using the antisymmetrized <pq||rs> (no L, no factor of 2 -- this yields all
                # spin states, singlets and triplets, not just the spin-adapted singlets)
                ERI = hbar.ccwfn.H.ERI
                H = ERI[v,o,o,v].swapaxes(0,1).swapaxes(0,2).copy()
            else:
                L = hbar.ccwfn.H.L
                H = L[v,o,o,v].swapaxes(0,1).swapaxes(0,2).copy()
            H += contract('ab,ij->iajb', F[v,v], np.eye(no))
            H -= contract('ij,ab->iajb', F[o,o], np.eye(nv))
            eps, c = np.linalg.eigh(np.reshape(H, (no*nv,no*nv)))
        # Use eigenvectors of singles-singles block of hbar (mimics Psi4)
        elif method == 'HBAR_SS':
            if hbar.ccwfn.orbital_basis == 'spinorbital':
                # spin-orbital singles-singles block: single antisymmetrized Hovvo (no Hovov,
                # no factor of 2)
                H = hbar.Hovvo.swapaxes(1,2).swapaxes(2,3).copy()
            else:
                H = (2.0 * hbar.Hovvo.swapaxes(1,2).swapaxes(2,3) - hbar.Hovov.swapaxes(1,3)).copy()
            H += contract('ab,ij->iajb', hbar.Hvv, np.eye(no))
            H -= contract('ij,ab->iajb', hbar.Hoo, np.eye(nv))
            eps, c = np.linalg.eig(np.reshape(H, (no*nv,no*nv)))
            # The HBAR singles block is non-Hermitian, so eig can return complex conjugate
            # eigenpairs for (near-)degenerate roots (e.g. the spin-orbital triplets). Keep the
            # real span: complex guess vectors otherwise seed spurious near-zero roots in the
            # Davidson. The real parts of a conjugate pair span the same real subspace, and the
            # QR orthonormalization in solve_eom removes any resulting redundancy.
            eps = eps.real; c = c.real
            idx = eps.argsort()
            eps = eps[idx]; c = c[:,idx]

        # Build list of guess vectors (C1 tensors)
        guesses = np.reshape(c.T[slice(0,M),:], (M, no, nv)).copy()

        return eps[:M], guesses


    def s_r1(self, hbar, C1, C2):
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
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._so_s_r1(hbar, C1, C2)
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

    def _so_s_r1(self, hbar, C1, C2):
        """Spin-orbital singles sigma (right), sigma = HBAR * C.  Built from the
        antisymmetrized spin-orbital HBAR (single Hovvo, no Hovov; <pq||rs> for the ERI)."""
        contract = hbar.contract
        s1 = contract('ie,ae->ia', C1, hbar.Hvv)
        s1 -= contract('mi,ma->ia', hbar.Hoo, C1)
        s1 += contract('maei,me->ia', hbar.Hovvo, C1)
        s1 += contract('imae,me->ia', C2, hbar.Hov)
        s1 += 0.5 * contract('imef,amef->ia', C2, hbar.Hvovv)
        s1 -= 0.5 * contract('mnie,mnae->ia', hbar.Hooov, C2)
        return s1.copy()


    def s_r2(self, hbar, C1, C2):
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
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._so_s_r2(hbar, C1, C2)
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

    def _so_s_r2(self, hbar, C1, C2):
        """Spin-orbital doubles sigma (right), sigma = HBAR * C.  Built already antisymmetric
        under i<->j and a<->b (explicit permutations), from the antisymmetrized spin-orbital
        HBAR (single Hovvo, no Hovov) and the <pq||rs> ERI."""
        contract = hbar.contract
        ERI = hbar.ccwfn.H.ERI
        t2 = hbar.ccwfn.t2
        o = hbar.o
        v = hbar.v

        # C1-into-doubles coupling intermediates (built from C1 and the C2 x <mn||ef> pieces)
        Zvv = contract('amef,mf->ae', hbar.Hvovv, C1)
        Zvv -= 0.5 * contract('mnaf,mnef->ae', C2, ERI[o,o,v,v])
        Zoo = -contract('mnie,ne->mi', hbar.Hooov, C1)
        Zoo -= 0.5 * contract('mnef,inef->mi', ERI[o,o,v,v], C2)

        # terms already antisymmetric in the surviving pair get only the other permutation
        tmp = contract('ie,abej->ijab', C1, hbar.Hvvvo)          # antisym ab (Hvvvo)
        s2 = tmp - tmp.swapaxes(0,1)
        tmp = -contract('mbij,ma->ijab', hbar.Hovoo, C1)         # antisym ij (Hovoo)
        s2 += tmp - tmp.swapaxes(2,3)
        tmp = contract('ijeb,ae->ijab', t2, Zvv)                 # antisym ij (t2)
        s2 += tmp - tmp.swapaxes(2,3)
        tmp = contract('mi,mjab->ijab', Zoo, t2)                 # antisym ab (t2)
        s2 += tmp - tmp.swapaxes(0,1)
        tmp = contract('ijeb,ae->ijab', C2, hbar.Hvv)            # antisym ij (C2)
        s2 += tmp - tmp.swapaxes(2,3)
        tmp = -contract('mi,mjab->ijab', hbar.Hoo, C2)           # antisym ab (C2)
        s2 += tmp - tmp.swapaxes(0,1)
        # fully antisymmetric ladder terms
        s2 += 0.5 * contract('mnij,mnab->ijab', hbar.Hoooo, C2)
        s2 += 0.5 * contract('ijef,abef->ijab', C2, hbar.Hvvvv)
        # ring term: single antisymmetrized Hovvo, full P(ij)P(ab)
        tmp = contract('imae,mbej->ijab', C2, hbar.Hovvo)
        s2 += (tmp - tmp.swapaxes(0,1) - tmp.swapaxes(2,3)
               + tmp.swapaxes(0,1).swapaxes(2,3))
        return s2.copy()


    def build_Goo(self, t2, l2):
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            return 0.5 * contract('mnef,inef->mi', t2, l2)
        return contract('mjab,ijab->mi', t2, l2)


    def build_Gvv(self, t2, l2):
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            return -0.5 * contract('mnef,mnaf->ae', t2, l2)
        return -1.0 * contract('ijeb,ijab->ae', t2, l2)


    def s_l1(self, hbar, C1, C2, Goo, Gvv):
        """
        Build the singles components of the sigma = C * HBAR vector

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
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._so_s_l1(hbar, C1, C2, Goo, Gvv)
        contract = hbar.contract

        s1 = contract('ie,ea->ia', C1, hbar.Hvv)
        s1 = s1 - contract('ma,im->ia', C1, hbar.Hoo)
        s1 = s1 + contract('imef,efam->ia', C2, hbar.Hvvvo)
        s1 = s1 - contract('mnae,iemn->ia', C2, hbar.Hovoo)
        s1 = s1 + contract('me,ieam->ia', C1, (2.0 * hbar.Hovvo - hbar.Hovov.swapaxes(2,3)))
        s1 = s1 - 2.0 * contract('ef,eifa->ia', Gvv, hbar.Hvovv)
        s1 = s1 + contract('ef,eiaf->ia', Gvv, hbar.Hvovv)
        s1 = s1 - 2.0 * contract('mn,mina->ia', Goo, hbar.Hooov)
        s1 = s1 + contract('mn,imna->ia', Goo, hbar.Hooov)
        return s1.copy()


    def s_l2(self, hbar, C1, C2, Goo, Gvv):
        """
        Build the doubles components of the sigma = C * HBAR vector

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
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._so_s_l2(hbar, C1, C2, Goo, Gvv)
        contract = hbar.contract
        L = hbar.ccwfn.H.L
        t2 = hbar.ccwfn.t2
        o = hbar.o
        v = hbar.v

        s2 = 2.0 * contract('ia,jb->ijab', C1, hbar.Hov)
        s2 = s2 - contract('ja,ib->ijab', C1, hbar.Hov)
        s2 = s2 + 2.0 * contract('ie,ejab->ijab', C1, hbar.Hvovv)
        s2 = s2 - contract('ie,ejba->ijab', C1, hbar.Hvovv)
        s2 = s2 - 2.0 * contract('mb,jima->ijab', C1, hbar.Hooov)
        s2 = s2 + contract('mb,ijma->ijab', C1, hbar.Hooov)
        s2 = s2 + contract('ijeb,ea->ijab', C2, hbar.Hvv)
        s2 = s2 - contract('mjab,im->ijab', C2, hbar.Hoo)
        s2 = s2 + 0.5 * contract('mnab,ijmn->ijab', C2, hbar.Hoooo)
        s2 = s2 + 0.5 * contract('ijef,efab->ijab', C2, hbar.Hvvvv)
        s2 = s2 + contract('mjeb,ieam->ijab', C2, (2.0 * hbar.Hovvo - hbar.Hovov.swapaxes(2,3)))
        s2 = s2 - contract('mibe,jema->ijab', C2, hbar.Hovov)
        s2 = s2 - contract('mieb,jeam->ijab', C2, hbar.Hovvo)
        s2 = s2 + contract('ae,ijeb->ijab', Gvv, L[o,o,v,v])
        s2 = s2 - contract('mi,mjab->ijab', Goo, L[o,o,v,v])
        s2 = s2 + s2.swapaxes(0,1).swapaxes(2,3)

        return s2.copy()

    def _so_s_l1(self, hbar, C1, C2, Goo, Gvv):
        """Spin-orbital singles sigma (left), sigma = C * HBAR.  The spin-orbital lambda-singles
        structure (cclambda._so_r_L1) without the inhomogeneous Hov term (EOM is homogeneous),
        with l1/l2 -> C1/C2; single antisymmetrized Hovvo, no Hovov."""
        contract = hbar.contract
        s1 = contract('ie,ea->ia', C1, hbar.Hvv)
        s1 = s1 - contract('ma,im->ia', C1, hbar.Hoo)
        s1 = s1 + 0.5 * contract('imef,efam->ia', C2, hbar.Hvvvo)
        s1 = s1 - 0.5 * contract('mnae,iemn->ia', C2, hbar.Hovoo)
        s1 = s1 + contract('me,ieam->ia', C1, hbar.Hovvo)
        s1 = s1 - contract('ef,eifa->ia', Gvv, hbar.Hvovv)
        s1 = s1 - contract('mn,mina->ia', Goo, hbar.Hooov)
        return s1.copy()

    def _so_s_l2(self, hbar, C1, C2, Goo, Gvv):
        """Spin-orbital doubles sigma (left), sigma = C * HBAR.  The spin-orbital lambda-doubles
        structure (cclambda._so_r_L2) without the inhomogeneous <ij||ab> term, with l1/l2 -> C1/C2;
        built already antisymmetric under i<->j and a<->b (no trailing symmetrization), from the
        antisymmetrized spin-orbital HBAR (single Hovvo, no Hovov) and the <pq||rs> ERI."""
        contract = hbar.contract
        ERI = hbar.ccwfn.H.ERI
        o = hbar.o
        v = hbar.v
        s2 = (contract('ia,jb->ijab', C1, hbar.Hov) - contract('ja,ib->ijab', C1, hbar.Hov))
        s2 = s2 + (contract('jb,ia->ijab', C1, hbar.Hov) - contract('ib,ja->ijab', C1, hbar.Hov))
        s2 = s2 + (contract('ijae,eb->ijab', C2, hbar.Hvv) - contract('ijbe,ea->ijab', C2, hbar.Hvv))
        s2 = s2 - (contract('imab,jm->ijab', C2, hbar.Hoo) - contract('jmab,im->ijab', C2, hbar.Hoo))
        s2 = s2 + 0.5 * contract('ijef,efab->ijab', C2, hbar.Hvvvv)
        s2 = s2 + 0.5 * contract('mnab,ijmn->ijab', C2, hbar.Hoooo)
        s2 = s2 + (contract('ie,ejab->ijab', C1, hbar.Hvovv) - contract('je,eiab->ijab', C1, hbar.Hvovv))
        s2 = s2 - (contract('ma,ijmb->ijab', C1, hbar.Hooov) - contract('mb,ijma->ijab', C1, hbar.Hooov))
        tmp = contract('imae,jebm->ijab', C2, hbar.Hovvo)
        s2 = s2 + (tmp - tmp.swapaxes(0,1) - tmp.swapaxes(2,3) + tmp.swapaxes(0,1).swapaxes(2,3))
        s2 = s2 + (contract('be,ijae->ijab', Gvv, ERI[o,o,v,v]) - contract('ae,ijbe->ijab', Gvv, ERI[o,o,v,v]))
        s2 = s2 - (contract('mj,imab->ijab', Goo, ERI[o,o,v,v]) - contract('mi,jmab->ijab', Goo, ERI[o,o,v,v]))
        return s2.copy()
