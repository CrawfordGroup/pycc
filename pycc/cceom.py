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
        _, C1 = self._guess(nguess, guess)
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
                Goo = self._build_Goo(t2, C2)
                Gvv = self._build_Gvv(t2, C2)
                s1 = self.s_l1(hbar, C1, C2, Goo, Gvv)
                s2 = self.s_l2(hbar, C1, C2, Goo, Gvv)
            out[k, :s1_len] = np.asarray(s1).ravel()
            out[k, s1_len:] = np.asarray(s2).ravel()
        return out

    def _guess(self, M, method):
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
        r"""
        Build the singles components of the right EOM-CC sigma vector sigma = HBAR * C
        (the homogeneous analogue of the ccresponse r_X1 Jacobian: no perturbation
        source, no frequency), projected onto singles.

        Parameters
        ----------
        hbar : PyCC cchbar object
        C1, C2 : NumPy arrays
            the singles and doubles vectors for the current guess

        Returns
        -------
        s1 : NumPy array
            the singles components of sigma

        Notes
        -----
        Spin-adapted (spatial) form (repeated indices summed); Hbar_* are hbar blocks::

            sigma_ia = C1_ie Hbar_ae - Hbar_mi C1_ma
                     + C1_me (2 Hbar_maei - Hbar_maie)
                     + Hbar_me (2 C2_miea - C2_imea)
                     + C2_imef (2 Hbar_amef - Hbar_amfe)
                     - (2 Hbar_mnie - Hbar_nmie) C2_mnae

        .. math::

            \begin{aligned}
            \sigma^{a}_{i} &= C^{e}_{i}\bar{H}_{ae} - \bar{H}_{mi} C^{a}_{m}
                + C^{e}_{m}(2\bar{H}_{maei} - \bar{H}_{maie}) \\
            &\quad + \bar{H}_{me}(2 C^{ea}_{mi} - C^{ea}_{im})
                + C^{ef}_{im}(2\bar{H}_{amef} - \bar{H}_{amfe})
                - (2\bar{H}_{mnie} - \bar{H}_{nmie}) C^{ae}_{mn}
            \end{aligned}
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
        r"""Spin-orbital singles sigma (right), sigma = HBAR * C.  Built from the
        antisymmetrized spin-orbital HBAR (single Hovvo, no Hovov; <pq||rs> for the ERI);
        the L-combinations of the spatial s_r1 collapse to single blocks and a 1/2 appears
        on the C2 ladders.

        Notes
        -----
        Spin-orbital form (repeated indices summed; Hbar_amef, Hbar_mnie antisymmetrized)::

            sigma_ia = C1_ie Hbar_ae - Hbar_mi C1_ma + Hbar_maei C1_me
                     + Hbar_me C2_imae
                     + 1/2 C2_imef Hbar_amef - 1/2 Hbar_mnie C2_mnae

        .. math::

            \sigma^{a}_{i} = C^{e}_{i}\bar{H}_{ae} - \bar{H}_{mi} C^{a}_{m}
                + \bar{H}_{maei} C^{e}_{m} + \bar{H}_{me} C^{ae}_{im}
                + \tfrac{1}{2} C^{ef}_{im}\bar{H}_{amef} - \tfrac{1}{2}\bar{H}_{mnie} C^{ae}_{mn}
        """
        contract = hbar.contract
        s1 = contract('ie,ae->ia', C1, hbar.Hvv)
        s1 -= contract('mi,ma->ia', hbar.Hoo, C1)
        s1 += contract('maei,me->ia', hbar.Hovvo, C1)
        s1 += contract('imae,me->ia', C2, hbar.Hov)
        s1 += 0.5 * contract('imef,amef->ia', C2, hbar.Hvovv)
        s1 -= 0.5 * contract('mnie,mnae->ia', hbar.Hooov, C2)
        return s1.copy()


    def s_r2(self, hbar, C1, C2):
        r"""
        Build the doubles components of the right EOM-CC sigma vector sigma = HBAR * C
        (the homogeneous analogue of the ccresponse r_X2 Jacobian), projected onto
        doubles. P(ij,ab) f_ijab = f_ijab + f_jiba restores the ijab<->jiba symmetry.

        Parameters
        ----------
        hbar : PyCC cchbar object
        C1, C2 : NumPy arrays
            the singles and doubles vectors for the current guess

        Returns
        -------
        s2 : NumPy array
            the doubles components of sigma

        Notes
        -----
        Spin-adapted (spatial) form (repeated indices summed); L_mnef = 2<mn|ef> -
        <mn|fe>, and Zae/Zmi are the C1/C2-dressed vv/oo intermediates::

            Zae = (2 Hbar_amef - Hbar_amfe) C1_mf - L_mnef C2_mnaf
            Zmi = -(2 Hbar_mnie - Hbar_nmie) C1_ne - L_mnef C2_inef
            sigma_ijab = P(ij,ab)[ C1_ie Hbar_abej - C1_ma Hbar_mbij
                       + Zae t2_ijeb + Zmi t2_mjab
                       + C2_ijeb Hbar_ae - C2_mjab Hbar_mi
                       + 1/2 C2_mnab Hbar_mnij + 1/2 C2_ijef Hbar_abef
                       - C2_imeb Hbar_maje - C2_imea Hbar_mbej
                       + 2 C2_miea Hbar_mbej - C2_miea Hbar_mbje ]

        .. math::

            \begin{aligned}
            Z_{ae} &= (2\bar{H}_{amef} - \bar{H}_{amfe}) C^{f}_{m} - L_{mnef} C^{af}_{mn},
                \qquad Z_{mi} = -(2\bar{H}_{mnie} - \bar{H}_{nmie}) C^{e}_{n} - L_{mnef} C^{ef}_{in} \\
            \sigma^{ab}_{ij} &= \mathcal{P}(ij,ab)\Big[ C^{e}_{i}\bar{H}_{abej} - C^{a}_{m}\bar{H}_{mbij}
                + Z_{ae} t^{eb}_{ij} + Z_{mi} t^{ab}_{mj} \\
            &\quad + C^{eb}_{ij}\bar{H}_{ae} - C^{ab}_{mj}\bar{H}_{mi}
                + \tfrac{1}{2} C^{ab}_{mn}\bar{H}_{mnij} + \tfrac{1}{2} C^{ef}_{ij}\bar{H}_{abef} \\
            &\quad - C^{eb}_{im}\bar{H}_{maje} - C^{ea}_{im}\bar{H}_{mbej}
                + 2 C^{ea}_{mi}\bar{H}_{mbej} - C^{ea}_{mi}\bar{H}_{mbje} \Big]
            \end{aligned}

        with :math:`\mathcal{P}(ij,ab) f_{ijab} = f_{ijab} + f_{jiba}`.
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
        r"""Spin-orbital doubles sigma (right), sigma = HBAR * C.  Built already antisymmetric
        under i<->j and a<->b (explicit permutations), from the antisymmetrized spin-orbital
        HBAR (single Hovvo, no Hovov) and the <pq||rs> ERI.

        Notes
        -----
        Spin-orbital form (repeated indices summed; P(pq) X_pq = X_pq - X_qp; each term
        carries only the permutation its factors do not already supply)::

            Zae = Hbar_amef C1_mf - 1/2 C2_mnaf <mn||ef>
            Zmi = -Hbar_mnie C1_ne - 1/2 <mn||ef> C2_inef
            sigma_ijab = P(ij)[ C1_ie Hbar_abej ] - P(ab)[ C1_ma Hbar_mbij ]
                       + P(ab)[ Zae t2_ijeb ] + P(ij)[ Zmi t2_mjab ]
                       + P(ab)[ C2_ijeb Hbar_ae ] - P(ij)[ Hbar_mi C2_mjab ]
                       + 1/2 Hbar_mnij C2_mnab + 1/2 C2_ijef Hbar_abef
                       + P(ij)P(ab)[ C2_imae Hbar_mbej ]

        .. math::

            \begin{aligned}
            Z_{ae} &= \bar{H}_{amef} C^{f}_{m} - \tfrac{1}{2} C^{af}_{mn}\langle mn\Vert ef\rangle,
                \qquad Z_{mi} = -\bar{H}_{mnie} C^{e}_{n} - \tfrac{1}{2}\langle mn\Vert ef\rangle C^{ef}_{in} \\
            \sigma^{ab}_{ij} &= \mathcal{P}(ij)\, C^{e}_{i}\bar{H}_{abej} - \mathcal{P}(ab)\, C^{a}_{m}\bar{H}_{mbij}
                + \mathcal{P}(ab)\, Z_{ae} t^{eb}_{ij} + \mathcal{P}(ij)\, Z_{mi} t^{ab}_{mj} \\
            &\quad + \mathcal{P}(ab)\, C^{eb}_{ij}\bar{H}_{ae} - \mathcal{P}(ij)\, \bar{H}_{mi} C^{ab}_{mj}
                + \tfrac{1}{2}\bar{H}_{mnij} C^{ab}_{mn} + \tfrac{1}{2} C^{ef}_{ij}\bar{H}_{abef} \\
            &\quad + \mathcal{P}(ij)\mathcal{P}(ab)\, C^{ae}_{im}\bar{H}_{mbej}
            \end{aligned}

        with :math:`\mathcal{P}(pq) X_{pq} = X_{pq} - X_{qp}`.
        """
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


    def _build_Goo(self, t2, l2):
        r"""Occupied-occupied G intermediate contracting t2 with l2 (called with
        l2 = C2 by the left sigma builders s_l1/s_l2).

        Notes
        -----
        Spin-adapted (spatial) and spin-orbital forms::

            spatial: G_mi = t2_mjab l2_ijab
            so:      G_mi = 1/2 t2_mnef l2_inef

        .. math::

            G_{mi} = t^{ab}_{mj} l^{ab}_{ij}
            \qquad\text{(SO: } G_{mi} = \tfrac{1}{2}\, t^{ef}_{mn} l^{ef}_{in}\text{)}
        """
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            return 0.5 * contract('mnef,inef->mi', t2, l2)
        return contract('mjab,ijab->mi', t2, l2)


    def _build_Gvv(self, t2, l2):
        r"""Virtual-virtual G intermediate contracting t2 with l2 (called with
        l2 = C2 by the left sigma builders s_l1/s_l2).

        Notes
        -----
        Spin-adapted (spatial) and spin-orbital forms::

            spatial: G_ae = -t2_ijeb l2_ijab
            so:      G_ae = -1/2 t2_mnef l2_mnaf

        .. math::

            G_{ae} = -t^{eb}_{ij} l^{ab}_{ij}
            \qquad\text{(SO: } G_{ae} = -\tfrac{1}{2}\, t^{ef}_{mn} l^{af}_{mn}\text{)}
        """
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            return -0.5 * contract('mnef,mnaf->ae', t2, l2)
        return -1.0 * contract('ijeb,ijab->ae', t2, l2)


    def s_l1(self, hbar, C1, C2, Goo, Gvv):
        r"""
        Build the singles components of the left EOM-CC sigma vector sigma = C * HBAR
        (the homogeneous analogue of the ccresponse r_Y1 Jacobian), projected onto
        singles. Gvv/Goo are the C2-dressed intermediates _build_Gvv(t2, C2),
        _build_Goo(t2, C2).

        Parameters
        ----------
        hbar : PyCC cchbar object
        C1, C2 : NumPy arrays
            the singles and doubles vectors for the current guess
        Goo, Gvv : NumPy arrays
            the oo/vv G intermediates (_build_Goo(t2, C2), _build_Gvv(t2, C2))

        Returns
        -------
        s1 : NumPy array
            the singles components of sigma

        Notes
        -----
        Spin-adapted (spatial) form (repeated indices summed); Hbar_* are hbar blocks::

            sigma_ia = C1_ie Hbar_ea - C1_ma Hbar_im
                     + (2 Hbar_ieam - Hbar_iema) C1_me
                     + C2_imef Hbar_efam - C2_mnae Hbar_iemn
                     - (2 Hbar_eifa - Hbar_eiaf) Gvv_ef
                     - (2 Hbar_mina - Hbar_imna) Goo_mn

        .. math::

            \begin{aligned}
            \sigma^{a}_{i} &= C^{e}_{i}\bar{H}_{ea} - C^{a}_{m}\bar{H}_{im}
                + (2\bar{H}_{ieam} - \bar{H}_{iema}) C^{e}_{m} \\
            &\quad + C^{ef}_{im}\bar{H}_{efam} - C^{ae}_{mn}\bar{H}_{iemn}
                - (2\bar{H}_{eifa} - \bar{H}_{eiaf}) G_{ef}
                - (2\bar{H}_{mina} - \bar{H}_{imna}) G_{mn}
            \end{aligned}
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
        r"""
        Build the doubles components of the left EOM-CC sigma vector sigma = C * HBAR
        (the homogeneous analogue of the ccresponse r_Y2 Jacobian), projected onto
        doubles. P(ij,ab) f_ijab = f_ijab + f_jiba restores the ijab<->jiba symmetry;
        Gvv/Goo are _build_Gvv(t2, C2)/_build_Goo(t2, C2).

        Parameters
        ----------
        hbar : PyCC cchbar object
        C1, C2 : NumPy arrays
            the singles and doubles vectors for the current guess
        Goo, Gvv : NumPy arrays
            the oo/vv G intermediates (_build_Goo(t2, C2), _build_Gvv(t2, C2))

        Returns
        -------
        s2 : NumPy array
            the doubles components of sigma

        Notes
        -----
        Spin-adapted (spatial) form (repeated indices summed); L_ijeb = 2<ij|eb> -
        <ij|be>, the (2 Hbar_ieam - Hbar_iema) ring is the L-combination of the
        ovvo/ovov blocks::

            sigma_ijab = P(ij,ab)[ 2 C1_ia Hbar_jb - C1_ja Hbar_ib
                       + 2 C1_ie Hbar_ejab - C1_ie Hbar_ejba
                       - 2 C1_mb Hbar_jima + C1_mb Hbar_ijma
                       + C2_ijeb Hbar_ea - C2_mjab Hbar_im
                       + 1/2 C2_mnab Hbar_ijmn + 1/2 C2_ijef Hbar_efab
                       + C2_mjeb (2 Hbar_ieam - Hbar_iema)
                       - C2_mibe Hbar_jema - C2_mieb Hbar_jeam
                       + Gvv_ae L_ijeb - Goo_mi L_mjab ]

        .. math::

            \begin{aligned}
            \sigma^{ab}_{ij} &= \mathcal{P}(ij,ab)\Big[ 2 C^{a}_{i}\bar{H}_{jb} - C^{a}_{j}\bar{H}_{ib}
                + 2 C^{e}_{i}\bar{H}_{ejab} - C^{e}_{i}\bar{H}_{ejba}
                - 2 C^{b}_{m}\bar{H}_{jima} + C^{b}_{m}\bar{H}_{ijma} \\
            &\quad + C^{eb}_{ij}\bar{H}_{ea} - C^{ab}_{mj}\bar{H}_{im}
                + \tfrac{1}{2} C^{ab}_{mn}\bar{H}_{ijmn} + \tfrac{1}{2} C^{ef}_{ij}\bar{H}_{efab} \\
            &\quad + C^{eb}_{mj}(2\bar{H}_{ieam} - \bar{H}_{iema})
                - C^{be}_{mi}\bar{H}_{jema} - C^{eb}_{mi}\bar{H}_{jeam}
                + G_{ae} L_{ijeb} - G_{mi} L_{mjab} \Big]
            \end{aligned}

        with :math:`\mathcal{P}(ij,ab) f_{ijab} = f_{ijab} + f_{jiba}`.
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
        r"""Spin-orbital singles sigma (left), sigma = C * HBAR.  The spin-orbital lambda-singles
        structure (cclambda._so_r_L1) without the inhomogeneous Hov term (EOM is homogeneous),
        with l1/l2 -> C1/C2; single antisymmetrized Hovvo, no Hovov.

        Notes
        -----
        Spin-orbital form (repeated indices summed); Gvv = _build_Gvv(t2, C2),
        Goo = _build_Goo(t2, C2)::

            sigma_ia = C1_ie Hbar_ea - C1_ma Hbar_im + Hbar_ieam C1_me
                     + 1/2 C2_imef Hbar_efam - 1/2 C2_mnae Hbar_iemn
                     - Gvv_ef Hbar_eifa - Goo_mn Hbar_mina

        .. math::

            \begin{aligned}
            \sigma^{a}_{i} &= C^{e}_{i}\bar{H}_{ea} - C^{a}_{m}\bar{H}_{im} + \bar{H}_{ieam} C^{e}_{m} \\
            &\quad + \tfrac{1}{2} C^{ef}_{im}\bar{H}_{efam} - \tfrac{1}{2} C^{ae}_{mn}\bar{H}_{iemn}
                - \bar{H}_{eifa} G_{ef} - \bar{H}_{mina} G_{mn}
            \end{aligned}

        with :math:`G_{ef}` = _build_Gvv(t2, C2), :math:`G_{mn}` = _build_Goo(t2, C2).
        """
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
        r"""Spin-orbital doubles sigma (left), sigma = C * HBAR.  The spin-orbital lambda-doubles
        structure (cclambda._so_r_L2) without the inhomogeneous <ij||ab> term, with l1/l2 -> C1/C2;
        built already antisymmetric under i<->j and a<->b (no trailing symmetrization), from the
        antisymmetrized spin-orbital HBAR (single Hovvo, no Hovov) and the <pq||rs> ERI.

        Notes
        -----
        Spin-orbital form (repeated indices summed; P(pq) X_pq = X_pq - X_qp; each term
        carries only the permutation its factors do not already supply); Gvv =
        _build_Gvv(t2, C2), Goo = _build_Goo(t2, C2)::

            sigma_ijab = P(ij)[ C1_ia Hbar_jb ] + P(ij)[ C1_jb Hbar_ia ]
                       + P(ab)[ C2_ijae Hbar_eb ] - P(ij)[ C2_imab Hbar_jm ]
                       + 1/2 C2_ijef Hbar_efab + 1/2 C2_mnab Hbar_ijmn
                       + P(ij)[ C1_ie Hbar_ejab ] - P(ab)[ C1_ma Hbar_ijmb ]
                       + P(ij)P(ab)[ C2_imae Hbar_jebm ]
                       + P(ab)[ Gvv_be <ij||ae> ] - P(ij)[ Goo_mj <im||ab> ]

        .. math::

            \begin{aligned}
            \sigma^{ab}_{ij} &= \mathcal{P}(ij)\, C^{a}_{i}\bar{H}_{jb} + \mathcal{P}(ij)\, C^{b}_{j}\bar{H}_{ia}
                + \mathcal{P}(ab)\, C^{ae}_{ij}\bar{H}_{eb} - \mathcal{P}(ij)\, C^{ab}_{im}\bar{H}_{jm} \\
            &\quad + \tfrac{1}{2} C^{ef}_{ij}\bar{H}_{efab} + \tfrac{1}{2} C^{ab}_{mn}\bar{H}_{ijmn}
                + \mathcal{P}(ij)\, C^{e}_{i}\bar{H}_{ejab} - \mathcal{P}(ab)\, C^{a}_{m}\bar{H}_{ijmb} \\
            &\quad + \mathcal{P}(ij)\mathcal{P}(ab)\, C^{ae}_{im}\bar{H}_{jebm}
                + \mathcal{P}(ab)\, G_{be}\langle ij\Vert ae\rangle - \mathcal{P}(ij)\, G_{mj}\langle im\Vert ab\rangle
            \end{aligned}

        with :math:`\mathcal{P}(pq) X_{pq} = X_{pq} - X_{qp}`, :math:`G_{be}` =
        _build_Gvv(t2, C2), :math:`G_{mj}` = _build_Goo(t2, C2).
        """
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

    def _orthonormalize(self, rows):
        """Return an orthonormal row basis for the row space of ``rows`` (M, sigma_len), dropping
        linearly dependent rows.  Used to condition the initial guess block and the restart Ritz
        vectors."""
        Q, Rm = np.linalg.qr(np.asarray(rows).real.T)
        keep = np.abs(np.diag(Rm)) > 1e-8
        return Q.T[keep].copy()
