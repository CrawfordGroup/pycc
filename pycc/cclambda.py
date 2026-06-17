"""
cclambda.py: Lambda-amplitude Solver
"""

from __future__ import annotations

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import time
from typing import TYPE_CHECKING

import numpy as np
from opt_einsum import contract
from .utils import helper_diis, zeros, zeros_like, clone, sqrt
from pycc.ccwfn import HAS_TORCH
if HAS_TORCH:
    import torch
from .cctriples import t3c_ijk, l3_ijk, l3_ijk_alt, t3_pert_ijk

if TYPE_CHECKING:
    from pycc.ccwfn import CCwfn
    from pycc.cchbar import cchbar


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
    def __init__(self, ccwfn: "CCwfn", hbar: "cchbar") -> None:
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
        self.contract = self.ccwfn.contract

        if self.ccwfn.orbital_basis == 'spinorbital':
            # Spin-orbital guess: l1 = t1, l2 = t2 (the spin-adapted 2*t1 / 2(2t2-t2.T)
            # form has no analog without L).
            self.l1 = clone(self.ccwfn.t1)
            self.l2 = clone(self.ccwfn.t2)
        else:
            self.l1 = 2.0 * self.ccwfn.t1
            self.l2 = 2.0 * (2.0 * self.ccwfn.t2 - self.ccwfn.t2.swapaxes(2, 3))

    def solve_lambda(self, e_conv: float = 1e-7, r_conv: float = 1e-7, maxiter: int = 100, max_diis: int = 8, start_diis: int = 1) -> float:
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
        contract = self.ccwfn.contract
        lambda_tstart = time.time()

        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccwfn.nv
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2
        l1 = self.l1
        l2 = self.l2
        Dia = self.ccwfn.Dia
        Dijab = self.ccwfn.Dijab
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        L = self.ccwfn.H.L if self.ccwfn.orbital_basis == 'spatial' else None

        if self.ccwfn.orbital_basis == 'spinorbital' and self.ccwfn.model == 'CC3':
            raise NotImplementedError("Spin-orbital Lambda currently supports CCSD/CCD, "
                                      "not CC3.")

        Hov = self.hbar.Hov
        Hvv = self.hbar.Hvv
        Hoo = self.hbar.Hoo
        Hoooo = self.hbar.Hoooo
        Hvvvv = self.hbar.Hvvvv
        Hvovv = self.hbar.Hvovv
        Hooov = self.hbar.Hooov
        Hovvo = self.hbar.Hovvo
        Hovov = self.hbar.Hovov if self.ccwfn.orbital_basis == 'spatial' else None
        Hvvvo = self.hbar.Hvvvo
        Hovoo = self.hbar.Hovoo

        lecc = self.pseudoenergy(o, v, ERI, l2)

        print("\nLCC Iter %3d: LCC PseudoE = %.15f  dE = % .5E" % (0, lecc, -lecc))

        diis = helper_diis(l1, l2, max_diis, self.ccwfn.precision)
 
        contract = self.contract

        if self.ccwfn.model == 'CC3':
            # T-dependent CC3 intermediates: t1/t2 are fixed during the Lambda
            # solve, so build them once here and reuse every iteration.
            cc3_ints = self._build_cc3_lambda_intermediates(o, v, t1, t2, F, ERI, L)

        for niter in range(1, maxiter+1):
            lecc_last = lecc

            l1 = self.l1
            l2 = self.l2

            Goo = self.build_Goo(t2, l2)
            Gvv = self.build_Gvv(t2, l2)
            r1 = self.r_L1(o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)
            r2 = self.r_L2(o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)
   
            if self.ccwfn.model == 'CC3':
                Y1, Y2 = self._cc3_lambda_triples(o, v, l1, l2, t2, F, ERI, L, cc3_ints)
                r1 += Y1
                r2 += Y2 + Y2.swapaxes(0,1).swapaxes(2,3)

            if self.ccwfn.local is not None:
                inc1, inc2 = self.ccwfn.Local.filter_amps(r1, r2)
                self.l1 += inc1
                self.l2 += inc2
                rms = contract('ia,ia->', inc1, inc1)
                rms += contract('ijab,ijab->', inc2, inc2)
                rms = sqrt(rms)
            else:
                self.l1 += r1/Dia
                self.l2 += r2/Dijab
                rms = contract('ia,ia->', r1/Dia, r1/Dia)
                rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                rms = sqrt(rms)

            lecc = self.pseudoenergy(o, v, ERI, self.l2)
            ediff = lecc - lecc_last
            print("LCC Iter %3d: LCC PseudoE = %.15f  dE = % .5E  rms = % .5E" % (niter, lecc, ediff, rms))
            
            if HAS_TORCH and isinstance(self.l1, torch.Tensor):
                if ((torch.abs(ediff) < e_conv) and torch.abs(rms) < r_conv):
                    print("\nLambda-CC has converged in %.3f seconds.\n" % (time.time() - lambda_tstart))
                    return lecc
            else:
                if ((abs(ediff) < e_conv) and abs(rms) < r_conv):
                    print("\nLambda-CC has converged in %.3f seconds.\n" % (time.time() - lambda_tstart))
                    return lecc

            diis.add_error_vector(self.l1, self.l2)
            if niter >= start_diis:
                self.l1, self.l2 = diis.extrapolate(self.l1, self.l2)

        if HAS_TORCH and isinstance(r1, torch.Tensor):
            del Goo, Gvv, Hoo, Hvv, Hov, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov

        if (HAS_TORCH and isinstance(r1, torch.Tensor)) & (self.ccwfn.model == 'CC3'):
            del cc3_ints
           
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
        contract = self.ccwfn.contract

        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccwfn.nv
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

        if self.ccwfn.model == 'CC3':
            cc3_ints = self._build_cc3_lambda_intermediates(o, v, t1, t2, F, ERI, L, real_time=self.ccwfn.real_time)
            Y1, Y2 = self._cc3_lambda_triples(o, v, l1, l2, t2, F, ERI, L, cc3_ints, real_time=self.ccwfn.real_time)
            r1 += Y1
            r2 += Y2 + Y2.swapaxes(0,1).swapaxes(2,3)

            if HAS_TORCH and isinstance(r1, torch.Tensor):
                del cc3_ints
       
        if HAS_TORCH and isinstance(r1, torch.Tensor):
            del Goo, Gvv, Hoo, Hvv, Hov, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov
                                             
        return r1, r2

    def _build_cc3_lambda_intermediates(self, o, v, t1, t2, F, ERI, L, real_time=False):
        """Build the T-dependent CC3 intermediates shared by the Lambda equations.

        Constructs the CC3 W-intermediates (t3 and l3) and the ``Zmndi``/``Zmdfa``
        triples intermediates. These depend only on the T-amplitudes (not Lambda),
        so ``solve_lambda`` builds them once before its iteration loop while
        ``residuals`` builds them per call.

        Parameters
        ----------
        o, v : slice
            occupied/virtual orbital slices
        t1, t2 : ndarray or torch.Tensor
            current T1/T2 amplitudes
        F, ERI, L : ndarray or torch.Tensor
            Fock matrix, two-electron integrals, and L = 2*ERI - ERI.swapaxes
        real_time : bool
            if True, subtract the explicit time-dependent perturbation from the
            connected triples (real-time CC3 path)

        Returns
        -------
        dict
            the W-intermediates plus ``Zmndi`` and ``Zmdfa``
        """
        contract = self.contract
        no = self.ccwfn.no
        nv = self.ccwfn.nv

        # Intermediates for t3
        Fov = self.ccwfn.build_Fme(o, v, F, L, t1)
        Woooo = self.ccwfn.build_cc3_Wmnij(o, v, ERI, t1)
        Wovoo = self.ccwfn.build_cc3_Wmbij(o, v, ERI, t1, Woooo)
        Wooov = self.ccwfn.build_cc3_Wmnie(o, v, ERI, t1)
        Wvovv = self.ccwfn.build_cc3_Wamef(o, v, ERI, t1)
        Wvvvo = self.ccwfn.build_cc3_Wabei(o, v, ERI, t1)
        # Additional intermediates for l3
        Wovov = self.build_cc3_Wmbje(o, v, ERI, t1)
        Wovvo = self.build_cc3_Wmbej(o, v, ERI, t1)
        Wvvvv = self.build_cc3_Wabef(o, v, ERI, t1)

        # Building intermediates in t3l1. Zmdfa's second axis is virtual
        # (shape no,nv,nv,nv), so it is allocated directly rather than padded.
        Zmndi = zeros((no, no, nv, no), like=t2)
        Zmdfa = zeros((no, nv, nv, nv), like=t2)
        for m in range(no):
            for n in range(no):
                for l in range(no):
                    t3_lmn = t3c_ijk(o, v, l, m, n, t2, Wvvvo, Wovoo, F, contract, WithDenom=True)
                    if real_time is True:
                        V = F - clone(self.ccwfn.H.F)
                        t3_lmn -= t3_pert_ijk(o, v, l, m, n, t2, V, F, contract)
                    Zmndi[m,n] += contract('def,ief->di', t3_lmn, ERI[o,l,v,v])
                    Zmndi[m,n] -= contract('fed,ief->di', t3_lmn, L[o,l,v,v])
                    Zmdfa[m] += contract('def,ea->dfa', t3_lmn, ERI[n,l,v,v])
                    Zmdfa[m] -= contract('dfe,ea->dfa', t3_lmn, L[n,l,v,v])

        return {'Fov': Fov, 'Woooo': Woooo, 'Wovoo': Wovoo, 'Wooov': Wooov,
                'Wvovv': Wvovv, 'Wvvvo': Wvvvo, 'Wovov': Wovov, 'Wovvo': Wovvo,
                'Wvvvv': Wvvvv, 'Zmndi': Zmndi, 'Zmdfa': Zmdfa}

    def _cc3_lambda_triples(self, o, v, l1, l2, t2, F, ERI, L, ints, real_time=False):
        """Compute the CC3 triples contributions (Y1, Y2) to the Lambda residuals.

        Uses the T-dependent intermediates from
        :meth:`_build_cc3_lambda_intermediates` together with the current
        Lambda amplitudes. Shared by ``solve_lambda`` (each iteration, with the
        same ``ints``) and ``residuals`` (a single evaluation).

        Parameters
        ----------
        o, v : slice
            occupied/virtual orbital slices
        l1, l2 : ndarray or torch.Tensor
            current L1/L2 amplitudes
        t2 : ndarray or torch.Tensor
            current T2 amplitudes
        F, ERI, L : ndarray or torch.Tensor
            Fock matrix, two-electron integrals, and L = 2*ERI - ERI.swapaxes
        ints : dict
            intermediates from :meth:`_build_cc3_lambda_intermediates`
        real_time : bool
            if True, subtract the explicit time-dependent perturbation from the
            connected triples (real-time CC3 path)

        Returns
        -------
        Y1, Y2 : ndarray or torch.Tensor
            the CC3 triples contributions to the L1 and L2 residuals
        """
        contract = self.contract
        no = self.ccwfn.no
        nv = self.ccwfn.nv

        Fov = ints['Fov']
        Woooo = ints['Woooo']
        Wovoo = ints['Wovoo']
        Wooov = ints['Wooov']
        Wvovv = ints['Wvovv']
        Wvvvo = ints['Wvvvo']
        Wovov = ints['Wovov']
        Wovvo = ints['Wovvo']
        Wvvvv = ints['Wvvvv']
        Zmndi = ints['Zmndi']
        Zmdfa = ints['Zmdfa']

        # Y residual accumulators and the CC3 Z-intermediates. Y1/Y2/Znf match
        # a Lambda array's shape; Zbide/Zblad_* have a virtual leading axis
        # (nv,no,nv,nv) and Zjlma/Zjlid_* an all-but-last occupied shape
        # (no,no,no,nv), so they are allocated by explicit shape.
        Y1 = zeros_like(l1)
        Y2 = zeros_like(l2)
        # t3l1
        Znf = zeros_like(l1)
        #l3l1+l3l2
        Zbide = zeros((nv, no, nv, nv), like=l2)
        Zblad_1 = zeros((nv, no, nv, nv), like=l2)
        Zblad_2 = zeros((nv, no, nv, nv), like=l2)
        Zjlma = zeros((no, no, no, nv), like=l2)
        Zjlid_1 = zeros((no, no, no, nv), like=l2)
        Zjlid_2 = zeros((no, no, no, nv), like=l2)
        # t3l1
        for l in range(no):
            for m in range(no):
                for n in range(no):
                    t3_lmn = t3c_ijk(o, v, l, m, n, t2, Wvvvo, Wovoo, F, contract, WithDenom=True)
                    if real_time is True:
                        V = F - clone(self.ccwfn.H.F)
                        t3_lmn -= t3_pert_ijk(o, v, l, m, n, t2, V, F, contract)
                    Znf[n] += contract('de,def->f', l2[l,m], (t3_lmn - t3_lmn.swapaxes(0,2)))
        for m in range(no):
            Y1 += contract('idf,dfa->ia', l2[:,m], Zmdfa[m])
            Y1 += contract('iaf,f->ia', L[o,m,v,v], Znf[m])
            for n in range(no):
                Y1 += contract('ad,di->ia', l2[m,n], Zmndi[m,n,:,:])
        # end of t3l1
        #l3l1+l3l2
        for i in range(no):
            for j in range(no):
                for k in range(no):
                    l3_kij = l3_ijk(k, i, j, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True)
                    # l3l1_Z_build
                    Zbide[:,i,:,:] += contract('bc,cde->bde', t2[j,k], l3_kij)
                    Zblad_1[:,i,:,:] += contract('bc,cad->bad', t2[j,k], l3_kij)
                    Zblad_2[:,i,:,:] += contract('bc,cda->bad', t2[j,k], l3_kij)
                    Zjlma[:,i,j,:] += contract('jbc,cab->ja', t2[:,k,:,:], l3_kij)
                    Zjlid_1[:,i,j,:] += contract('jbc,cbd->jd', t2[:,k,:,:], l3_kij)
                    Zjlid_2[:,i,j,:] += contract('jbc,cdb->jd', t2[:,k,:,:], l3_kij)
                    # l3l2
                    Y2[i,j] += contract('deb,eda->ab', l3_kij, Wvvvo[:,:,:,k])
                    Y2[i] -= contract('dab,ld->lab', l3_kij, Wovoo[:,:,j,k])
        # l3l1
        Y1 += contract('bide,deab->ia', Zbide, Wvvvv)
        for j in range(no):
            for l in range(no):
                for m in range(no):
                    Y1 += contract('a,i->ia', Zjlma[j,l,m], Woooo[:,j,l,m])
        for j in range(no):
            for l in range(no):
                Y1 -= contract('id,da->ia', Zjlid_1[j,l,:,:], Wovov[j,:,l,:])
                Y1 -= contract('id,da->ia', Zjlid_2[j,l,:,:], Wovvo[j,:,:,l])
        for l in range(no):
            Y1 -= contract('bad,idb->ia', Zblad_1[:,l,:,:], Wovov[:,:,l,:])
            Y1 -= contract('bad,idb->ia', Zblad_2[:,l,:,:], Wovvo[:,:,:,l])
        # end l3l1+l3l2

        return Y1, Y2

    def build_Goo(self, t2, l2):
        """Build the G_mi occupied density intermediate (t2-weighted lambda).

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no)
            Indexed [m, i].

        Notes
        -----
        ::

            G_mi = t2_mjab l2_ijab
        """
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            return 0.5 * contract('mnef,inef->mi', t2, l2)
        return contract('mjab,ijab->mi', t2, l2)


    def build_Gvv(self, t2, l2):
        """Build the G_ae virtual density intermediate (t2-weighted lambda).

        Returns
        -------
        ndarray or torch.Tensor, shape (nv, nv)
            Indexed [a, e].

        Notes
        -----
        ::

            G_ae = - t2_ijeb l2_ijab
        """
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            return -0.5 * contract('mnef,mnaf->ae', t2, l2)
        return -1.0 * contract('ijeb,ijab->ae', t2, l2)


    def r_L1(self, o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
        """Compute the L1 (lambda singles) residual.

        The lambda equations are linear; solve_lambda drives this residual to
        zero. H_* are the HBAR blocks, G_* the t2-weighted lambda densities.
        Returns zeros for CCD (no singles); CC2 and CCSD(T) modify/extend the
        terms below.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv)
            L1 residual indexed [i, a].

        Notes
        -----
        CCSD form (repeated indices summed)::

            r_l1_ia = 2 H_ia + l1_ie H_ea - l1_ma H_im
                    + l2_imef H_efam - l2_mnae H_iemn
                    + l1_me (2 H_ieam - H_iema)
                    - G_ef (2 H_eifa - H_eiaf)
                    - G_mn (2 H_mina - H_imna)
        """
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._r_L1_spinorbital(o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hvvvo,
                                          Hovoo, Hvovv, Hooov, Gvv, Goo)
        if self.ccwfn.model == 'CCD':
            r_l1 = zeros_like(l1)
        else:
            r_l1 = 2.0 * clone(Hov)

            # Add (T) contributions to L1
            if self.ccwfn.model == 'CCSD(T)':
                r_l1 = r_l1 + self.ccwfn.S1

            r_l1 = r_l1 + contract('ie,ea->ia', l1, Hvv)
            r_l1 = r_l1 - contract('ma,im->ia', l1, Hoo)          
            r_l1 = r_l1 + contract('imef,efam->ia', l2, Hvvvo)
            r_l1 = r_l1 - contract('mnae,iemn->ia', l2, Hovoo)
            r_l1 = r_l1 + contract('me,ieam->ia', l1, (2.0 * Hovvo - Hovov.swapaxes(2,3)))
            if self.ccwfn.model == 'CC2':
                tmp = contract('me,nmfe->nf', l1, self.ccwfn.t2)
                r_l1 = r_l1 + contract('nf,inaf->ia', tmp, (2 * self.ccwfn.H.L[o,o,v,v]))
                tmp = contract('me,mnfe->nf', l1, self.ccwfn.build_tau(self.ccwfn.t1, self.ccwfn.t2))
                r_l1 = r_l1 - contract('nf,inaf->ia', tmp, (2 * self.ccwfn.H.ERI[o,o,v,v]))
                r_l1 = r_l1 + contract('nf,inaf->ia', tmp, self.ccwfn.H.ERI[o,o,v,v].swapaxes(2,3))
            else:
                r_l1 = r_l1 - 2.0 * contract('ef,eifa->ia', Gvv, Hvovv)
                r_l1 = r_l1 + contract('ef,eiaf->ia', Gvv, Hvovv)
                r_l1 = r_l1 - 2.0 * contract('mn,mina->ia', Goo, Hooov)
                r_l1 = r_l1 + contract('mn,imna->ia', Goo, Hooov)
        return r_l1

    def r_L2(self, o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
        """Compute the L2 (lambda doubles) residual.

        The lambda equations are linear; solve_lambda drives this residual to
        zero. The expression below is symmetrized as r_l2_ijab += r_l2_jiba on
        return. H_* are the HBAR blocks, G_* the t2-weighted lambda densities.
        CCD drops the l1 terms; CC2 and CCSD(T) modify/extend the terms below.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no, nv, nv)
            L2 residual indexed [i, j, a, b].

        Notes
        -----
        CCSD form, before the i<->j / a<->b symmetrization (repeated indices
        summed)::

            r_l2_ijab = L_ijab
                      + 2 l1_ia H_jb - l1_ja H_ib
                      + 2 l1_ie H_ejab - l1_ie H_ejba
                      - 2 l1_mb H_jima + l1_mb H_ijma
                      + l2_ijeb H_ea - l2_mjab H_im
                      + 1/2 l2_mnab H_ijmn + 1/2 l2_ijef H_efab
                      + l2_mjeb (2 H_ieam - H_iema)
                      - l2_mibe H_jema - l2_mieb H_jeam
                      + G_ae L_ijeb - G_mi L_mjab
        """
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._r_L2_spinorbital(o, v, l1, l2, self.ccwfn.H.ERI, Hov, Hvv, Hoo,
                                          Hoooo, Hvvvv, Hovvo, Hvvvo, Hovoo, Hvovv, Hooov,
                                          Gvv, Goo)
        if self.ccwfn.model == 'CCD':
            r_l2 = clone(L[o,o,v,v], device=self.ccwfn.device1)

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
            r_l2 = clone(L[o,o,v,v], device=self.ccwfn.device1)

            # Add (T) contributions to L2
            if self.ccwfn.model == 'CCSD(T)':
                r_l2 = r_l2 + 0.5 * self.ccwfn.S2

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

    # Additional intermediates needed for CC3 lambda equations
    def build_cc3_Wmbje(self, o, v, ERI, t1):
        """Build the CC3 W_mbje intermediate (T1-dressed integrals).

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv, no, nv)
            Indexed [m, b, j, e].

        Notes
        -----
        Repeated indices summed::

            W_mbje = <mb|je> + t_jf <mb|fe> - t_nb <mn|je>
                            - t_jf t_nb <mn|fe>
        """
        contract = self.contract
        W = clone(ERI[o,v,o,v], device=self.ccwfn.device1)
        W = W + contract('mbfe,jf->mbje', ERI[o,v,v,v], t1)
        W = W - contract('mnje,nb->mbje', ERI[o,o,o,v], t1)
        W = W - contract('mnfe,jf,nb->mbje', ERI[o,o,v,v], t1, t1)
        return W

    def build_cc3_Wmbej(self, o, v, ERI, t1):
        """Build the CC3 W_mbej intermediate (T1-dressed integrals).

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv, nv, no)
            Indexed [m, b, e, j].

        Notes
        -----
        Repeated indices summed::

            W_mbej = <mb|ej> + t_jf <mb|ef> - t_nb <mn|ej>
                            - t_jf t_nb <mn|ef>
        """
        contract = self.contract
        W = clone(ERI[o,v,v,o], device=self.ccwfn.device1)
        W = W + contract('mbef,jf->mbej', ERI[o,v,v,v], t1)
        W = W - contract('mnej,nb->mbej', ERI[o,o,v,o], t1)
        W = W - contract('mnef,jf,nb->mbej', ERI[o,o,v,v], t1, t1)
        return W

    def build_cc3_Wabef(self, o, v, ERI, t1):
        """Build the CC3 W_abef intermediate (T1-dressed integrals).

        Returns
        -------
        ndarray or torch.Tensor, shape (nv, nv, nv, nv)
            Indexed [a, b, e, f].

        Notes
        -----
        Repeated indices summed::

            W_abef = <ab|ef> - t_ma <mb|ef> - t_mb <ma|fe>
                            + t_ma t_nb <mn|ef>
        """
        contract = self.contract
        W = clone(ERI[v,v,v,v], device=self.ccwfn.device1)
        tmp = contract('mbef,ma->abef', ERI[o,v,v,v], t1)
        W = W - tmp - tmp.swapaxes(0,1).swapaxes(2,3)
        W = W + contract('mnef,ma,nb->abef', ERI[o,o,v,v], t1, t1)
        return W
                                         
    def _r_L1_spinorbital(self, o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hvvvo, Hovoo,
                          Hvovv, Hooov, Gvv, Goo):
        """Spin-orbital L1 (lambda singles) residual, built from the antisymmetrized
        spin-orbital HBAR blocks (no Hovov)."""
        contract = self.contract
        r_l1 = clone(Hov)
        r_l1 = r_l1 + contract('ie,ea->ia', l1, Hvv)
        r_l1 = r_l1 - contract('ma,im->ia', l1, Hoo)
        r_l1 = r_l1 + 0.5 * contract('imef,efam->ia', l2, Hvvvo)
        r_l1 = r_l1 - 0.5 * contract('mnae,iemn->ia', l2, Hovoo)
        r_l1 = r_l1 + contract('me,ieam->ia', l1, Hovvo)
        r_l1 = r_l1 - contract('ef,eifa->ia', Gvv, Hvovv)
        r_l1 = r_l1 - contract('mn,mina->ia', Goo, Hooov)
        return r_l1

    def _r_L2_spinorbital(self, o, v, l1, l2, ERI, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo,
                          Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
        """Spin-orbital L2 (lambda doubles) residual. Built as the full residual,
        already antisymmetric in i<->j and a<->b (no separate symmetrization)."""
        contract = self.contract
        r_l2 = clone(ERI[o,o,v,v])
        r_l2 = r_l2 + (contract('ia,jb->ijab', l1, Hov) - contract('ja,ib->ijab', l1, Hov))
        r_l2 = r_l2 + (contract('jb,ia->ijab', l1, Hov) - contract('ib,ja->ijab', l1, Hov))
        r_l2 = r_l2 + (contract('ijae,eb->ijab', l2, Hvv) - contract('ijbe,ea->ijab', l2, Hvv))
        r_l2 = r_l2 - (contract('imab,jm->ijab', l2, Hoo) - contract('jmab,im->ijab', l2, Hoo))
        r_l2 = r_l2 + 0.5 * contract('ijef,efab->ijab', l2, Hvvvv)
        r_l2 = r_l2 + 0.5 * contract('mnab,ijmn->ijab', l2, Hoooo)
        r_l2 = r_l2 + (contract('ie,ejab->ijab', l1, Hvovv) - contract('je,eiab->ijab', l1, Hvovv))
        r_l2 = r_l2 - (contract('ma,ijmb->ijab', l1, Hooov) - contract('mb,ijma->ijab', l1, Hooov))
        tmp = contract('imae,jebm->ijab', l2, Hovvo)
        r_l2 = r_l2 + (tmp - tmp.swapaxes(0,1) - tmp.swapaxes(2,3)
                       + tmp.swapaxes(0,1).swapaxes(2,3))
        r_l2 = r_l2 + (contract('be,ijae->ijab', Gvv, ERI[o,o,v,v])
                       - contract('ae,ijbe->ijab', Gvv, ERI[o,o,v,v]))
        r_l2 = r_l2 - (contract('mj,imab->ijab', Goo, ERI[o,o,v,v])
                       - contract('mi,jmab->ijab', Goo, ERI[o,o,v,v]))
        return r_l2

    def pseudoenergy(self, o, v, ERI, l2):
        """Compute the CC pseudoenergy from the L2 amplitudes.

        Returns
        -------
        float
            The lambda pseudoenergy 1/2 <ij|ab> l2_ijab.
        """
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            return 0.25 * contract('ijab,ijab->', ERI[o,o,v,v], l2)
        return 0.5 * contract('ijab,ijab->',ERI[o,o,v,v], l2)
