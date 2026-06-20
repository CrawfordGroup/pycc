"""
ccresponse.py: CC Response Functions
"""

from __future__ import annotations

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import time
from typing import TYPE_CHECKING

import numpy as np
from .utils import helper_diis
from .cclambda import cclambda
from .cctriples import t3c_ijk_so, t3c_abc_so, l3_ijk_so
from ._typing import Tensor

if TYPE_CHECKING:
    from pycc.ccwfn import CCwfn
    from pycc.ccdensity import ccdensity

class ccresponse(object):
    """
    An RHF-CC Response Property Object.

    Methods
    -------
    linresp():
        Compute a CC linear response function.
    solve_right():
        Solve the right-hand perturbed wave function equations.
    pertcheck():
        Check first-order perturbed wave functions for all available perturbation operators.
    """

    def __init__(self, ccdensity: "ccdensity", omega1: float = 0, omega2: float = 0) -> None:
        """
        Parameters
        ----------
        ccdensity : PyCC ccdensity object
            Contains all components of the CC one- and two-electron densities, as well as references to the underlying ccwfn, cchbar, and cclambda objects
        omega1 : scalar
            The first external field frequency (for linear and quadratic response functions)
        omega2 : scalar
            The second external field frequency (for quadratic response functions)

        Returns
        -------
        None
        """

        self.ccwfn = ccdensity.ccwfn
        self.cclambda = ccdensity.cclambda
        self.H = self.ccwfn.H
        self.hbar = self.cclambda.hbar
        self.contract = self.ccwfn.contract

        # Cartesian indices
        self.cart = ["X", "Y", "Z"]

        # Similarity-transformed property integrals are built lazily, on first access
        # (see _build_pertbar / _PertbarCache), so each response function constructs only
        # the operators it actually uses -- polarizability needs MU, optical rotation
        # needs MU and M/M* -- rather than all of them (MU, M, M*, P, P*, Q) up front.
        self.pertbar = _PertbarCache(self)

        # HBAR-based denominators
        eps_occ = np.diag(self.hbar.Hoo)
        eps_vir = np.diag(self.hbar.Hvv)
        self.Dia = eps_occ.reshape(-1,1) - eps_vir
        self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

    def _build_pertbar(self, key):
        """Build the similarity-transformed perturbation operator for a single key
        (e.g. 'MU_X', 'M_Y', 'M*_Z', 'P_X', 'Q_XY'). Called lazily by ``self.pertbar``
        on first access, so a response function pays only for the operators it uses.

        The magnetic-dipole (M) and velocity-gauge (P) integrals are stored *pure
        imaginary* in hamiltonian.py (H.m, H.p = i * real) for the RT-CC code; the i is
        factored out here so the perturbed amplitudes stay real (the property values are
        bilinear in the perturbation, so dropping the i changes nothing). np.real(-1.0j *
        pert) returns a real-dtype operator; applied to the conjugates too, so M* = -M
        (P* = -P) stays distinct from M (P).
        """
        ax = {"X": 0, "Y": 1, "Z": 2}
        name, comp = key.split("_", 1)
        if name == "MU":
            pert = self.H.mu[ax[comp]]
        elif name == "M":
            pert = np.real(-1.0j * self.H.m[ax[comp]])
        elif name == "M*":
            pert = np.real(-1.0j * np.conj(self.H.m[ax[comp]]))
        elif name == "P":
            pert = np.real(-1.0j * self.H.p[ax[comp]])
        elif name == "P*":
            pert = np.real(-1.0j * np.conj(self.H.p[ax[comp]]))
        elif name == "Q":
            a1, a2 = sorted((ax[comp[0]], ax[comp[1]]))
            # H.Q is ordered XX, XY, XZ, YY, YZ, ZZ
            qidx = {(0,0): 0, (0,1): 1, (0,2): 2, (1,1): 3, (1,2): 4, (2,2): 5}[(a1, a2)]
            pert = self.H.Q[qidx]
        else:
            raise KeyError("Unknown perturbation operator key: %r" % key)
        return pertbar(pert, self.ccwfn)

    def pertcheck(self, omega: float, e_conv: float = 1e-13, r_conv: float = 1e-13, maxiter: int = 200, max_diis: int = 8, start_diis: int = 1):
        """
        Build first-order perturbed wave functions for all available perturbations and return a dict of their converged pseudoresponse values.  Primarily for testing purposes.

        Parameters
        ----------
        omega: float
            The external field frequency.
        e_conv : float
            convergence condition for the pseudoresponse value (default if 1e-13)
        r_conv : float
            convergence condition for perturbed wave function rmsd (default if 1e-13)
        maxiter : int
            maximum allowed number of iterations of the wave function equations (default is 100)
        max_diis : int
            maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
        start_diis : int
            earliest iteration to start DIIS extrapolations (default is 1)

        Returns
        -------
        check: dictionary
            Converged pseudoresponse values for all available perturbations.
        """
        # dictionaries for perturbed wave functions and test pseudoresponses
        X = {}
        check = {}

        # Electric-dipole (length)
        for axis in range(3):
            pertkey = "MU_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Magnetic-dipole
        for axis in range(3):
            pertkey = "M_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Complex-conjugate of magnetic-dipole
        for axis in range(3):
            pertkey = "M*_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Electric-dipole (velocity)
        for axis in range(3):
            pertkey = "P_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Complex-conjugate of electric-dipole (velocity)
        for axis in range(3):
            pertkey = "P*_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Traceless quadrupole
        for axis1 in range(3):
            for axis2 in range(3):
                pertkey = "Q_" + self.cart[axis1] + self.cart[axis2]
                X_key = pertkey + "_" + f"{omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar
                if (omega != 0.0):
                    X_key = pertkey + "_" + f"{-omega:0.6f}"
                    print("Solving right-hand perturbed wave function for %s:" % (X_key))
                    X[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                    check[X_key] = polar

        return check


    def solve_right(self, pertbar: "pertbar", omega: float, e_conv: float = 1e-12, r_conv: float = 1e-12, maxiter: int = 200, max_diis: int = 7, start_diis: int = 1):
        solver_start = time.time()

        Dia = self.Dia
        Dijab = self.Dijab

        # initial guess
        X1 = pertbar.Avo.T/(Dia + omega)
        X2 = pertbar.Avvoo/(Dijab + omega)

        pseudo = self.pseudoresponse(pertbar, X1, X2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        diis = helper_diis(X1, X2, max_diis)
        contract = self.ccwfn.contract

        self.X1 = X1
        self.X2 = X2

        cc3_ints = None
        if self.ccwfn.model == 'CC3':
            if self.ccwfn.orbital_basis != 'spinorbital':
                raise NotImplementedError(
                    "Spatial CC3 response is not implemented; use the spin-orbital "
                    "path (orbital_basis='spinorbital').")
            cc3_ints = self._cc3_response_setup_spinorbital(pertbar)

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            X1 = self.X1
            X2 = self.X2

            r1 = self.r_X1(pertbar, omega)
            r2 = self.r_X2(pertbar, omega)

            if self.ccwfn.model == 'CC3':
                z1, z2 = self._cc3_iter_spinorbital(pertbar, omega, cc3_ints)
                r1 += z1
                r2 += z2

            self.X1 += r1/(Dia + omega)
            self.X2 += r2/(Dijab + omega)

            rms = contract('ia,ia->', np.conj(r1/(Dia+omega)), r1/(Dia+omega))
            rms += contract('ijab,ijab->', np.conj(r2/(Dijab+omega)), r2/(Dijab+omega))
            rms = np.sqrt(rms)

            pseudo = self.pseudoresponse(pertbar, self.X1, self.X2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")

            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                if self.ccwfn.model == 'CC3':
                    X3 = self._cc3_build_X3_spinorbital(pertbar, omega, cc3_ints)
                    return [self.X1, self.X2, X3], pseudo
                return [self.X1, self.X2], pseudo

            diis.add_error_vector(self.X1, self.X2)
            if niter >= start_diis:
                self.X1, self.X2 = diis.extrapolate(self.X1, self.X2)

    def r_X1(self, pertbar, omega):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        X1 = self.X1
        X2 = self.X2
        hbar = self.hbar

        if self.ccwfn.orbital_basis == 'spinorbital':
            r_X1 = (pertbar.Avo.T - omega * X1).copy()
            r_X1 += contract('ie,ae->ia', X1, hbar.Hvv)
            r_X1 -= contract('ma,mi->ia', X1, hbar.Hoo)
            r_X1 += contract('me,maei->ia', X1, hbar.Hovvo)
            r_X1 += contract('me,imae->ia', hbar.Hov, X2)
            r_X1 += 0.5 * contract('imef,amef->ia', X2, hbar.Hvovv)
            r_X1 -= 0.5 * contract('mnae,mnie->ia', X2, hbar.Hooov)
            return r_X1

        r_X1 = (pertbar.Avo.T - omega * X1).copy()
        r_X1 += contract('ie,ae->ia', X1, hbar.Hvv)
        r_X1 -= contract('ma,mi->ia', X1, hbar.Hoo)
        r_X1 += 2.0*contract('me,maei->ia', X1, hbar.Hovvo)
        r_X1 -= contract('me,maie->ia', X1, hbar.Hovov)
        r_X1 += contract('me,miea->ia', hbar.Hov, (2.0*X2 - X2.swapaxes(0,1)))
        r_X1 += contract('imef,amef->ia', X2, (2.0*hbar.Hvovv - hbar.Hvovv.swapaxes(2,3)))
        r_X1 -= contract('mnae,mnie->ia', X2, (2.0*hbar.Hooov - hbar.Hooov.swapaxes(0,1)))

        return r_X1

    def r_X2(self, pertbar, omega):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        X1 = self.X1
        X2 = self.X2
        t2 = self.ccwfn.t2
        hbar = self.hbar

        if self.ccwfn.orbital_basis == 'spinorbital':
            ERI = self.H.ERI
            Zvv = contract('amef,me->af', hbar.Hvovv, X1)
            Zoo = contract('mnie,me->ni', hbar.Hooov, X1)
            Yoo = 0.5 * contract('mnef,mjef->nj', ERI[o,o,v,v], X2)
            Yvv = 0.5 * contract('mnef,mneb->fb', ERI[o,o,v,v], X2)
            r_X2 = (pertbar.Avvoo - omega * X2).copy()
            r_X2 += contract('ie,abej->ijab', X1, hbar.Hvvvo) - contract('je,abei->ijab', X1, hbar.Hvvvo)
            r_X2 -= contract('ma,mbij->ijab', X1, hbar.Hovoo) - contract('mb,maij->ijab', X1, hbar.Hovoo)
            r_X2 += contract('ni,njab->ijab', Zoo, t2) - contract('nj,niab->ijab', Zoo, t2)
            r_X2 -= contract('af,ijfb->ijab', Zvv, t2) - contract('bf,ijfa->ijab', Zvv, t2)
            r_X2 -= contract('nj,inab->ijab', Yoo, t2) - contract('ni,jnab->ijab', Yoo, t2)
            r_X2 -= contract('fb,ijaf->ijab', Yvv, t2) - contract('fa,ijbf->ijab', Yvv, t2)
            r_X2 += contract('ijae,be->ijab', X2, hbar.Hvv) - contract('ijbe,ae->ijab', X2, hbar.Hvv)
            r_X2 -= contract('imab,mj->ijab', X2, hbar.Hoo) - contract('jmab,mi->ijab', X2, hbar.Hoo)
            r_X2 += 0.5 * contract('mnab,mnij->ijab', X2, hbar.Hoooo)
            r_X2 += 0.5 * contract('ijef,abef->ijab', X2, hbar.Hvvvv)
            tmp = contract('imae,mbej->ijab', X2, hbar.Hovvo)
            r_X2 += tmp - tmp.swapaxes(0,1) - tmp.swapaxes(2,3) + tmp.swapaxes(0,1).swapaxes(2,3)
            return r_X2

        L = self.H.L
        Zvv = contract('amef,mf->ae', (2.0*hbar.Hvovv - hbar.Hvovv.swapaxes(2,3)), X1)
        Zvv -= contract('mnef,mnaf->ae', L[o,o,v,v], X2)

        Zoo = -1.0*contract('mnie,ne->mi', (2.0*hbar.Hooov - hbar.Hooov.swapaxes(0,1)), X1)
        Zoo -= contract('mnef,inef->mi', L[o,o,v,v], X2)

        r_X2 = 0.5 * (pertbar.Avvoo - omega * X2)
        r_X2 += contract('ie,abej->ijab', X1, hbar.Hvvvo)
        r_X2 -= contract('ma,mbij->ijab', X1, hbar.Hovoo)
        r_X2 += contract('mi,mjab->ijab', Zoo, t2)
        r_X2 += contract('ae,ijeb->ijab', Zvv, t2)
        r_X2 += contract('ijeb,ae->ijab', X2, hbar.Hvv)
        r_X2 -= contract('mjab,mi->ijab', X2, hbar.Hoo)
        r_X2 += 0.5*contract('mnab,mnij->ijab', X2, hbar.Hoooo)
        r_X2 += 0.5*contract('ijef,abef->ijab', X2, hbar.Hvvvv)
        r_X2 -= contract('imeb,maje->ijab', X2, hbar.Hovov)
        r_X2 -= contract('imea,mbej->ijab', X2, hbar.Hovvo)
        r_X2 += 2.0*contract('miea,mbej->ijab', X2, hbar.Hovvo)
        r_X2 -= contract('miea,mbje->ijab', X2, hbar.Hovov)

        r_X2 = r_X2 + r_X2.swapaxes(0,1).swapaxes(2,3)

        return r_X2

    def pseudoresponse(self, pertbar: "pertbar", X1: Tensor, X2: Tensor):
        contract = self.ccwfn.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            polar1 = contract('ai,ia->', pertbar.Avo, X1)
            polar2 = 0.25 * contract('ijab,ijab->', pertbar.Avvoo, X2)
            return -2.0 * (polar1 + polar2)
        polar1 = 2.0 * contract('ai,ia->', np.conj(pertbar.Avo), X1)
        polar2 = contract('ijab,ijab->', np.conj(pertbar.Avvoo), (2.0*X2 - X2.swapaxes(2,3)))

        return -2.0*(polar1 + polar2)

    def _cc3_response_setup_spinorbital(self, pertbar):
        """Build the T-dependent spin-orbital CC3 response intermediates that are
        reused across solve_right iterations: the CC3 W-intermediates and the
        once-only Yoovo/Yovvv (ground-state T3 . ERI) and Zovoo/Zvvvo
        (perturbation . T2). Ports socc CC3_noniter (IJK). Returns a dict."""
        contract = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        t1, t2 = self.ccwfn.t1, self.ccwfn.t2
        no = self.ccwfn.no

        Woooo = self.ccwfn._so_build_Woooo_CC3(o, v, ERI, t1)
        Wovoo = self.ccwfn._so_build_Wovoo_CC3(o, v, ERI, t1, Woooo)
        Wvvvo = self.ccwfn._so_build_Wvvvo_CC3(o, v, ERI, t1)
        Wvvvv = self.ccwfn._so_build_Wvvvv_CC3(o, v, ERI, t1)
        Wovvo = self.ccwfn._so_build_Wovvo_CC3(o, v, ERI, t1)

        Yoovo = np.zeros_like(ERI[o,o,v,o])
        Yovvv = np.zeros_like(ERI[o,v,v,v])
        for i in range(no):
            for j in range(no):
                for k in range(no):
                    t3 = t3c_ijk_so(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                    Yoovo[i,j] -= 0.5 * contract('abc,lbc->al', t3, ERI[o,k,v,v])
                    Yovvv[i] -= 0.5 * contract('abc,dc->abd', t3, ERI[j,k,v,v])
        Zovoo = 0.5 * contract('ld,jkdc->lcjk', pertbar.Aov, t2)
        Zvvvo = -0.5 * contract('ld,lkbc->bcdk', pertbar.Aov, t2)

        return {'Fov': self.hbar.Hov, 'Woooo': Woooo, 'Wovoo': Wovoo,
                'Wvvvo': Wvvvo, 'Wvvvv': Wvvvv, 'Wovvo': Wovvo,
                'Yoovo': Yoovo, 'Yovvv': Yovvv, 'Zovoo': Zovoo, 'Zvvvo': Zvvvo}

    def _cc3_iter_spinorbital(self, pertbar, omega, ints):
        """Per-iteration spin-orbital CC3 contributions (z1, z2) to the perturbed
        r_X1/r_X2. Rebuilds the perturbed triples X3 (loop-over-ijk and -abc, no
        stored X3) from the current X1/X2 and folds them back. Ports socc
        CC3_iter (IJK + ABC)."""
        contract = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        t2 = self.ccwfn.t2
        hbar = self.hbar
        X1, X2 = self.X1, self.X2
        no, nv = X1.shape

        Woooo, Wovoo = ints['Woooo'], ints['Wovoo']
        Wvvvo, Wvvvv, Wovvo = ints['Wvvvo'], ints['Wvvvv'], ints['Wovvo']
        Yoovo, Yovvv = ints['Yoovo'], ints['Yovvv']
        Zovoo, Zvvvo = ints['Zovoo'], ints['Zvvvo']

        z1 = np.zeros_like(X1)
        z2 = np.zeros_like(X2)

        # <mu2|[[H~,T3],X1]|0> --> X2 (the X1-dressed once-only pieces)
        Yov = contract('ld,klcd->kc', X1, ERI[o,o,v,v])
        tmp = contract('ld,ijal->ijad', X1, Yoovo)
        z2 += tmp - tmp.swapaxes(2,3)
        tmp = contract('ld,iabd->ilab', X1, Yovvv)
        z2 += tmp - tmp.swapaxes(0,1)

        # <mu3|[[H~,T2],X1]|0> --> X3 dressings (X1-dressed W intermediates)
        Zbcdk = contract('ke,bcde->bcdk', X1, Wvvvv)
        tmp = -contract('lb,lcdk->bcdk', X1, Wovvo)
        Zbcdk += tmp - tmp.swapaxes(0,1)
        Zlcjk = -contract('mc,lmjk->lcjk', X1, Woooo)
        tmp = contract('jd,lcdk->lcjk', X1, Wovvo)
        Zlcjk += tmp - tmp.swapaxes(2,3)

        occ = np.diag(F)[o]
        vir = np.diag(F)[v]

        # occupied-batched (IJK): X3 from [A,T3]_Avv + [[A,T2],T2]+[[H,T2],X1] + [H,X2]
        for i in range(no):
            for j in range(no):
                for k in range(no):
                    t3 = t3c_ijk_so(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                    z2[i,j] += contract('abc,c->ab', t3, Yov[k])

                    tmp = contract('abc,dc->abd', t3, pertbar.Avv)
                    x3 = tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)
                    denom = (occ[i] + occ[j] + occ[k] + omega
                             - (vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir))
                    x3 = x3/denom
                    x3 += t3c_ijk_so(o, v, i, j, k, t2, Zvvvo+Zbcdk, Zovoo+Zlcjk, F, contract, omega)
                    x3 += t3c_ijk_so(o, v, i, j, k, X2, Wvvvo, Wovoo, F, contract, omega)

                    z1[i] += 0.25 * contract('abc,bc->a', x3, ERI[j,k,v,v])
                    z2[i,j] += contract('abc,c->ab', x3, hbar.Hov[k])
                    tmp = 0.5 * contract('abc,dbc->ad', x3, hbar.Hvovv[:,k,:,:])
                    z2[i,j] += tmp - tmp.swapaxes(0,1)
                    for l in range(no):
                        tmp = -0.5 * contract('abc,c->ab', x3, hbar.Hooov[j,k,l,:])
                        z2[i,l] += tmp
                        z2[l,i] -= tmp

        # virtual-batched (ABC): X3 from [A,T3]_Aoo
        y1 = np.zeros_like(z1.T)
        y2 = np.zeros_like(z2.T)
        for a in range(nv):
            for b in range(nv):
                for c in range(nv):
                    t3 = t3c_abc_so(o, v, a, b, c, t2, Wvvvo, Wovoo, F, contract)
                    tmp = -contract('ijk,kl->ijl', t3, pertbar.Aoo)
                    x3 = tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)
                    denom = (occ.reshape(-1,1,1) + occ.reshape(-1,1) + occ + omega
                             - (vir[a] + vir[b] + vir[c]))
                    x3 = x3/denom

                    y1[a] += 0.25 * contract('ijk,jk->i', x3, ERI[o,o,b+no,c+no])
                    y2[a,b] += contract('ijk,k->ij', x3, hbar.Hov[:,c])
                    tmp = -0.5 * contract('ijk,jkl->il', x3, hbar.Hooov[:,:,:,c])
                    y2[a,b] += tmp - tmp.swapaxes(0,1)
                    for d in range(nv):
                        tmp = 0.5 * contract('ijk,k->ij', x3, hbar.Hvovv[d,:,b,c])
                        y2[a,d] += tmp
                        y2[d,a] -= tmp

        z1 += y1.T
        z2 += y2.T
        return z1, z2

    def _cc3_build_X3_spinorbital(self, pertbar, omega, ints):
        """Build and return the full converged spin-orbital perturbed triples X3
        from the converged X1/X2 (self.X1/self.X2).

        This replicates the X3-block construction inside _cc3_iter_spinorbital
        (occupied-batched IJK + virtual-batched ABC, omega-shifted denominators)
        but accumulates into the full X3[ijkabc] array instead of folding into
        z1/z2. Called once after solve_right converges so the response-function
        terms (Phase B) can contract against a stored X3. Counterpart of socc
        CC3_iter_full's stored self.X3 (store_triples=True path)."""
        contract = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        F = self.ccwfn.H.F
        t2 = self.ccwfn.t2
        X1, X2 = self.X1, self.X2
        no, nv = X1.shape

        Woooo, Wovoo = ints['Woooo'], ints['Wovoo']
        Wvvvo, Wvvvv, Wovvo = ints['Wvvvo'], ints['Wvvvv'], ints['Wovvo']
        Zovoo, Zvvvo = ints['Zovoo'], ints['Zvvvo']

        # X1-dressed W intermediates (same as _cc3_iter_spinorbital)
        Zbcdk = contract('ke,bcde->bcdk', X1, Wvvvv)
        tmp = -contract('lb,lcdk->bcdk', X1, Wovvo)
        Zbcdk += tmp - tmp.swapaxes(0,1)
        Zlcjk = -contract('mc,lmjk->lcjk', X1, Woooo)
        tmp = contract('jd,lcdk->lcjk', X1, Wovvo)
        Zlcjk += tmp - tmp.swapaxes(2,3)

        occ = np.diag(F)[o]
        vir = np.diag(F)[v]

        X3 = np.zeros((no, no, no, nv, nv, nv), dtype=X2.dtype)

        # occupied-batched (IJK): X3 from [A,T3]_Avv + dressed-t3 ([[H,T2],X1]) + [H,X2]
        for i in range(no):
            for j in range(no):
                for k in range(no):
                    t3 = t3c_ijk_so(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                    tmp = contract('abc,dc->abd', t3, pertbar.Avv)
                    x3 = tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)
                    denom = (occ[i] + occ[j] + occ[k] + omega
                             - (vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir))
                    x3 = x3/denom
                    x3 += t3c_ijk_so(o, v, i, j, k, t2, Zvvvo+Zbcdk, Zovoo+Zlcjk, F, contract, omega)
                    x3 += t3c_ijk_so(o, v, i, j, k, X2, Wvvvo, Wovoo, F, contract, omega)
                    X3[i,j,k] += x3

        # virtual-batched (ABC): X3 from [A,T3]_Aoo
        for a in range(nv):
            for b in range(nv):
                for c in range(nv):
                    t3 = t3c_abc_so(o, v, a, b, c, t2, Wvvvo, Wovoo, F, contract)
                    tmp = -contract('ijk,kl->ijl', t3, pertbar.Aoo)
                    x3 = tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)
                    denom = (occ.reshape(-1,1,1) + occ.reshape(-1,1) + occ + omega
                             - (vir[a] + vir[b] + vir[c]))
                    x3 = x3/denom
                    X3[:,:,:,a,b,c] += x3

        return X3

    # ==================================================================
    # Symmetric linear response -- the X-only formulation: only the
    # right-hand perturbed amplitudes X(P,+/-w) are solved (no left-hand Y).
    # Top-level entry points polarizability() and optrot() dispatch the
    # basis-specific assembly through linresp_sym(). The asymmetric (X and Y)
    # machinery is retained, deprecated for linear response, at the bottom of
    # the class for the future quadratic response function.
    #
    # The linear response function <<A;B>>_w generally requires the following
    # right-hand perturbed wave functions and frequencies:
    #     A(-w), A*(w), B(w), B*(-w)
    # If the external field is static (w=0), then we need:
    #     A(0), A*(0), B(0), B*(0)
    # If the perturbation A is real and B is pure imaginary:
    #     A(-w), A(w), B(w), B*(-w)
    # or vice versa:
    #     A(-w), A*(w), B(w), B(-w)
    # If the perturbations are both real and the field is static:
    #     A(0), B(0)
    # If the perturbations are identical then:
    #     A(w), A*(-w)  or  A(0), A*(0)
    # If the perturbations are identical, the field is dynamic and the operator
    # is real:
    #     A(-w), A(w)
    # If the perturbations are identical, the field is static and the operator
    # is real:
    #     A(0)
    # ==================================================================

    def polarizability(self, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200,
                       max_diis=7, start_diis=1):
        """Dipole polarizability tensor (length gauge) at frequency omega via the
        symmetric response function (right-hand perturbed amplitudes only):

            alpha_w = -<<mu;mu>>_w
                    = -<0|(1+L){ [muBAR,X(mu,-w)] + [muBAR,X(mu,w)]
                                 + [[HBAR,X(mu,-w)],X(mu,w)] }|0>

        Returns a 3x3 array. Basis-agnostic: all basis-specific work lives in
        solve_right and linresp_sym."""
        args = (e_conv, r_conv, maxiter, max_diis, start_diis)
        Xp, Xm = [], []
        # solve_right returns (X, pseudo) where X is the list [X1, X2] (CCSD) or
        # [X1, X2, X3] (CC3); the CC3 terms in linresp_sym pick up X[2].
        for axis in range(3):
            A = self.pertbar["MU_" + self.cart[axis]]
            X, _ = self.solve_right(A, omega, *args)
            Xp.append([a.copy() for a in X])
            if omega == 0.0:
                Xm.append([a.copy() for a in X])
            else:
                X, _ = self.solve_right(A, -omega, *args)
                Xm.append([a.copy() for a in X])

        polar = np.zeros((3, 3))
        for a in range(3):
            A = self.pertbar["MU_" + self.cart[a]]
            for b in range(3):
                B = self.pertbar["MU_" + self.cart[b]]
                polar[a, b] = -1.0 * self.linresp_sym(A, Xm[a], B, Xp[b])
        return polar

    def optrot(self, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200,
               max_diis=7, start_diis=1):
        """Optical-rotation tensor (length gauge) at frequency omega via the symmetric
        response function (right-hand perturbed amplitudes only):

            G'_w = <<mu;m>>_w
                 = (1/2) <0|(1+L){ [muBAR,X(m,w)] + [mBAR,X(mu,-w)]
                                   + [[HBAR,X(mu,-w)],X(m,w)] }|0>
                 + (1/2) <0|(1+L){ [muBAR,X(m*,-w)] + [m*BAR,X(mu,w)]
                                   + [[HBAR,X(m*,-w)],X(mu,w)] }|0>

        assembled from the right-hand perturbed amplitudes X(mu,-w), X(m,+w),
        X(mu,+w), X(m*,-w). Returns a 3x3 array. Basis-agnostic: all basis-specific
        work lives in solve_right and linresp_sym.
        """
        if omega == 0.0:
            raise ValueError("Optical rotation requires a nonzero field frequency.")
        args = (e_conv, r_conv, maxiter, max_diis, start_diis)

        Xmu_p, Xmu_m, Xm_p, Xmstar_m = [], [], [], []
        # solve_right returns (X, pseudo) where X is [X1, X2(, X3)] (see
        # polarizability).
        for axis in range(3):
            X, _ = self.solve_right(self.pertbar["MU_" + self.cart[axis]], omega, *args)
            Xmu_p.append([a.copy() for a in X])
            X, _ = self.solve_right(self.pertbar["MU_" + self.cart[axis]], -omega, *args)
            Xmu_m.append([a.copy() for a in X])
            X, _ = self.solve_right(self.pertbar["M_" + self.cart[axis]], omega, *args)
            Xm_p.append([a.copy() for a in X])
            X, _ = self.solve_right(self.pertbar["M*_" + self.cart[axis]], -omega, *args)
            Xmstar_m.append([a.copy() for a in X])

        tensor = np.zeros((3, 3))
        # (1/2) <<mu; m>>_omega  with X(mu,-omega) and X(m,+omega)
        for a in range(3):
            A = self.pertbar["MU_" + self.cart[a]]
            for b in range(3):
                B = self.pertbar["M_" + self.cart[b]]
                tensor[a, b] = 0.5 * self.linresp_sym(A, Xmu_m[a], B, Xm_p[b])
        # (1/2) <<mu; m*>>_{-omega}  with X(mu,+omega) and X(m*,-omega)
        for a in range(3):
            A = self.pertbar["MU_" + self.cart[a]]
            for b in range(3):
                B = self.pertbar["M*_" + self.cart[b]]
                tensor[a, b] += 0.5 * self.linresp_sym(A, Xmu_p[a], B, Xmstar_m[b])
        return tensor

    # ------------------------------------------------------------------
    # Symmetric linear-response assembly. Each component dispatches on
    # orbital_basis: the spin-orbital form (_*_spinorbital) is implemented;
    # the spin-adapted (spatial) form is phase 9a-ii (stubbed). The call
    # structure (linresp_sym -> LCX / LHX1Y1 / LHX2Y2 / LHX1Y2) is identical
    # for both bases; only the tensor contractions differ.
    # ------------------------------------------------------------------

    def linresp_sym(self, A, X_A, B, X_B):
        """Half of the symmetric CC linear-response function for one-electron
        perturbations A and B at frequency omega (w):

            <<A;B>>_w = (1/2) <0|(1+L){ [ABAR,X(B,w)] + [BBAR,X(A,-w)]
                                        + [[HBAR,X(A,-w)],X(B,w)] }|0>

        The other half -- using the complex conjugates of the operators and the
        swapped frequencies -- comes from a separate call (see optrot; for the real,
        symmetric polarizability the two halves are equal). This is for specific
        Cartesian components: both the perturbed wave functions (X_A, X_B) and the
        similarity-transformed operators (A, B) are built by the caller.

        Parameters
        ----------
        A, B : pertbar
            Similarity-transformed left- (A) and right-hand (B) perturbation operators.
        X_A, X_B : [singles, doubles]
            Right-hand perturbed wave functions for A (at -w) and B (at +w).

        Returns
        -------
        float
            The requested component of the linear-response tensor (sum of the LCX,
            HXY, LHX1Y1, LHX2Y2, and LHX1Y2 contributions).

        Notes
        -----
        Spin-adapted (spatial) assembly (the spin-orbital path has the same
        structure; each component dispatches to its own basis-specific code). The
        only explicit contraction is the HXY direct term: spin-summing the
        antisymmetrized <ij||ab> X_A1_ia X_B1_jb over the (sigma_i, sigma_j) spin
        cases gives 4<ij|ab> - 2<ij|ba> = 2 L_ijab (L = 2<ij|ab> - <ij|ba>) -- the
        direct integral survives all four spin combinations, the exchange only the
        two with sigma_i == sigma_j::

            <<A;B>>_w = LCX(A, X_B) + LCX(B, X_A)              (LCX)
                      + 2 L_ijab X_A1_ia X_B1_jb               (HXY)
                      + LHX1Y1(X_A, X_B)                       (LHX1Y1)
                      + LHX2Y2(X_A, X_B)                       (LHX2Y2)
                      + LHX1Y2(X_A, X_B) + LHX1Y2(X_B, X_A)    (LHX1Y2)
        """
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._linresp_sym_spinorbital(A, X_A, B, X_B)
        # spin-adapted (spatial) assembly. Identical in structure to the spin-orbital
        # path: LCX/LHX1Y1/LHX2Y2/LHX1Y2 each dispatch to their own spatial code. The
        # only basis-specific piece is the HXY direct term. Spin-summing the
        # antisymmetrized <ij||ab> X_A[ia] X_B[jb] over the (sigma_i, sigma_j) spin
        # cases gives 4<ij|ab> - 2<ij|ba> = 2*L (L = 2<ij|ab> - <ij|ba> = self.H.L):
        # the direct integral survives all four spin combinations, the exchange only
        # the two with sigma_i == sigma_j.
        o, v = self.ccwfn.o, self.ccwfn.v
        contract = self.contract
        L = self.H.L
        polar = self.LCX(A, X_B) + self.LCX(B, X_A)
        polar += 2.0 * contract('ijab,ia,jb->', L[o,o,v,v], X_A[0], X_B[0])
        polar += self.LHX1Y1(X_A, X_B)
        polar += self.LHX2Y2(X_A, X_B)
        polar += self.LHX1Y2(X_A, X_B)
        polar += self.LHX1Y2(X_B, X_A)
        return polar

    def _linresp_sym_spinorbital(self, A, X_A, B, X_B):
        o, v = self.ccwfn.o, self.ccwfn.v
        contract = self.contract
        ERI = self.H.ERI
        # <0|(1+L)[ABAR,X_B]|0> + <0|(1+L)[BBAR,X_A]|0>
        polar = self.LCX(A, X_B) + self.LCX(B, X_A)
        # <0|[[HBAR,X1_A],X1_B]|0>
        polar += contract('ijab,ia,jb->', ERI[o,o,v,v], X_A[0], X_B[0])
        # <0|L[[HBAR,X1_A],X1_B]|0>
        polar += self.LHX1Y1(X_A, X_B)
        # <0|L[[HBAR,X2_A],X2_B]|0>
        polar += self.LHX2Y2(X_A, X_B)
        # <0|L[[HBAR,X1_A],X2_B]|0>
        polar += self.LHX1Y2(X_A, X_B)
        # <0|L[[HBAR,X1_B],X2_A]|0>
        polar += self.LHX1Y2(X_B, X_A)
        return polar

    def LCX(self, pert, X):
        """One-particle-density (LCX) term of the symmetric response function:
        <0|(1+L)[Abar, X]|0>. (Diagram labels match _LCX_spinorbital; this is the same
        quantity as the <0|(1+L)[Abar,X(B)]|0> contribution of linresp_asym (polar2).)

        Notes
        -----
        Spin-adapted (spatial) form (repeated indices summed). A_* are the
        similarity-transformed perturbation blocks (pert.Aov / Avv / Aoo / Avvvo /
        Aovoo)::

            LCX = 2 A_ia X1_ia                                         (diagram 1)
                + l1_ia X1_ic A_ac - l1_ia X1_ka A_ki                  (diagrams 2, 3)
                + l1_ia A_jb (2 X2_ijab - X2_ijba)                     (diagram 8)
                + l2_ijbc A_bcaj X1_ia                                 (diagram 4)
                - 1/2 l2_ijab A_kbij X1_ka - 1/2 l2_ijab A_kaji X1_kb  (diagram 5)
                + 1/2 l2_ijab X2_ijac A_bc + 1/2 l2_ijab X2_ijcb A_ac  (diagram 6)
                - 1/2 l2_ijab X2_kjab A_ki - 1/2 l2_ijab X2_kiba A_kj  (diagram 7)
        """
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._LCX_spinorbital(pert, X)
        # spin-adapted (spatial) LCX
        contract = self.contract
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        X1, X2 = X[0], X[1]
        # <0|[Abar, X1]|0>                                          (diagram 1)
        polar = 2.0 * contract('ia,ia->', pert.Aov, X1)
        # <0|L1[Abar, X1]|0>                                        (diagrams 2, 3)
        tmp = contract('ia,ic->ac', l1, X1)
        polar += contract('ac,ac->', tmp, pert.Avv)
        tmp = contract('ia,ka->ik', l1, X1)
        polar -= contract('ik,ki->', tmp, pert.Aoo)
        # <0|L1[Abar, X2]|0>                                        (diagram 8)
        tmp = contract('ia,jb->ijab', l1, pert.Aov)
        polar += 2.0 * contract('ijab,ijab->', tmp, X2)
        polar -= contract('ijab,ijba->', tmp, X2)
        # <0|L2[Abar, X1]|0>                                        (diagrams 4, 5)
        tmp = contract('ijbc,bcaj->ia', l2, pert.Avvvo)
        polar += contract('ia,ia->', tmp, X1)
        tmp = contract('ijab,kbij->ak', l2, pert.Aovoo)
        polar -= 0.5 * contract('ak,ka->', tmp, X1)
        tmp = contract('ijab,kaji->bk', l2, pert.Aovoo)
        polar -= 0.5 * contract('bk,kb->', tmp, X1)
        # <0|L2[Abar, X2]|0>                                        (diagrams 6, 7)
        tmp = contract('ijab,kjab->ik', l2, X2)
        polar -= 0.5 * contract('ik,ki->', tmp, pert.Aoo)
        tmp = contract('ijab,kiba->jk', l2, X2)
        polar -= 0.5 * contract('jk,kj->', tmp, pert.Aoo)
        tmp = contract('ijab,ijac->bc', l2, X2)
        polar += 0.5 * contract('bc,bc->', tmp, pert.Avv)
        tmp = contract('ijab,ijcb->ac', l2, X2)
        polar += 0.5 * contract('ac,ac->', tmp, pert.Avv)
        return polar

    def _LCX_spinorbital(self, pert, X):
        contract = self.contract
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        X1, X2 = X[0], X[1]
        polar = contract('ia,ia->', pert.Aov, X1)              # diagram 1
        tmp = contract('ae,ie->ia', pert.Avv, X1)              # diagram 2
        tmp -= contract('mi,ma->ia', pert.Aoo, X1)             # diagram 3
        tmp += contract('me,imae->ia', pert.Aov, X2)           # diagram 8
        polar += contract('ia,ia->', l1, tmp)
        tmp = contract('abej,ie->ijab', pert.Avvvo, X1)        # diagram 4
        tmp -= contract('mbij,ma->ijab', pert.Aovoo, X1)       # diagram 5
        tmp += contract('be,ijae->ijab', pert.Avv, X2)         # diagram 6
        tmp -= contract('mj,imab->ijab', pert.Aoo, X2)         # diagram 7
        polar += 0.5 * contract('ijab,ijab->', l2, tmp)
        return polar

    def LHX1Y1(self, X, Y):
        """X1*Y1 (LHX1Y1) term of the symmetric response function:
        <0|L[[HBAR,X1],Y1]|0>. (Diagram labels match _LHX1Y1_spinorbital.)

        Notes
        -----
        Spin-adapted (spatial) form (repeated indices summed). H_* are hbar blocks
        (H_me = H_ov, H_amef = H_vovv, H_mnie = H_ooov, H_mnij = H_oooo,
        H_abef = H_vvvv, H_mbej = H_ovvo, H_maje = H_ovov); L_mnef = 2<mn|ef> -
        <mn|fe>; HvL_amef = 2 H_amef - H_amfe and HoL_mnie = 2 H_mnie - H_nmie are
        the L-combinations of H_vovv / H_ooov. The singles enter through
        tau_ijab = X1_ia Y1_jb + Y1_ia X1_jb (both orderings, since X1 != Y1)::

            # l1-part (diagrams 1, 2, 7-10)
            tmp_ia = -H_me tau_imea + HvL_amef tau_imef - HoL_mnie tau_mnae
            # l2-part (diagrams 3-6, 11-14)
            Z_fb   = 1/2 L_mnef tau_mneb
            Z_nj   = 1/2 L_mnef tau_mjef
            ring   = -tau_imea H_mbej - tau_imeb H_maje               (diagrams 5, 6)
            tmp_ijab = 1/2 H_mnij X1_ma Y1_nb + 1/2 H_abef X1_ie Y1_jf
                     - t2_ijaf Z_fb - t2_inab Z_nj + 1/2 ring_ijab
            LHX1Y1 = l1_ia tmp_ia + 2 l2_ijab tmp_ijab
        """
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._LHX1Y1_spinorbital(X, Y)
        # spin-adapted (spatial) LHX1Y1
        contract = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        t2 = self.ccwfn.t2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        hbar = self.hbar
        L = self.H.L
        X1, Y1 = X[0], Y[0]
        tau = contract('ia,jb->ijab', X1, Y1) + contract('ia,jb->ijab', Y1, X1)
        # L1 part (diagrams 1-2, 7-10): naive Hov, L-combinations on Hvovv/Hooov
        HvL = 2.0 * hbar.Hvovv - hbar.Hvovv.swapaxes(2,3)
        HoL = 2.0 * hbar.Hooov - hbar.Hooov.swapaxes(0,1)
        tmp = -1.0 * contract('me,imea->ia', hbar.Hov, tau)
        tmp += contract('amef,imef->ia', HvL, tau)
        tmp -= contract('mnie,mnae->ia', HoL, tau)
        polar = contract('ia,ia->', l1, tmp)
        # L2 part (diagrams 3-6, 11-14). The ring (diagrams 5,6) follows the
        # _r_T2_ccsd t1*t1 disconnected ring, but with both singles orderings
        # (X1(x)Y1 and Y1(x)X1, here folded into tau) since X1 != Y1 in general.
        Zvv = 0.5 * contract('mnef,mneb->fb', L[o,o,v,v], tau)
        Zoo = 0.5 * contract('mnef,mjef->nj', L[o,o,v,v], tau)
        ring = (-contract('imea,mbej->ijab', tau, hbar.Hovvo)
                - contract('imeb,maje->ijab', tau, hbar.Hovov))
        tmp = 0.5 * contract('mnij,ma,nb->ijab', hbar.Hoooo, X1, Y1)
        tmp += 0.5 * contract('abef,ie,jf->ijab', hbar.Hvvvv, X1, Y1)
        tmp -= contract('ijaf,fb->ijab', t2, Zvv)
        tmp -= contract('inab,nj->ijab', t2, Zoo)
        tmp += 0.5 * ring
        polar += 2.0 * contract('ijab,ijab->', l2, tmp)
        return polar

    def _LHX1Y1_spinorbital(self, X, Y):
        contract = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        t2 = self.ccwfn.t2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        hbar = self.hbar
        ERI = self.H.ERI
        X1, Y1 = X[0], Y[0]
        tau = contract('ia,jb->ijab', X1, Y1) + contract('ia,jb->ijab', Y1, X1)
        tmp = -1.0 * contract('me,imea->ia', hbar.Hov, tau)        # diagrams 1 and 2
        tmp += contract('amef,imef->ia', hbar.Hvovv, tau)          # diagrams 7 and 8
        tmp -= contract('mnie,mnae->ia', hbar.Hooov, tau)          # diagrams 9 and 10
        polar = contract('ia,ia->', l1, tmp)
        Zvv = 0.5 * contract('mnef,mneb->fb', ERI[o,o,v,v], tau)
        Zoo = 0.5 * contract('mnef,mjef->nj', ERI[o,o,v,v], tau)
        tmp = 0.5 * contract('mnij,ma,nb->ijab', hbar.Hoooo, X1, Y1)   # diagram 3
        tmp += 0.5 * contract('abef,ie,jf->ijab', hbar.Hvvvv, X1, Y1)  # diagram 4
        tmp -= contract('mbej,imea->ijab', hbar.Hovvo, tau)        # diagrams 5 and 6
        tmp -= contract('ijaf,fb->ijab', t2, Zvv)                  # diagrams 11 and 12
        tmp -= contract('inab,nj->ijab', t2, Zoo)                  # diagrams 13 and 14
        polar += contract('ijab,ijab->', l2, tmp)
        return polar

    def LHX2Y2(self, X, Y):
        """X2*Y2 (LHX2Y2) term of the symmetric response function:
        <0|L[[HBAR,X2],Y2]|0>. (Diagram labels match _LHX2Y2_spinorbital.)

        Notes
        -----
        Spin-adapted (spatial) form (repeated indices summed). <mn|ef> = ERI[o,o,v,v],
        L_mnef = 2<mn|ef> - <mn|fe>; W_mbje[e<->j] swaps the last two axes; W_mbie is
        W_mbje with j -> i. Diagram 1 is the particle-hole ring (the 1/2-weighted
        W_mbej / W_mbje built from Y2, contracted with X2 through the _r_T2_ccsd
        three-term ring); diagrams 2-7 are the oo/vv ladders, symmetrized over
        X2 <-> Y2::

            W_mbej = -1/2 Y2_jnfb <mn|ef> + 1/2 Y2_njfb L_mnef
            W_mbje =  1/2 Y2_jnfb <mn|fe>
            tmp_ijab = (X2_imae - X2_imea) W_mbej + X2_imae (W_mbej + W_mbje[e<->j])
                     + X2_mjae W_mbie                                   (diagram 1)
                     + 1/2 (1/2 <mn|ef> X2_ijef) Y2_mnab
                     + 1/2 (1/2 <mn|ef> Y2_ijef) X2_mnab               (diagrams 2, 3)
                     + 1/2 (-L_mnef Y2_mnbf) X2_ijae
                     + 1/2 (-L_mnef X2_mnbf) Y2_ijae                   (diagrams 4, 6)
                     + 1/2 (-L_mnef Y2_jnef) X2_imab
                     + 1/2 (-L_mnef X2_jnef) Y2_imab                   (diagrams 5, 7)
            LHX2Y2 = 2 l2_ijab tmp_ijab
        """
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._LHX2Y2_spinorbital(X, Y)
        # spin-adapted (spatial) LHX2Y2
        contract = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        l2 = self.cclambda.l2
        ERI = self.H.ERI[o, o, v, v]
        L = self.H.L[o, o, v, v]
        X2, Y2 = X[1], Y[1]
        # diagram 1 (ph-ring): the T2-equation Wmbej/Wmbje intermediates (the 1/2-
        # weighted ones from build_Wmbej/build_Wmbje) built from Y2, contracted with X2
        # via the three-term _r_T2_ccsd ring.
        Wmbej = -0.5 * contract('jnfb,mnef->mbej', Y2, ERI) + 0.5 * contract('njfb,mnef->mbej', Y2, L)
        Wmbje = 0.5 * contract('jnfb,mnfe->mbje', Y2, ERI)
        tmp = contract('imae,mbej->ijab', X2 - X2.swapaxes(2,3), Wmbej)
        tmp += contract('imae,mbej->ijab', X2, Wmbej + Wmbje.swapaxes(2,3))
        tmp += contract('mjae,mbie->ijab', X2, Wmbje)
        # diagrams 2,3 (oo ladder)
        tmp += 0.5 * contract('mnij,mnab->ijab', 0.5 * contract('mnef,ijef->mnij', ERI, X2), Y2)
        tmp += 0.5 * contract('mnij,mnab->ijab', 0.5 * contract('mnef,ijef->mnij', ERI, Y2), X2)
        # diagrams 4,6 (vv ladder)
        tmp += 0.5 * contract('eb,ijae->ijab', -contract('mnef,mnbf->eb', L, Y2), X2)
        tmp += 0.5 * contract('eb,ijae->ijab', -contract('mnef,mnbf->eb', L, X2), Y2)
        # diagrams 5,7 (oo ladder)
        tmp += 0.5 * contract('mj,imab->ijab', -contract('mnef,jnef->mj', L, Y2), X2)
        tmp += 0.5 * contract('mj,imab->ijab', -contract('mnef,jnef->mj', L, X2), Y2)
        return 2.0 * contract('ijab,ijab->', l2, tmp)

    def _LHX2Y2_spinorbital(self, X, Y):
        contract = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        l2 = self.cclambda.l2
        ERI = self.H.ERI
        X2, Y2 = X[1], Y[1]
        Zovvo = contract('mnef,njfb->mbej', ERI[o,o,v,v], Y2)
        Zoooo_A = 0.25 * contract('mnef,ijef->mnij', ERI[o,o,v,v], X2)
        Zoooo_B = 0.25 * contract('mnef,ijef->mnij', ERI[o,o,v,v], Y2)
        Zvv_A = -0.5 * contract('mnef,mnbf->eb', ERI[o,o,v,v], Y2)
        Zvv_B = -0.5 * contract('mnef,mnbf->eb', ERI[o,o,v,v], X2)
        Zoo_A = -0.5 * contract('mnef,jnef->mj', ERI[o,o,v,v], Y2)
        Zoo_B = -0.5 * contract('mnef,jnef->mj', ERI[o,o,v,v], X2)
        tmp = contract('mbej,imae->ijab', Zovvo, X2)              # diagram 1
        tmp += 0.25 * contract('mnij,mnab->ijab', Zoooo_A, Y2)    # diagram 2
        tmp += 0.25 * contract('mnij,mnab->ijab', Zoooo_B, X2)    # diagram 3
        tmp += 0.5 * contract('eb,ijae->ijab', Zvv_A, X2)         # diagram 4
        tmp += 0.5 * contract('eb,ijae->ijab', Zvv_B, Y2)         # diagram 6
        tmp += 0.5 * contract('mj,imab->ijab', Zoo_A, X2)         # diagram 5
        tmp += 0.5 * contract('mj,imab->ijab', Zoo_B, Y2)         # diagram 7
        return contract('ijab,ijab->', l2, tmp)

    def LHX1Y2(self, X, Y):
        """X1*Y2 (LHX1Y2) term of the symmetric response function:
        <0|L[[HBAR,X1],Y2]|0>. (Diagram labels match _LHX1Y2_spinorbital.)

        Notes
        -----
        Spin-adapted (spatial) form (repeated indices summed). X1 = X[0], Y2 = Y[1];
        H_me = H_ov, and H_bmfe / H_bmef are the H_vovv block while H_nmje / H_mnje /
        H_mnie are the H_ooov block (read with the index slots shown); L_mnef =
        2<mn|ef> - <mn|fe>; HvL_amef = 2 H_amef - H_amfe, HoL_mnie = 2 H_mnie -
        H_nmie, Y2L_ijab = 2 Y2_ijab - Y2_ijba. The l2-part voov ring (diagrams 6, 8)
        is the _r_T2_ccsd three-term ring with X1-dressed ph intermediates W_mbej /
        W_mbje (W_mbje[e<->j] swaps the last two axes; W_mbie is W_mbje with j -> i)
        and Y2 as the external doubles::

            # l1-part
            Z_nf  = L_mnef X1_me
            Z_ea  = -L_mnef Y2_mnaf
            Z_mi  = -L_mnef Y2_inef
            tmp_ia = Z_nf Y2L_nifa + Z_ea X1_ie + Z_mi X1_ma           (diagrams 3, 4, 5)
            # l2-part
            Z_mi   = H_me X1_ie + HoL_mnie X1_ne
            Z_ea   = H_me X1_ma - HvL_amef X1_mf
            Z_mnij = H_mnie X1_je
            Z_ijam = H_amef Y2_ijef
            tmp_ijab = -1/2 Z_mi Y2_mjab - 1/2 Z_ea Y2_ijeb           (diagrams 1/7, 2/9)
                     + 1/2 Z_mnij Y2_mnab - 1/2 Z_ijam X1_mb          (diagrams 10, 11)
            W_mbej =  X1_jf H_bmfe - X1_nb H_nmje
            W_mbje = -X1_jf H_bmef + X1_nb H_mnje
            ring = (Y2_imae - Y2_imea) W_mbej + Y2_imae (W_mbej + W_mbje[e<->j])
                 + Y2_mjae W_mbie                                      (diagrams 6, 8)
            tmp_ijab += 1/2 ring_ijab
            LHX1Y2 = l1_ia tmp_ia + 2 l2_ijab tmp_ijab
        """
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self._LHX1Y2_spinorbital(X, Y)
        # spin-adapted (spatial) LHX1Y2
        contract = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        hbar = self.hbar
        L = self.H.L[o, o, v, v]
        X1, Y2 = X[0], Y[1]
        HvL = 2.0 * hbar.Hvovv - hbar.Hvovv.swapaxes(2, 3)
        HoL = 2.0 * hbar.Hooov - hbar.Hooov.swapaxes(0, 1)
        Y2L = 2.0 * Y2 - Y2.swapaxes(2, 3)
        # l1-part (diagrams 3,4,5)
        Zov = contract('mnef,me->nf', L, X1)                       # L3
        tmp1 = contract('nf,nifa->ia', Zov, Y2L)
        Zvv = -contract('mnef,mnaf->ea', L, Y2)                    # L4
        tmp1 += contract('ea,ie->ia', Zvv, X1)
        Zoo = -contract('mnef,inef->mi', L, Y2)                    # L5
        tmp1 += contract('mi,ma->ia', Zoo, X1)
        polar = contract('ia,ia->', l1, tmp1)
        # l2-part (diagrams 1,2,6,7,8,9,10,11)
        Zoo = contract('me,ie->mi', hbar.Hov, X1) + contract('mnie,ne->mi', HoL, X1)
        tmp2 = -0.5 * contract('mi,mjab->ijab', Zoo, Y2)          # L1_7
        Zvv = contract('me,ma->ea', hbar.Hov, X1) - contract('amef,mf->ea', HvL, X1)
        tmp2 += -0.5 * contract('ea,ijeb->ijab', Zvv, Y2)        # L2_9
        Zoooo = contract('mnie,je->mnij', hbar.Hooov, X1)         # L10
        tmp2 += 0.5 * contract('mnij,mnab->ijab', Zoooo, Y2)
        Zoovo = contract('amef,ijef->ijam', hbar.Hvovv, Y2)       # L11
        tmp2 += -0.5 * contract('ijam,mb->ijab', Zoovo, X1)
        # L6_8 (voov ring, diagrams 6,8): one ph ring, structurally the LHX2Y2-D1
        # three-term ring with the X1-dressed ph intermediate (both Hvovv and Hooov
        # dressings, mirroring build_Hovvo/build_Hovov) and Y2 as external doubles.
        Wmbej = contract('jf,bmfe->mbej', X1, hbar.Hvovv) - contract('nb,nmje->mbej', X1, hbar.Hooov)
        Wmbje = -contract('jf,bmef->mbje', X1, hbar.Hvovv) + contract('nb,mnje->mbje', X1, hbar.Hooov)
        ring = contract('imae,mbej->ijab', Y2 - Y2.swapaxes(2, 3), Wmbej)
        ring += contract('imae,mbej->ijab', Y2, Wmbej + Wmbje.swapaxes(2, 3))
        ring += contract('mjae,mbie->ijab', Y2, Wmbje)
        tmp2 += 0.5 * ring
        polar += 2.0 * contract('ijab,ijab->', l2, tmp2)
        return polar

    def _LHX1Y2_spinorbital(self, X, Y):
        contract = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        hbar = self.hbar
        ERI = self.H.ERI
        X1, Y2 = X[0], Y[1]
        Zov = contract('mnef,me->nf', ERI[o,o,v,v], X1)
        Zvv = -0.5 * contract('mnef,mnaf->ea', ERI[o,o,v,v], Y2)
        Zoo = -0.5 * contract('mnef,inef->mi', ERI[o,o,v,v], Y2)
        tmp = contract('nf,nifa->ia', Zov, Y2)                    # diagram 3
        tmp += contract('ea,ie->ia', Zvv, X1)                     # diagram 4
        tmp += contract('mi,ma->ia', Zoo, X1)                     # diagram 5
        polar = contract('ia,ia->', l1, tmp)
        Zoo = contract('me,ie->mi', hbar.Hov, X1)
        Zoo += contract('mnie,ne->mi', hbar.Hooov, X1)
        Zvv = contract('me,ma->ea', hbar.Hov, X1)
        Zvv -= contract('amef,mf->ea', hbar.Hvovv, X1)
        Zvoov = contract('anfe,if->anie', hbar.Hvovv, X1)
        Zvoov -= contract('mnie,ma->anie', hbar.Hooov, X1)
        Zoooo = 0.5 * contract('mnie,je->mnij', hbar.Hooov, X1)
        Zoovo = 0.5 * contract('amef,ijef->ijam', hbar.Hvovv, Y2)
        tmp = -0.5 * contract('mi,mjab->ijab', Zoo, Y2)           # diagrams 1 and 7
        tmp -= 0.5 * contract('ea,ijeb->ijab', Zvv, Y2)           # diagrams 2 and 9
        tmp += contract('anie,njeb->ijab', Zvoov, Y2)             # diagrams 6 and 8
        tmp += 0.5 * contract('mnij,mnab->ijab', Zoooo, Y2)       # diagram 10
        tmp -= 0.5 * contract('ijam,mb->ijab', Zoovo, X1)         # diagram 11
        polar += contract('ijab,ijab->', l2, tmp)
        return polar

    # ==================================================================
    # DEPRECATED for linear response: the *asymmetric* formulation, which
    # solves both the right- (X) and left-hand (Y) perturbed amplitudes.
    # Superseded by the symmetric path above (polarizability / optrot /
    # linresp_sym, X only). Retained, not deleted: the quadratic response
    # function will need the left-hand (Y) contributions.
    # ==================================================================

    def linresp_asym(self, pertkey_a: str, X1_B: Tensor, X2_B: Tensor, Y1_B: Tensor, Y2_B: Tensor):
        """
	Calculate the CC linear response function for polarizability at field-frequency omega(w1).

	The linear response function, <<A;B(w1)>> generally reuires the following perturbed wave functions and frequencies:
	
	Parameters
	----------
	pertkey_a: string
		String identifying the one-electron perturbation, A along a cartesian axis
	

	Return
	------
	polar: float
	     A value of the chosen linear response function corresponding to compute polariazabiltity in a specified cartesian diresction.
        """

        contract = self.ccwfn.contract

        # Defining the l1 and l2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2

        # Please refer to eqn 78 of [Crawford: https://crawford.chem.vt.edu/wp-content/uploads/2022/06/cc_response.pdf].
        # Writing H(1)(omega) = B, T(1)(omega) = X, L(1)(omega) = y
        # <<A;B>> = <0|Y(B) * A_bar|0> + <0| (1 + L(0))[A_bar, X(B)}|0>
        #                 polar1                polar2
        polar1 = 0
        polar2 = 0
        pertbar_A = self.pertbar[pertkey_a]
        Avvoo = pertbar_A.Avvoo.swapaxes(0, 2).swapaxes(1, 3)
        # <0|Y1(B) * A_bar|0>
        polar1 += contract("ai, ia -> ", pertbar_A.Avo, Y1_B)
        # <0|Y2(B) * A_bar|0>
        polar1 += 0.25 * contract("abij, ijab -> ", Avvoo, Y2_B)
        polar1 += 0.25 * contract("baji, ijab -> ", Avvoo, Y2_B)
        # <0|[A_bar, X(B)]|0>
        polar2 += 2.0 * contract("ia, ia -> ", pertbar_A.Aov, X1_B)
        # <0|L1(0) [A_bar, X2(B)]|0>
        tmp = contract("ia, ic -> ac", l1, X1_B)
        polar2 += contract("ac, ac -> ", tmp, pertbar_A.Avv)
        tmp = contract("ia, ka -> ik", l1, X1_B)
        polar2 -= contract("ik, ki -> ", tmp, pertbar_A.Aoo)
        # <0|L1(0)[a_bar, X2(B)]|0>
        tmp = contract("ia, jb -> ijab", l1, pertbar_A.Aov)
        polar2 += 2.0 * contract("ijab, ijab -> ", tmp, X2_B)
        polar2 += -1.0 * contract("ijab, ijba -> ", tmp, X2_B)
        # <0|L2(0)[A_bar, X1(B)]|0>
        tmp = contract("ijbc, bcaj -> ia", l2, pertbar_A.Avvvo)
        polar2 += contract("ia, ia -> ", tmp, X1_B)
        tmp = contract("ijab, kbij -> ak", l2, pertbar_A.Aovoo)
        polar2 -= 0.5 * contract("ak, ka -> ", tmp, X1_B)
        tmp = contract("ijab, kaji -> bk", l2, pertbar_A.Aovoo)
        polar2 -= 0.5 * contract("bk, kb -> ", tmp, X1_B)
        # <0|L2(0)[A_bar, X1(B)]|0>
        tmp = contract("ijab, kjab -> ik", l2, X2_B)
        polar2 -= 0.5 * contract("ik, ki -> ", tmp, pertbar_A.Aoo)
        tmp = contract("ijab, kiba-> jk", l2, X2_B)
        polar2 -= 0.5 * contract("jk, kj -> ", tmp, pertbar_A.Aoo)
        tmp = contract("ijab, ijac -> bc", l2, X2_B)
        polar2 += 0.5 * contract("bc, bc -> ", tmp, pertbar_A.Avv)
        tmp = contract("ijab, ijcb -> ac", l2, X2_B)
        polar2 += 0.5 * contract("ac, ac -> ", tmp, pertbar_A.Avv)

        return -1.0 * (polar1 + polar2)

    def solve_left(self, pertbar: "pertbar", omega: float, e_conv: float = 1e-12, r_conv: float = 1e-12, maxiter: int = 200, max_diis: int = 7, start_diis: int = 1):
        """
	Notes
	-----
	The first-order lambda equations are partition into two expressions: inhomogeneous (in_Y1 and in_Y2) and homogeneous terms (r_Y1 and r_Y2),
	the inhomogeneous terms contains only terms that are not changing over the iterative process of obtaining the solutions for these equations. 
	Therefore, it is computed only once and is called when solving for the homogeneous terms.
        """
 
        solver_start = time.time()

        Dia = self.Dia
        Dijab = self.Dijab

        # initial guess
        X1_guess = pertbar.Avo.T/(Dia + omega)
        X2_guess = pertbar.Avvoo/(Dijab + omega)

        # initial guess
        Y1 = 2.0 * X1_guess.copy()
        Y2 = 4.0 * X2_guess.copy()
        Y2 -= 2.0 * X2_guess.copy().swapaxes(2,3)              

        # need to understand this
        pseudo = self.pseudoresponse(pertbar, Y1, Y2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")
        
        diis = helper_diis(Y1, Y2, max_diis)
        contract = self.ccwfn.contract

        self.Y1 = Y1
        self.Y2 = Y2 
        
        # uses updated X1 and X2
        self.im_Y1 = self.in_Y1(pertbar, self.X1, self.X2)
        self.im_Y2 = self.in_Y2(pertbar, self.X1, self.X2)

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo
            
            Y1 = self.Y1
            Y2 = self.Y2
            
            r1 = self.r_Y1(pertbar, omega)
            r2 = self.r_Y2(pertbar, omega)
            
            self.Y1 += r1/(Dia + omega)
            self.Y2 += r2/(Dijab + omega)
            
            rms = contract('ia,ia->', np.conj(r1/(Dia+omega)), r1/(Dia+omega))
            rms += contract('ijab,ijab->', np.conj(r2/(Dijab+omega)), r2/(Dijab+omega))
            rms = np.sqrt(rms)
            
            pseudo = self.pseudoresponse(pertbar, self.Y1, self.Y2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")
                
            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                return self.Y1, self.Y2 , pseudo
            
            diis.add_error_vector(self.Y1, self.Y2)
            if niter >= start_diis:
                self.Y1, self.Y2 = diis.extrapolate(self.Y1, self.Y2)

    def in_Y1(self, pertbar, X1, X2):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L
 
        # <O|A_bar|phi^a_i> good
        r_Y1 = 2.0 * pertbar.Aov.copy()
        # <O|L1(0)|A_bar|phi^a_i> good
        r_Y1 -= contract('im,ma->ia', pertbar.Aoo, l1)
        r_Y1 += contract('ie,ea->ia', l1, pertbar.Avv)
        # <O|L2(0)|A_bar|phi^a_i>
        r_Y1 += contract('imfe,feam->ia', l2, pertbar.Avvvo)
   
        # can combine the next two to swapaxes type contraction
        r_Y1 -= 0.5 * contract('ienm,mnea->ia', pertbar.Aovoo, l2)
        r_Y1 -= 0.5 * contract('iemn,mnae->ia', pertbar.Aovoo, l2)

        # <O|[Hbar(0), X1]|phi^a_i> good
        r_Y1 += 2.0 * contract('imae,me->ia', L[o,o,v,v], X1)

        # <O|L1(0)|[Hbar(0), X1]|phi^a_i>
        tmp = -1.0 * contract('ma,ie->miae', hbar.Hov, l1)
        tmp -= contract('ma,ie->miae', l1, hbar.Hov)
        tmp -= 2.0 * contract('mina,ne->miae', hbar.Hooov, l1)

        tmp += contract('imna,ne->miae', hbar.Hooov, l1)

       #can combine the next two to swapaxes type contraction
        tmp -= 2.0 * contract('imne,na->miae', hbar.Hooov, l1)
        tmp += contract('mine,na->miae', hbar.Hooov, l1)

        #can combine the next two to swapaxes type contraction
        tmp += 2.0 * contract('fmae,if->miae', hbar.Hvovv, l1)
        tmp -= contract('fmea,if->miae', hbar.Hvovv, l1)

        #can combine the next two to swapaxes type contraction
        tmp += 2.0 * contract('fiea,mf->miae', hbar.Hvovv, l1)
        tmp -= contract('fiae,mf->miae', hbar.Hvovv, l1)
        r_Y1 += contract('miae,me->ia', tmp, X1)

        # <O|L1(0)|[Hbar(0), X2]|phi^a_i> good

        #can combine the next two to swapaxes type contraction
        tmp = 2.0 * contract('mnef,nf->me', X2, l1)
        tmp -= contract('mnfe,nf->me', X2, l1)
        r_Y1 += contract('imae,me->ia', L[o,o,v,v], tmp)
        r_Y1 -= contract('ni,na->ia', cclambda.build_Goo(X2, L[o,o,v,v]), l1)
        r_Y1 += contract('ie,ea->ia', l1, cclambda.build_Gvv(L[o,o,v,v], X2))

        # <O|L2(0)|[Hbar(0), X1]|phi^a_i> good

        # can reorganize these next four to two swapaxes type contraction
        tmp = -1.0 * contract('nief,mfna->iema', l2, hbar.Hovov)
        tmp -= contract('ifne,nmaf->iema', hbar.Hovov, l2)
        tmp -= contract('inef,mfan->iema', l2, hbar.Hovvo)
        tmp -= contract('ifen,nmfa->iema', hbar.Hovvo, l2)

        #can combine the next two to swapaxes type contraction
        tmp += 0.5 * contract('imfg,fgae->iema', l2, hbar.Hvvvv)
        tmp += 0.5 * contract('imgf,fgea->iema', l2, hbar.Hvvvv)

        #can combine the next two to swapaxes type contraction
        tmp += 0.5 * contract('imno,onea->iema', hbar.Hoooo, l2)
        tmp += 0.5 * contract('mino,noea->iema', hbar.Hoooo, l2)
        r_Y1 += contract('iema,me->ia', tmp, X1)

       #contains regular Gvv as well as Goo, think about just calling it from cclambda instead of generating it
        tmp = contract('nb,fb->nf', X1, cclambda.build_Gvv(l2, t2))
        r_Y1 += contract('inaf,nf->ia', L[o,o,v,v], tmp)
        tmp = contract('me,fa->mefa', X1, cclambda.build_Gvv(l2, t2))
        r_Y1 += contract('mief,mefa->ia', L[o,o,v,v], tmp)
        tmp  =  contract('me,ni->meni', X1, cclambda.build_Goo(t2, l2))
        r_Y1 -= contract('meni,mnea->ia', tmp, L[o,o,v,v])
        tmp  =  contract('jf,nj->fn', X1, cclambda.build_Goo(t2, l2))
        r_Y1 -= contract('inaf,fn->ia', L[o,o,v,v], tmp)

        # <O|L2(0)|[Hbar(0), X2]|phi^a_i>
        r_Y1 -= contract('mi,ma->ia', cclambda.build_Goo(X2, l2), hbar.Hov)
        r_Y1 += contract('ie,ea->ia', hbar.Hov, cclambda.build_Gvv(l2, X2))
        tmp   = contract('imfg,mnef->igne', l2, X2)
        r_Y1 -= contract('igne,gnea->ia', tmp, hbar.Hvovv)
        tmp   = contract('mifg,mnef->igne', l2, X2)
        r_Y1 -= contract('igne,gnae->ia', tmp, hbar.Hvovv)
        tmp   = contract('mnga,mnef->gaef', l2, X2)
        r_Y1 -= contract('gief,gaef->ia', hbar.Hvovv, tmp)

        #can combine the next two to swapaxes type contraction
        tmp   = 2.0 * contract('gmae,mnef->ganf', hbar.Hvovv, X2)
        tmp  -= contract('gmea,mnef->ganf', hbar.Hvovv, X2)
        r_Y1 += contract('nifg,ganf->ia', l2, tmp)

        #can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('giea,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))
        r_Y1 += contract('giae,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))
        tmp   = contract('oief,mnef->oimn', l2, X2)
        r_Y1 += contract('oimn,mnoa->ia', tmp, hbar.Hooov)
        tmp   = contract('mofa,mnef->oane', l2, X2)
        r_Y1 += contract('inoe,oane->ia', hbar.Hooov, tmp)
        tmp   = contract('onea,mnef->oamf', l2, X2)
        r_Y1 += contract('miof,oamf->ia', hbar.Hooov, tmp)

        #can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('mioa,mo->ia', hbar.Hooov, cclambda.build_Goo(X2, l2))
        r_Y1 += contract('imoa,mo->ia', hbar.Hooov, cclambda.build_Goo(X2, l2))

        #can combine the next two to swapaxes type contraction
        tmp   = -2.0 * contract('imoe,mnef->ionf', hbar.Hooov, X2)
        tmp  += contract('mioe,mnef->ionf', hbar.Hooov, X2)
        r_Y1 += contract('ionf,nofa->ia', tmp, l2) 

        return r_Y1

    def r_Y1(self, pertbar, omega):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        Y1 = self.Y1 
        Y2 = self.Y2
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L

        #imhomogenous terms
        r_Y1 = self.im_Y1.copy()
        #homogenous terms appearing in Y1 equations
        r_Y1 += omega * Y1
        r_Y1 += contract('ie,ea->ia', Y1, hbar.Hvv)
        r_Y1 -= contract('im,ma->ia', hbar.Hoo, Y1)
        r_Y1 += 2.0 * contract('ieam,me->ia', hbar.Hovvo, Y1)
        r_Y1 -= contract('iema,me->ia', hbar.Hovov, Y1)
        r_Y1 += contract('imef,efam->ia', Y2, hbar.Hvvvo)
        r_Y1 -= contract('iemn,mnae->ia', hbar.Hovoo, Y2)

        #can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('eifa,ef->ia', hbar.Hvovv, cclambda.build_Gvv(t2, Y2))
        r_Y1 += contract('eiaf,ef->ia', hbar.Hvovv, cclambda.build_Gvv(t2, Y2))

        #can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('mina,mn->ia', hbar.Hooov, cclambda.build_Goo(t2, Y2))
        r_Y1 += contract('imna,mn->ia', hbar.Hooov, cclambda.build_Goo(t2, Y2))

        return r_Y1

    def in_Y2(self, pertbar, X1, X2):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        #X1 = self.X1
        #X2 = self.X2
        Y1 = self.Y1
        Y2 = self.Y2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L
        ERI = self.H.ERI

        # Inhomogenous terms appearing in Y2 equations
        # <O|L1(0)|A_bar|phi^ab_ij> good

        #next two turn to swapaxes contraction
        r_Y2  = 2.0 * contract('ia,jb->ijab', l1, pertbar.Aov.copy())
        r_Y2 -= contract('ja,ib->ijab', l1, pertbar.Aov)

        # <O|L2(0)|A_bar|phi^ab_ij> good
        r_Y2 += contract('ijeb,ea->ijab', l2, pertbar.Avv)
        r_Y2 -= contract('im,mjab->ijab', pertbar.Aoo, l2)

        # <O|L1(0)|[Hbar(0), X1]|phi^ab_ij> good
        tmp   = contract('me,ja->meja', X1, l1)
        r_Y2 -= contract('mieb,meja->ijab', L[o,o,v,v], tmp)
        tmp   = contract('me,mb->eb', X1, l1)
        r_Y2 -= contract('ijae,eb->ijab', L[o,o,v,v], tmp)
        tmp   = contract('me,ie->mi', X1, l1)
        r_Y2 -= contract('mi,jmba->ijab', tmp, L[o,o,v,v])
        tmp   = 2.0 *contract('me,jb->mejb', X1, l1)
        r_Y2 += contract('imae,mejb->ijab', L[o,o,v,v], tmp)

        # <O|L2(0)|[Hbar(0), X1]|phi^ab_ij> 
        tmp   = contract('me,ma->ea', X1, hbar.Hov)
        r_Y2 -= contract('ijeb,ea->ijab', l2, tmp)
        tmp   = contract('me,ie->mi', X1, hbar.Hov)
        r_Y2 -= contract('mi,jmba->ijab', tmp, l2)
        tmp   = contract('me,ijef->mijf', X1, l2)
        r_Y2 -= contract('mijf,fmba->ijab', tmp, hbar.Hvovv)
        tmp   = contract('me,imbf->eibf', X1, l2)
        r_Y2 -= contract('eibf,fjea->ijab', tmp, hbar.Hvovv)
        tmp   = contract('me,jmfa->ejfa', X1, l2)
        r_Y2 -= contract('fibe,ejfa->ijab', hbar.Hvovv, tmp)

        #swapaxes contraction
        tmp   = 2.0 * contract('me,fmae->fa', X1, hbar.Hvovv)
        tmp  -= contract('me,fmea->fa', X1, hbar.Hvovv)
        r_Y2 += contract('ijfb,fa->ijab', l2, tmp)

        #swapaxes contraction
        tmp   = 2.0 * contract('me,fiea->mfia', X1, hbar.Hvovv)
        tmp  -= contract('me,fiae->mfia', X1, hbar.Hvovv)
        r_Y2 += contract('mfia,jmbf->ijab', tmp, l2)
        tmp   = contract('me,jmna->ejna', X1, hbar.Hooov)
        r_Y2 += contract('ineb,ejna->ijab', l2, tmp)

        tmp   = contract('me,mjna->ejna', X1, hbar.Hooov)
        r_Y2 += contract('nieb,ejna->ijab', l2, tmp)
        tmp   = contract('me,nmba->enba', X1, l2)
        r_Y2 += contract('jine,enba->ijab', hbar.Hooov, tmp)

        #swapaxes
        tmp   = 2.0 * contract('me,mina->eina', X1, hbar.Hooov)
        tmp  -= contract('me,imna->eina', X1, hbar.Hooov)
        r_Y2 -= contract('eina,njeb->ijab', tmp, l2)

        #swapaxes
        tmp   = 2.0 * contract('me,imne->in', X1, hbar.Hooov)
        tmp  -= contract('me,mine->in', X1, hbar.Hooov)
        r_Y2 -= contract('in,jnba->ijab', tmp, l2)

        # <O|L2(0)|[Hbar(0), X2]|phi^ab_ij>
        tmp   = 0.5 * contract('ijef,mnef->ijmn', l2, X2)
        r_Y2 += contract('ijmn,mnab->ijab', tmp, ERI[o,o,v,v])
        tmp   = 0.5 * contract('ijfe,mnef->ijmn', ERI[o,o,v,v], X2)
        r_Y2 += contract('ijmn,mnba->ijab', tmp, l2)
        tmp   = contract('mifb,mnef->ibne', l2, X2)
        r_Y2 += contract('ibne,jnae->ijab', tmp, ERI[o,o,v,v])
        tmp   = contract('imfb,mnef->ibne', l2, X2)
        r_Y2 += contract('ibne,njae->ijab', tmp, ERI[o,o,v,v])
        tmp   = contract('mjfb,mnef->jbne', l2, X2)
        r_Y2 -= contract('jbne,inae->ijab', tmp, L[o,o,v,v])

        #temp intermediate?
        r_Y2 -= contract('in,jnba->ijab', cclambda.build_Goo(L[o,o,v,v], X2), l2)
        r_Y2 += contract('ijfb,af->ijab', l2, cclambda.build_Gvv(X2, L[o,o,v,v]))
        r_Y2 += contract('ijae,be->ijab', L[o,o,v,v], cclambda.build_Gvv(X2, l2))
        r_Y2 -= contract('imab,jm->ijab', L[o,o,v,v], cclambda.build_Goo(l2, X2))
        tmp   = contract('nifb,mnef->ibme', l2, X2)
        r_Y2 -= contract('ibme,mjea->ijab', tmp, L[o,o,v,v])
        tmp   = 2.0 * contract('njfb,mnef->jbme', l2, X2)
        r_Y2 += contract('imae,jbme->ijab', L[o,o,v,v], tmp)

        return r_Y2

    def r_Y2(self, pertbar, omega):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        Y1 = self.Y1
        Y2 = self.Y2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L
        ERI = self.H.ERI

        #inhomogenous terms
        r_Y2 = self.im_Y2.copy()

        # Homogenous terms now!
        # a factor of 0.5 because of the relation/comment just above
        # and due to the fact that Y2_ijab = Y2_jiba
        r_Y2 += 0.5 * omega * self.Y2.copy()
        r_Y2 += 2.0 * contract('ia,jb->ijab', Y1, hbar.Hov)
        r_Y2 -= contract('ja,ib->ijab', Y1, hbar.Hov)
        r_Y2 += contract('ijeb,ea->ijab', Y2, hbar.Hvv)
        r_Y2 -= contract('im,mjab->ijab', hbar.Hoo, Y2)
        r_Y2 += 0.5 * contract('ijmn,mnab->ijab', hbar.Hoooo, Y2)
        r_Y2 += 0.5 * contract('ijef,efab->ijab', Y2, hbar.Hvvvv)
        r_Y2 += 2.0 * contract('ie,ejab->ijab', Y1, hbar.Hvovv)
        r_Y2 -= contract('ie,ejba->ijab', Y1, hbar.Hvovv)
        r_Y2 -= 2.0 * contract('mb,jima->ijab', Y1, hbar.Hooov)
        r_Y2 += contract('mb,ijma->ijab', Y1, hbar.Hooov)
        r_Y2 += 2.0 * contract('ieam,mjeb->ijab', hbar.Hovvo, Y2)
        r_Y2 -= contract('iema,mjeb->ijab', hbar.Hovov, Y2)
        r_Y2 -= contract('mibe,jema->ijab', Y2, hbar.Hovov)
        r_Y2 -= contract('mieb,jeam->ijab', Y2, hbar.Hovvo)
        r_Y2 += contract('ijeb,ae->ijab', L[o,o,v,v], cclambda.build_Gvv(t2, Y2))
        r_Y2 -= contract('mi,mjab->ijab', cclambda.build_Goo(t2, Y2), L[o,o,v,v])

        r_Y2 = r_Y2 + r_Y2.swapaxes(0,1).swapaxes(2,3)

        return r_Y2


class pertbar(object):
    def __init__(self, pert: Tensor, ccwfn: "CCwfn") -> None:
        o = ccwfn.o
        v = ccwfn.v
        t1 = ccwfn.t1
        t2 = ccwfn.t2
        contract = ccwfn.contract

        if ccwfn.orbital_basis == 'spinorbital':
            # Spin-orbital similarity-transformed perturbation (no RHF spin factors).
            self.Aov = pert[o,v].copy()
            self.Aoo = pert[o,o].copy()
            self.Aoo += contract('ie,me->mi', t1, pert[o,v])
            self.Avv = pert[v,v].copy()
            self.Avv -= contract('ma,me->ae', t1, pert[o,v])
            self.Avo = pert[v,o].copy()
            self.Avo += contract('ie,ae->ai', t1, pert[v,v])
            self.Avo -= contract('ma,mi->ai', t1, pert[o,o])
            self.Avo += contract('miea,me->ai', t2, pert[o,v])
            self.Avo -= contract('ie,ma,me->ai', t1, t1, pert[o,v])
            self.Aovoo = contract('ijeb,me->mbij', t2, pert[o,v])
            self.Avvvo = -1.0 * contract('miab,me->abei', t2, pert[o,v])
            # Stored oovv-ordered to match X2/t2/l2; full (antisymmetric) form.
            self.Avvoo = (contract('ijae,be->ijab', t2, self.Avv)
                          - contract('ijbe,ae->ijab', t2, self.Avv))
            self.Avvoo -= (contract('imab,mj->ijab', t2, self.Aoo)
                           - contract('jmab,mi->ijab', t2, self.Aoo))
            if ccwfn.model == 'CC3':
                # CC3: ground-state T3 contribution to Avvoo (loop-over-ijk, no
                # stored T3). Avvoo[ij] += <kc> t3_ijkabc.
                F = ccwfn.H.F
                ERI = ccwfn.H.ERI
                Woooo = ccwfn._so_build_Woooo_CC3(o, v, ERI, t1)
                Wovoo = ccwfn._so_build_Wovoo_CC3(o, v, ERI, t1, Woooo)
                Wvvvo = ccwfn._so_build_Wvvvo_CC3(o, v, ERI, t1)
                no = ccwfn.no
                for i in range(no):
                    for j in range(no):
                        for k in range(no):
                            t3 = t3c_ijk_so(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                            self.Avvoo[i,j] += contract('c,abc->ab', pert[k,v], t3)
            return

        self.Aov = pert[o,v].copy()

        self.Aoo = pert[o,o].copy()
        self.Aoo += contract('ie,me->mi', t1, pert[o,v])

        self.Avv = pert[v,v].copy()
        self.Avv -= contract('ma,me->ae', t1, pert[o,v])

        self.Avo = pert[v,o].copy()
        self.Avo += contract('ie,ae->ai', t1, pert[v,v])
        self.Avo -= contract('ma,mi->ai', t1, pert[o,o])
        self.Avo += contract('miea,me->ai', (2.0*t2 - t2.swapaxes(2,3)), pert[o,v])
        self.Avo -= contract('ie,ma,me->ai', t1, t1, pert[o,v])

        self.Aovoo = contract('ijeb,me->mbij', t2, pert[o,v])

        self.Avvvo = -1.0*contract('miab,me->abei', t2, pert[o,v])

        # Avvoo is the permutationally symmetric (un-halved) form, unlike the
        # implementation in ugacc. The compensating 0.5 lives in r_X2 (the residual
        # constant term), and linresp_asym carries 0.25 on its Y2 contractions.
        self.Avvoo = contract('ijeb,ae->ijab', t2, self.Avv)
        self.Avvoo -= contract('mjab,mi->ijab', t2, self.Aoo)
        self.Avvoo = self.Avvoo + self.Avvoo.swapaxes(0,1).swapaxes(2,3)


class _PertbarCache(dict):
    """A dict of similarity-transformed perturbation operators (pertbar objects) that
    builds each entry on first access via ``ccresponse._build_pertbar``. This defers
    pertbar construction so a response function only builds the operators it uses,
    instead of building all of them (MU, M, M*, P, P*, Q) in the constructor."""
    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def __missing__(self, key):
        value = self._owner._build_pertbar(key)
        self[key] = value
        return value
