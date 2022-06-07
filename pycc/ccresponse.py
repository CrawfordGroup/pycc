"""
ccresponse.py: CC Response Functions
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
import time
from .utils import helper_diis

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

    def __init__(self, ccdensity, omega1 = 0, omega2 = 0):
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

        # Build dictionary of similarity-transformed property integrals
        self.pertbar = {}

        # Electric-dipole operator (length)
        for axis in range(3):
            key = "MU_" + self.cart[axis]
            self.pertbar[key] = pertbar(self.H.mu[axis], self.ccwfn)

        # Magnetic-dipole operator
        for axis in range(3):
            key = "M_" + self.cart[axis]
            self.pertbar[key] = pertbar(self.H.m[axis], self.ccwfn)

        # Complex-conjugate of magnetic-dipole operator
        for axis in range(3):
            key = "M*_" + self.cart[axis]
            self.pertbar[key] = pertbar(np.conj(self.H.m[axis]), self.ccwfn)

        # Electric-dipole operator (velocity)
        for axis in range(3):
            key = "P_" + self.cart[axis]
            self.pertbar[key] = pertbar(self.H.p[axis], self.ccwfn)

        # Complex-conjugate of electric-dipole operator (velocity)
        for axis in range(3):
            key = "P*_" + self.cart[axis]
            self.pertbar[key] = pertbar(np.conj(self.H.p[axis]), self.ccwfn)

        # Traceless quadrupole
        ij = 0
        for axis1 in range(3):
            for axis2 in range(axis1,3):
                key = "Q_" + self.cart[axis1] + self.cart[axis2]
                self.pertbar[key] = pertbar(self.H.Q[ij], self.ccwfn)
                if (axis1 != axis2):
                    key2 = "Q_" + self.cart[axis2] + self.cart[axis1]
                    self.pertbar[key2] = self.pertbar[key]
                ij += 1

        # HBAR-based denominators
        eps_occ = np.diag(self.hbar.Hoo)
        eps_vir = np.diag(self.hbar.Hvv)
        self.Dia = eps_occ.reshape(-1,1) - eps_vir
        self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

    def pertcheck(self, omega, e_conv=1e-13, r_conv=1e-13, maxiter=200, max_diis=8, start_diis=1):
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
        X1 = {}
        X2 = {}
        check = {}

        # Electric-dipole (length)
        for axis in range(3):
            pertkey = "MU_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Magnetic-dipole
        for axis in range(3):
            pertkey = "M_" + self.cart[axis]            
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Complex-conjugate of magnetic-dipole
        for axis in range(3):
            pertkey = "M*_" + self.cart[axis] 
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Electric-dipole (velocity)
        for axis in range(3):
            pertkey = "P_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Complex-conjugate of electric-dipole (velocity)
        for axis in range(3):
            pertkey = "P*_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Traceless quadrupole
        for axis1 in range(3):
            for axis2 in range(3):
                pertkey = "Q_" + self.cart[axis1] + self.cart[axis2]
                X_key = pertkey + "_" + f"{omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar
                if (omega != 0.0):
                    X_key = pertkey + "_" + f"{-omega:0.6f}"
                    print("Solving right-hand perturbed wave function for %s:" % (X_key))
                    X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                    check[X_key] = polar

        return check

    def linresp(self, A, B, omega, e_conv=1e-13, r_conv=1e-13, maxiter=200, max_diis=8, start_diis=1):
        """
        Calculate the CC linear-response function for one-electron perturbations A and B at field-frequency omega (w).

        The linear response function, <<A;B>>w, generally requires the following perturbed wave functions and frequencies: 
            A(-w), A*(w), B(w), B*(-w)
        If the external field is static (w=0), then we need:
            A(0), A*(0), B(0), B*(0)
        If the perturbation A is real and B is pure imaginary:
            A(-w), A(w), B(w), B*(-w)
        or vice versa:
            A(-w), A*(w), B(w), B(-w)
        If the perturbations are both real and the field is static:
            A(0), B(0)
        If the perturbations are identical then:
            A(w), A*(-w) or A(0), A*(0)
        If the perturbations are identical, the field is dynamic and the operator is real: 
            A(-w), A(w)
        If the perturbations are identical, the field is static and the operator is real: 
            A(0)

        Parameters:
        -----------
        A: string
            String identifying the left-hand perturbation operator.  
        B: string
            String identifying the right-hand perturbation operator.  
        NB: Allowed values for A and B are: 
            "MU": Electric dipole operator (length)
            "P": Electric dipole operator (velocity)
            "P*": Complex conjugate of electric dipole operator (velocity)
            "M": Magnetic dipole operator
            "M*": Complex conjugate of Magnetic dipole operator
            "Q": Traceless quadrupole operator
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

        Returns:
        --------
        linresp: NumPy array
            A 3x3 or 9 x 3 x 3 array of values of the chosen linear response function.
        """

        A = A.upper()
        B = B.upper()

        # dictionaries for perturbed wave functions
        X1 = {} 
        X2 = {}
        for axis in range(3):
            # A(-w) or A(0)
            pertkey = A + "_" + self.cart[axis]
            X_key = pertkey + "_" + f"{-omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)

            # A(w) or A*(w) 
            if (omega != 0.0):
                if (np.iscomplexobj(self.pertbar[pertkey].Aoo)):
                    pertkey = A + "*_" + self.cart[axis]
                X_key = pertkey + "_" + f"{omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)


        if (B != A):
            for axis in range(3):
                pertkey = B + "_" + self.cart[axis]
                X_key = pertkey + "_" + f"{omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check.append(polar)
                if (omega != 0.0):
                    X_key = pertkey + "_" + f"{-omega:0.6f}"
                    print("Solving right-hand perturbed wave function for %s:" % (X_key))
                    X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                    check.append(polar)

    def solve_right(self, pertbar, omega, e_conv=1e-13, r_conv=1e-13, maxiter=200, max_diis=8, start_diis=1):
        solver_start = time.time()

        Dia = self.Dia
        Dijab = self.Dijab

        # initial guess
        X1 = pertbar.Avo.T/(Dia + omega)
        X2 = (pertbar.Avvoo+pertbar.Avvoo.swapaxes(0,1).swapaxes(2,3))/(Dijab + omega)

        pseudo = self.pseudoresponse(pertbar, X1, X2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        diis = helper_diis(X1, X2, max_diis)
        contract = self.ccwfn.contract

        self.X1 = X1
        self.X2 = X2

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            X1 = self.X1
            X2 = self.X2

            r1 = self.r_X1(pertbar, omega)
            r2 = self.r_X2(pertbar, omega)

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
                return self.X1, self.X2, pseudo

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
        L = self.H.L

        Zvv = contract('amef,mf->ae', (2.0*hbar.Hvovv - hbar.Hvovv.swapaxes(2,3)), X1)
        Zvv -= contract('mnef,mnaf->ae', L[o,o,v,v], X2)

        Zoo = -1.0*contract('mnie,ne->mi', (2.0*hbar.Hooov - hbar.Hooov.swapaxes(0,1)), X1)
        Zoo -= contract('mnef,inef->mi', L[o,o,v,v], X2)

        r_X2 = pertbar.Avvoo - 0.5 * omega*X2
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

    def pseudoresponse(self, pertbar, X1, X2):
        contract = self.ccwfn.contract
        polar1 = 2.0 * contract('ai,ia->', np.conj(pertbar.Avo), X1)
        polar2 = 2.0 * contract('ijab,ijab->', np.conj(pertbar.Avvoo), (2.0*X2 - X2.swapaxes(2,3)))

        return -2.0*(polar1 + polar2)
        

class pertbar(object):
    def __init__(self, pert, ccwfn):
        o = ccwfn.o
        v = ccwfn.v
        t1 = ccwfn.t1
        t2 = ccwfn.t2
        contract = ccwfn.contract

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
  
        # This is intended to match ugacc, but I need to re-derive all of these expressions for consistency
        # Also note the incorrectness of my tensor notation for Avvoo
        self.Avvoo = contract('ijeb,ae->ijab', t2, self.Avv)
        self.Avvoo -= contract('mjab,mi->ijab', t2, self.Aoo)
        # self.Avvoo = 0.5*(self.Avvoo + self.Avvoo.swapaxes(0,1).swapaxes(2,3))
