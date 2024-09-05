"""
ccresponse.py: CC Response Functions
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
import time
from .utils import helper_diis
from .cclambda import cclambda
from .hamiltonian import Hamiltonian

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
                X_2[pertkey] = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check.append(polar)
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check.append(polar)
                if (omega != 0.0):
                    X_key = pertkey + "_" + f"{-omega:0.6f}"
                    print("Solving right-hand perturbed wave function for %s:" % (X_key))
                    X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                    check.append(polar)


    def linresp_sym(self, pertkey_a, pertkey_b, X1_A, X2_A, X1_B, X2_B):
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
        o = self.ccwfn.o
        v = self.ccwfn.v

        # Defining the l1 and l2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        hbar = self.hbar
        L = self.ccwfn.H.L
        F = self.ccwfn.H.F
        t2 = self.ccwfn.t2
        ERI = self.ccwfn.H.ERI

        # Please refer to eqn 45 of [Crawford: 10.1002/wcms.1406].
        # Writing H(1)(omega) = B, T^(1)(omega) = X, Lambda = L^(0)
        # <<A;B>> = <0|(1 + L^(0)) { [\bar{A}^(0), X^(1)(B)] + [\bar{B}^(1), X^(1)(B)] + [[\bar{B}^(0), X^(1)(B)], X^(1)(B)] } |0>

        # <0|(1 + L^(0)) [\bar{A}^(0), X^(1)(B)] |0>
        # <0| [\bar{A}^(0), X^(1)(B)] |0> + <0| L^(0) [\bar{A}^(0), X^(1)(B)] |0>
        #           First term                      Second term

        polar = 0.0
        pertbar_A = self.pertbar[pertkey_a]
        pertbar_B = self.pertbar[pertkey_b]
        Aoovv = pertbar_B.Avvoo.swapaxes(0, 2).swapaxes(1, 3)

        # First term (The contribution is only from the X1 amplitude and no contribution for the X2 amplitudes)
        polar += 2.0 * contract("ia, ia -> ", pertbar_A.Aov, X1_B)
        # Second term (contributions from the L1 and X1)
        temp = contract("ia, ic -> ac", l1, X1_B)
        polar += contract("ac, ac -> ", temp, pertbar_A.Avv)
        temp = contract("ia, ka -> ik", l1, X1_B)
        polar -= contract("ik, ki -> ", temp, pertbar_A.Aoo)
        # Second term (contributions from the L1 and X2)
        temp = contract("ia, jb -> ijab", l1, pertbar_A.Aov)
        polar += 2.0 * contract("ijab, ijab -> ", temp, X2_B)
        polar += -1.0 * contract("ijab, ijba -> ", temp, X2_B)
        # Second term (contributions from the L2 and X1)
        temp = contract("ijbc, bcaj -> ia", l2, pertbar_A.Avvvo)
        polar += contract("ia, ia -> ", temp, X1_B)
        temp = contract("ijab, kbij -> ak", l2, pertbar_A.Aovoo)
        polar -= 0.5 * contract("ak, ka -> ", temp, X1_B)
        temp = contract("ijab, kaji -> bk", l2, pertbar_A.Aovoo)
        polar -= 0.5 * contract("bk, kb -> ", temp, X1_B)
        # Second term (contributions from the L2 and X2)
        temp = contract("ijab, kjab -> ik", l2, X2_B)
        polar -= 0.5 * contract("ik, ki -> ", temp, pertbar_A.Aoo)
        temp = contract("ijab, kiba-> jk", l2, X2_B)
        polar -= 0.5 * contract("jk, kj -> ", temp, pertbar_A.Aoo)
        temp = contract("ijab, ijac -> bc", l2, X2_B)
        polar += 0.5 * contract("bc, bc -> ", temp, pertbar_A.Avv)
        temp = contract("ijab, ijcb -> ac", l2, X2_B)
        polar += 0.5 * contract("ac, ac -> ", temp, pertbar_A.Avv)

        # <0|(1 + L^(0)) [\bar{B}^(1), X^(1)(B)] |0>
        # <0| [\bar{B}^(1), X^(1)(B)] |0> + <0| L^(0) [\bar{B}^(0), X^(1)(B)] |0>
        #           Third term                      Fourth term

        # Third term (The contribution is only from the X1 amplitude and no contribution for the X2 amplitudes)
        polar += 2.0 * contract("ia, ia -> ", pertbar_B.Aov, X1_A)
        # Fourth term (contributions from the L1 and X1)
        temp = contract("ia, ic -> ac", l1, X1_A)
        polar += contract("ac, ac -> ", temp, pertbar_B.Avv)
        temp = contract("ia, ka -> ik", l1, X1_A)
        polar -= contract("ik, ki -> ", temp, pertbar_B.Aoo)
        # Fourth term (contributions from the L1 and X2)
        temp = contract("ia, jb -> ijab", l1, pertbar_B.Aov)
        polar += 2.0 * contract("ijab, ijab -> ", temp, X2_A)
        polar += -1.0 * contract("ijab, ijba -> ", temp, X2_A)
        # Fourth term (contributions from the L2 and X1)
        temp = contract("ijbc, bcaj -> ia", l2, pertbar_B.Avvvo)
        polar += contract("ia, ia -> ", temp, X1_A)
        temp = contract("ijab, kbij -> ak", l2, pertbar_B.Aovoo)
        polar -= 0.5 * contract("ak, ka -> ", temp, X1_A)
        temp = contract("ijab, kaji -> bk", l2, pertbar_B.Aovoo)
        polar -= 0.5 * contract("bk, kb -> ", temp, X1_A)
        # Fourth term (contributions from the L2 and X2)
        temp = contract("ijab, kjab -> ik", l2, X2_A)
        polar -= 0.5 * contract("ik, ki -> ", temp, pertbar_B.Aoo)
        temp = contract("ijab, kiba-> jk", l2, X2_A)
        polar -= 0.5 * contract("jk, kj -> ", temp, pertbar_B.Aoo)
        temp = contract("ijab, ijac -> bc", l2, X2_A)
        polar += 0.5 * contract("bc, bc -> ", temp, pertbar_B.Avv)
        temp = contract("ijab, ijcb -> ac", l2, X2_A)
        polar += 0.5 * contract("ac, ac -> ", temp, pertbar_B.Avv)

        # <0|(1 + L^(0))[[\bar{B}^(0), X^(1)(B)], X^(1)(B)]|0>
        # <0|[[\bar{B}^(0), X^(1)(B)], X^(1)(B)]|0> + <0| L^(0) [[\bar{B}^(0), X^(1)(B)], X^(1)(B)]|0>
        #               Fifth term                          Sixth term

        # Expanding the permutational operator and implementing it explicitly
        temp = contract("ijab, ia -> jb", L[o, o, v, v], X1_A)
        polar += 2.0 * contract("jb, jb -> ", temp, X1_B)
        # # #
        temp = contract("je, ja -> ea", X1_A, hbar.Hov)
        temp = contract("ea, ma -> me", temp, X1_B)
        polar -= contract("me, me -> ", temp, l1)
        temp = contract("je, ja -> ea", X1_B, hbar.Hov)
        temp = contract("ea, ma -> me", temp, X1_A)
        polar -= contract("me, me -> ", temp, l1)
        #
        temp = contract("jima, je -> ieam", (2 * hbar.Hooov), X1_A)
        temp = contract("ieam, ia -> me", temp, X1_B)
        polar -= contract("me, me -> ", temp, l1)
        temp = contract("jima, je -> ieam", (2 * hbar.Hooov), X1_B)
        temp = contract("ieam, ia -> me", temp, X1_A)
        polar -= contract("me, me -> ", temp, l1)
        temp = contract("jiem, ia -> ajem", hbar.Hooov.swapaxes(2, 3), X1_A)
        temp = contract("ajem, je -> ma", temp, X1_B)
        polar += contract("ma, ma -> ", temp, l1)
        temp = contract("jiem, ia -> ajem", hbar.Hooov.swapaxes(2, 3), X1_B)
        temp = contract("ajem, je -> ma", temp, X1_A)
        polar += contract("ma, ma -> ", temp, l1)
        # #
        temp = contract("ejab, jb -> ea", (2.0 * hbar.Hvovv), X1_A)
        temp = contract("ea, ma -> me", temp, X1_B)
        polar += contract("me, me -> ", temp, l1)
        temp = contract("ejab, jb -> ea", hbar.Hvovv.swapaxes(2, 3), X1_A)
        temp = contract("ea, ma -> me", temp, X1_B)
        polar -= contract("me, me -> ", temp, l1)
        temp = contract("ejba, ja -> be", (2.0 * hbar.Hvovv), X1_B)
        temp = contract("be, mb -> me", temp, X1_A)
        polar += contract("me, me -> ", temp, l1)
        temp = contract("ejba, ja -> be", hbar.Hvovv.swapaxes(2, 3), X1_B)
        temp = contract("be, mb -> me", temp, X1_A)
        polar -= contract("me, me -> ", temp, l1)

        temp = contract("jfma, je -> amef", hbar.Hovov, X1_A)
        temp = contract("amef, na -> mnef", temp, X1_B)
        polar -= contract("mnef, mnef -> ", temp, l2)

        temp = contract("jfma, je -> amef", hbar.Hovov, X1_B)
        temp = contract("amef, na -> mnef", temp, X1_A)
        polar -= contract("mnef, mnef -> ", temp, l2)

        temp = contract("jeam, na -> mnej", hbar.Hovvo, X1_A)
        temp = contract("mnej, jc -> mnec", temp, X1_B)
        polar -= contract("mnec, mnec -> ", temp, l2)

        temp = contract("jeam, na -> mnej", hbar.Hovvo, X1_B)
        temp = contract("mnej, jc -> mnec", temp, X1_A)
        polar -= contract("mnec, mnec -> ", temp, l2)

        temp = contract("abef, jf -> abej", hbar.Hvvvv, X1_A)
        temp = contract("abej, ie -> ijab", temp, X1_B)
        polar += 0.5 * contract("ijab, ijab  -> ", temp, l2)

        temp = contract("abef, jf -> abej", hbar.Hvvvv, X1_B)
        temp = contract("abej, ie -> ijab", temp, X1_A)
        polar += 0.5 * contract("ijab, ijab  -> ", temp, l2)

        temp = contract("mnij, ma -> anij", hbar.Hoooo, X1_A)
        temp = contract("anij, nb -> ijab", temp, X1_B)
        polar += 0.5 * contract("ijab, ijab -> ", temp, l2)

        temp = contract("mnij, ma -> anij", hbar.Hoooo, X1_B)
        temp = contract("anij, nb -> ijab", temp, X1_A)
        polar += 0.5 * contract("ijab, ijab -> ", temp, l2)

        Goo = contract('mjab,ijab->mi', t2, l2)
        Gvv = -1.0 * contract('ijeb,ijab->ae', t2, l2)
        r2_Gvv = contract('ae,ijeb->ijab', Gvv, L[o, o, v, v])
        r2_Goo = -1.0 * contract('mi,mjab->ijab', Goo, L[o, o, v, v])
        r2_Gvv = r2_Gvv + r2_Gvv.swapaxes(0, 1).swapaxes(2, 3)
        r2_Goo = r2_Goo + r2_Goo.swapaxes(0, 1).swapaxes(2, 3)
        polar += contract('ijab,ia,jb', r2_Gvv, X1_A, X1_B)     #Gvv
        # print("Gvv\n", polar_Gvv)
        polar += contract('ijab,ia,jb', r2_Goo, X1_A, X1_B)     #Goo
        # print("Goo\n", polar_Goo)

        # # Begin HX_2Y_2
        temp = contract("ikac, ijab -> kjbc", L[o, o, v, v], X2_A)
        temp = contract("kjbc, klcd -> jlbd", temp, X2_B)
        polar += 2 * contract("jlbd, jlbd -> ", temp, l2)

        temp = contract("ijac, ijab -> bc", L[o, o, v, v], X2_A)
        temp = contract("bc, klcd -> klbd", temp, X2_B)
        polar -= contract("klbd, klbd -> ", temp, l2)
        temp = contract("ijac, ikac -> jk", L[o, o, v, v], X2_A)
        temp = contract("jk, jlbd -> klbd", temp, X2_B)
        polar -= contract("klbd, klbd -> ", temp, l2)
        temp = contract("ijac, jkbc -> ikab", L[o, o, v, v], X2_A)
        temp = contract("ikab, ilad -> klbd", temp, X2_B)
        polar -= contract("klbd, klbd -> ", temp, l2)

        temp = contract("ikad, klcd -> ilac", L[o, o, v, v], X2_B)
        temp = contract("ilac, ijab -> jlbc", temp, X2_A)
        polar -= contract("jlbc, jlbc -> ", temp, l2)
        temp = contract("ikad, ikac -> cd", L[o, o, v, v], X2_B)
        temp = contract("cd, jlbd -> jlbc", temp, X2_A)
        polar -= contract("jlbc, jlbc -> ", temp, l2)
        temp = contract("ikad, ilad -> kl", L[o, o, v, v], X2_B)
        temp = contract("kl, jkbc -> jlbc", temp, X2_A)
        polar -= contract("jlbc, jlbc -> ", temp, l2)


        temp = contract("klab, ijab -> klij", ERI[o, o, v, v].copy(), X2_A)
        temp = contract("klij, klcd -> ijcd", temp, X2_B)
        polar += 0.5 * contract("ijcd, ijcd -> ", temp, l2)

        temp = contract("klab, ijab -> klij", ERI[o, o, v, v].copy(), X2_B)
        temp = contract("klij, klcd -> ijcd", temp, X2_A)
        polar += 0.5 * contract("ijcd, ijcd -> ", temp, l2)

        temp = contract("klab, ikac -> ilbc", ERI[o, o, v, v].copy(), X2_A)
        temp = contract("ilbc, jlbd -> ijcd", temp, X2_B)
        polar += 0.5 * contract("ijcd, ijcd -> ", temp, l2)

        temp = contract("klab, ikac -> ilbc", ERI[o, o, v, v].copy(), X2_B)
        temp = contract("ilbc, jlbd -> ijcd", temp, X2_A)
        polar += 0.5 * contract("ijcd, ijcd -> ", temp, l2)

        temp = contract("klab, ilad -> ikbd", ERI[o, o, v, v].copy(), X2_A)
        temp = contract("ikbd, jkbc -> ijcd", temp, X2_B)
        polar += 0.5 * contract("ijcd, ijcd -> ", temp, l2)

        temp = contract("klab, ilad -> ikbd", ERI[o, o, v, v].copy(), X2_B)
        temp = contract("ikbd, jkbc -> ijcd", temp, X2_A)
        polar += 0.5 * contract("ijcd, ijcd -> ", temp, l2)
        # # End HX_{2}Y_{2}

        # Begin L_{1}HX_{1}Y_{2}
        temp = contract("ijac, ia -> jc", L[o, o, v, v], X1_A)
        temp = contract("jc, jkbc -> kb", temp, X2_B)
        polar -= contract("kb, kb", temp, l1)
        temp = contract("ijac, ia -> jc", L[o, o, v, v], X1_B)
        temp = contract("jc, jkbc -> kb", temp, X2_A)
        polar -= contract("kb, kb", temp, l1)
        temp = contract("ijac, ia -> jc", L[o, o, v, v], X1_A)
        temp = contract("jc, jkcb -> kb", temp, X2_B)
        polar += 2.0 * contract("kb, kb", temp, l1)
        temp = contract("ijac, ia -> jc", L[o, o, v, v], X1_B)
        temp = contract("jc, jkcb -> kb", temp, X2_A)
        polar += 2.0 * contract("kb, kb", temp, l1)

        temp = contract("ijac, ikac -> jk", L[o, o, v, v], X2_B)
        temp = contract("jk, jb -> kb", temp, X1_A)
        polar -= contract("kb, kb -> ", temp, l1)
        temp = contract("ijac, ikac -> jk", L[o, o, v, v], X2_A)
        temp = contract("jk, jb -> kb", temp, X1_B)
        polar -= contract("kb, kb -> ", temp, l1)

        temp = contract("ijac, ijab -> bc", L[o, o, v, v], X2_B)
        temp = contract("bc, kc -> kb", temp, X1_A)
        polar -= contract("kb, kb -> ", temp, l1)
        temp = contract("ijac, ijab -> bc", L[o, o, v, v], X2_A)
        temp = contract("bc, kc -> kb", temp, X1_B)
        polar -= contract("kb, kb -> ", temp, l1)
        # End L_{1}HX_{1}Y_{2}

        # Begin L_{2}HX_{1}Y_{2}
        # <O|L2(A)[[Hbar(0),X2(B)],X1(C)]]|0>
        tmp = contract("klcd,lkdb->cb", X2_B, l2)
        tmp = contract("jb,cb->jc", X1_A, tmp)
        polar -= contract("jc,jc->", tmp, hbar.Hov)

        tmp = contract("klcd,ljdc->kj", X2_B, l2)
        tmp = contract("kj,jb->kb", tmp, X1_A)
        polar -= contract("kb,kb->", tmp, hbar.Hov)

        # down
        tmp = contract('lkda,klcd->ac', l2, X2_B)
        tmp2 = contract('jb,ajcb->ac', X1_A, hbar.Hvovv)
        polar += 2.0 * contract('ac,ac->', tmp, tmp2)

        tmp = contract('lkda,klcd->ac', l2, X2_B)
        tmp2 = contract('jb,ajbc->ac', X1_A, hbar.Hvovv)
        polar -= contract('ac,ac->', tmp, tmp2)

        tmp = contract('jb,ljda->lbda', X1_A, l2)

        # swapaxes
        tmp2 = 2.0 * contract('klcd,akbc->ldab', X2_B, hbar.Hvovv)
        tmp2 -= contract('klcd,akcb->ldab', X2_B, hbar.Hvovv)
        polar += contract('lbda,ldab->', tmp, tmp2)

        tmp = contract('ia,fkba->fkbi', X1_A, hbar.Hvovv)
        tmp = contract('fkbi,jifc->kjbc', tmp, l2)
        polar -= contract('jkbc,kjbc->', X2_B, tmp)

        tmp = contract('ia,fjac->fjic', X1_A, hbar.Hvovv)
        tmp = contract('fjic,ikfb->jkbc', tmp, l2)
        polar -= contract('jkbc,jkbc->', X2_B, tmp)

        tmp = contract('ia,jkfa->jkfi', X1_A, l2)
        tmp2 = contract('jkbc,fibc->jkfi', X2_B, hbar.Hvovv)
        polar -= contract('jkfi,jkfi->', tmp2, tmp)

        tmp = contract('jb,kjib->ki', X1_A, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_B, l2)
        polar -= 2.0 * contract('ki,ki->', tmp, tmp2)

        tmp = contract('jb,jkib->ki', X1_A, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_B, l2)
        polar += contract('ki,ki->', tmp, tmp2)

        tmp = 2.0 * contract('jkic,klcd->jild', hbar.Hooov, X2_B)
        tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_B)
        tmp = contract('jild,jb->bild', tmp, X1_A)
        polar -= contract('bild,ilbd->', tmp, l2)

        tmp = contract('ia,jkna->jkni', X1_A, hbar.Hooov)
        tmp2 = contract('jkbc,nibc->jkni', X2_B, l2)
        polar += contract('jkni,jkni->', tmp2, tmp)

        tmp = contract('ia,nkab->nkib', X1_A, l2)
        tmp = contract('jkbc,nkib->jnic', X2_B, tmp)
        polar += contract('jnic,ijnc->', tmp, hbar.Hooov)

        tmp = contract('ia,nkba->nkbi', X1_A, l2)
        tmp = contract('jkbc,nkbi->jnci', X2_B, tmp)
        polar += contract('jnci,jinc->', tmp, hbar.Hooov)

        tmp = contract("klcd,lkdb->cb", X2_A, l2)
        tmp = contract("jb,cb->jc", X1_B, tmp)
        polar -= contract("jc,jc->", tmp, hbar.Hov)

        tmp = contract("klcd,ljdc->kj", X2_A, l2)
        tmp = contract("kj,jb->kb", tmp, X1_B)
        polar -= contract("kb,kb->", tmp, hbar.Hov)

        # down
        tmp = contract('lkda,klcd->ac', l2, X2_A)
        tmp2 = contract('jb,ajcb->ac', X1_B, hbar.Hvovv)
        polar += 2.0 * contract('ac,ac->', tmp, tmp2)

        tmp = contract('lkda,klcd->ac', l2, X2_A)
        tmp2 = contract('jb,ajbc->ac', X1_B, hbar.Hvovv)
        polar -= contract('ac,ac->', tmp, tmp2)

        tmp = contract('jb,ljda->lbda', X1_B, l2)

        # swapaxes
        tmp2 = 2.0 * contract('klcd,akbc->ldab', X2_A, hbar.Hvovv)
        tmp2 -= contract('klcd,akcb->ldab', X2_A, hbar.Hvovv)
        polar += contract('lbda,ldab->', tmp, tmp2)

        tmp = contract('ia,fkba->fkbi', X1_B, hbar.Hvovv)
        tmp = contract('fkbi,jifc->kjbc', tmp, l2)
        polar -= contract('jkbc,kjbc->', X2_A, tmp)

        tmp = contract('ia,fjac->fjic', X1_B, hbar.Hvovv)
        tmp = contract('fjic,ikfb->jkbc', tmp, l2)
        polar -= contract('jkbc,jkbc->', X2_A, tmp)

        tmp = contract('ia,jkfa->jkfi', X1_B, l2)
        tmp2 = contract('jkbc,fibc->jkfi', X2_A, hbar.Hvovv)
        polar -= contract('jkfi,jkfi->', tmp2, tmp)

        tmp = contract('jb,kjib->ki', X1_B, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_A, l2)
        polar -= 2.0 * contract('ki,ki->', tmp, tmp2)

        tmp = contract('jb,jkib->ki', X1_B, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_A, l2)
        polar += contract('ki,ki->', tmp, tmp2)

        tmp = 2.0 * contract('jkic,klcd->jild', hbar.Hooov, X2_A)
        tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_A)
        tmp = contract('jild,jb->bild', tmp, X1_B)
        polar -= contract('bild,ilbd->', tmp, l2)

        tmp = contract('ia,jkna->jkni', X1_B, hbar.Hooov)
        tmp2 = contract('jkbc,nibc->jkni', X2_A, l2)
        polar += contract('jkni,jkni->', tmp2, tmp)

        tmp = contract('ia,nkab->nkib', X1_B, l2)
        tmp = contract('jkbc,nkib->jnic', X2_A, tmp)
        polar += contract('jnic,ijnc->', tmp, hbar.Hooov)

        tmp = contract('ia,nkba->nkbi', X1_B, l2)
        tmp = contract('jkbc,nkbi->jnci', X2_A, tmp)
        polar += contract('jnci,jinc->', tmp, hbar.Hooov)
        # End L_{2}HX_{1}Y_{2}

        return polar#-1.0 * polar

    def linresp_asym(self, pertkey_a, X1_B, X2_B, Y1_B, Y2_B):
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
        # # <0|Y1(B) * A_bar|0>
        polar1 += contract("ai, ia -> ", pertbar_A.Avo, Y1_B)
        # <0|Y2(B) * A_bar|0>
        polar1 += 0.5 * contract("abij, ijab -> ", Avvoo, Y2_B)
        polar1 += 0.5 * contract("baji, ijab -> ", Avvoo, Y2_B)
        # <0|[A_bar, X(B)]|0>
        polar2 += 2.0 * contract("ia, ia -> ", pertbar_A.Aov, X1_B)
        # <0|L1(0) [A_bar, X1(B)]|0>
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
        # <0|L2(0)[A_bar, X2(B)]|0>
        tmp = contract("ijab, kjab -> ik", l2, X2_B)
        polar2 -= 0.5 * contract("ik, ki -> ", tmp, pertbar_A.Aoo)
        tmp = contract("ijab, kiba-> jk", l2, X2_B)
        polar2 -= 0.5 * contract("jk, kj -> ", tmp, pertbar_A.Aoo)
        tmp = contract("ijab, ijac -> bc", l2, X2_B)
        polar2 += 0.5 * contract("bc, bc -> ", tmp, pertbar_A.Avv)
        tmp = contract("ijab, ijcb -> ac", l2, X2_B)
        polar2 += 0.5 * contract("ac, ac -> ", tmp, pertbar_A.Avv)

        return -1.0 * (polar1 + polar2)


    def solve_right(self, pertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
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

    def solve_left(self, pertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
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

        # Note that Avvoo is permutationally symmetric, unlike the implementation in ugacc
        self.Avvoo = contract('ijeb,ae->ijab', t2, self.Avv)
        self.Avvoo -= contract('mjab,mi->ijab', t2, self.Aoo)
        self.Avvoo = 0.5*(self.Avvoo + self.Avvoo.swapaxes(0,1).swapaxes(2,3))
