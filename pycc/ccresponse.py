"""
ccresponse.py: CC Response Functions
"""
from opt_einsum import contract

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
import time
from .utils import helper_diis
from .cclambda import cclambda

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
        # self.l1 = self.cclambda.l1
        # self.l2 = self.cclambda.l2


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
                #X_2[pertkey] = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check.append(polar)
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check.append(polar)
                if (omega != 0.0):
                    X_key = pertkey + "_" + f"{-omega:0.6f}"
                    print("Solving right-hand perturbed wave function for %s:" % (X_key))
                    X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                    check.append(polar)


    def linresp_asym(self, pertkey_a, pertkey_b, X1_B, X2_B, Y1_B, Y2_B):

        # Defining the l1 and l2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2

        # Grab X and Y amplitudes corresponding to perturbation B, omega1
        # X1_B = ccpert_X_B[0]
        # X2_B = ccpert_X_B[1]
        # Y1_B = ccpert_Y_B[0]
        # Y2_B = ccpert_Y_B[1]

        # Please refer to eqn 78 of [Crawford:xxxx].
        # Writing H(1)(omega) = B, T(1)(omega) = X, L(1)(omega) = y
        # <<A;B>> = <0|Y(B) * A_bar|0> + <0| (1 + L(0))[A_bar, X(B)}|0>
        #                 polar1                polar2
        polar1 = 0
        polar2 = 0
        # <0|Y1(B) * A_bar|0>
        pertbar_A = self.pertbar[pertkey_a]
        pertbar_B = self.pertbar[pertkey_b]
        Avvoo = pertbar_A.Avvoo.swapaxes(0,2).swapaxes(1,3)
        polar1 += contract("ai, ia -> ", pertbar_A.Avo, Y1_B)
        # <0|Y2(B) * A_bar|0>
        polar1 += 0.5 * contract("abij, ijab -> ", Avvoo, Y2_B)
        polar1 += 0.5 * contract("baji, ijab -> ", Avvoo, Y2_B)
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








    def pert_quadresp(self, omega1, omega2, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):

        self.ccpert_om1_X = {}
        self.ccpert_om2_X = {}
        self.ccpert_om_sum_X = {}
        
        self.ccpert_om1_2nd_X = {}
        self.ccpert_om2_2nd_X = {}
        self.ccpert_om_sum_2nd_X = {}
       
        self.ccpert_om1_Y = {} 
        self.ccpert_om2_Y = {} 
        self.ccpert_om_sum_Y = {} 
     
        self.ccpert_om1_2nd_Y = {} 
        self.ccpert_om2_2nd_Y = {} 
        self.ccpert_om_sum_2nd_Y = {}
 
        omega_sum = -(omega1 + omega2) 
   
        for axis in range(0, 3):
            
            pertkey = "MU_" + self.cart[axis]
            #X_key = pertkey + "_" + f"{omega1:0.6f}"
             
            print("Solving right-hand perturbed wave function for omega1 %s:" % (pertkey)) 
            self.ccpert_om1_X[pertkey] = self.solve_right(self.pertbar[pertkey], omega1, e_conv, r_conv, maxiter, max_diis, start_diis)

            print("Solving left-hand perturbed wave function for %s:" % (pertkey))
            self.ccpert_om1_Y[pertkey] = self.solve_left(self.pertbar[pertkey], omega1, e_conv, r_conv, maxiter, max_diis, start_diis)

            print("Solving right-hand perturbed wave function for omega2 %s:" % (pertkey))
            self.ccpert_om2_X[pertkey] = self.solve_right(self.pertbar[pertkey], omega2, e_conv, r_conv, maxiter, max_diis, start_diis)

            print("Solving left-hand perturbed wave function for %s:" % (pertkey))
            self.ccpert_om2_Y[pertkey] = self.solve_left(self.pertbar[pertkey], omega2, e_conv, r_conv, maxiter, max_diis, start_diis)

            print("Solving right-hand perturbed wave function for omega_sum %s:" % (pertkey))
            self.ccpert_om_sum_X[pertkey] = self.solve_right(self.pertbar[pertkey], omega_sum, e_conv, r_conv, maxiter, max_diis, start_diis)
            #print("solved X", self.ccpert_om_sum_X[pertkey][0]) 
            
            print("Solving left-hand perturbed wave function for %s:" % (pertkey))
            self.ccpert_om_sum_Y[pertkey] = self.solve_left(self.pertbar[pertkey], omega_sum, e_conv, r_conv, maxiter, max_diis, start_diis)
            #self.ccpert_om_sum_Y[pertkey] = self.solve_left(self.pertbar[pertkey], omega_sum, self.ccpert_om_sum_X[pertkey][0], self.ccpert_om_sum_X[pertkey][1], e_conv, r_conv, maxiter, max_diis, start_diis)
            #print("solved Y", self.ccpert_om_sum_Y[pertkey][0])

            print("Solving right-hand perturbed wave function for -omega1 %s:" % (pertkey))
            self.ccpert_om1_2nd_X[pertkey] = self.solve_right(self.pertbar[pertkey], -omega1, e_conv, r_conv, maxiter, max_diis, start_diis)

            print("Solving left-hand perturbed wave function for %s:" % (pertkey))
            self.ccpert_om1_2nd_Y[pertkey] = self.solve_left(self.pertbar[pertkey], -omega1, e_conv, r_conv, maxiter, max_diis, start_diis)

            print("Solving right-hand perturbed wave function for -omega2 %s:" % (pertkey))
            self.ccpert_om2_2nd_X[pertkey] = self.solve_right(self.pertbar[pertkey], -omega2, e_conv, r_conv, maxiter, max_diis, start_diis)

            print("Solving left-hand perturbed wave function for %s:" % (pertkey))
            self.ccpert_om2_2nd_Y[pertkey] = self.solve_left(self.pertbar[pertkey], -omega2, e_conv, r_conv, maxiter, max_diis, start_diis)

            print("Solving right-hand perturbed wave function for -omega_sum %s:" % (pertkey))
            self.ccpert_om_sum_2nd_X[pertkey] = self.solve_right(self.pertbar[pertkey], -omega_sum, e_conv, r_conv, maxiter, max_diis, start_diis)

            print("Solving left-hand perturbed wave function for %s:" % (pertkey))
            self.ccpert_om_sum_2nd_Y[pertkey] = self.solve_left(self.pertbar[pertkey], -omega_sum, e_conv, r_conv, maxiter, max_diis, start_diis)

       # self.ccpert_om1_X = ccpert_om1_X
       # self.ccpert_om2_X = ccpert_om2_X
       # self.ccpert_om_sum_X =ccpert_om_sum_X

       # self.ccpert_om1_2nd_X = ccpert_om1_2nd_X
       # self.ccpert_om2_2nd_X = ccpert_om2_2nd_X
       # self.ccpert_om_sum_2nd_X = ccpert_om_sum_2nd_X
    
       # self.ccpert_om1_Y = ccpert_om1_Y
       # self.ccpert_om2_Y = ccpert_om2_Y
       # self.ccpert_om_sum_Y =ccpert_om_sum_Y

       # self.ccpert_om1_2nd_Y = ccpert_om1_2nd_Y
       # self.ccpert_om2_2nd_Y = ccpert_om2_2nd_Y
       # self.ccpert_om_sum_2nd_Y = ccpert_om_sum_2nd_Y

    def quadraticresp(self, pertkey_a, pertkey_b, pertkey_c, ccpert_X_A, ccpert_X_B, ccpert_X_C, ccpert_Y_A, ccpert_Y_B, ccpert_Y_C):
        contract = self.contract

        o = self.ccwfn.o
        v = self.ccwfn.v
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2 
        l1 = self.cclambda.l1 
        l2 = self.cclambda.l2  
        # Grab X and Y amplitudes corresponding to perturbation A, omega_sum
        X1_A = ccpert_X_A[0]
        X2_A = ccpert_X_A[1]
        Y1_A = ccpert_Y_A[0]
        Y2_A = ccpert_Y_A[1]
        # Grab X and Y amplitudes corresponding to perturbation B, omega1
        X1_B = ccpert_X_B[0]
        X2_B = ccpert_X_B[1]
        Y1_B = ccpert_Y_B[0]
        Y2_B = ccpert_Y_B[1]
        # Grab X and Y amplitudes corresponding to perturbation C
        X1_C = ccpert_X_C[0]
        X2_C = ccpert_X_C[1]
        Y1_C = ccpert_Y_C[0]
        Y2_C = ccpert_Y_C[1]
        # Grab pert integrals
        pertbar_A = self.pertbar[pertkey_a]
        pertbar_B = self.pertbar[pertkey_b]
        pertbar_C = self.pertbar[pertkey_c]

        #Grab H_bar, L and ERI
        hbar = self.hbar
        L = self.H.L
        ERI = self.H. ERI

        self.hyper = 0.0
        self.LAX = 0.0
        self.LAX2 = 0.0
        self.LAX3 = 0.0
        self.LAX4 = 0.0
        self.LAX5 = 0.0
        self.LAX6 = 0.0
        self.LHX1Y1 = 0.0
        self.LHX1Y2 = 0.0
        self.LHX1X2 = 0.0
        self.LHX2Y2 = 0.0

        # <0|L1(B)[A_bar, X1(C)]|0> good
        tmp = contract('ia,ic->ac', Y1_B, X1_C)
        self.LAX += contract('ac,ac->',tmp, pertbar_A.Avv)
        tmp = contract('ia,ka->ik', Y1_B, X1_C)
        self.LAX -= contract('ik,ki->', tmp, pertbar_A.Aoo)

        # <0|L1(B)[A_bar, X2(C)]|0>
        tmp = contract('ia,jb->ijab', Y1_B, pertbar_A.Aov)

        #swapaxes
        self.LAX += 2.0 * contract('ijab,ijab->', tmp, X2_C)
        self.LAX -= contract('ijab,ijba->',tmp, X2_C)

        # <0|L2(B)[A_bar, X1(C)]|0>
        tmp = contract('ijbc,bcaj->ia', Y2_B, pertbar_A.Avvvo)
        self.LAX += contract('ia,ia->', tmp, X1_C)
        tmp = contract('ijab,kbij->ak', Y2_B, pertbar_A.Aovoo)
        self.LAX -= contract('ak,ka->', tmp, X1_C)
        # <0|L2(B)[A_bar, X2(C)]|0>
        tmp = contract('ijab,kjab->ik', Y2_B, X2_C)
        self.LAX -= contract('ik,ki->', tmp, pertbar_A.Aoo)
        tmp = contract('ijab,ijac->bc', Y2_B, X2_C)
        self.LAX += contract('bc,bc->', tmp, pertbar_A.Avv)

        self.hyper += self.LAX # good

        # <0|L1(C)[A_bar, X1(B)]|0> good
        tmp = contract('ia,ic->ac', Y1_C, X1_B)
        self.LAX2 += contract('ac,ac->', tmp, pertbar_A.Avv)
        tmp = contract('ia,ka->ik', Y1_C, X1_B)
        self.LAX2 -= contract('ik,ki->',tmp, pertbar_A.Aoo)

        # <0|L1(C)[A_bar, X2(B)]|0>
        tmp = contract('ia,jb->ijab', Y1_C, pertbar_A.Aov)
        
        #swapaxes
        self.LAX2 += 2.0 * contract('ijab,ijab->', tmp, X2_B)
        self.LAX2 -= contract('ijab,ijba->', tmp, X2_B)

        # <0|L2(C)[A_bar, X1(B)]|0>
        tmp = contract('ijbc,bcaj->ia', Y2_C, pertbar_A.Avvvo)
        self.LAX2 += contract('ia,ia->', tmp, X1_B)
        tmp = contract('ijab,kbij->ak', Y2_C, pertbar_A.Aovoo)
        self.LAX2 -= contract('ak,ka->', tmp, X1_B)

        # <0|L2(C)[A_bar, X2(B)]|0>
        tmp = contract('ijab,kjab->ik', Y2_C, X2_B)
        self.LAX2 -= contract('ik,ki->', tmp, pertbar_A.Aoo)
        tmp = contract('ijab,ijac->bc', Y2_C, X2_B)
        self.LAX2 += contract('bc,bc->', tmp, pertbar_A.Avv)

        self.hyper += self.LAX2 #good

        # <0|L1(A)[B_bar,X1(C)]|0>
        tmp = contract('ia,ic->ac', Y1_A, X1_C)
        self.LAX3 += contract('ac,ac->',tmp, pertbar_B.Avv)
        tmp = contract('ia,ka->ik', Y1_A, X1_C)
        self.LAX3 -= contract('ik,ki->', tmp, pertbar_B.Aoo)
        # <0|L1(A)[B_bar, X2(C)]|0>
        tmp = contract('ia,jb->ijab', Y1_A, pertbar_B.Aov)

        #swapaxes 
        self.LAX3 += 2.0 * contract('ijab,ijab->', tmp, X2_C)
        self.LAX3 -= contract('ijab,ijba->', tmp, X2_C)
        # <0|L2(A)[B_bar, X1(C)]|0>
        tmp = contract('ijbc,bcaj->ia', Y2_A, pertbar_B.Avvvo)
        self.LAX3 += contract('ia,ia->',tmp, X1_C)
        tmp = contract('ijab,kbij->ak', Y2_A, pertbar_B.Aovoo)
        self.LAX3 -= contract('ak,ka->', tmp, X1_C)
        # <0|L2(A)[B_bar, X2(C)]|0>
        tmp = contract('ijab,kjab->ik', Y2_A, X2_C)
        self.LAX3 -= contract('ik,ki->', tmp, pertbar_B.Aoo)
        tmp = contract('ijab,ijac->bc', Y2_A, X2_C)
        self.LAX3 += contract('bc,bc->', tmp, pertbar_B.Avv)

        self.hyper += self.LAX3

        # <0|L1(C)|[B_bar,X1(A)]|0>
        tmp = contract('ia,ic->ac', Y1_C, X1_A)
        self.LAX4 += contract('ac,ac->', tmp, pertbar_B.Avv)
        tmp = contract('ia,ka->ik', Y1_C, X1_A)
        self.LAX4 -= contract('ik,ki->', tmp, pertbar_B.Aoo)
        # <0|L1(C)[B_bar, X2(A)]|0>
        tmp = contract('ia,jb->ijab',Y1_C, pertbar_B.Aov)        
        #swapaxes
        self.LAX4 += 2.0 * contract('ijab,ijab->',tmp, X2_A)
        self.LAX4 -= contract('ijab,ijba->', tmp, X2_A)
        # <0|L2(C)[B_bar, X1(A)]|0>
        tmp = contract('ijbc,bcaj->ia', Y2_C, pertbar_B.Avvvo)
        self.LAX4 += contract('ia,ia->', tmp, X1_A)
        tmp = contract('ijab,kbij->ak', Y2_C, pertbar_B.Aovoo)
        self.LAX4 -= contract('ak,ka->', tmp, X1_A)
        # <0|L2(C)[B_bar, X2(A)]|0>
        tmp = contract('ijab,kjab->ik', Y2_C, X2_A)
        self.LAX4 -= contract('ik,ki->', tmp, pertbar_B.Aoo)
        tmp = contract('ijab,kiba->jk', Y2_C, X2_A)
        tmp = contract('ijab,ijac->bc', Y2_C, X2_A)
        self.LAX4 += contract('bc,bc->', tmp, pertbar_B.Avv)

        self.hyper += self.LAX4 #good

        # <0|L1(A)[C_bar,X1(B)]|0>
        tmp = contract('ia,ic->ac', Y1_A, X1_B)
        self.LAX5 += contract('ac,ac->', tmp, pertbar_C.Avv)
        tmp = contract('ia,ka->ik', Y1_A, X1_B)
        self.LAX5 -= contract('ik,ki->', tmp, pertbar_C.Aoo)
        # <0|L1(A)[C_bar, X2(B)]|0>
        tmp = contract('ia,jb->ijab', Y1_A, pertbar_C.Aov)

        #swapaxes
        self.LAX5 += 2.0 * contract('ijab,ijab->', tmp, X2_B)
        self.LAX5 -= contract('ijab,ijba->', tmp, X2_B)
        # <0|L2(A)[C_bar, X1(B)]|0>
        tmp = contract('ijbc,bcaj->ia', Y2_A, pertbar_C.Avvvo)
        self.LAX5 += contract('ia,ia->', tmp, X1_B)
        tmp = contract('ijab,kbij->ak', Y2_A, pertbar_C.Aovoo)
        self.LAX5 -= contract('ak,ka->', tmp, X1_B)
        # <0|L2(A)[C_bar, X2(B)]|0>
        tmp = contract('ijab,kjab->ik', Y2_A, X2_B)
        self.LAX5 -= contract('ik,ki->', tmp, pertbar_C.Aoo)
        tmp = contract('ijab,ijac->bc', Y2_A, X2_B)
        self.LAX5 += contract('bc,bc->', tmp, pertbar_C.Avv)

        self.hyper += self.LAX5

        # <0|L1(B)|[C_bar,X1(A)]|0>
        tmp = contract('ia,ic->ac', Y1_B, X1_A)
        self.LAX6 += contract('ac,ac->', tmp, pertbar_C.Avv)
        tmp = contract('ia,ka->ik', Y1_B, X1_A)
        self.LAX6 -= contract('ik,ki->', tmp, pertbar_C.Aoo)
        # <0|L1(B)[C_bar, X2(A)]|0>
        tmp = contract('ia,jb->ijab', Y1_B, pertbar_C.Aov)

        #swapaxes
        self.LAX6 += 2.0 * contract('ijab,ijab->', tmp, X2_A)
        self.LAX6 -= contract('ijab,ijba->', tmp, X2_A)
        # <0|L2(B)[C_bar, X1(A)]|0>
        tmp = contract('ijbc,bcaj->ia', Y2_B, pertbar_C.Avvvo)
        self.LAX6 += contract('ia,ia->', tmp, X1_A)
        tmp = contract('ijab,kbij->ak', Y2_B, pertbar_C.Aovoo)
        self.LAX6 -= contract('ak,ka->', tmp, X1_A)
        # <0|L2(B)[C_bar, X2(A)]|0>
        tmp = contract('ijab,kjab->ik', Y2_B, X2_A)
        self.LAX6 -= np.einsum('ik,ki->', tmp, pertbar_C.Aoo)
        tmp = contract('ijab,ijac->bc', Y2_B, X2_A)
        self.LAX6 += contract('bc,bc->', tmp, pertbar_C.Avv)

        self.hyper += self.LAX6 #good

        self.Fz1 = 0
        self.Fz2 = 0
        self.Fz3 = 0

        # <0|L1(0)[[A_bar,X1(B)],X1(C)]|0>
        tmp = contract('ia,ja->ij', X1_B, pertbar_A.Aov)
        tmp2 = contract('ib,jb->ij', l1, X1_C)
        self.Fz1 -= contract('ij,ij->', tmp2, tmp)

        tmp = contract('jb,ib->ij', X1_C, pertbar_A.Aov)
        tmp2 = contract('ia,ja->ij', X1_B, l1)
        self.Fz1 -= contract('ij,ij->', tmp2, tmp)

        # <0|L2(0)[[A_bar,X1(B)],X2(C)]|0>
        tmp = contract('ia,ja->ij', X1_B, pertbar_A.Aov)
        tmp2 = contract('jkbc,ikbc->ij', X2_C, l2)
        self.Fz1 -= contract('ij,ij->',tmp2,tmp)

        tmp = contract('ia,jkac->jkic', X1_B, l2)
        tmp = contract('jkbc,jkic->ib', X2_C, tmp)
        self.Fz1 -= contract('ib,ib->', tmp, pertbar_A.Aov)

        # <0|L2(0)[[A_bar,X2(B)],X1(C)]|0>
        tmp = contract('ia,ja->ij', X1_C, pertbar_A.Aov)
        tmp2 = contract('jkbc,ikbc->ij', X2_B, l2)
        self.Fz1 -= contract('ij,ij->', tmp2, tmp)

        tmp = contract('ia,jkac->jkic', X1_C, l2)
        tmp = contract('jkbc,jkic->ib', X2_B, tmp)
        self.Fz1 -= contract('ib,ib->', tmp, pertbar_A.Aov)

        # <0|L1(0)[B_bar,X1(A)],X1(C)]|0>
        tmp = contract('ia,ja->ij', X1_A, pertbar_B.Aov)
        tmp2 = contract('ib,jb->ij', l1, X1_C)
        self.Fz2 -= contract('ij,ij->', tmp2, tmp)

        tmp = contract('jb,ib->ij', X1_C, pertbar_B.Aov)
        tmp2 = contract('ia,ja->ij', X1_A, l1)
        self.Fz2 -= contract('ij,ij->', tmp2, tmp)

        # <0|L2(0)[[B_bar,X1(A)],X2(C)]|0>
        tmp = contract('ia,ja->ij', X1_A, pertbar_B.Aov)
        tmp2 = contract('jkbc,ikbc->ij', X2_C, l2)
        self.Fz2 -= contract('ij,ij->', tmp2, tmp)

        tmp = contract('ia,jkac->jkic', X1_A, l2)
        tmp = contract('jkbc,jkic->ib', X2_C, tmp)
        self.Fz2 -= contract('ib,ib->', tmp, pertbar_B.Aov)

        # <0|L2(0)[[B_bar,X2(A)],X1(C)]|0>
        tmp = contract('ia,ja->ij', X1_C, pertbar_B.Aov)
        tmp2 = contract('jkbc,ikbc->ij', X2_A, l2)
        self.Fz2 -= contract('ij,ij->', tmp2, tmp)

        tmp = contract('ia,jkac->jkic', X1_C, l2)
        tmp = contract('jkbc,jkic->ib', X2_A, tmp)
        self.Fz2 -= contract('ib,ib->', tmp, pertbar_B.Aov)

        # <0|L1(0)[C_bar,X1(A)],X1(B)]|0>
        tmp = contract('ia,ja->ij', X1_A, pertbar_C.Aov)
        tmp2 = contract('ib,jb->ij', l1, X1_B)
        self.Fz3 -= contract('ij,ij->', tmp2, tmp)

        tmp = contract('jb,ib->ij', X1_B, pertbar_C.Aov)
        tmp2 = contract('ia,ja->ij', X1_A, l1)
        self.Fz3 -= contract('ij,ij->', tmp2, tmp)

        # <0|L2(0)[[C_bar,X1(A)],X2(B)]|0>
        tmp = contract('ia,ja->ij', X1_A, pertbar_C.Aov)
        tmp2 = contract('jkbc,ikbc->ij', X2_B, l2)
        self.Fz3 -= contract('ij,ij->', tmp2, tmp)

        tmp = contract('ia,jkac->jkic', X1_A, l2)
        tmp = contract('jkbc,jkic->ib', X2_B, tmp)
        self.Fz3 -= contract('ib,ib->', tmp, pertbar_C.Aov)

        # <0|L2(0)[[C_bar,X2(A)],X1(B)]|0>
        tmp = contract('ia,ja->ij', X1_B, pertbar_C.Aov)
        tmp2 = contract('jkbc,ikbc->ij', X2_A, l2)
        self.Fz3 -= contract('ij,ij->', tmp2, tmp)

        tmp = contract('ia,jkac->jkic', X1_B, l2)
        tmp = contract('jkbc,jkic->ib', X2_A, tmp)
        self.Fz3 -= contract('ib,ib->', tmp, pertbar_C.Aov)

        self.hyper += self.Fz1+self.Fz2+self.Fz3

        self.G = 0
        # <L1(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>
        tmp = contract('ia,ijac->jc', X1_A, L[o,o,v,v])
        tmp = contract('kc,jc->jk', X1_C, tmp)
        tmp2 = contract('jb,kb->jk', X1_B, l1)
        self.G -= contract('jk,jk->', tmp2, tmp)

        tmp = contract('ia,ikab->kb', X1_A, L[o,o,v,v])
        tmp = contract('jb,kb->jk', X1_B, tmp)
        tmp2 = contract('jc,kc->jk', l1, X1_C)
        self.G -= contract('jk,jk->', tmp2, tmp)

        tmp = contract('jb,jkba->ka', X1_B, L[o,o,v,v])
        tmp = contract('ia,ka->ki', X1_A, tmp)
        tmp2 = contract('kc,ic->ki', X1_C, l1)
        self.G -= contract('ki,ki->', tmp2, tmp)

        tmp = contract('jb,jibc->ic', X1_B, L[o,o,v,v])
        tmp = contract('kc,ic->ki', X1_C, tmp)
        tmp2 = contract('ka,ia->ki', l1, X1_A)
        self.G -= contract('ki,ki->', tmp2, tmp)

        tmp = contract('kc,kicb->ib', X1_C, L[o,o,v,v])
        tmp = contract('jb,ib->ji', X1_B, tmp)
        tmp2 = contract('ja,ia->ji', l1, X1_A)
        self.G -= contract('ji,ji->', tmp2, tmp)

        tmp = contract('kc,kjca->ja', X1_C, L[o,o,v,v])
        tmp = contract('ia,ja->ji', X1_A, tmp)
        tmp2 = contract('jb,ib->ji', X1_B, l1)
        self.G -= contract('ji,ji->', tmp2, tmp)

        # <L2(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>
        tmp = contract('jb,klib->klij', X1_A, hbar.Hooov)
        tmp2  = contract('ld,ijcd->ijcl', X1_C, l2)
        tmp2  = contract('kc,ijcl->ijkl', X1_B, tmp2)
        self.G += contract('ijkl,klij->', tmp2, tmp)

        tmp = contract('jb,lkib->lkij', X1_A, hbar.Hooov)
        tmp2 = contract('ld,ijdc->ijlc', X1_C, l2)
        tmp2 = contract('kc,ijlc->ijlk', X1_B, tmp2)
        self.G += contract('ijlk,lkij->', tmp2, tmp)

        tmp = contract('kc,jlic->jlik', X1_B, hbar.Hooov)
        tmp2  = contract('jb,ikbd->ikjd', X1_A, l2)
        tmp2  = contract('ld,ikjd->ikjl', X1_C, tmp2)
        self.G += contract('ikjl,jlik->', tmp2, tmp)

        tmp = contract('kc,ljic->ljik', X1_B, hbar.Hooov)
        tmp2  = contract('jb,ikdb->ikdj', X1_A, l2)
        tmp2  = contract('ld,ikdj->iklj', X1_C, tmp2)
        self.G += contract('iklj,ljik->', tmp2, tmp)

        tmp = contract('ld,jkid->jkil', X1_C, hbar.Hooov)
        tmp2  = contract('jb,ilbc->iljc', X1_A, l2)
        tmp2  = contract('kc,iljc->iljk', X1_B, tmp2)
        self.G += contract('iljk,jkil->', tmp2, tmp)

        tmp = contract('ld,kjid->kjil', X1_C, hbar.Hooov)
        tmp2  = contract('jb,ilcb->ilcj', X1_A, l2)
        tmp2  = contract('kc,ilcj->ilkj', X1_B, tmp2)
        self.G += contract('ilkj,kjil->', tmp2, tmp)

        tmp = contract('jb,albc->aljc', X1_A, hbar.Hvovv)
        tmp = contract('kc,aljc->aljk', X1_B, tmp)
        tmp2  = contract('ld,jkad->jkal', X1_C, l2)
        self.G -= contract('jkal,aljk->', tmp2, tmp)

        tmp = contract('jb,alcb->alcj', X1_A, hbar.Hvovv)
        tmp = contract('kc,alcj->alkj', X1_B, tmp)
        tmp2  = contract('ld,jkda->jkla', X1_C, l2)
        self.G -= contract('jkla,alkj->', tmp2, tmp)

        tmp = contract('jb,akbd->akjd', X1_A, hbar.Hvovv)
        tmp = contract('ld,akjd->akjl', X1_C, tmp)
        tmp2  = contract('kc,jlac->jlak', X1_B, l2)
        self.G -= contract('jlak,akjl->', tmp2, tmp)

        tmp = contract('jb,akdb->akdj', X1_A, hbar.Hvovv)
        tmp = contract('ld,akdj->aklj', X1_C, tmp)
        tmp2  = contract('kc,jlca->jlka', X1_B, l2)
        self.G -= contract('jlka,aklj->', tmp2, tmp)

        tmp = contract('kc,ajcd->ajkd', X1_B, hbar.Hvovv)
        tmp = contract('ld,ajkd->ajkl', X1_C, tmp)
        tmp2  = contract('jb,klab->klaj', X1_A, l2)
        self.G -= contract('klaj,ajkl->', tmp2, tmp)

        tmp = contract('kc,ajdc->ajdk', X1_B, hbar.Hvovv)
        tmp = contract('ld,ajdk->ajlk', X1_C, tmp)
        tmp2  = contract('jb,klba->klja', X1_A, l2)
        self.G -= contract('klja,ajlk->', tmp2, tmp)

        # <L2(0)|[[[H_bar,X2(A)],X1(B)],X1(C)]|0>
        tmp = contract('kc,jlbc->jlbk', X1_B, l2)
        tmp2 = contract('ld,ikad->ikal', X1_C, L[o,o,v,v])
        tmp2 = contract('ijab,ikal->jlbk', X2_A, tmp2)
        self.G -= contract('jlbk,jlbk->', tmp, tmp2)

        tmp = contract('ld,jkbd->jkbl', X1_C, l2)
        tmp2 = contract('kc,ilac->ilak', X1_B, L[o,o,v,v])
        tmp2 = contract('ijab,ilak->jkbl', X2_A, tmp2)
        self.G -= contract('jkbl,jkbl->',tmp,tmp2)

        tmp = contract('ijab,jibd->ad', X2_A, l2)
        tmp = contract('ld,ad->la', X1_C, tmp)
        tmp2 = contract('klca,kc->la', L[o,o,v,v], X1_B)
        self.G -= contract('la,la->', tmp, tmp2)

        tmp = contract('ijab,jlba->il', X2_A, l2)
        tmp2 = contract('kc,kicd->id', X1_B, L[o,o,v,v])
        tmp2 = contract('ld,id->il', X1_C, tmp2)
        self.G -= contract('il,il->', tmp, tmp2)

        tmp = contract('ijab,jkba->ik', X2_A, l2)
        tmp2 = contract('ld,lidc->ic', X1_C, L[o,o,v,v])
        tmp2 = contract('kc,ic->ik', X1_B, tmp2)
        self.G -= contract('ik,ik->', tmp, tmp2)

        tmp = contract('ijab,jibc->ac', X2_A, l2)
        tmp = contract('ac,kc->ka', tmp, X1_B)
        tmp2 = contract('ld,lkda->ka', X1_C, L[o,o,v,v])
        self.G -= contract('ka,ka->', tmp, tmp2)

        # Goovv -> ERIoovv
        tmp = contract('ijab,klab->ijkl',X2_A, ERI[o,o,v,v])
        tmp2 = contract('kc,ijcd->ijkd', X1_B, l2)
        tmp2 = contract('ld,ijkd->ijkl', X1_C, tmp2)
        self.G += contract('ijkl,ijkl->',tmp,tmp2)

        tmp = contract('kc,jlac->jlak', X1_B, ERI[o,o,v,v])
        tmp = contract('ijab,jlak->ilbk', X2_A, tmp)
        tmp2 = contract('ikbd,ld->ilbk', l2, X1_C)
        self.G += contract('ilbk,ilbk->', tmp, tmp2)

        tmp = contract('kc,ljac->ljak', X1_B, ERI[o,o,v,v])
        tmp = contract('ijab,ljak->ilbk', X2_A, tmp)
        tmp2 = contract('ikdb,ld->ilbk', l2, X1_C)
        self.G += contract('ilbk,ilbk->', tmp, tmp2)

        tmp = contract('ld,jkad->jkal', X1_C, ERI[o,o,v,v])
        tmp = contract('ijab,jkal->ikbl', X2_A, tmp)
        tmp2 = contract('kc,ilbc->ilbk', X1_B, l2)
        self.G += contract('ikbl,ilbk->', tmp, tmp2)

        tmp = contract('ld,kjad->kjal', X1_C, ERI[o,o,v,v])
        tmp = contract('ijab,kjal->iklb', X2_A, tmp)
        tmp2 = contract('kc,ilcb->ilkb', X1_B, l2)
        self.G += contract('iklb,ilkb->', tmp, tmp2)

        tmp = contract('kc,ijcd->ijkd', X1_B, ERI[o,o,v,v])
        tmp = contract('ld,ijkd->ijkl', X1_C, tmp)
        tmp2 = contract('ijab,klab->ijkl', X2_A, l2)
        self.G += contract('ijkl,ijkl->', tmp, tmp2)

        # <L2(0)|[[[H_bar,X1(A)],X2(B)],X1(C)]|0>
        tmp = contract('kc,jlbc->jlbk', X1_A, l2)
        tmp2 = contract('ld,ikad->ikal', X1_C, L[o,o,v,v])
        tmp2 = contract('ijab,ikal->jlbk', X2_B, tmp2)
        self.G -= contract('jlbk,jlbk->', tmp, tmp2)

        tmp = contract('ld,jkbd->jkbl', X1_C, l2)
        tmp2 = contract('kc,ilac->ilak', X1_A, L[o,o,v,v])
        tmp2 = contract('ijab,ilak->jkbl',X2_B, tmp2)
        self.G -= contract('jkbl,jkbl->', tmp, tmp2)

        tmp = contract('ijab,jibd->ad', X2_B, l2)
        tmp = contract('ld,ad->la', X1_C, tmp)
        tmp2 = contract('klca,kc->la', L[o,o,v,v], X1_A)
        self.G -= contract('la,la->', tmp, tmp2)

        tmp = contract('ijab,jlba->il', X2_B, l2)
        tmp2 = contract('kc,kicd->id', X1_A, L[o,o,v,v])
        tmp2 = contract('ld,id->il', X1_C, tmp2)
        self.G -= contract('il,il->', tmp, tmp2)

        tmp = contract('ijab,jkba->ik', X2_B, l2)
        tmp2 = contract('ld,lidc->ic', X1_C, L[o,o,v,v])
        tmp2 = contract('kc,ic->ik', X1_A, tmp2)
        self.G -= contract('ik,ik->', tmp, tmp2)

        tmp = contract('ijab,jibc->ac', X2_B, l2)
        tmp = contract('ac,kc->ka', tmp, X1_A)
        tmp2 = contract('ld,lkda->ka', X1_C, L[o,o,v,v])
        self.G -= contract('ka,ka->', tmp, tmp2)

        tmp = contract('ijab,klab->ijkl', X2_B, ERI[o,o,v,v])
        tmp2 = contract('kc,ijcd->ijkd', X1_A, l2)
        tmp2 = contract('ld,ijkd->ijkl', X1_C, tmp2)
        self.G += contract('ijkl,ijkl->', tmp, tmp2)

        tmp = contract('kc,jlac->jlak', X1_A, ERI[o,o,v,v])
        tmp = contract('ijab,jlak->ilbk', X2_B, tmp)
        tmp2 = contract('ikbd,ld->ilbk', l2, X1_C)
        self.G += contract('ilbk,ilbk->', tmp, tmp2)

        tmp  = contract('kc,ljac->ljak', X1_A, ERI[o,o,v,v])
        tmp  = contract('ijab,ljak->ilbk', X2_B, tmp)
        tmp2 = contract('ikdb,ld->ilbk', l2, X1_C)
        self.G += contract('ilbk,ilbk->', tmp, tmp2)

        tmp = contract('ld,jkad->jkal', X1_C, ERI[o,o,v,v])
        tmp = contract('ijab,jkal->ikbl', X2_B, tmp)
        tmp2 = contract('kc,ilbc->ilbk', X1_A, l2)
        self.G += contract('ikbl,ilbk->', tmp, tmp2)

        tmp = contract('ld,kjad->kjal', X1_C, ERI[o,o,v,v])
        tmp = contract('ijab,kjal->iklb', X2_B, tmp)
        tmp2 = contract('kc,ilcb->ilkb', X1_A, l2)
        self.G += contract('iklb,ilkb->', tmp, tmp2)

        tmp = contract('kc,ijcd->ijkd', X1_A, ERI[o,o,v,v])
        tmp = contract('ld,ijkd->ijkl', X1_C, tmp)
        tmp2 = contract('ijab,klab->ijkl', X2_B, l2)
        self.G += contract('ijkl,ijkl->', tmp, tmp2)

        # <L2(0)|[[[H_bar,X1(A)],X1(B)],X2(C)]|0>
        tmp = contract('kc,jlbc->jlbk', X1_A, l2)
        tmp2 = contract('ld,ikad->ikal', X1_B, L[o,o,v,v])
        tmp2 = contract('ijab,ikal->jlbk', X2_C, tmp2)
        self.G -= contract('jlbk,jlbk->', tmp, tmp2)

        tmp = contract('ld,jkbd->jkbl', X1_B, l2)
        tmp2 = contract('kc,ilac->ilak', X1_A, L[o,o,v,v])
        tmp2 = contract('ijab,ilak->jkbl', X2_C, tmp2)
        self.G -= contract('jkbl,jkbl->', tmp, tmp2)

        tmp = contract('ijab,jibd->ad', X2_C, l2)
        tmp = contract('ld,ad->la', X1_B, tmp)
        tmp2 = contract('klca,kc->la', L[o,o,v,v], X1_A)
        self.G -= contract('la,la->', tmp, tmp2)

        tmp = contract('ijab,jlba->il', X2_C, l2)
        tmp2 = contract('kc,kicd->id', X1_A, L[o,o,v,v])
        tmp2 = contract('ld,id->il', X1_B, tmp2)
        self.G -= contract('il,il->', tmp, tmp2)

        tmp = contract('ijab,jkba->ik', X2_C, l2)
        tmp2 = contract('ld,lidc->ic', X1_B, L[o,o,v,v])
        tmp2 = contract('kc,ic->ik', X1_A, tmp2)
        self.G -= contract('ik,ik->', tmp, tmp2)

        tmp = contract('ijab,jibc->ac', X2_C, l2)
        tmp = contract('ac,kc->ka', tmp, X1_A)
        tmp2 = contract('ld,lkda->ka', X1_B, L[o,o,v,v])
        self.G -= contract('ka,ka->', tmp, tmp2)

        tmp = contract('ijab,klab->ijkl', X2_C, ERI[o,o,v,v])
        tmp2 = contract('kc,ijcd->ijkd', X1_A, l2)
        tmp2 = contract('ld,ijkd->ijkl', X1_B, tmp2)
        self.G += contract('ijkl,ijkl->', tmp, tmp2)

        tmp = contract('kc,jlac->jlak', X1_A, ERI[o,o,v,v])
        tmp = contract('ijab,jlak->ilbk', X2_C, tmp)
        tmp2 = contract('ikbd,ld->ilbk', l2, X1_B)
        self.G += contract('ilbk,ilbk->', tmp, tmp2)

        tmp  = contract('kc,ljac->ljak', X1_A, ERI[o,o,v,v])
        tmp  = contract('ijab,ljak->ilbk', X2_C, tmp)
        tmp2 = contract('ikdb,ld->ilbk', l2, X1_B)
        self.G += contract('ilbk,ilbk->', tmp, tmp2)

        tmp = contract('ld,jkad->jkal', X1_B, ERI[o,o,v,v])
        tmp = contract('ijab,jkal->ikbl', X2_C, tmp)
        tmp2 = contract('kc,ilbc->ilbk', X1_A, l2)
        self.G += contract('ikbl,ilbk->', tmp, tmp2)

        tmp = contract('ld,kjad->kjal', X1_B, ERI[o,o,v,v])
        tmp = contract('ijab,kjal->iklb', X2_C, tmp)
        tmp2 = contract('kc,ilcb->ilkb', X1_A, l2)
        self.G += contract('iklb,ilkb->', tmp, tmp2)

        tmp = contract('kc,ijcd->ijkd', X1_A, ERI[o,o,v,v])
        tmp = contract('ld,ijkd->ijkl', X1_B, tmp)
        tmp2 = contract('ijab,klab->ijkl', X2_C, l2)
        self.G += contract('ijkl,ijkl->', tmp, tmp2)

        self.hyper += self.G #good

        self.Bcon1 = 0
        # <O|L1(A)[[Hbar(0),X1(B),X1(C)]]|0>
        tmp  = -1.0* contract('jc,kb->jkcb', hbar.Hov, Y1_A)
        tmp -= contract('jc,kb->jkcb', Y1_A, hbar.Hov)

        #swapaxes
        tmp -= 2.0* contract('kjib,ic->jkcb', hbar.Hooov, Y1_A)
        tmp += contract('jkib,ic->jkcb', hbar.Hooov, Y1_A)

        #swapaxes
        tmp -= 2.0* contract('jkic,ib->jkcb', hbar.Hooov, Y1_A)
        tmp += np.einsum('kjic,ib->jkcb', hbar.Hooov, Y1_A)

        # swapaxes 
        tmp += 2.0* contract('ajcb,ka->jkcb', hbar.Hvovv, Y1_A)
        tmp -= contract('ajbc,ka->jkcb', hbar.Hvovv, Y1_A)

        # swapaxes
        tmp += 2.0* contract('akbc,ja->jkcb', hbar.Hvovv, Y1_A)
        tmp -= contract('akcb,ja->jkcb', hbar.Hvovv, Y1_A)

        tmp2 = contract('miae,me->ia', tmp, X1_B)
        self.Bcon1 += contract('ia,ia->', tmp2, X1_C)

        # <O|L2(A)|[[Hbar(0),X1(B)],X1(C)]|0>
        tmp   = -1.0* contract('janc,nkba->jckb', hbar.Hovov, Y2_A)
        tmp  -= contract('kanb,njca->jckb', hbar.Hovov, Y2_A)
        tmp  -= contract('jacn,nkab->jckb', hbar.Hovvo, Y2_A)
        tmp  -= contract('kabn,njac->jckb', hbar.Hovvo, Y2_A)
        tmp  += 0.5* contract('fabc,jkfa->jckb', hbar.Hvvvv, Y2_A)
        tmp  += 0.5* contract('facb,kjfa->jckb', hbar.Hvvvv, Y2_A)
        tmp  += 0.5* contract('kjin,nibc->jckb', hbar.Hoooo, Y2_A)
        tmp  += 0.5* contract('jkin,nicb->jckb', hbar.Hoooo, Y2_A)
        tmp2 = contract('iema,me->ia', tmp, X1_B)
        self.Bcon1 += contract('ia,ia->', tmp2, X1_C)

        tmp = contract('ijab,ijdb->ad', t2, Y2_A)
        tmp = contract('ld,ad->la', X1_C, tmp)
        tmp = contract('la,klca->kc', tmp, L[o,o,v,v])
        self.Bcon1 -= contract('kc,kc->', tmp, X1_B)

        tmp = contract('ijab,jlba->il', t2, Y2_A)
        tmp2 = contract('kc,kicd->id', X1_B, L[o,o,v,v])
        tmp2 = contract('id,ld->il', tmp2, X1_C)
        self.Bcon1 -= contract('il,il->', tmp2, tmp)

        tmp = contract('ijab,jkba->ik', t2, Y2_A)
        tmp2 = contract('ld,lidc->ic', X1_C, L[o,o,v,v])
        tmp2 = contract('ic,kc->ik', tmp2, X1_B)
        self.Bcon1 -= contract('ik,ik->', tmp2, tmp)

        tmp = contract('ijab,ijcb->ac', t2, Y2_A)
        tmp = contract('kc,ac->ka', X1_B, tmp)
        tmp2 = contract('ld,lkda->ka', X1_C, L[o,o,v,v])
        self.Bcon1 -= contract('ka,ka->', tmp2, tmp)

        # <O|L2(A)[[Hbar(0),X2(B)],X2(C)]|0>
        tmp = contract("klcd,ijcd->ijkl", X2_C, Y2_A)
        tmp = contract("ijkl,ijab->klab", tmp, X2_B)
        self.Bcon1 += 0.5* contract('klab,klab->', tmp, ERI[o,o,v,v])

        tmp = contract("ijab,ikbd->jkad", X2_B, Y2_A)
        tmp = contract("jkad,klcd->jlac", tmp, X2_C)
        self.Bcon1 += contract('jlac,jlac->',tmp, ERI[o,o,v,v])

        tmp = contract("klcd,ikdb->licb", X2_C, Y2_A)
        tmp = contract("licb,ijab->ljca", tmp, X2_B)
        self.Bcon1 += contract('ljca,ljac->', tmp, ERI[o,o,v,v])

        tmp = contract("ijab,klab->ijkl", X2_B, Y2_A)
        tmp = contract("ijkl,klcd->ijcd", tmp, X2_C)
        self.Bcon1 += 0.5* contract('ijcd,ijcd->',tmp, ERI[o,o,v,v])

        tmp = contract("ijab,ijac->bc", X2_B, L[o,o,v,v])
        tmp = contract("bc,klcd->klbd", tmp, X2_C)
        self.Bcon1 -= contract("klbd,klbd->", tmp, Y2_A)
        tmp = contract("ijab,ikab->jk", X2_B, L[o,o,v,v])
        tmp = contract("jk,klcd->jlcd", tmp, X2_C)
        self.Bcon1 -= contract("jlcd,jlcd->", tmp, Y2_A)
        tmp = contract("ikbc,klcd->ilbd", L[o,o,v,v], X2_C)
        tmp = contract("ilbd,ijab->jlad", tmp, X2_B)
        self.Bcon1 -= contract("jlad,jlad->", tmp, Y2_A)
        tmp = contract("ijab,jlbc->ilac", X2_B, Y2_A)
        tmp = contract("ilac,klcd->ikad", tmp, X2_C)
        self.Bcon1 -= contract("ikad,ikad->", tmp, L[o,o,v,v])
        tmp = contract("klca,klcd->ad", L[o,o,v,v], X2_C)
        tmp = contract("ad,ijdb->ijab", tmp, Y2_A)
        self.Bcon1 -= contract("ijab,ijab->", tmp, X2_B)
        tmp = contract("kicd,klcd->il",L[o,o,v,v], X2_C)
        tmp = contract("ijab,il->ljab", X2_B, tmp)
        self.Bcon1 -= contract("ljab,ljab->", tmp, Y2_A)

        tmp = contract("klcd,ikac->lida", X2_C, Y2_A)
        tmp = contract("lida,jlbd->ijab", tmp, L[o,o,v,v])
        self.Bcon1 += 2.0* contract("ijab,ijab->", tmp, X2_B)

        # <O|L1(A)[[Hbar(0),X1(B)],X2(C)]]|0>
        tmp  = 2.0* contract("jkbc,kc->jb", X2_C, Y1_A)
        tmp -= contract("jkcb,kc->jb", X2_C, Y1_A)
        tmp = contract('ijab,jb->ia', L[o,o,v,v], tmp)
        self.Bcon1 += contract("ia,ia->", tmp, X1_B)

        tmp = contract("jkbc,jkba->ca", X2_C, L[o,o,v,v])
        tmp = contract("ia,ca->ic", X1_B, tmp)
        self.Bcon1 -= contract("ic,ic->", tmp, Y1_A)

        tmp = contract("jkbc,jibc->ki", X2_C, L[o,o,v,v])
        tmp = contract("ki,ia->ka", tmp, X1_B)
        self.Bcon1 -= contract("ka,ka->", tmp, Y1_A)

        # <O|L2(A)[[Hbar(0),X1(B)],X2(C)]]|0>
        tmp = contract("klcd,lkdb->cb", X2_C, Y2_A)
        tmp = contract("jb,cb->jc", X1_B, tmp)
        self.Bcon1 -= contract("jc,jc->", tmp, hbar.Hov)

        tmp = contract("klcd,ljdc->kj", X2_C, Y2_A)
        tmp = contract("kj,jb->kb", tmp, X1_B)
        self.Bcon1 -= contract("kb,kb->", tmp, hbar.Hov)

        tmp = contract('lkda,klcd->ac', Y2_A, X2_C)
        tmp2 = contract('jb,ajcb->ac', X1_B, hbar.Hvovv)
        self.Bcon1 += 2.0* contract('ac,ac->', tmp, tmp2)

        tmp = contract('lkda,klcd->ac', Y2_A, X2_C)
        tmp2 = contract('jb,ajbc->ac', X1_B, hbar.Hvovv)
        self.Bcon1 -= contract('ac,ac->', tmp, tmp2)

        tmp = contract('jb,ljda->lbda', X1_B, Y2_A)
        tmp2 = 2.0* contract('klcd,akbc->ldab', X2_C, hbar.Hvovv)
        tmp2 -= contract('klcd,akcb->ldab', X2_C, hbar.Hvovv)
        self.Bcon1 += contract('lbda,ldab->', tmp, tmp2)

        tmp = contract('ia,fkba->fkbi', X1_B, hbar.Hvovv)
        tmp = contract('fkbi,jifc->kjbc', tmp, Y2_A)
        self.Bcon1 -= contract('jkbc,kjbc->', X2_C, tmp)

        tmp = contract('ia,fjac->fjic', X1_B, hbar.Hvovv)
        tmp = contract('fjic,ikfb->jkbc', tmp, Y2_A)
        self.Bcon1 -= contract('jkbc,jkbc->', X2_C, tmp)

        tmp = contract('ia,jkfa->jkfi', X1_B, Y2_A)
        tmp2 = contract('jkbc,fibc->jkfi', X2_C, hbar.Hvovv)
        self.Bcon1 -= contract('jkfi,jkfi->', tmp2, tmp)

        tmp = contract('jb,kjib->ki', X1_B, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_C, Y2_A)
        self.Bcon1 -= 2.0*contract('ki,ki->', tmp, tmp2)

        tmp = contract('jb,jkib->ki', X1_B, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_C, Y2_A)
        self.Bcon1 += contract('ki,ki->', tmp, tmp2)

        tmp  = 2.0* contract('jkic,klcd->jild', hbar.Hooov, X2_C)
        tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_C)
        tmp  = contract('jild,jb->bild', tmp, X1_B)
        self.Bcon1 -= contract('bild,ilbd->', tmp, Y2_A)

        tmp  = contract('ia,jkna->jkni', X1_B, hbar.Hooov)
        tmp2  = contract('jkbc,nibc->jkni', X2_C, Y2_A)
        self.Bcon1 += contract('jkni,jkni->', tmp2, tmp)

        tmp  = contract('ia,nkab->nkib', X1_B, Y2_A)
        tmp  = contract('jkbc,nkib->jnic', X2_C, tmp)
        self.Bcon1 += contract('jnic,ijnc->', tmp, hbar.Hooov)

        tmp  = contract('ia,nkba->nkbi', X1_B, Y2_A)
        tmp  = contract('jkbc,nkbi->jnci', X2_C, tmp)
        self.Bcon1 += contract('jnci,jinc->', tmp, hbar.Hooov)

        # <O|L1(A)[[Hbar(0),X2(B)],X1(C)]]|0>
        #swapaxes
        tmp  = 2.0* contract("jkbc,kc->jb", X2_B, Y1_A)
        tmp -= contract("jkcb,kc->jb", X2_B, Y1_A)
        tmp = contract('ijab,jb->ia', L[o,o,v,v], tmp)
        self.Bcon1 += contract("ia,ia->", tmp, X1_C)

        tmp = contract("jkbc,jkba->ca", X2_B, L[o,o,v,v])
        tmp = contract("ia,ca->ic", X1_C, tmp)
        self.Bcon1 -= contract("ic,ic->", tmp, Y1_A)

        tmp = contract("jkbc,jibc->ki", X2_B, L[o,o,v,v])
        tmp = contract("ki,ia->ka", tmp, X1_C)
        self.Bcon1 -= contract("ka,ka->", tmp, Y1_A)

        # <O|L2(A)[[Hbar(0),X2(B)],X1(C)]]|0>
        tmp = contract("klcd,lkdb->cb", X2_B, Y2_A)
        tmp = contract("jb,cb->jc", X1_C, tmp)
        self.Bcon1 -= contract("jc,jc->", tmp, hbar.Hov)

        tmp = contract("klcd,ljdc->kj", X2_B, Y2_A)
        tmp = contract("kj,jb->kb", tmp, X1_C)
        self.Bcon1 -= contract("kb,kb->", tmp, hbar.Hov)

        tmp = contract('lkda,klcd->ac', Y2_A, X2_B)
        tmp2 = contract('jb,ajcb->ac', X1_C, hbar.Hvovv)
        self.Bcon1 += 2.0* contract('ac,ac->', tmp, tmp2)

        tmp = contract('lkda,klcd->ac', Y2_A, X2_B)
        tmp2 = contract('jb,ajbc->ac', X1_C, hbar.Hvovv)
        self.Bcon1 -= contract('ac,ac->', tmp, tmp2)

        tmp = contract('jb,ljda->lbda', X1_C, Y2_A)

        #swapaxes
        tmp2 = 2.0* contract('klcd,akbc->ldab', X2_B, hbar.Hvovv)
        tmp2 -= contract('klcd,akcb->ldab', X2_B, hbar.Hvovv)
        self.Bcon1 += contract('lbda,ldab->', tmp, tmp2)

        tmp = contract('ia,fkba->fkbi', X1_C, hbar.Hvovv)
        tmp = contract('fkbi,jifc->kjbc', tmp, Y2_A)
        self.Bcon1 -= contract('jkbc,kjbc->', X2_B, tmp)

        tmp = contract('ia,fjac->fjic', X1_C, hbar.Hvovv)
        tmp = contract('fjic,ikfb->jkbc', tmp, Y2_A)
        self.Bcon1 -= contract('jkbc,jkbc->', X2_B, tmp)

        tmp = contract('ia,jkfa->jkfi', X1_C, Y2_A)
        tmp2 = contract('jkbc,fibc->jkfi', X2_B, hbar.Hvovv)
        self.Bcon1 -= contract('jkfi,jkfi->', tmp2, tmp)

        tmp = contract('jb,kjib->ki', X1_C, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_B, Y2_A)
        self.Bcon1 -= 2.0* contract('ki,ki->', tmp, tmp2)

        tmp = contract('jb,jkib->ki', X1_C, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_B, Y2_A)
        self.Bcon1 += contract('ki,ki->', tmp, tmp2)

        tmp  = 2.0* contract('jkic,klcd->jild', hbar.Hooov, X2_B)
        tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_B)
        tmp  = contract('jild,jb->bild', tmp, X1_C)
        self.Bcon1 -= contract('bild,ilbd->', tmp, Y2_A)

        tmp  = contract('ia,jkna->jkni', X1_C, hbar.Hooov)
        tmp2  = contract('jkbc,nibc->jkni', X2_B, Y2_A)
        self.Bcon1 += contract('jkni,jkni->', tmp2, tmp)

        tmp  = contract('ia,nkab->nkib', X1_C, Y2_A)
        tmp  = contract('jkbc,nkib->jnic', X2_B, tmp)
        self.Bcon1 += contract('jnic,ijnc->', tmp, hbar.Hooov)

        tmp  = contract('ia,nkba->nkbi', X1_C, Y2_A)
        tmp  = contract('jkbc,nkbi->jnci', X2_B, tmp)
        self.Bcon1 += contract('jnci,jinc->', tmp, hbar.Hooov)

        self.Bcon2 = 0
        # <O|L1(B)[[Hbar(0),X1(A),X1(C)]]|0>
        tmp  = -1.0* contract('jc,kb->jkcb', hbar.Hov, Y1_B)
        tmp -= contract('jc,kb->jkcb', Y1_B, hbar.Hov)

        #swapaxes
        tmp -= 2.0* contract('kjib,ic->jkcb',hbar.Hooov, Y1_B)
        tmp += contract('jkib,ic->jkcb', hbar.Hooov, Y1_B)
       
        #swapaxes
        tmp -= 2.0* contract('jkic,ib->jkcb', hbar.Hooov, Y1_B)
        tmp += contract('kjic,ib->jkcb', hbar.Hooov, Y1_B)

        tmp += 2.0* contract('ajcb,ka->jkcb', hbar.Hvovv, Y1_B)
        tmp -= contract('ajbc,ka->jkcb', hbar.Hvovv, Y1_B)
        tmp += 2.0* contract('akbc,ja->jkcb', hbar.Hvovv, Y1_B)
        tmp -= contract('akcb,ja->jkcb', hbar.Hvovv, Y1_B)

        tmp2 = contract('miae,me->ia', tmp, X1_A)
        self.Bcon2 += contract('ia,ia->', tmp2, X1_C)

        # <O|L2(B)|[[Hbar(0),X1(A)],X1(C)]|0>
        tmp   = -1.0* contract('janc,nkba->jckb', hbar.Hovov, Y2_B)
        tmp  -= contract('kanb,njca->jckb', hbar.Hovov, Y2_B)
        tmp  -= contract('jacn,nkab->jckb', hbar.Hovvo, Y2_B)
        tmp  -= contract('kabn,njac->jckb', hbar.Hovvo, Y2_B)

        # swapaxes?
        tmp  += 0.5*contract('fabc,jkfa->jckb', hbar.Hvvvv, Y2_B)
        tmp  += 0.5*contract('facb,kjfa->jckb', hbar.Hvvvv, Y2_B)

        # swapaxes?
        tmp  += 0.5* contract('kjin,nibc->jckb', hbar.Hoooo, Y2_B)
        tmp  += 0.5* contract('jkin,nicb->jckb', hbar.Hoooo, Y2_B)
        tmp2 = contract('iema,me->ia', tmp, X1_A)
        self.Bcon2 += contract('ia,ia->', tmp2, X1_C)

        tmp = contract('ijab,ijdb->ad', t2, Y2_B)
        tmp = contract('ld,ad->la', X1_C, tmp)
        tmp = contract('la,klca->kc', tmp, L[o,o,v,v])
        self.Bcon2 -= contract('kc,kc->', tmp, X1_A)

        tmp = contract('ijab,jlba->il', t2, Y2_B)
        tmp2 = contract('kc,kicd->id', X1_A, L[o,o,v,v])
        tmp2 = contract('id,ld->il', tmp2, X1_C)
        self.Bcon2 -= contract('il,il->', tmp2, tmp)

        tmp = contract('ijab,jkba->ik', t2, Y2_B)
        tmp2 = contract('ld,lidc->ic', X1_C, L[o,o,v,v])
        tmp2 = contract('ic,kc->ik', tmp2, X1_A)
        self.Bcon2 -= contract('ik,ik->', tmp2, tmp)

        tmp = contract('ijab,ijcb->ac', t2, Y2_B)
        tmp = contract('kc,ac->ka', X1_A, tmp)
        tmp2 = contract('ld,lkda->ka', X1_C, L[o,o,v,v])
        self.Bcon2 -= contract('ka,ka->', tmp2, tmp)

        # <O|L2(B)[[Hbar(0),X2(A)],X2(C)]|0>
        tmp = contract("klcd,ijcd->ijkl", X2_C, Y2_B)
        tmp = contract("ijkl,ijab->klab", tmp, X2_A)
        self.Bcon2 += 0.5* contract('klab,klab->', tmp, ERI[o,o,v,v])

        tmp = contract("ijab,ikbd->jkad", X2_A, Y2_B)
        tmp = contract("jkad,klcd->jlac", tmp, X2_C)
        self.Bcon2 += contract('jlac,jlac->', tmp, ERI[o,o,v,v])

        tmp = contract("klcd,ikdb->licb", X2_C, Y2_B)
        tmp = contract("licb,ijab->ljca", tmp, X2_A)
        self.Bcon2 += contract('ljca,ljac->', tmp, ERI[o,o,v,v])

        tmp = contract("ijab,klab->ijkl", X2_A, Y2_B)
        tmp = contract("ijkl,klcd->ijcd", tmp, X2_C)
        self.Bcon2 += 0.5* contract('ijcd,ijcd->', tmp, ERI[o,o,v,v])

        tmp = contract("ijab,ijac->bc", X2_A, L[o,o,v,v])
        tmp = contract("bc,klcd->klbd", tmp, X2_C)
        self.Bcon2 -= contract("klbd,klbd->", tmp, Y2_B)
        tmp = contract("ijab,ikab->jk", X2_A, L[o,o,v,v])
        tmp = contract("jk,klcd->jlcd", tmp, X2_C)
        self.Bcon2 -= contract("jlcd,jlcd->", tmp, Y2_B)
        tmp = contract("ikbc,klcd->ilbd", L[o,o,v,v], X2_C)
        tmp = contract("ilbd,ijab->jlad", tmp, X2_A)
        self.Bcon2 -= contract("jlad,jlad->", tmp, Y2_B)
        tmp = contract("ijab,jlbc->ilac", X2_A, Y2_B)
        tmp = contract("ilac,klcd->ikad", tmp, X2_C)
        self.Bcon2 -= contract("ikad,ikad->", tmp, L[o,o,v,v])
        tmp = contract("klca,klcd->ad", L[o,o,v,v], X2_C)
        tmp = contract("ad,ijdb->ijab", tmp, Y2_B)
        self.Bcon2 -= contract("ijab,ijab->", tmp, X2_A)
        tmp = contract("kicd,klcd->il", L[o,o,v,v], X2_C)
        tmp = contract("ijab,il->ljab", X2_A, tmp)
        self.Bcon2 -= contract("ljab,ljab->", tmp, Y2_B)

        tmp = contract("klcd,ikac->lida", X2_C, Y2_B)
        tmp = contract("lida,jlbd->ijab", tmp, L[o,o,v,v])
        self.Bcon2 += 2.0* contract("ijab,ijab->", tmp, X2_A)

        # <O|L1(B)[[Hbar(0),X1(A)],X2(C)]]|0>
        #swapaxes
        tmp = 2.0* contract("jkbc,kc->jb", X2_C, Y1_B)
        tmp -= contract("jkcb,kc->jb", X2_C, Y1_B)
        tmp = contract('ijab,jb->ia', L[o,o,v,v], tmp)
        self.Bcon2 += contract("ia,ia->", tmp, X1_A)

        tmp = contract("jkbc,jkba->ca", X2_C, L[o,o,v,v])
        tmp = contract("ia,ca->ic", X1_A, tmp)
        self.Bcon2 -= contract("ic,ic->", tmp, Y1_B)

        tmp = contract("jkbc,jibc->ki", X2_C, L[o,o,v,v])
        tmp = contract("ki,ia->ka", tmp, X1_A)
        self.Bcon2 -= contract("ka,ka->", tmp, Y1_B)

        # <O|L2(B)[[Hbar(0),X1(A)],X2(C)]]|0>
        tmp = contract("klcd,lkdb->cb", X2_C, Y2_B)
        tmp = contract("jb,cb->jc", X1_A, tmp)
        self.Bcon2 -= contract("jc,jc->", tmp, hbar.Hov)

        tmp = contract("klcd,ljdc->kj", X2_C, Y2_B)
        tmp = contract("kj,jb->kb", tmp, X1_A)
        self.Bcon2 -= contract("kb,kb->", tmp, hbar.Hov)

        tmp = contract('lkda,klcd->ac', Y2_B, X2_C)
        tmp2 = contract('jb,ajcb->ac', X1_A, hbar.Hvovv)
        self.Bcon2 += 2.0* contract('ac,ac->', tmp, tmp2)

        tmp = contract('lkda,klcd->ac', Y2_B, X2_C)
        tmp2 = contract('jb,ajbc->ac', X1_A, hbar.Hvovv)
        self.Bcon2 -= contract('ac,ac->', tmp, tmp2)

        tmp = contract('jb,ljda->lbda', X1_A, Y2_B)

        #swapaxes
        tmp2 = 2.0* contract('klcd,akbc->ldab', X2_C, hbar.Hvovv)
        tmp2 -= contract('klcd,akcb->ldab', X2_C, hbar.Hvovv)
        self.Bcon2 += contract('lbda,ldab->', tmp, tmp2)

        tmp = contract('ia,fkba->fkbi', X1_A, hbar.Hvovv)
        tmp = contract('fkbi,jifc->kjbc', tmp, Y2_B)
        self.Bcon2 -= contract('jkbc,kjbc->', X2_C, tmp)

        tmp = contract('ia,fjac->fjic', X1_A, hbar.Hvovv)
        tmp = contract('fjic,ikfb->jkbc', tmp, Y2_B)
        self.Bcon2 -= contract('jkbc,jkbc->', X2_C, tmp)

        tmp = contract('ia,jkfa->jkfi', X1_A, Y2_B)
        tmp2 = contract('jkbc,fibc->jkfi', X2_C, hbar.Hvovv)
        self.Bcon2 -= contract('jkfi,jkfi->', tmp2, tmp)

        tmp = contract('jb,kjib->ki', X1_A, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_C, Y2_B)
        self.Bcon2 -= 2.0* contract('ki,ki->', tmp, tmp2)

        tmp = contract('jb,jkib->ki', X1_A, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_C, Y2_B)
        self.Bcon2 += contract('ki,ki->', tmp, tmp2)

        #swapaxes
        tmp  = 2.0* contract('jkic,klcd->jild', hbar.Hooov, X2_C)
        tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_C)
        tmp  = contract('jild,jb->bild', tmp, X1_A)
        self.Bcon2 -= contract('bild,ilbd->', tmp, Y2_B)

        tmp  = contract('ia,jkna->jkni', X1_A, hbar.Hooov)
        tmp2  = contract('jkbc,nibc->jkni', X2_C, Y2_B)
        self.Bcon2 += contract('jkni,jkni->', tmp2, tmp)

        tmp  = contract('ia,nkab->nkib', X1_A, Y2_B)
        tmp  = contract('jkbc,nkib->jnic', X2_C, tmp)
        self.Bcon2 += contract('jnic,ijnc->', tmp, hbar.Hooov)

        tmp  = contract('ia,nkba->nkbi', X1_A, Y2_B)
        tmp  = contract('jkbc,nkbi->jnci', X2_C,tmp)
        self.Bcon2 += contract('jnci,jinc->', tmp, hbar.Hooov)

        # <O|L1(B)[[Hbar(0),X2(A)],X1(C)]]|0>
        # swapaxes
        tmp  = 2.0* contract("jkbc,kc->jb", X2_A, Y1_B)
        tmp -= contract("jkcb,kc->jb", X2_A, Y1_B)
        tmp = contract('ijab,jb->ia', L[o,o,v,v], tmp)
        self.Bcon2 += contract("ia,ia->", tmp, X1_C)

        tmp = contract("jkbc,jkba->ca", X2_A, L[o,o,v,v])
        tmp = contract("ia,ca->ic", X1_C, tmp)
        self.Bcon2 -= contract("ic,ic->", tmp, Y1_B)

        tmp = contract("jkbc,jibc->ki", X2_A, L[o,o,v,v])
        tmp = contract("ki,ia->ka", tmp, X1_C)
        self.Bcon2 -= contract("ka,ka->", tmp, Y1_B)

        # <O|L2(B)[[Hbar(0),X2(A)],X1(C)]]|0>
        tmp = contract("klcd,lkdb->cb", X2_A, Y2_B)
        tmp = contract("jb,cb->jc", X1_C, tmp)
        self.Bcon2 -= contract("jc,jc->", tmp, hbar.Hov)

        tmp = contract("klcd,ljdc->kj", X2_A, Y2_B)
        tmp = contract("kj,jb->kb", tmp, X1_C)
        self.Bcon2 -= contract("kb,kb->",tmp, hbar.Hov)

        tmp = contract('lkda,klcd->ac', Y2_B, X2_A)
        tmp2 = contract('jb,ajcb->ac', X1_C, hbar.Hvovv)
        self.Bcon2 += 2.0* contract('ac,ac->', tmp, tmp2)

        tmp = contract('lkda,klcd->ac', Y2_B, X2_A)
        tmp2 = contract('jb,ajbc->ac', X1_C, hbar.Hvovv)
        self.Bcon2 -= contract('ac,ac->', tmp, tmp2)

        tmp = contract('jb,ljda->lbda', X1_C, Y2_B)

        #swapaxes
        tmp2 = 2.0* contract('klcd,akbc->ldab', X2_A, hbar.Hvovv)
        tmp2 -= contract('klcd,akcb->ldab', X2_A, hbar.Hvovv)
        self.Bcon2 += contract('lbda,ldab->', tmp, tmp2)

        tmp = contract('ia,fkba->fkbi', X1_C, hbar.Hvovv)
        tmp = contract('fkbi,jifc->kjbc', tmp, Y2_B)
        self.Bcon2 -= contract('jkbc,kjbc->', X2_A, tmp)

        tmp = contract('ia,fjac->fjic', X1_C, hbar.Hvovv)
        tmp = contract('fjic,ikfb->jkbc', tmp, Y2_B)
        self.Bcon2 -= contract('jkbc,jkbc->', X2_A, tmp)

        tmp = contract('ia,jkfa->jkfi', X1_C, Y2_B)
        tmp2 = contract('jkbc,fibc->jkfi', X2_A, hbar.Hvovv)
        self.Bcon2 -= contract('jkfi,jkfi->', tmp2, tmp)

        tmp = contract('jb,kjib->ki', X1_C, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_A, Y2_B)
        self.Bcon2 -= 2.0* contract('ki,ki->', tmp, tmp2)

        tmp = contract('jb,jkib->ki', X1_C, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_A, Y2_B)
        self.Bcon2 += contract('ki,ki->', tmp, tmp2)

        tmp  = 2.0* contract('jkic,klcd->jild', hbar.Hooov, X2_A)
        tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_A)
        tmp  = contract('jild,jb->bild', tmp, X1_C)
        self.Bcon2 -= contract('bild,ilbd->', tmp, Y2_B)

        tmp  = contract('ia,jkna->jkni', X1_C, hbar.Hooov)
        tmp2  = contract('jkbc,nibc->jkni', X2_A, Y2_B)
        self.Bcon2 += contract('jkni,jkni->', tmp2, tmp)

        tmp  = contract('ia,nkab->nkib', X1_C, Y2_B)
        tmp  = contract('jkbc,nkib->jnic', X2_A, tmp)
        self.Bcon2 += contract('jnic,ijnc->', tmp, hbar.Hooov)

        tmp  = contract('ia,nkba->nkbi', X1_C, Y2_B)
        tmp  = contract('jkbc,nkbi->jnci', X2_A, tmp)
        self.Bcon2 += contract('jnci,jinc->', tmp, hbar.Hooov)

        self.Bcon3 = 0
        # <0|L1(C)[[Hbar(0),X1(A),X1(B)]]|0>
        tmp  = -1.0* contract('jc,kb->jkcb', hbar.Hov, Y1_C)
        tmp -= contract('jc,kb->jkcb', Y1_C, hbar.Hov)

        #swapaxes
        tmp -= 2.0* contract('kjib,ic->jkcb', hbar.Hooov, Y1_C)
        tmp += contract('jkib,ic->jkcb', hbar.Hooov, Y1_C)

        #swapaxes
        tmp -= 2.0* contract('jkic,ib->jkcb', hbar.Hooov, Y1_C)
        tmp += contract('kjic,ib->jkcb', hbar.Hooov, Y1_C)

        #swapaxes
        tmp += 2.0* contract('ajcb,ka->jkcb', hbar.Hvovv, Y1_C)
        tmp -= contract('ajbc,ka->jkcb', hbar.Hvovv, Y1_C)
        tmp += 2.0* contract('akbc,ja->jkcb', hbar.Hvovv, Y1_C)
        tmp -= contract('akcb,ja->jkcb', hbar.Hvovv, Y1_C)

        tmp2 = contract('miae,me->ia', tmp, X1_A)
        self.Bcon3 += contract('ia,ia->', tmp2, X1_B)

        # <0|L2(C)|[[Hbar(0),X1(A)],X1(B)]|0>
        tmp   = -1.0* contract('janc,nkba->jckb', hbar.Hovov, Y2_C)
        tmp  -= contract('kanb,njca->jckb', hbar.Hovov, Y2_C)
        tmp  -= contract('jacn,nkab->jckb', hbar.Hovvo, Y2_C)
        tmp  -= contract('kabn,njac->jckb', hbar.Hovvo, Y2_C)

        #swapaxes?
        tmp  += 0.5* contract('fabc,jkfa->jckb', hbar.Hvvvv, Y2_C)
        tmp  += 0.5* contract('facb,kjfa->jckb', hbar.Hvvvv, Y2_C)

        #swapaxes?
        tmp  += 0.5* contract('kjin,nibc->jckb', hbar.Hoooo, Y2_C)
        tmp  += 0.5* contract('jkin,nicb->jckb', hbar.Hoooo, Y2_C)
        tmp2 = contract('iema,me->ia', tmp, X1_A)
        self.Bcon3 += contract('ia,ia->', tmp2, X1_B)

        tmp = contract('ijab,ijdb->ad', t2, Y2_C)
        tmp = contract('ld,ad->la', X1_B, tmp)
        tmp = contract('la,klca->kc', tmp, L[o,o,v,v])
        self.Bcon3 -= contract('kc,kc->',tmp, X1_A)

        tmp = contract('ijab,jlba->il', t2, Y2_C)
        tmp2 = contract('kc,kicd->id', X1_A, L[o,o,v,v])
        tmp2 = contract('id,ld->il', tmp2, X1_B)
        self.Bcon3 -= contract('il,il->', tmp2, tmp)

        tmp = contract('ijab,jkba->ik', t2, Y2_C)
        tmp2 = contract('ld,lidc->ic', X1_B, L[o,o,v,v])
        tmp2 = contract('ic,kc->ik', tmp2, X1_A)
        self.Bcon3 -= contract('ik,ik->', tmp2, tmp)

        tmp = contract('ijab,ijcb->ac', t2, Y2_C)
        tmp = contract('kc,ac->ka', X1_A, tmp)
        tmp2 = contract('ld,lkda->ka', X1_B, L[o,o,v,v])
        self.Bcon3 -= contract('ka,ka->', tmp2, tmp)

        # <0|L2(C)[[Hbar(0),X2(A)],X2(B)]|0>
        tmp = contract("klcd,ijcd->ijkl", X2_B, Y2_C)
        tmp = contract("ijkl,ijab->klab", tmp, X2_A)
        self.Bcon3 += 0.5* contract('klab,klab->', tmp, ERI[o,o,v,v])

        tmp = contract("ijab,ikbd->jkad", X2_A, Y2_C)
        tmp = contract("jkad,klcd->jlac", tmp, X2_B)
        self.Bcon3 += contract('jlac,jlac->', tmp, ERI[o,o,v,v])

        tmp = contract("klcd,ikdb->licb", X2_B, Y2_C)
        tmp = contract("licb,ijab->ljca", tmp, X2_A)
        self.Bcon3 += contract('ljca,ljac->', tmp, ERI[o,o,v,v])

        tmp = contract("ijab,klab->ijkl", X2_A, Y2_C)
        tmp = contract("ijkl,klcd->ijcd", tmp, X2_B)
        self.Bcon3 += 0.5* contract('ijcd,ijcd->', tmp, ERI[o,o,v,v])

        tmp = contract("ijab,ijac->bc", X2_A, L[o,o,v,v])
        tmp = contract("bc,klcd->klbd", tmp, X2_B)
        self.Bcon3 -= contract("klbd,klbd->", tmp, Y2_C)
        tmp = contract("ijab,ikab->jk", X2_A, L[o,o,v,v])
        tmp = contract("jk,klcd->jlcd", tmp, X2_B)
        self.Bcon3 -= contract("jlcd,jlcd->", tmp, Y2_C)
        tmp = contract("ikbc,klcd->ilbd", L[o,o,v,v], X2_B)
        tmp = contract("ilbd,ijab->jlad", tmp, X2_A)
        self.Bcon3 -= contract("jlad,jlad->", tmp, Y2_C)
        tmp = contract("ijab,jlbc->ilac", X2_A, Y2_C)
        tmp = contract("ilac,klcd->ikad", tmp, X2_B)
        self.Bcon3 -= contract("ikad,ikad->", tmp, L[o,o,v,v])
        tmp = contract("klca,klcd->ad", L[o,o,v,v], X2_B)
        tmp = contract("ad,ijdb->ijab", tmp, Y2_C)
        self.Bcon3 -= contract("ijab,ijab->", tmp, X2_A)
        tmp = contract("kicd,klcd->il", L[o,o,v,v], X2_B)
        tmp = contract("ijab,il->ljab", X2_A, tmp)
        self.Bcon3 -= contract("ljab,ljab->", tmp, Y2_C)

        tmp = contract("klcd,ikac->lida", X2_B, Y2_C)
        tmp = contract("lida,jlbd->ijab", tmp, L[o,o,v,v])
        self.Bcon3 += 2.0* contract("ijab,ijab->", tmp, X2_A)

        # <0|L1(C)[[Hbar(0),X1(A)],X2(B)]]|0>
        #swapaxes
        tmp = 2.0 * contract("jkbc,kc->jb", X2_B, Y1_C)
        tmp -= contract("jkcb,kc->jb", X2_B, Y1_C)
        tmp = contract('ijab,jb->ia', L[o,o,v,v], tmp)
        self.Bcon3 += contract("ia,ia->", tmp, X1_A)

        tmp = contract("jkbc,jkba->ca", X2_B, L[o,o,v,v])
        tmp = contract("ia,ca->ic", X1_A, tmp)
        self.Bcon3 -= contract("ic,ic->", tmp, Y1_C)

        tmp = contract("jkbc,jibc->ki", X2_B, L[o,o,v,v])
        tmp = contract("ki,ia->ka", tmp, X1_A)
        self.Bcon3 -= contract("ka,ka->", tmp, Y1_C)

        # <0|L2(C)[[Hbar(0),X1(A)],X2(B)]]|0>
        tmp = contract("klcd,lkdb->cb", X2_B, Y2_C)
        tmp = contract("jb,cb->jc", X1_A, tmp)
        self.Bcon3 -= contract("jc,jc->", tmp, hbar.Hov)

        tmp = contract("klcd,ljdc->kj", X2_B, Y2_C)
        tmp = contract("kj,jb->kb", tmp, X1_A)
        self.Bcon3 -= contract("kb,kb->", tmp, hbar.Hov)

        tmp = contract('lkda,klcd->ac', Y2_C, X2_B)
        tmp2 = contract('jb,ajcb->ac', X1_A, hbar.Hvovv)
        self.Bcon3 += 2.0* contract('ac,ac->', tmp, tmp2)

        tmp = contract('lkda,klcd->ac', Y2_C, X2_B)
        tmp2 = contract('jb,ajbc->ac', X1_A, hbar.Hvovv)
        self.Bcon3 -= contract('ac,ac->', tmp, tmp2)

        tmp = contract('jb,ljda->lbda', X1_A, Y2_C)

        #swapaxes
        tmp2 = 2.0* contract('klcd,akbc->ldab', X2_B, hbar.Hvovv)
        tmp2 -= contract('klcd,akcb->ldab', X2_B, hbar.Hvovv)
        self.Bcon3 += contract('lbda,ldab->', tmp, tmp2)

        tmp = contract('ia,fkba->fkbi', X1_A, hbar.Hvovv)
        tmp = contract('fkbi,jifc->kjbc', tmp, Y2_C)
        self.Bcon3 -= contract('jkbc,kjbc->', X2_B, tmp)

        tmp = contract('ia,fjac->fjic', X1_A, hbar.Hvovv)
        tmp = contract('fjic,ikfb->jkbc', tmp, Y2_C)
        self.Bcon3 -= contract('jkbc,jkbc->', X2_B, tmp)

        tmp = contract('ia,jkfa->jkfi', X1_A, Y2_C)
        tmp2 = contract('jkbc,fibc->jkfi', X2_B, hbar.Hvovv)
        self.Bcon3 -= contract('jkfi,jkfi->', tmp2, tmp)

        tmp = contract('jb,kjib->ki', X1_A, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_B, Y2_C)
        self.Bcon3 -= 2.0* contract('ki,ki->', tmp, tmp2)

        tmp = contract('jb,jkib->ki', X1_A, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_B, Y2_C)
        self.Bcon3 += contract('ki,ki->', tmp, tmp2)

        #swapaxes
        tmp  = 2.0* contract('jkic,klcd->jild', hbar.Hooov, X2_B)
        tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_B)
        tmp  = contract('jild,jb->bild', tmp, X1_A)
        self.Bcon3 -= contract('bild,ilbd->', tmp, Y2_C)

        tmp  = contract('ia,jkna->jkni', X1_A, hbar.Hooov)
        tmp2  = contract('jkbc,nibc->jkni', X2_B, Y2_C)
        self.Bcon3 += contract('jkni,jkni->', tmp2, tmp)

        tmp  = contract('ia,nkab->nkib', X1_A, Y2_C)
        tmp  = contract('jkbc,nkib->jnic', X2_B, tmp)
        self.Bcon3 += contract('jnic,ijnc->', tmp, hbar.Hooov)

        tmp  = contract('ia,nkba->nkbi', X1_A, Y2_C)
        tmp  = contract('jkbc,nkbi->jnci', X2_B, tmp)
        self.Bcon3 += contract('jnci,jinc->', tmp, hbar.Hooov)

        # <0|L1(C)[[Hbar(0),X2(A)],X1(B)]]|0>
        tmp = 2.0* contract("jkbc,kc->jb", X2_A, Y1_C)
        tmp -= contract("jkcb,kc->jb", X2_A, Y1_C)
        tmp = contract('ijab,jb->ia', L[o,o,v,v], tmp)
        self.Bcon3 += contract("ia,ia->", tmp, X1_B)

        tmp = contract("jkbc,jkba->ca", X2_A, L[o,o,v,v])
        tmp = contract("ia,ca->ic", X1_B, tmp)
        self.Bcon3 -= contract("ic,ic->", tmp, Y1_C)

        tmp = contract("jkbc,jibc->ki", X2_A, L[o,o,v,v])
        tmp = contract("ki,ia->ka", tmp, X1_B)
        self.Bcon3 -= contract("ka,ka->", tmp, Y1_C)

        # <0|L1(C)[[Hbar(0),X2(A)],X1(B)]]|0>
        tmp = contract("klcd,lkdb->cb", X2_A, Y2_C)
        tmp = contract("jb,cb->jc", X1_B, tmp)
        self.Bcon3 -= contract("jc,jc->", tmp, hbar.Hov)

        tmp = contract("klcd,ljdc->kj", X2_A, Y2_C)
        tmp = contract("kj,jb->kb", tmp, X1_B)
        self.Bcon3 -= contract("kb,kb->",tmp, hbar.Hov)

        tmp = contract('lkda,klcd->ac', Y2_C, X2_A)
        tmp2 = contract('jb,ajcb->ac', X1_B, hbar.Hvovv)
        self.Bcon3 += 2.0* contract('ac,ac->', tmp, tmp2)

        tmp = contract('lkda,klcd->ac', Y2_C, X2_A)
        tmp2 = contract('jb,ajbc->ac', X1_B, hbar.Hvovv)
        self.Bcon3 -= contract('ac,ac->', tmp, tmp2)

        tmp = contract('jb,ljda->lbda', X1_B, Y2_C)

        #swapaxes
        tmp2 = 2.0* contract('klcd,akbc->ldab', X2_A, hbar.Hvovv)
        tmp2 -= contract('klcd,akcb->ldab', X2_A, hbar.Hvovv)
        self.Bcon3 += contract('lbda,ldab->', tmp, tmp2)

        tmp = contract('ia,fkba->fkbi', X1_B, hbar.Hvovv)
        tmp = contract('fkbi,jifc->kjbc', tmp, Y2_C)
        self.Bcon3 -= contract('jkbc,kjbc->', X2_A, tmp)

        tmp = contract('ia,fjac->fjic', X1_B, hbar.Hvovv)
        tmp = contract('fjic,ikfb->jkbc', tmp, Y2_C)
        self.Bcon3 -= contract('jkbc,jkbc->', X2_A, tmp)

        tmp = contract('ia,jkfa->jkfi', X1_B, Y2_C)
        tmp2 = contract('jkbc,fibc->jkfi', X2_A, hbar.Hvovv)
        self.Bcon3 -= contract('jkfi,jkfi->', tmp2, tmp)

        tmp = contract('jb,kjib->ki', X1_B, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_A, Y2_C)
        self.Bcon3 -= 2.0*contract('ki,ki->', tmp, tmp2)

        tmp = contract('jb,jkib->ki', X1_B, hbar.Hooov)
        tmp2 = contract('klcd,ilcd->ki', X2_A, Y2_C)
        self.Bcon3 += contract('ki,ki->', tmp, tmp2)

        #swapaxes
        tmp  = 2.0* contract('jkic,klcd->jild', hbar.Hooov, X2_A)
        tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_A)
        tmp  = contract('jild,jb->bild', tmp, X1_B)
        self.Bcon3 -= contract('bild,ilbd->', tmp, Y2_C)

        tmp  = contract('ia,jkna->jkni', X1_B, hbar.Hooov)
        tmp2  = contract('jkbc,nibc->jkni', X2_A, Y2_C)
        self.Bcon3 += contract('jkni,jkni->', tmp2, tmp)

        tmp  = contract('ia,nkab->nkib', X1_B, Y2_C)
        tmp  = contract('jkbc,nkib->jnic', X2_A, tmp)
        self.Bcon3 += contract('jnic,ijnc->', tmp, hbar.Hooov)

        tmp  = contract('ia,nkba->nkbi', X1_B, Y2_C)
        tmp  = contract('jkbc,nkbi->jnci', X2_A, tmp)
        self.Bcon3 += contract('jnci,jinc->', tmp, hbar.Hooov)

        self.hyper += self.Bcon1 + self.Bcon2 + self.Bcon3

        return self.hyper

    def hyperpolar(self):
        solver_start = time.time()
        
        ccpert_om1_X = self.ccpert_om1_X
        ccpert_om2_X = self.ccpert_om2_X
        ccpert_om_sum_X = self.ccpert_om_sum_X

        ccpert_om1_2nd_X = self.ccpert_om1_2nd_X
        ccpert_om2_2nd_X = self.ccpert_om2_2nd_X
        ccpert_om_sum_2nd_X = self.ccpert_om_sum_2nd_X
  
        ccpert_om1_Y = self.ccpert_om1_Y
        ccpert_om2_Y = self.ccpert_om2_Y
        ccpert_om_sum_Y = self.ccpert_om_sum_Y

        ccpert_om1_2nd_Y = self.ccpert_om1_2nd_Y
        ccpert_om2_2nd_Y = self.ccpert_om2_2nd_Y
        ccpert_om_sum_2nd_Y = self.ccpert_om_sum_2nd_Y

        hyper_AB_1st = np.zeros((3,3,3))
        hyper_AB_2nd = np.zeros((3,3,3))
        hyper_AB = np.zeros((3,3,3))

        for a in range(0, 3):
            pertkey_a = "MU_" + self.cart[a]
            for b in range(0, 3):
                pertkey_b = "MU_" + self.cart[b]
                for c in range(0, 3):
                    pertkey_c = "MU_" + self.cart[c]

                    hyper_AB_1st[a,b,c] = self.quadraticresp(pertkey_a, pertkey_b, pertkey_c, ccpert_om_sum_X[pertkey_a], ccpert_om1_X[pertkey_b], ccpert_om2_X[pertkey_c],  ccpert_om_sum_Y[pertkey_a], ccpert_om1_Y[pertkey_b], ccpert_om2_Y[pertkey_c] )
                    hyper_AB_2nd[a,b,c] = self.quadraticresp(pertkey_a, pertkey_b, pertkey_c, ccpert_om_sum_2nd_X[pertkey_a], ccpert_om1_2nd_X[pertkey_b], ccpert_om2_2nd_X[pertkey_c],  ccpert_om_sum_2nd_Y[pertkey_a], ccpert_om1_2nd_Y[pertkey_b], ccpert_om2_2nd_Y[pertkey_c])
                    hyper_AB[a,b,c] = (hyper_AB_1st[a,b,c] + hyper_AB_2nd[a,b,c] )/2

        self.hyper_AB = hyper_AB
    
        print("\Beta_zxx = %10.12lf" %(hyper_AB[2,0,0]))
        print("\Beta_xzx = %10.12lf" %(hyper_AB[0,2,0]))
        print("\Beta_xxz = %10.12lf" %(hyper_AB[0,0,2]))
        print("\Beta_zyy = %10.12lf" %(hyper_AB[2,1,1]))
        print("\Beta_yzy = %10.12lf" %(hyper_AB[1,2,1]))
        print("\Beta_yyz = %10.12lf" %(hyper_AB[1,1,2]))
        print("\Beta_zzz = %10.12lf" %(hyper_AB[2,2,2]))
        print("\n First Dipole Hyperpolarizability computed in %.3f seconds.\n" % (time.time() - solver_start))

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
        solver_start = time.time()

        Dia = self.Dia
        Dijab = self.Dijab

        # initial guess
        X1_guess = pertbar.Avo.T/(Dia + omega)
        X2_guess = pertbar.Avvoo/(Dijab + omega)
        #print("guess X1", X1_guess)
        #print("guess X2", X2_guess)
        #print("X1 used for inital Y1", self.X1)
        #print("X2 used for initial Y2", self.X2)

        # initial guess
        Y1 = 2.0 * X1_guess.copy()
        Y2 = 4.0 * X2_guess.copy()
        Y2 -= 2.0 * X2_guess.copy().swapaxes(2,3)              
        #print("initial Y1", Y1)
        #print("inital Y2", Y2) 
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
        print("Im_y1 density", np.sqrt(np.einsum('ia, ia ->', self.im_Y1, self.im_Y1)))
        #print("im_Y1", self.im_Y1)
        #print("im_Y2", self.im_Y2)  
        for niter in range(1, maxiter+1):
            pseudo_last = pseudo
            
            Y1 = self.Y1
            Y2 = self.Y2
            
            r1 = self.r_Y1(pertbar, omega)
            r2 = self.r_Y2(pertbar, omega)
            
            self.Y1 += r1/(Dia + omega)
            self.Y2 += r2/(Dijab + omega)
            
            rms = contract('ia,ia->', np.conj(r1/(Dia+omega)), r1/(Dia+omega))
            #print("rms for r1/energy density", rms)
            rms += contract('ijab,ijab->', np.conj(r2/(Dijab+omega)), r2/(Dijab+omega))
            rms = np.sqrt(rms)
            
            # need to undertsand this 
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

        # Inhomogenous terms appearing in Y1 equations
        #seems like these imhomogenous terms are computing at the beginning and not involve in the iteration itself
        #may require moving to a sperate function
 
        # <O|A_bar|phi^a_i> good
        r_Y1 = 2.0 * pertbar.Aov.copy()
        # <O|L1(0)|A_bar|phi^a_i> good
        r_Y1 -= contract('im,ma->ia', pertbar.Aoo, l1)
        r_Y1 += contract('ie,ea->ia', l1, pertbar.Avv)
        # <O|L2(0)|A_bar|phi^a_i>
        r_Y1 += contract('imfe,feam->ia', l2, pertbar.Avvvo)
   
        #can combine the next two to swapaxes type contraction
        r_Y1 -= 0.5 * contract('ienm,mnea->ia', pertbar.Aovoo, l2)
        r_Y1 -= 0.5 * contract('iemn,mnae->ia', pertbar.Aovoo, l2)

        # <O|[Hbar(0), X1]|phi^a_i> good
        r_Y1 +=  2.0 * contract('imae,me->ia', L[o,o,v,v], X1)

        # <O|L1(0)|[Hbar(0), X1]|phi^a_i>
        tmp  = -1.0 * contract('ma,ie->miae', hbar.Hov, l1)
        tmp -= contract('ma,ie->miae', l1, hbar.Hov)
        tmp -= 2.0 * contract('mina,ne->miae', hbar.Hooov, l1)

        #double check this one
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
        tmp  = 2.0 * contract('mnef,nf->me', X2, l1)
        tmp  -= contract('mnfe,nf->me', X2, l1)
        r_Y1 += contract('imae,me->ia', L[o,o,v,v], tmp)
        #print("Goo denisty", np.sqrt(np.einsum('ij, ij ->', cclambda.build_Goo(X2, L[o,o,v,v]), cclambda.build_Goo(X2, L[o,o,v,v]))))
        #print("l1 density", np.sqrt(np.einsum('ia, ia ->', l1, l1)))
        r_Y1 -= contract('ni,na->ia', cclambda.build_Goo(X2, L[o,o,v,v]), l1)
        r_Y1 += contract('ie,ea->ia', l1, cclambda.build_Gvv(L[o,o,v,v], X2))

        # <O|L2(0)|[Hbar(0), X1]|phi^a_i> good

        # can reorganize thesenext four to two swapaxes type contraction
        tmp   = -1.0 * contract('nief,mfna->iema', l2, hbar.Hovov)
        tmp  -= contract('ifne,nmaf->iema', hbar.Hovov, l2)
        tmp  -= contract('inef,mfan->iema', l2, hbar.Hovvo)
        tmp  -= contract('ifen,nmfa->iema', hbar.Hovvo, l2)

        #can combine the next two to swapaxes type contraction
        tmp  += 0.5 * contract('imfg,fgae->iema', l2, hbar.Hvvvv)
        tmp  += 0.5 * contract('imgf,fgea->iema', l2, hbar.Hvvvv)

        #can combine the next two to swapaxes type contraction
        tmp  += 0.5 * contract('imno,onea->iema', hbar.Hoooo, l2)
        tmp  += 0.5 * contract('mino,noea->iema', hbar.Hoooo, l2)
        r_Y1 += contract('iema,me->ia', tmp, X1)

       #contains regular Gvv as well as Goo, think about just calling it from cclambda instead of generating it
        tmp  =  contract('nb,fb->nf', X1, cclambda.build_Gvv(l2, t2))
        r_Y1 += contract('inaf,nf->ia', L[o,o,v,v], tmp)
        tmp  =  contract('me,fa->mefa', X1, cclambda.build_Gvv(l2, t2))
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

#        r_Y2 = r_Y2 + r_Y2.swapaxes(0,1).swapaxes(2,3)        

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
