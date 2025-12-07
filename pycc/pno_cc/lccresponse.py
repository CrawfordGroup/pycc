"""
ccresponse.py: LPNO CC Response Functions
"""
import itertools

from opt_einsum import contract

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
import time 
from time import process_time
from ..utils import helper_diis

class lccresponse(object):
    """
    An RHF-CC Response Property Object.

    Methods
    -------
    lquadresp():
        Compute a CC quadratic response function.
    lhyperpolar():
        Compute a first electric dipole hyperpolarizability average. 
    local_solve_right():
        Solve the right-hand perturbed wave function equations.
    local_solve_left(): 
        Solve the left-hand perturbed wave function equations.
    pert_lquadresp():
        Obtain the solutions of the right- and left-hand perturbed wave function equations for the CC quadritc response function. 
    Note
    ------
    For now, the only properties that can use LPNO-CC is electric-field based with inputs of two frequencies: omega1 and omega2
    such as second harmonic generation (omega1 = omega2) or optical refractivity (omega2 = -omega1)

    """

    def __init__(self, ccwfn, cclambda, omega1 = 0, omega2 = 0):
        """
        Parameters
        ----------
        ccwfn: PyCC ccwfn object
            contains references to the one- and two- electron integrals as well as Local and lccwfn object 
        cclambda: PyCC lcclambda object
            contains the references to the amplitudes and hbar terms
        omega1 : scalar
            The first external field frequency (for linear and quadratic response functions)
        omega2 : scalar
            The second external field frequency (for quadratic response functions)

        Returns
        -------
        None
        """

        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.H = self.ccwfn.H
        self.hbar = self.cclambda.hbar
        self.contract = self.ccwfn.contract
        self.no = self.ccwfn.no
        self.psuedoresponse = []
 
        self.lccwfn = ccwfn.lccwfn
        self.Local = self.ccwfn.Local       

        #initialize variables for timing each function
        self.lX1_t = 0
        self.lX2_t = 0
        self.lY1_t = 0
        self.lY2_t = 0
        self.pseudoresponse_t = 0
        self.LAX_t = 0 
        self.Fz_t = 0
        self.Bcon_t = 0 
        self.G_t = 0

        self.eps_occ = np.diag(self.hbar.Hoo)
        self.eps_lvir = []
        for i in range(self.ccwfn.no):
            for j in range(self.ccwfn.no):
                ij = i*self.ccwfn.no + j
                self.eps_lvir.append(np.diag(self.hbar.Hvv[ij])) 

        # Cartesian indices
        self.cart = ["X", "Y", "Z"]

        # Build dictionary of similarity-transformed property integrals
        self.lpertbar = {}

        self.pertbar_t = process_time()
        # Electric-dipole operator (length)
        for axis in range(3):
            key = "MU_" + self.cart[axis]
            self.lpertbar[key] = lpertbar(self.H.mu[axis], self.ccwfn, self.lccwfn)
                    
        self.pertbar_t = process_time() - self.pertbar_t 

    def pert_lquadresp(self, omega1, omega2, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
        """
        Build first-order perturbed wave functions (left- and right-hand) for the electric dipole operator (Mu)

        Parameters
        ----------
        omega1: float
            First external field frequency.
        omega2: float
            Second external field frequency.
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

        To Do
        -----
        Organize to only compute the neccesary perturbed wave functions.  
        """

        #dictionaries for perturbed waves functions
        self.lccpert_om1_X = {}
        self.lccpert_om2_X = {}
        self.lccpert_om_sum_X = {}

        self.lccpert_om1_2nd_X = {}
        self.lccpert_om2_2nd_X = {}
        self.lccpert_om_sum_2nd_X = {}

        self.lccpert_om1_Y = {}
        self.lccpert_om2_Y = {}
        self.lccpert_om_sum_Y = {}
    
        self.lccpert_om1_2nd_Y = {}
        self.lccpert_om2_2nd_Y = {}
        self.lccpert_om_sum_2nd_Y = {}

        omega_sum = -(omega1 + omega2)
  
        for axis in range(0, 3):

            pertkey = "MU_" + self.cart[axis]

            print("Solving right-hand perturbed wave function for omega1 %s:" % (pertkey))
            self.lccpert_om1_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], omega1, e_conv, r_conv, maxiter) #, max_diis, start_diis)

            print("Solving left-hand perturbed wave function for omega1 %s:" % (pertkey))
            self.lccpert_om1_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], omega1, e_conv, r_conv, maxiter) #, max_diis, start_diis)

            print("Solving right-hand perturbed wave function for omega2 %s:" % (pertkey))
            self.lccpert_om2_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], omega2, e_conv, r_conv, maxiter) #, max_diis, start_diis)

            print("Solving left-hand perturbed wave function for omega2 %s:" % (pertkey))
            self.lccpert_om2_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], omega2, e_conv, r_conv, maxiter) #, max_diis, start_diis)

            print("Solving right-hand perturbed wave function for omega_sum %s:" % (pertkey))
            self.lccpert_om_sum_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], omega_sum, e_conv, r_conv, maxiter) #, max_diis, start_diis)

            print("Solving left-hand perturbed wave function for omega_sum%s:" % (pertkey))
            self.lccpert_om_sum_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], omega_sum, e_conv, r_conv, maxiter) #, max_diis, start_diis)

            print("Solving right-hand perturbed wave function for -omega1 %s:" % (pertkey))
            self.lccpert_om1_2nd_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], -omega1, e_conv, r_conv, maxiter) # , max_diis, start_diis)

            print("Solving left-hand perturbed wave function for -omega1 %s:" % (pertkey))
            self.lccpert_om1_2nd_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], -omega1, e_conv, r_conv, maxiter) #, max_diis, start_diis)

            print("Solving right-hand perturbed wave function for -omega2 %s:" % (pertkey))
            self.lccpert_om2_2nd_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], -omega2, e_conv, r_conv, maxiter) # , max_diis, start_diis)

            print("Solving left-hand perturbed wave function for -omega2 %s:" % (pertkey))
            self.lccpert_om2_2nd_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], -omega2, e_conv, r_conv, maxiter) #, max_diis, start_diis)

            print("Solving right-hand perturbed wave function for -omega_sum %s:" % (pertkey))
            self.lccpert_om_sum_2nd_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], -omega_sum, e_conv, r_conv, maxiter) #, max_diis, start_diis)

            print("Solving left-hand perturbed wave function for -omega_sum %s:" % (pertkey))
            self.lccpert_om_sum_2nd_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], -omega_sum, e_conv, r_conv, maxiter) # , max_diis, start_diis)


    def lquadraticresp(self, pertkey_a, pertkey_b, pertkey_c, ccpert_X_A, ccpert_X_B, ccpert_X_C, ccpert_Y_A, ccpert_Y_B, ccpert_Y_C):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2
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
        # Grab X and Y amplitudes corresponding to perturbation C, omega2
        X1_C = ccpert_X_C[0]
        X2_C = ccpert_X_C[1]
        Y1_C = ccpert_Y_C[0]
        Y2_C = ccpert_Y_C[1]
        # Grab pert integrals
        pertbar_A = self.lpertbar[pertkey_a]
        pertbar_B = self.lpertbar[pertkey_b]
        pertbar_C = self.lpertbar[pertkey_c]

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
        self.Fz1 = 0.0
        self.Fz2 = 0.0
        self.Fz3 = 0.0
    

        #LAX expressions       
        # BAC
        self.LAX = self.comp_lLAX(X1_C, X2_C, Y1_B, Y2_B, pertbar_A)
        # CAB
        self.LAX2 = self.comp_lLAX(X1_B, X2_B, Y1_C, Y2_C, pertbar_A)
        # ABC
        self.LAX3 = self.comp_lLAX(X1_C, X2_C, Y1_A, Y2_A, pertbar_B)
        # CBA
        self.LAX4 = self.comp_lLAX(X1_A, X2_A, Y1_C, Y2_C, pertbar_B)
        # ACB
        self.LAX5 = self.comp_lLAX(X1_B, X2_B, Y1_A, Y2_A, pertbar_C)
        # BCA
        self.LAX6 = self.comp_lLAX(X1_A, X2_A, Y1_B, Y2_B, pertbar_C)
      
        self.hyper += self.LAX + self.LAX2 + self.LAX3 + self.LAX4 + self.LAX5 + self.LAX6

        #Fz expressions
        #BCA
        self.Fz1 = self.comp_lFz(X1_B, X1_C, X2_B, X2_C, pertbar_A) 
        #ACB
        self.Fz2 = self.comp_lFz(X1_A, X1_C, X2_A, X2_C, pertbar_B)
        #ABC
        self.Fz3 = self.comp_lFz(X1_A, X1_B, X2_A, X2_B, pertbar_C)

        self.hyper += self.Fz1 + self.Fz2 + self.Fz3
        
        #Bcon expressions
        self.Bcon1 = 0 
        self.Bcon2 = 0 
        self.Bcon3 = 0 

        #ABC
        self.Bcon1 = self.comp_lBcon(Y1_A, X1_B, X1_C, Y2_A, X2_B, X2_C, hbar)
        #BAC
        self.Bcon2 = self.comp_lBcon(Y1_B, X1_A, X1_C, Y2_B, X2_A, X2_C, hbar)
        #CAB
        self.Bcon3 = self.comp_lBcon(Y1_C, X1_A, X1_B, Y2_C, X2_A, X2_B, hbar)
        
        self.hyper += self.Bcon1 + self.Bcon2 + self.Bcon3

        self.G = 0
        
        G_start = process_time()
        QL = self.Local.QL
        Sijmn = self.Local.Sijmn 
        #<L1(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>        
        for i in range(no):
            ii= i*no + i
  
            for j in range(no): 
                jj = j*no + j 
                ij = i*no + j 
                iijj = ii*(no*no) + jj

                for k in range(no):
                    kk = k*no + k
                    ik = i*no + k
                    jk = j*no + k
                    jjkk = jj*(no*no) + kk
                    iikk = ii*(no*no) + kk
                    ijkk = ij*(no*no) + kk
                    ikjj = ik*(no*no) + jj
                    jkkk = jk*(no*no) + kk

                    tmp = contract('a,ac->c', X1_A[i], QL[ii].T @ L[i,j,v,v] @ QL[kk])
                    tmp = contract('c,c->', X1_C[k], tmp)
                    tmp2 = contract('b,b->', X1_B[j], l1[k] @ Sijmn[jjkk].T)         
                    self.G -= tmp2 * tmp
          
                    tmp = contract('a,ab->b', X1_A[i], QL[ii].T @ L[i,k,v,v] @ QL[jj])
                    tmp = contract('b,b->', X1_B[j], tmp)
                    tmp2 = contract('c,c->', l1[j] @ Sijmn[jjkk], X1_C[k])
                    self.G -= tmp2 * tmp

                    tmp = contract('b,ba->a', X1_B[j], QL[jj].T @ L[j,k,v,v] @ QL[ii])
                    tmp = contract('a,a->', X1_A[i], tmp)
                    tmp2 = contract('c,c->', X1_C[k], l1[i] @ Sijmn[iikk])
                    self.G -= tmp2 * tmp

                    tmp = contract('b,bc->c', X1_B[j], QL[jj].T @ L[j,i,v,v] @ QL[kk])
                    tmp = contract('c,c->', X1_C[k], tmp)
                    tmp2 = contract('a,a->', l1[k] @ Sijmn[iikk].T , X1_A[i])
                    self.G -= tmp2 * tmp

                    tmp = contract('c,cb->b', X1_C[k], QL[kk].T @ L[k,i,v,v] @ QL[jj])
                    tmp = contract('b,b->', X1_B[j], tmp)
                    tmp2 = contract('a,a->', l1[j] @ Sijmn[iijj].T , X1_A[i])
                    self.G -= tmp2 * tmp

                    tmp = contract('c,ca->a', X1_C[k], QL[kk].T @ L[k,j,v,v] @ QL[ii])
                    tmp = contract('a,a->', X1_A[i], tmp)
                    tmp2 = contract('b,b->', X1_B[j], l1[i] @ Sijmn[iijj])
                    self.G -= tmp2 * tmp

                    for l in range(no): 
                        ll = l*no + l 
                        il = i*no + l
                        ijll = ij*(no*no) + ll
                        ikll = ik*(no*no) + ll
                        iljj = il*(no*no) + jj
                        ilkk = il*(no*no) + kk 
                        jkll = jk*(no*no) + ll

                        # <L2(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>
                        tmp = contract('b,b->', X1_A[j], hbar.Hooov[jj][k,l,i])
                        tmp2  = contract('d,cd->c', X1_C[l], Sijmn[ijkk].T @ l2[ij] @ Sijmn[ijll])
                        tmp2  = contract('c,c->', X1_B[k], tmp2)
                        self.G = self.G + (tmp2 * tmp)

                        tmp = contract('b,b->', X1_A[j], hbar.Hooov[jj][l,k,i])
                        tmp2 = contract('d,dc->c', X1_C[l], Sijmn[ijll].T @ l2[ij] @ Sijmn[ijkk])
                        tmp2 = contract('c,c->', X1_B[k], tmp2)
                        self.G = self.G + (tmp2 * tmp)

                        tmp = contract('c,c->', X1_B[k], hbar.Hooov[kk][j,l,i])
                        tmp2  = contract('b,bd->d', X1_A[j], Sijmn[ikjj].T @ l2[ik] @ Sijmn[ikll])
                        tmp2  = contract('d,d->', X1_C[l], tmp2)
                        self.G = self.G + (tmp2 * tmp)
                        
                        tmp = contract('c,c->', X1_B[k], hbar.Hooov[kk][l,j,i])
                        tmp2  = contract('b,db->d', X1_A[j], Sijmn[ikll].T @ l2[ik] @ Sijmn[ikjj])
                        tmp2  = contract('d,d->', X1_C[l], tmp2)
                        self.G = self.G + (tmp2 * tmp)

                        tmp = contract('d,d->', X1_C[l], hbar.Hooov[ll][j,k,i])
                        tmp2  = contract('b,bc->c', X1_A[j], Sijmn[iljj].T @ l2[il] @ Sijmn[ilkk])
                        tmp2  = contract('c,c->', X1_B[k], tmp2)
                        self.G = self.G + (tmp2 * tmp)

                        tmp = contract('d,d->', X1_C[l], hbar.Hooov[ll][k,j,i])
                        tmp2  = contract('b,cb->c', X1_A[j], Sijmn[ilkk].T @ l2[il] @ Sijmn[iljj])
                        tmp2  = contract('c,c->', X1_B[k], tmp2)
                        self.G = self.G + (tmp2 * tmp)

        for j in range(no): 
            jj = j*no + j 
            
            for k in range(no): 
                kk = k*no + k 
                jk = j*no + k 

                for l in range(no): 
                    ll = l*no + l
                    jl = j*no + l
                    kl = k*no + l
                    lk = l*no + k
                    lj = l*no + j
                    jkll = jk*(no*no) + ll
                    jlll = jl*(no*no) + ll
                    jlkk = jl*(no*no) + kk
                    jjkl = jj*(no*no) + kl
                    jkl = j*(no*no) + kl
                    jlk = j*(no*no) + lk
                    klj = k*(no*no) + lj
   
                    tmp = contract('b,abc->ac', X1_A[j], hbar.Halbc[jkl])
                    tmp = contract('c,ac->a', X1_B[k], tmp)
                    tmp2  = contract('d,ad->a', X1_C[l], l2[jk] @ Sijmn[jkll])
                    self.G = self.G - contract('a,a->', tmp2, tmp)

                    #can I just do previous Hvovv.swapaxes(1,2)?
                    #Hvovv = Hvovv.swapaxes(1,2) - answer is no
                    tmp = contract('b,acb->ac', X1_A[j], hbar.Halcb[jkl])
                    tmp = contract('c,ac->a', X1_B[k], tmp)
                    tmp2  = contract('d,da->a', X1_C[l], Sijmn[jkll].T @ l2[jk])
                    self.G = self.G - contract('a,a->', tmp2, tmp)

                    tmp = contract('b,abd->ad', X1_A[j], hbar.Halbc[jlk])
                    tmp = contract('d,ad->a', X1_C[l], tmp)
                    tmp2  = contract('c,ac->a', X1_B[k], l2[jl] @ Sijmn[jlkk])
                    self.G = self.G - contract('a,a->', tmp2, tmp)

                    tmp = contract('b,adb->ad', X1_A[j], hbar.Halcb[jlk])
                    tmp = contract('d,ad->a', X1_C[l], tmp)
                    tmp2  = contract('c,ca->a', X1_B[k], Sijmn[jlkk].T @ l2[jl])
                    self.G = self.G - contract('a,a->', tmp2, tmp)

                    tmp = contract('c,acd->ad', X1_B[k], hbar.Halbc[klj])
                    tmp = contract('d,ad->a', X1_C[l], tmp)
                    tmp2  = contract('b,ab->a', X1_A[j], l2[kl] @ Sijmn[jjkl].T)
                    self.G = self.G - contract('a,a->', tmp2, tmp)

                    tmp = contract('c,adc->ad', X1_B[k], hbar.Halcb[klj])
                    tmp = contract('d,ad->a', X1_C[l], tmp)
                    tmp2  = contract('b,ba->a', X1_A[j], Sijmn[jjkl] @ l2[kl])
                    self.G = self.G - contract('a,a->', tmp2, tmp)
       
        G_end = process_time()
        self.G_t += G_end - G_start

        #LHX2Y1Z1
        self.G += self.comp_lLHXYZ(X2_A, X1_B, X1_C)
        #LHX1Y2Z1
        self.G += self.comp_lLHXYZ(X2_B, X1_A, X1_C)
        #LHX1Y1Z2
        self.G += self.comp_lLHXYZ(X2_C, X1_A, X1_B)
 
        self.hyper += self.G
        return self.hyper

    def comp_lLAX(self, X1_C, X2_C, Y1_B, Y2_B, pertbar_A):
        LAX_start = process_time()
        contract = self.contract
        no = self.ccwfn.no

        LAX = 0 
        Sijmn = self.Local.Sijmn
        for i in range(no):
            ii = i*no + i

            # <0|L1(B)[A_bar, X1(C)]|0>
            tmp = contract('a,c->ac', Y1_B[i], X1_C[i])
            LAX += contract('ac,ac->',tmp, pertbar_A.Avv[ii].copy())
            for m in range(no):
                mm = m*no + m
                iimm = ii*(no*no) + mm

                tmp = contract('a,a->', Y1_B[i], Sijmn[iimm] @ X1_C[m])
                LAX -= tmp * pertbar_A.Aoo[m,i]

            for j in range(no):
                ij = i*no + j
                ijii = ij*(no*no) + ii

                # <0|L1(B)[A_bar, X2(C)]|0>
                tmp = contract('a,b->ab', Sijmn[ijii] @ Y1_B[i], pertbar_A.Aov[ij][j])
                LAX += contract('ab,ab->', tmp, 2.0 * X2_C[ij] - X2_C[ij].swapaxes(0,1))

            for j in range(no):
                ij = i*no + j
                ijii = ij*(no*no) + ii
                tmp = contract('bc,bca->a', Y2_B[ij], pertbar_A.Avvvj_ii[ij])
                LAX += contract('a,a->', tmp, X1_C[i])

                for m in range(no):
                    mm = m*no + m
                    ijmm = ij*(no*no)+mm

                    tmp = contract('ab,b->a', Y2_B[ij], pertbar_A.Aovoo[ij][m])
                    LAX -= contract('a,a->', tmp, X1_C[m] @ Sijmn[ijmm].T)

            for j in range(no):
                ij = i*no + j

                #Second Term
                tmp = contract('ab,ac->bc', Y2_B[ij], X2_C[ij])
                LAX += contract('bc,bc->', tmp, pertbar_A.Avv[ij])
                for m in range(no):
                    mj = m*no + j
                    ijmj = ij*(no*no) + mj

                    tmp = contract('ab, ab->', Y2_B[ij], Sijmn[ijmj] @ X2_C[mj] @ Sijmn[ijmj].T)
                    LAX -= tmp * pertbar_A.Aoo[m,i]    

        LAX_end = process_time()
        self.LAX_t += LAX_end - LAX_start
        return LAX

    def comp_lFz(self, X1_B, X1_C, X2_B, X2_C, pertbar_A):
        Fz_start = process_time()
        contract = self.contract
        no = self.ccwfn.no
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2

        Fz = 0
        Sijmn = self.Local.Sijmn
        for i in range(no):
            ii = i*no + i 
            for j in range(no):
                ij = i*no + j   
                jj = j*no + j
                iijj = ii*(no*no)+jj

                # <0|L1(0)[[A_bar,X1(B)],X1(C)]|0>
                tmp2 =  contract('a,a->', X1_B[i], pertbar_A.Aov[ii][j]) 
                tmp1 = contract('b,b->', l1[i] @ Sijmn[iijj], X1_C[j]) 
                Fz -= tmp2 * tmp1

                tmp = contract('b,b->', X1_C[j], pertbar_A.Aov[jj][i])
                tmp1 = contract('a,a->', X1_B[i], Sijmn[iijj] @ l1[j])   
                Fz -= tmp * tmp1

                # <0|L2(0)[[A_bar,X1(B)],X2(C)]|0>
                tmp1 = 0
                tmp = 0
                
                for m in range(no):
                    jm = j*no + m
                    im = i*no + m
                    jmim = jm*(no*no) + im
                    jmii = jm*(no*no) + ii 
    
                    tmp1 = contract('bc,bc->', X2_C[jm], Sijmn[jmim] @ l2[im] @ Sijmn[jmim].T)
                    
                    #second term                    
                    tmp = contract('a, ac->c', X1_B[i], Sijmn[jmii].T @ l2[jm]) 
                    tmp = contract('bc, c->b', X2_C[jm], tmp)
                    Fz -= contract('b,b->', tmp, pertbar_A.Aov[jm][i]) 

                    #first term
                    tmp2 = contract('a,a->', X1_B[i], pertbar_A.Aov[ii][j]) 
                    Fz -= tmp2*tmp1
                
                # <0|L2(0)[[A_bar,X2(B)],X1(C)]|0>
                tmp1 = 0
                tmp = 0
                tmp2 = contract('a,a->', X1_C[i], pertbar_A.Aov[ii][j])
                for m in range(no):
                    jm = j*no + m
                    im = i*no + m
                    jmii = jm*(no*no) + ii
                    jmim = jm*(no*no) + im 
    
                    tmp1 =  contract('bc,bc->', X2_B[jm], Sijmn[jmim] @ l2[im] @ Sijmn[jmim].T)
                    
                    #second term                    
                    tmp = contract('a, ac->c', X1_C[i], Sijmn[jmii].T @ l2[jm]) 
                    tmp = contract('bc, c->b', X2_B[jm], tmp) 
                    Fz -= contract('b,b->', tmp, pertbar_A.Aov[jm][i]) 
 
                    #first term
                    Fz -= tmp2*tmp1
        Fz_end = process_time()
        self.Fz_t += Fz_end - Fz_start
        return Fz

    def comp_lBcon(self, Y1_A, X1_B, X1_C, Y2_A, X2_B, X2_C, hbar):
        Bcon_start = process_time()
        contract = self.contract
        no = self.ccwfn.no
        Bcon = 0 
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2
        l1 = self.cclambda.l1
        Sijmn = self.Local.Sijmn
        QL = self.Local.QL
        ERIoovv = self.Local.ERIoovv
        ERI = self.H.ERI
        L = self.H.L
        v = self.ccwfn.v

        # <O|L1(A)[[Hbar(0),X1(B),X1(C)]]|0>
        for j in range(no): 
            jj = j*no + j
 
            for k in range(no): 
                kk = k*no + k 
                jk = j*no + k
                jjkk = jj*(no*no) + kk
                kkj = kk*no + j
                jjk = jj*no + k 

                tmp = -1.0 * contract('c,b->cb', hbar.Hov[kk][j], Y1_A[k])  
                tmp2 = contract('cb, b -> c', tmp, Sijmn[jjkk].T @ X1_B[j]) 
                Bcon = Bcon + contract('c,c->', tmp2, X1_C[k])     

                tmp = -1.0 * contract('c, b -> cb', Y1_A[j], hbar.Hov[jj][k]) 
                tmp2 = contract('cb, b ->c', tmp, X1_B[j]) 
                Bcon = Bcon + contract('c,c->', tmp2, Sijmn[jjkk] @ X1_C[k])

                for i in range(no): 
                    ii = i*no + i
                    iikk = ii*(no*no) + kk
                    iijj = ii*(no*no) + jj
                   
                    tmp = contract('b,c->cb', -2.0 * hbar.Hooov[jj][k,j,i] + hbar.Hooov[jj][j,k,i], Y1_A[i]) 
                    tmp2 = contract('cb,b ->c', tmp, X1_B[j]) 
                    Bcon = Bcon + contract('c,c->', tmp2, Sijmn[iikk] @ X1_C[k]) 

                    tmp = contract('c,b->cb', -2.0 * hbar.Hooov[kk][j,k,i] + hbar.Hooov[kk][k,j,i], Y1_A[i])
                    tmp2 = contract('cb,b ->c', tmp, Sijmn[iijj] @ X1_B[j])
                    Bcon = Bcon + contract('c,c->', tmp2, X1_C[k])

                tmp = contract('acb, a -> cb', 2.0 * hbar.Hamef[kkj] - hbar.Hamfe[kkj].swapaxes(1,2), Y1_A[k]) 
                tmp2 = contract('cb, b ->c', tmp, X1_B[j])
                Bcon = Bcon + contract('c,c->', tmp2, X1_C[k])

                tmp = contract('abc, a -> cb', 2.0 * hbar.Hamef[jjk] - hbar.Hamfe[jjk].swapaxes(1,2), Y1_A[j])
                tmp2 = contract('cb, b ->c', tmp, X1_B[j])
                Bcon = Bcon + contract('c,c->', tmp2, X1_C[k])

        # # <O|L2(A)|[[Hbar(0),X1(B)],X1(C)]|0>
        for j in range(no):
            jj = j* no + j 
 
            for n in range(no):
                nn = n*no + n
                nj = n*no + j 

                for k in range(no): 
                    kk = k*no + k
                    nk = n*no + k 
                    nj = n*no + j 
                    kj = k*no + j  
                    jk = j*no + k
                    jjnk = jj*(no*no) + nk
                    njkk = nj*(no*no) + kk
                    jkn = jk*no + n 
                    kjn = kj*no + n 
                       
                    tmp  = -1.0 * contract('ac,ba->cb', hbar.Hovov_ni[kjn], Y2_A[nk])
                    tmp2 = contract('cb, c -> b', tmp, X1_B[k]) 
                    Bcon = Bcon + contract('b,b->', tmp2, Sijmn[jjnk].T @ X1_C[j]) 
 
                    tmp = -1.0 * contract('ab, ca -> cb', hbar.Hovov_ni[jkn], Y2_A[nj])  
                    tmp2 = contract('cb, c -> b', tmp, Sijmn[njkk] @ X1_B[k])
                    Bcon = Bcon + contract('b,b->', tmp2, X1_C[j])

                    tmp = -1.0 * contract('ac, ab -> cb', hbar.Hovvo_ni[kjn], Y2_A[nk]) 
                    tmp2 = contract('cb, c -> b', tmp, X1_B[k])
                    Bcon = Bcon + contract('b,b->', tmp2, Sijmn[jjnk].T @ X1_C[j])

                    tmp = -1.0 * contract('ab, ac -> cb', hbar.Hovvo_ni[jkn], Y2_A[nj])
                    tmp2 = contract('cb, c -> b', tmp, Sijmn[njkk] @ X1_B[k])
                    Bcon = Bcon + contract('b,b->', tmp2, X1_C[j])

                    for i in range(no):
                        ii = i*no + i 
                        ni = n*no + i  
                        _in = i*no + n
                        nijj = ni*(no*no) + jj 
                        nikk = ni*(no*no) + kk 

                        tmp = 0.5 * hbar.Hoooo[k,j,i,n] * Y2_A[ni].swapaxes(0,1) 
                        tmp2 = contract('cb, c -> b', tmp, Sijmn[nikk] @ X1_B[k])
                        Bcon = Bcon + contract('b,b->', tmp2, Sijmn[nijj] @ X1_C[j])

                        tmp = 0.5 * hbar.Hoooo[j,k,i,n] * Y2_A[ni]
                        tmp2 = contract('cb, c -> b', tmp, Sijmn[nikk] @ X1_B[k])
                        Bcon = Bcon + contract('b,b->', tmp2, Sijmn[nijj] @ X1_C[j])
 
        for j in range(no):
            jj = j*no + j
 
            for k in range(no):
                kk = k*no + k
                jk = j*no + k 
                kj = k*no + j
 
                tmp = 0.5 * contract('fabc, fa -> cb', hbar.Hvvvv_im[jk], Y2_A[jk])
                tmp2 = contract('cb, c -> b', tmp, X1_B[k])
                Bcon = Bcon + contract('b,b->', tmp2, X1_C[j])

                tmp = 0.5 * contract('facb, fa -> cb', hbar.Hvvvv_im[kj], Y2_A[kj])
                tmp2 = contract('cb, c -> b', tmp, X1_B[k])
                Bcon = Bcon + contract('b,b->', tmp2, X1_C[j])
     
        L = self.H.L
        ERIoovv = self.Local.ERIoovv
        Loovv = self.Local.Loovv

        for i in range(no): 
            ii = i*no + i 
            
            for j in range(no): 
                jj = j*no + j 
                ij = i*no + j 
 
                for k in range(no): 
                    kk = k*no + k 
                    jk = j*no + k 
                    ik = i*no + k 
                    jkkk = jk*(no*no) + kk
                    iikk = ii*(no*no) + kk
                    iijk = ii*(no*no) + jk

                    for l in range(no):
                        ll = l*no + l 
                        jl = j*no + l
                        lj = l*no + j 
                        kl = k*no + l
                        ijll = ij*(no*no) + ll     
                        ijjl = ij*(no*no) + jl
                        ijkk = ij*(no*no) + kk  
                        ijkl = ij*(no*no) + kl
                        ijjk = ij*(no*no) + jk
                        ikkl = ik*(no*no) + kl
                        ijik = ij*(no*no) + ik
                        jlkl = jl*(no*no) + kl
                        ijlj = ij*(no*no) + lj

                        tmp = contract('ab,db->ad', t2[ij], Y2_A[ij])
                        tmp = contract('d,ad->a', Sijmn[ijll] @ X1_C[l], tmp) 
                        tmp = contract('a,ca->c', tmp, QL[kk].T @ L[k,l,v,v] @ QL[ij]) 
                        Bcon = Bcon - contract('c,c->', tmp, X1_B[k]) 

                        tmp = contract('ab,ba->', Sijmn[ijjl].T @ t2[ij] @ Sijmn[ijjl], Y2_A[jl])
                        tmp2 = contract('c,cd->d', X1_B[k], QL[kk].T @ L[k,i,v,v] @ QL[ll])
                        tmp2 = contract('d,d->', tmp2, X1_C[l])
                        Bcon = Bcon - (tmp2 * tmp) 

                        tmp = contract('ab,ba->', Sijmn[ijjk].T @ t2[ij] @ Sijmn[ijjk], Y2_A[jk])
                        tmp2 = contract('d,dc->c', X1_C[l], QL[ll].T @ L[l,i,v,v] @ QL[kk])
                        tmp2 = contract('c,c->', tmp2, X1_B[k])
                        Bcon = Bcon - (tmp2 *tmp) 

                        tmp = contract('ab,cb->ac', t2[ij], Y2_A[ij])
                        tmp = contract('c,ac->a', Sijmn[ijkk] @ X1_B[k], tmp)
                        tmp2 = contract('d,da->a', X1_C[l], QL[ll].T @ L[l,k,v,v] @ QL[ij])
                        Bcon = Bcon - contract('a,a->', tmp2, tmp)

                        # <O|L2(A)[[Hbar(0),X2(B)],X2(C)]|0>
                        tmp = contract("cd,cd->", Sijmn[ijkl] @ X2_C[kl] @ Sijmn[ijkl].T, Y2_A[ij])
                        tmp = tmp * X2_B[ij]
                        Bcon = Bcon + 0.5* contract('ab,ab->', tmp, ERIoovv[ij][k,l].copy())
 
                        tmp = contract("ab,bd->ad", X2_C[ij] @ Sijmn[ijik] , Y2_A[ik])
                        tmp = contract("ad,cd->ac", tmp, X2_B[kl] @ Sijmn[ikkl].T)
                        Bcon = Bcon + contract('ac,ac->', tmp, QL[ij].T @ ERI[j,l,v,v].copy() @ QL[kl])

                        tmp = contract("cd,db->cb", X2_C[kl] @ Sijmn[ikkl].T , Y2_A[ik])
                        tmp = contract("cb,ab->ca", tmp, X2_B[ij] @ Sijmn[ijik])
                        Bcon = Bcon + contract('ca,ac->', tmp, QL[ij].T @ ERI[l,j,v,v].copy() @ QL[kl])
           
                        tmp = contract("ab,ab->", Sijmn[ijkl].T @ X2_B[ij] @ Sijmn[ijkl], Y2_A[kl])
                        tmp = tmp * X2_C[kl]
                        Bcon = Bcon + 0.5* contract('cd,cd->', tmp, ERIoovv[kl][i,j].copy())
                 
                        tmp = contract("ab,ac->bc", X2_B[ij] @ Sijmn[ijkl], QL[ij].T @ L[i,j,v,v] @ QL[kl])
                        tmp = contract("bc,cd->bd", tmp, X2_C[kl])
                        Bcon = Bcon - contract("bd,bd->", tmp, Y2_A[kl])

                        tmp = contract("ab,ab->", X2_B[ij], Loovv[ij][i,k])
                        tmp = tmp * (Sijmn[jlkl] @ X2_C[kl] @ Sijmn[jlkl].T)
                        Bcon = Bcon - contract("cd,cd->", tmp, Y2_A[jl])
 
                        tmp = contract("bc,cd->bd", QL[ij].T @ L[i,k,v,v] @ QL[kl], X2_C[kl] @ Sijmn[jlkl].T)
                        tmp = contract("bd,ab->ad", tmp, Sijmn[ijjl].T @ X2_B[ij])
                        Bcon = Bcon - contract("ad,ad->", tmp, Y2_A[jl])

                        tmp = contract("ab,bc->ac", X2_B[ij] @ Sijmn[ijjl], Y2_A[jl])
                        tmp = contract("ac,cd->ad", tmp, Sijmn[jlkl] @ X2_C[kl])
                        Bcon = Bcon - contract("ad,ad->", tmp, QL[ij].T @ L[i,k,v,v] @ QL[kl])

                        tmp = contract("ca,cd->ad", QL[kl].T @ L[k,l,v,v] @ QL[ij], X2_C[kl] @ Sijmn[ijkl].T)
                        tmp = contract("ad,db->ab", tmp, Y2_A[ij])
                        Bcon = Bcon - contract("ab,ab->", tmp, X2_B[ij])

                        tmp = contract("cd,cd->",Loovv[kl][k,i], X2_C[kl])
                        tmp = (Sijmn[ijlj].T @ X2_B[ij] @ Sijmn[ijlj]) * tmp
                        Bcon = Bcon - contract("ab,ab->", tmp, Y2_A[lj])
                       
                        tmp = contract("cd,ac->da", Sijmn[ikkl] @ X2_C[kl], Y2_A[ik])
                        tmp = contract("da,bd->ab", tmp, QL[ij].T @ L[j,l,v,v] @ QL[kl])
                        Bcon = Bcon + (2.0* contract("ab,ab->", tmp, Sijmn[ijik].T @ X2_B[ij]))

                    # <O|L1(A)[[Hbar(0),X1(B)],X2(C)]]|0>
                    tmp = contract('bc,c->b', 2.0 * (X2_C[jk] @ Sijmn[jkkk])  - (Sijmn[jkkk].T @ X2_C[jk]).swapaxes(0,1), Y1_A[k])  
                    tmp = contract('ab,b->a', QL[ii].T @ L[i,j,v,v] @ QL[jk], tmp)
                    Bcon = Bcon + contract("a,a->", tmp, X1_B[i])

                    tmp = contract("bc,ba->ca", X2_C[jk] @ Sijmn[iijk].T, QL[jk].T @ L[j,k,v,v] @ QL[ii])
                    tmp = contract("a,ca->c", X1_B[i], tmp)
                    Bcon = Bcon - contract("c,c->", tmp, Y1_A[i])

                    tmp = contract("bc,bc->", X2_C[jk], Loovv[jk][j,i])
                    tmp = tmp * (Sijmn[iikk].T @ X1_B[i])
                    Bcon= Bcon - contract("a,a->", tmp, Y1_A[k])
    
        # <O|L2(A)[[Hbar(0),X1(B)],X2(C)]]|0>
        for j in range(no): 
            jj = j*no + j    
            
            for k in range(no): 
                kk = k*no + k 
                jk = j*no + k 
                
                for l in range(no):
                    ll = l*no + l
                    kl = k*no + l 
                    lk = l*no + k
                    lj = l*no + j
                    jjlk = jj*(no*no) + lk 
                    kllj = kl*(no*no) + lj
                    kllk = kl*(no*no) + lk
                    lkj = lk*no + j 
                    klj = kl*no + j

                    tmp = contract('cd,db -> cb', X2_C[kl], Y2_A[lk]) 
                    tmp = contract('b, cb ->c', Sijmn[jjlk].T @ X1_B[j], tmp) 
                    Bcon = Bcon - contract('c,c->', tmp, hbar.Hov[kl][j]) 

                    tmp = contract('cd, dc ->', Sijmn[kllj].T @ X2_C[kl] @ Sijmn[kllj], Y2_A[lj]) 
                    tmp = tmp * X1_B[j]
                    Bcon = Bcon - contract('b,b ->', tmp, hbar.Hov[jj][k]) 
   
                    tmp = contract('da, cd -> ac', Y2_A[lk], X2_C[kl] @ Sijmn[kllk])           
                    tmp2 = contract('b,acb -> ac', X1_B[j], 2.0 * hbar.Hamef[lkj] - hbar.Hamfe[lkj].swapaxes(1,2))  
                    Bcon = Bcon + contract('ac,ac ->', tmp, tmp2)  

                    tmp = contract('b, da -> bda', X1_B[j], Y2_A[lj]) 
                    tmp2 = contract('cd, abc -> dab', X2_C[kl] @ Sijmn[kllj], 2.0 * hbar.Hfieb[klj] - hbar.Hfibe[klj].swapaxes(1,2))
                    Bcon = Bcon + contract('bda, dab ->', tmp, tmp2) 

                for i in range(no): 
                    ii = i*no + i 
                    ji = j*no + i 
                    ik = i*no + k
                    jkji = jk*(no*no) + ji
                    jkik = jk*(no*no) + ik                   
                    jkii = jk*(no*no) + ii
                    ij = i*no + j 
                    ijk = ij*no + k
                    ikj = ik*no + j 
 
                    tmp = contract('a, fba -> fb', X1_B[i], hbar.Hgnea[ijk]) 
                    tmp = contract('fb, fc -> bc', tmp, Y2_A[ji]) 
                    Bcon = Bcon - contract('bc,bc ->', X2_C[jk] @ Sijmn[jkji], tmp) 

                    tmp = contract('a, fac -> fc', X1_B[i], hbar.Hgnae[ikj]) 
                    tmp = contract('fc, fb -> bc', tmp, Y2_A[ik]) 
                    Bcon = Bcon - contract('bc, bc ->', Sijmn[jkik].T @ X2_C[jk], tmp)

                    tmp = contract('a, fa -> f', Sijmn[jkii] @ X1_B[i], Y2_A[jk]) 
                    tmp2 = contract('bc, fbc -> f', X2_C[jk], hbar.Hvovv_ij[jk][:,i]) 
                    Bcon = Bcon - contract('f,f->', tmp2, tmp) 
       
                    for l in range(no):
                        kl = k*no + l 
                        il = i*no + l 
                        klil = kl*(no*no) + il
                        jjil = jj*(no*no) + il 

                        tmp = contract('b,b->', X1_B[j], -2.0 * hbar.Hooov[jj][k,j,i] + hbar.Hooov[jj][j,k,i]) 
                        tmp2 = contract('cd, cd->', Sijmn[klil].T @ X2_C[kl] @ Sijmn[klil], Y2_A[il]) 
                        Bcon = Bcon + (tmp * tmp2) 
                             
                        tmp1 = contract('c, cd -> d', 2.0 * hbar.Hooov[kl][j,k,i] - hbar.Hooov[kl][k,j,i], X2_C[kl] @ Sijmn[klil]) 
                        tmp1 = contract('d, b-> bd', tmp1, X1_B[j] @ Sijmn[jjil])
                        Bcon = Bcon - contract('bd, bd ->', tmp1, Y2_A[il]) 
       
                    for n in range(no): 
                        ni = n*no + i
                        nn = n*no + n 
                        jkni = jk*(no*no) + ni 

                        tmp = contract('a, a->', X1_B[i], hbar.Hooov[ii][j,k,n]) 
                        tmp2 = contract('bc, bc ->', Sijmn[jkni].T @ X2_C[jk] @ Sijmn[jkni], Y2_A[ni])
                        Bcon = Bcon +  (tmp * tmp2) 
     
        for i in range(no):
            ii = i*no + i 
            for k in range(no):
                for n in range(no):
                    nn = n*no + n
                    nk = n*no + k
                    iink = ii*(no*no) + nk
 
                    for j in range(no):
                        jk = j*no + k 
                        nkjk = nk*(no*no) + jk

                        tmp = contract('a, ab -> b', Sijmn[iink].T @ X1_B[i], Y2_A[nk]) 
                        tmp = contract('bc, b -> c', Sijmn[nkjk] @ X2_C[jk], tmp) 
                        Bcon = Bcon + contract('c, c ->', tmp, hbar.Hooov[jk][i,j,n])
        
                        tmp = contract('a, ba -> b', Sijmn[iink].T @ X1_B[i], Y2_A[nk]) 
                        tmp = contract('bc, b -> c', Sijmn[nkjk] @ X2_C[jk], tmp) 
                        Bcon = Bcon + contract('c, c ->', tmp, hbar.Hooov[jk][j,i,n])            

        for i in range(no): 
            ii = i*no + i 
        
            for j in range(no):
                jj = j*no + j 
                ij = i*no + j 
               
                for k in range(no): 
                    kk = k*no + k
                    jk = j*no + k 
                    jkkk = jk*(no*no) + kk
                    iijk = ii*(no*no) + jk
                    iikk = ii*(no*no) + kk

                    #<O|L1(A)[[Hbar(0),X2(B)],X1(C)]]|0>
                    tmp = contract('bc,c->b', 2.0 * (X2_B[jk] @ Sijmn[jkkk])  - (Sijmn[jkkk].T @ X2_B[jk]).swapaxes(0,1), Y1_A[k])  
                    tmp = contract('ab,b->a', QL[ii].T @ L[i,j,v,v] @ QL[jk], tmp) 
                    Bcon = Bcon + contract("a,a->", tmp, X1_C[i])

                    tmp = contract("bc,ba->ca", X2_B[jk] @ Sijmn[iijk].T, QL[jk].T @ L[j,k,v,v] @ QL[ii])
                    tmp = contract("a,ca->c", X1_C[i], tmp) 
                    Bcon = Bcon - contract("c,c->", tmp, Y1_A[i])

                    tmp = contract("bc,bc->", X2_B[jk], Loovv[jk][j,i])
                    tmp = tmp * (Sijmn[iikk].T @ X1_C[i])
                    Bcon= Bcon - contract("a,a->", tmp, Y1_A[k])

        # <O|L2(A)[[Hbar(0),X2(B)],X1(C)]]|0>
        for k in range(no):
            for l in range(no): 
                    kl = k*no + l
                    lk = l*no + k 
                    kllk = kl*(no*no) + lk

                    for j in range(no): 
                        jj = j*no + j 
                        lj = l*no + j 
                        jjlk = jj*(no*no) + lk
                        kllj = kl*(no*no) + lj 
                        lkj = lk*no + j 
                        klj = kl*no + j 

                        tmp = contract('cd, db -> cb', X2_B[kl] @ Sijmn[kllk], Y2_A[lk])
                        tmp = contract('b, cb -> c', Sijmn[jjlk].T @ X1_C[j], tmp) 
                        Bcon = Bcon - contract('c,c->', tmp, hbar.Hov[kl][j]) 

                        tmp = contract('cd, dc ->', Sijmn[kllj].T @ X2_B[kl] @ Sijmn[kllj], Y2_A[lj])
                        tmp = tmp * X1_C[j] 
                        Bcon = Bcon - contract('b, b ->', tmp, hbar.Hov[jj][k]) 

                        tmp = contract('da, cd -> ac', Y2_A[lk], X2_B[kl] @ Sijmn[kllk])
                        tmp2 = contract('b,acb -> ac', X1_C[j], 2.0 * hbar.Hamef[lkj] - hbar.Hamfe[lkj].swapaxes(1,2))
                        Bcon = Bcon + contract('ac, ac->', tmp, tmp2) 

                        tmp = contract('b, da -> bda', X1_C[j], Y2_A[lj]) 
                        tmp2 = contract('cd, abc -> dab', X2_B[kl] @ Sijmn[kllj], 2.0 * hbar.Hfieb[klj] - hbar.Hfibe[klj].swapaxes(1,2)) 
                        Bcon = Bcon + contract('bda, dab ->', tmp, tmp2) 

        for i in range(no): 
            ii = i*no + i 
            for j in range(no): 
                jj = j*no + j 
                ji = j*no + i  
                ij = i*no + j 
 
                for k in range(no): 
                    kk = k*no + k 
                    jk = j*no + k 
                    ik = i*no + k 
                    jijk = ji*(no*no) + jk 
                    jkik = jk*(no*no) + ik
                    iijk = ii*(no*no) + jk
                    ijk = ij*no + k 
                    ikj = ik*no + j 

                    tmp = contract('a, fba -> fb', X1_C[i], hbar.Hgnea[ijk])
                    tmp = contract('fb, fc -> bc', tmp, Y2_A[ji]) 
                    Bcon = Bcon - contract('bc, bc ->', X2_B[jk] @ Sijmn[jijk].T, tmp)    
       
                    tmp = contract('a, fac -> fc', X1_C[i], hbar.Hgnae[ikj])
                    tmp = contract('fc, fb -> bc', tmp, Y2_A[ik]) 
                    Bcon = Bcon - contract('bc, bc ->', Sijmn[jkik].T @ X2_B[jk], tmp) 

                    tmp = contract('a, fa -> f', X1_C[i] @ Sijmn[iijk], Y2_A[jk])
                    tmp2 = contract('bc, fbc -> f', X2_B[jk], hbar.Hvovv_ij[jk][:,i])
                    Bcon = Bcon - contract('f, f ->', tmp2, tmp)

                    for l in range(no): 
                        il = i*no + l 
                        kl = k*no + l 
                        klil = kl*(no*no) + il
                        jjil = jj*(no*no) + il 

                        tmp = contract('b, b->', X1_C[j], -2.0 * hbar.Hooov[jj][k,j,i] + hbar.Hooov[jj][j,k,i]) 
                        tmp2 = contract('cd, cd->', Sijmn[klil].T @ X2_B[kl] @ Sijmn[klil], Y2_A[il]) 
                        Bcon = Bcon + (tmp * tmp2)                          
                  
                        tmp = contract('c, cd -> d', 2.0 * hbar.Hooov[kl][j,k,i] - hbar.Hooov[kl][k,j,i], X2_B[kl] @ Sijmn[klil]) 
                        tmp = contract('d, b -> bd', tmp, X1_C[j] @ Sijmn[jjil]) 
                        Bcon = Bcon - contract('bd, bd ->', tmp, Y2_A[il]) 
   
                    for n in range(no): 
                        nn = n*no + n
                        ni = n*no + i
                        nk = n*no + k  
                        jkni = jk*(no*no) + ni 
                        iink = ii*(no*no) + nk
                        jknk = jk*(no*no) + nk

                        tmp = contract('a, a ->', X1_C[i], hbar.Hooov[ii][j,k,n])
                        tmp2 = contract('bc, bc ->', Sijmn[jkni].T @ X2_B[jk] @ Sijmn[jkni], Y2_A[ni])
                        Bcon = Bcon + (tmp2 * tmp) 

                        tmp = contract('a, ab -> b', X1_C[i] @ Sijmn[iink], Y2_A[nk]) 
                        tmp = contract('bc, b -> c', Sijmn[jknk].T @ X2_B[jk], tmp) 
                        Bcon = Bcon + contract('c, c ->', tmp, hbar.Hooov[jk][i,j,n]) 

                        tmp = contract('a, ba -> b', X1_C[i] @ Sijmn[iink], Y2_A[nk])
                        tmp = contract('bc, b -> c', Sijmn[jknk].T @ X2_B[jk], tmp)
                        Bcon = Bcon + contract('c, c ->', tmp, hbar.Hooov[jk][j,i,n]) 
        
        Bcon_end = process_time()
        self.Bcon_t += Bcon_end - Bcon_start

        return Bcon

    def comp_lLHXYZ(self, X2_A, X1_B, X1_C):
        G_start = process_time()
        # <L2(0)|[[[H_bar,X2(A)],X1(B)],X1(C)]|0>
        # LHX2(A)Y1(B)Z1(C)
        v = self.ccwfn.v
        no = self.ccwfn.no 
        ERI = self.H.ERI
        L = self.H.L 
        l2 = self.cclambda.l2
        QL = self.Local.QL
        Sijmn = self.Local.Sijmn
        ERIoovv = self.Local.ERIoovv

        G = 0 
        for i in range(no):
            ii = i*no + i

            for j in range(no):
                 jj = j*no + j
                 ij = i*no + j
                 ji = j*no + i

                 for k in range(no):
                     kk = k*no + k
                     jk = j*no + k
                     ik = i*no + k

                     for l in range(no):
                         ll = l*no + l
                         jl = j*no + l
                         il = i*no + l
                         kl = k*no + l
                         jlkk = jl*(no*no) + kk
                         ijjl = ij*(no*no) + jl
                         jkll = jk*(no*no) + ll
                         ijjk = ij*(no*no) + jk
                         ijji = ij*(no*no) + ji
                         jill = ji*(no*no) + ll
                         jikk = ji*(no*no) + kk
                         ijll = ij*(no*no) + ll
                         ijkk = ij*(no*no) + kk
                         ijik = ij*(no*no) + ik
                         ikll = ik*(no*no) + ll
                         ijil = ij*(no*no) + il
                         ilkk = il*(no*no) + kk
                         ijil = ij*(no*no) + il
                         ijkl = ij*(no*no) + kl

                         tmp = contract('c,bc->b', X1_B[k], Sijmn[ijjl] @ l2[jl] @ Sijmn[jlkk])
                         tmp2 = contract('d,ad->a', X1_C[l], QL[ij].T @ L[i,k,v,v] @ QL[ll])
                         tmp2 = contract('ab,a->b', X2_A[ij], tmp2)
                         G -= contract('b,b->', tmp, tmp2)

                         tmp = contract('d,bd->b', X1_C[l], Sijmn[ijjk] @ l2[jk] @ Sijmn[jkll])
                         tmp2 = contract('c,ac->a', X1_B[k], QL[ij].T @ L[i,l,v,v] @ QL[kk])
                         tmp2 = contract('ab,a->b', X2_A[ij], tmp2)
                         G -= contract('b,b->',tmp,tmp2)

                         tmp = contract('ab,bd->ad', X2_A[ij], Sijmn[ijji] @ l2[ji] @ Sijmn[jill])
                         tmp = contract('d,ad->a', X1_C[l], tmp)
                         tmp2 = contract('ca,c->a', QL[kk].T @ L[k,l,v,v] @ QL[ij], X1_B[k])
                         G -= contract('a,a->', tmp, tmp2)

                         tmp = contract('ab,ba->', X2_A[ij], Sijmn[ijjl] @ l2[jl] @ Sijmn[ijjl].T)
                         tmp2 = contract('c,cd->d', X1_B[k], QL[kk].T @ L[k,i,v,v] @ QL[ll])
                         tmp2 = contract('d,d->', X1_C[l], tmp2)
                         G -= tmp * tmp2

                         tmp = contract('ab,ba->', X2_A[ij], Sijmn[ijjk] @ l2[jk] @ Sijmn[ijjk].T)
                         tmp2 = contract('d,dc->c', X1_C[l], QL[ll].T @ L[l,i,v,v] @ QL[kk])
                         tmp2 = contract('c,c->', X1_B[k], tmp2)
                         G -= tmp * tmp2

                         tmp = contract('ab,bc->ac', X2_A[ij], Sijmn[ijji].T @ l2[ji] @ Sijmn[jikk])
                         tmp = contract('ac,c->a', tmp, X1_B[k])
                         tmp2 = contract('d,da->a', X1_C[l], QL[ll].T @ L[l,k,v,v] @ QL[ij])
                         G -= contract('a,a->', tmp, tmp2)

                         tmp = contract('ab,ab->',X2_A[ij], ERIoovv[ij][k,l])
                         tmp2 = contract('c,cd->d', X1_B[k], Sijmn[ijkk].T @ l2[ij] @ Sijmn[ijll])
                         tmp2 = contract('d,d->', X1_C[l], tmp2)
                         G += tmp * tmp2

                         tmp = contract('c,ac->a', X1_B[k], QL[ij].T @ ERI[j,l,v,v] @ QL[kk])
                         tmp = contract('ab,a->b', X2_A[ij], tmp)
                         tmp2 = contract('bd,d->b', Sijmn[ijik] @ l2[ik] @ Sijmn[ikll], X1_C[l])
                         G += contract('b,b->', tmp, tmp2)

                         tmp = contract('c,ac->a', X1_B[k], QL[ij].T @ ERI[l,j,v,v] @ QL[kk])
                         tmp = contract('ab,a->b', X2_A[ij], tmp)
                         tmp2 = contract('db,d->b', Sijmn[ikll].T @ l2[ik] @ Sijmn[ijik].T, X1_C[l])
                         G += contract('b,b->', tmp, tmp2)

                         tmp = contract('d,ad->a', X1_C[l], QL[ij].T @ ERI[j,k,v,v] @ QL[ll])
                         tmp = contract('ab,a->b', X2_A[ij], tmp)
                         tmp2 = contract('c,bc->b', X1_B[k], Sijmn[ijil] @ l2[il] @ Sijmn[ilkk])
                         G += contract('b,b->', tmp, tmp2)

                         tmp = contract('d,ad->a', X1_C[l], QL[ij].T @ ERI[k,j,v,v] @ QL[ll])
                         tmp = contract('ab,a->b', X2_A[ij], tmp)
                         tmp2 = contract('c,cb->b', X1_B[k], Sijmn[ilkk].T @ l2[il] @ Sijmn[ijil].T)
                         G += contract('b,b->', tmp, tmp2)

                         tmp = contract('c,cd->d', X1_B[k], QL[kk].T @ ERI[i,j,v,v] @ QL[ll])
                         tmp = contract('d,d->', X1_C[l], tmp)
                         tmp2 = contract('ab,ab->', X2_A[ij], Sijmn[ijkl] @ l2[kl] @ Sijmn[ijkl].T)
                         G += tmp * tmp2

        G_end = process_time()
        self.G_t += G_end - G_start
        return G 

    def comp_LHXYZ(self, X2_A, X1_B, X1_C): 
        L = self.H.L
        ERI = self.H.ERI
        l2 = self.cclambda.l2
        o = self.ccwfn.o
        v = self.ccwfn.v

        G = 0 

        # <L2(0)|[[[H_bar,X2(A)],X1(B)],X1(C)]|0>
        tmp = contract('kc,jlbc->jlbk', X1_B, l2)
        tmp2 = contract('ld,ikad->ikal', X1_C, L[o,o,v,v])
        tmp2 = contract('ijab,ikal->jlbk', X2_A, tmp2)
        G -= contract('jlbk,jlbk->', tmp, tmp2)

        tmp = contract('ld,jkbd->jkbl', X1_C, l2)
        tmp2 = contract('kc,ilac->ilak', X1_B, L[o,o,v,v])
        tmp2 = contract('ijab,ilak->jkbl', X2_A, tmp2)
        G -= contract('jkbl,jkbl->',tmp,tmp2)

        tmp = contract('ijab,jibd->ad', X2_A, l2)
        tmp = contract('ld,ad->la', X1_C, tmp)
        tmp2 = contract('klca,kc->la', L[o,o,v,v], X1_B)
        G -= contract('la,la->', tmp, tmp2)

        tmp = contract('ijab,jlba->il', X2_A, l2)
        tmp2 = contract('kc,kicd->id', X1_B, L[o,o,v,v])
        tmp2 = contract('ld,id->il', X1_C, tmp2)
        G -= contract('il,il->', tmp, tmp2)

        tmp = contract('ijab,jkba->ik', X2_A, l2)
        tmp2 = contract('ld,lidc->ic', X1_C, L[o,o,v,v])
        tmp2 = contract('kc,ic->ik', X1_B, tmp2)
        G -= contract('ik,ik->', tmp, tmp2)

        tmp = contract('ijab,jibc->ac', X2_A, l2)
        tmp = contract('ac,kc->ka', tmp, X1_B)
        tmp2 = contract('ld,lkda->ka', X1_C, L[o,o,v,v])
        G -= contract('ka,ka->', tmp, tmp2)

        tmp = contract('ijab,klab->ijkl',X2_A, ERI[o,o,v,v])
        tmp2 = contract('kc,ijcd->ijkd', X1_B, l2)
        tmp2 = contract('ld,ijkd->ijkl', X1_C, tmp2)
        G += contract('ijkl,ijkl->',tmp,tmp2)

        tmp = contract('kc,jlac->jlak', X1_B, ERI[o,o,v,v])
        tmp = contract('ijab,jlak->ilbk', X2_A, tmp)
        tmp2 = contract('ikbd,ld->ilbk', l2, X1_C)
        G += contract('ilbk,ilbk->', tmp, tmp2)

        tmp = contract('kc,ljac->ljak', X1_B, ERI[o,o,v,v])
        tmp = contract('ijab,ljak->ilbk', X2_A, tmp)
        tmp2 = contract('ikdb,ld->ilbk', l2, X1_C)
        G += contract('ilbk,ilbk->', tmp, tmp2)

        tmp = contract('ld,jkad->jkal', X1_C, ERI[o,o,v,v])
        tmp = contract('ijab,jkal->ikbl', X2_A, tmp)
        tmp2 = contract('kc,ilbc->ilbk', X1_B, l2)
        G += contract('ikbl,ilbk->', tmp, tmp2)

        tmp = contract('ld,kjad->kjal', X1_C, ERI[o,o,v,v])
        tmp = contract('ijab,kjal->iklb', X2_A, tmp)
        tmp2 = contract('kc,ilcb->ilkb', X1_B, l2)
        G += contract('iklb,ilkb->', tmp, tmp2)

        tmp = contract('kc,ijcd->ijkd', X1_B, ERI[o,o,v,v])
        tmp = contract('ld,ijkd->ijkl', X1_C, tmp)
        tmp2 = contract('ijab,klab->ijkl', X2_A, l2)
        G += contract('ijkl,ijkl->', tmp, tmp2)

        return G 

    def lhyperpolar(self):
        """
        Return
        ------
        Beta_avg: float
            Hyperpolarizability average
        lhyper_AB: 3x3x3 tensor 
            Hyperpolarizability elements 
        """
        solver_start = time.time()

        lccpert_om1_X = self.lccpert_om1_X
        lccpert_om2_X = self.lccpert_om2_X
        lccpert_om_sum_X = self.lccpert_om_sum_X

        lccpert_om1_2nd_X = self.lccpert_om1_2nd_X
        lccpert_om2_2nd_X = self.lccpert_om2_2nd_X
        lccpert_om_sum_2nd_X = self.lccpert_om_sum_2nd_X

        lccpert_om1_Y = self.lccpert_om1_Y
        lccpert_om2_Y = self.lccpert_om2_Y
        lccpert_om_sum_Y = self.lccpert_om_sum_Y

        lccpert_om1_2nd_Y = self.lccpert_om1_2nd_Y
        lccpert_om2_2nd_Y = self.lccpert_om2_2nd_Y
        lccpert_om_sum_2nd_Y = self.lccpert_om_sum_2nd_Y

        lhyper_AB_1st = np.zeros((3,3,3))
        lhyper_AB_2nd = np.zeros((3,3,3))
        self.lhyper_AB = np.zeros((3,3,3))

        for a in range(0, 3):
            pertkey_a = "MU_" + self.cart[a]
            for b in range(0, 3):
                pertkey_b = "MU_" + self.cart[b]
                for c in range(0, 3):
                    pertkey_c = "MU_" + self.cart[c]

                    lhyper_AB_1st[a,b,c] = self.lquadraticresp(pertkey_a, pertkey_b, pertkey_c, 
                    lccpert_om_sum_X[pertkey_a], lccpert_om1_X[pertkey_b], lccpert_om2_X[pertkey_c], 
                    lccpert_om_sum_Y[pertkey_a], lccpert_om1_Y[pertkey_b], lccpert_om2_Y[pertkey_c] )
                    
                    lhyper_AB_2nd[a,b,c] = self.lquadraticresp(pertkey_a, pertkey_b, pertkey_c, 
                    lccpert_om_sum_2nd_X[pertkey_a], lccpert_om1_2nd_X[pertkey_b], lccpert_om2_2nd_X[pertkey_c], 
                    lccpert_om_sum_2nd_Y[pertkey_a], lccpert_om1_2nd_Y[pertkey_b], lccpert_om2_2nd_Y[pertkey_c])
                    
                    self.lhyper_AB[a,b,c] = (lhyper_AB_1st[a,b,c] + lhyper_AB_2nd[a,b,c] )/2

        Beta_avg = 0
        for i in range(0,3):
            Beta_avg += (self.lhyper_AB[2,i,i] + self.lhyper_AB[i,2,i] + self.lhyper_AB[i,i,2])/5

        print("Beta_zxx = %10.12lf" %(self.lhyper_AB[2,0,0]))
        print("Beta_xzx = %10.12lf" %(self.lhyper_AB[0,2,0]))
        print("Beta_xxz = %10.12lf" %(self.lhyper_AB[0,0,2]))
        print("Beta_zyy = %10.12lf" %(self.lhyper_AB[2,1,1]))
        print("Beta_yzy = %10.12lf" %(self.lhyper_AB[1,2,1]))
        print("Beta_yyz = %10.12lf" %(self.lhyper_AB[1,1,2]))
        print("Beta_zzz = %10.12lf" %(self.lhyper_AB[2,2,2]))

        print("Beta_avg = %10.12lf" %(Beta_avg))
        print("\n First Dipole Hyperpolarizability computed in %.3f seconds.\n" % (time.time() - solver_start))

        return self.lhyper_AB, Beta_avg

    def local_solve_right(self, lpertbar, omega, conv_hbar, e_conv=1e-12, r_conv=1e-12, maxiter=200):#max_diis=7, start_diis=1):
        solver_start = time.time()

        no = self.no
        contract =self.contract

        Avo = lpertbar.Avo.copy()
        Avvoo = lpertbar.Avvoo.copy()

        self.X1 = []
        self.X2 = []
        for i in range(no):
            ii = i * no + i

            lX1 = Avo[ii].copy()
            lX1 = lX1/ (self.H.F[i,i] - self.Local.eps[ii].reshape(-1,) + omega)

            #testing out the pertamps preconditioner
            #lX1 = lX1/ (self.H.F[i,i] - self.Local.eps[ii].reshape(-1,) + omega)

            self.X1.append(2.0 *lX1)

            for j in range(no):
                ij = i * no + j
                lX2 = Avvoo[ij].copy()
                lX2 = lX2/(self.H.F[i,i] + self.H.F[j,j] - self.Local.eps[ij].reshape(1,-1) - self.Local.eps[ij].reshape(-1,1) + omega)
                self.X2.append(2.0 *lX2)

        pseudo = self.local_pseudoresponse(lpertbar, self.X1, self.X2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        #diis = helper_diis(X1, X2, max_diis)

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            r1 = self.lr_X1(lpertbar, omega)
            r2 = self.lr_X2(lpertbar, conv_hbar, omega)

            rms = 0
            for i in range(no):
                ii = i * no + i

                self.X1[i] += r1[i] / (self.H.F[i,i] - self.Local.eps[ii].reshape(-1,) + omega)
                rms += contract('a,a->', np.conj(r1[i] /(self.H.F[i,i] - self.Local.eps[ii].reshape(-1,) + omega)), (r1[i] / (self.H.F[i,i] - self.Local.eps[ii].reshape(-1,) + omega)))

                for j in range(no):
                    ij = i*no + j

                    self.X2[ij] += r2[ij] / (self.H.F[i,i] + self.H.F[j,j] - self.Local.eps[ij].reshape(1,-1) - self.Local.eps[ij].reshape(-1,1)  + omega)
                    rms += contract('ab,ab->', np.conj(r2[ij]/(self.H.F[i,i] + self.H.F[j,j] - self.Local.eps[ij].reshape(1,-1) - self.Local.eps[ij].reshape(-1,1)  + omega)), 
                    r2[ij]/(self.H.F[i,i] + self.H.F[j,j] - self.Local.eps[ij].reshape(1,-1) - self.Local.eps[ij].reshape(-1,1)  + omega))

            rms = np.sqrt(rms)

            pseudo = self.local_pseudoresponse(lpertbar, self.X1, self.X2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")

            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.X1, self.X2, pseudo

            if niter == maxiter:
                print("\nPerturbed wave function not fully converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.X1, self.X2, pseudo

            #diis.add_error_vector(self.X1, self.X2)
            #if niter >= start_diis:
            #    self.X1, self.X2 = diis.extrapolate(self.X1, self.X2)    

    def local_solve_left(self, lpertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200): #, max_diis=7, start_diis=1):
        solver_start = time.time()
        no = self.no
        contract =self.contract

        Q = self.Local.Q
        L = self.Local.L

        QL = self.Local.QL
        Avo = lpertbar.Avo.copy()
        Avvoo = lpertbar.Avvoo.copy()

        #initial guess for Y 
        self.Y1 = []
        self.Y2 = []

        for i in range(no):
            ii = i * no + i
            QL_ii = Q[ii] @ L[ii]

            lX1 = Avo[ii].copy()
            lX1 /= (self.H.F[i,i] - self.Local.eps[ii].reshape(-1,) + omega)
            self.Y1.append(2.0 * lX1.copy())

            for j in range(no):
                ij = i * no + j

                lX2 = Avvoo[ij].copy()/(self.H.F[i,i] + self.H.F[j,j] - self.Local.eps[ij].reshape(1,-1) - self.Local.eps[ij].reshape(-1,1) + omega)
                self.Y2.append((4.0 * lX2.copy()) - (2.0 * lX2.copy().swapaxes(0,1)))

        pseudo = self.local_pseudoresponse(lpertbar, self.Y1, self.Y2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        ## uses updated X1 and X2
        self.im_Y1 = self.in_lY1(lpertbar, self.X1, self.X2)
        self.im_Y2 = self.in_lY2(lpertbar, self.X1, self.X2)

        #diis = helper_diis(X1, X2, max_diis)

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            r1 = self.lr_Y1(lpertbar, omega)
            r2 = self.lr_Y2(lpertbar, omega)

            rms = 0
            for i in range(no):
                ii = i * no + i

                self.Y1[i] += r1[i] / ( self.H.F[i,i] - self.Local.eps[ii].reshape(-1,) + omega)
                rms += contract('a,a->', np.conj(r1[i]/(self.H.F[i,i] - self.Local.eps[ii].reshape(-1,) + omega)), (r1[i]/(self.H.F[i,i] - self.Local.eps[ii].reshape(-1,) + omega)))

                for j in range(no):
                    ij = i*no + j

                    self.Y2[ij] += r2[ij] / (self.H.F[i,i] + self.H.F[j,j] - self.Local.eps[ij].reshape(1,-1) - self.Local.eps[ij].reshape(-1,1) + omega)
                    rms += contract('ab,ab->', np.conj(r2[ij]/(self.H.F[i,i] + self.H.F[j,j] - self.Local.eps[ij].reshape(1,-1) - self.Local.eps[ij].reshape(-1,1) + omega)), 
                    r2[ij]/(self.H.F[i,i] + self.H.F[j,j] - self.Local.eps[ij].reshape(1,-1) - self.Local.eps[ij].reshape(-1,1) + omega))

            rms = np.sqrt(rms)

            pseudo = self.local_pseudoresponse(lpertbar, self.Y1, self.Y2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")

            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.Y1, self.Y2, pseudo

            if niter == maxiter:
                print("\nPerturbed wave function not fully converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.Y1, self.Y2, pseudo

        #    #diis.add_error_vector(self.X1, self.X2)
        #    #if niter >= start_diis:
        #        #self.X1, self.X2 = diis.extrapolate(self.X1, self.X2)

    def lr_X1(self, lpertbar, omega):
        lX1_start = process_time()
        contract = self.contract
        no = self.ccwfn.no
        v = self.ccwfn.v
        hbar = self.hbar
        Avo = lpertbar.Avo
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2 
        ERI = self.H.ERI
        L = self.H.L 
        Sijmn = self.Local.Sijmn
        QL = self.Local.QL


        lr_X1_all = []
        for i in range(no):
            ii = i*no + i

            lr_X1 = (Avo[ii] - omega * self.X1[i]).copy()
            lr_X1 = lr_X1 + contract('e, ae ->a', self.X1[i], hbar.Hvv[ii]) 
            for m in range(no):
                mm = m*no + m 
                mi = m*no + i 
                im = i*no + m
                iimm = ii*(no*no) + mm
                iimi = ii*(no*no) + mi
 
                lr_X1 = lr_X1 - ((self.X1[m] @ Sijmn[iimm].T) * hbar.Hoo[m,i]) 
                
                lr_X1 = lr_X1 + contract('e, ae -> a', self.X1[m], 2.0 * hbar.Hovvo_mm[mi] - hbar.Hovov_mm[mi]) 
       
                lr_X1 = lr_X1 + 2.0 * contract('e, ea -> a', hbar.Hov[mi][m], self.X2[mi] @ Sijmn[iimi].T) 
                lr_X1 = lr_X1 - contract('e, ae -> a', hbar.Hov[mi][m], Sijmn[iimi] @ self.X2[mi]) 
 
                lr_X1 = lr_X1 + contract('ef, aef -> a', self.X2[im], 2.0 * hbar.Hvovv_ii[im][:,m,:,:] - hbar.Hvovv_ii[im][:,m,:,:].swapaxes(1,2))   

                for n in range(no):
                    mn = m*no + n
                    iimn = ii*(no*no) + mn

                    lr_X1 = lr_X1 - contract('ae, e -> a', Sijmn[iimn] @ self.X2[mn], 2.0 * hbar.Hooov[mn][m,n,i] - hbar.Hooov[mn][n,m,i]) 
            lr_X1_all.append(lr_X1)

        lX1_end = process_time()
        self.lX1_t += lX1_end - lX1_start

        return lr_X1_all

    def lr_X2(self, lpertbar, conv_hbar, omega):
        lX2_start = process_time()
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        X1 = self.X1
        X2 = self.X2
        t2 = self.lccwfn.t2
        hbar = self.hbar
        L = self.H.L

        dim = self.Local.dim
        QL = self.Local.QL

        Zoo = np.zeros((no,no))
        for i in range(no):
            for m in range(no):
                im = i*no + m
                for n in range(no):
                    imn = im*no + n
                    _in = i*no + n
                    Zoo[m,i] -= contract('n,n->', (2.0 * hbar.Hmnie[imn] - hbar.Hnmie[imn]), X1[n]) 
                    tmp = contract('ef, eE, fF->EF', L[m,n, v, v], QL[_in], QL[_in])  
                    Zoo[m,i] -= contract('ef,ef->', tmp, X2[_in])

        Zvv = []
        Sijmn = self.Local.Sijmn
        for i in range(no):
            for j in range(no):
                ij = i*no + j
                lZvv = np.zeros((dim[ij], dim[ij])) 
                for m in range(no):
                    mm = m*no + m
                    ijm = ij*no + m
                    mij = m*(no*no) + ij 
                    
                    lZvv += contract('aef,f->ae', (2.0 * hbar.Hvovv_imn[mij] - hbar.Hvovv_imns[mij].swapaxes(1,2)), X1[m]) 
                    for n in range(no):
                        mn = m*no + n
                        ijmn = ijm * no + n
                        tmp = contract('ef, eE, fF->EF', L[m,n,v,v], QL[ij], QL[mn])
                        lZvv -= contract('ef, af->ae', tmp, Sijmn[ijmn] @ X2[mn]) 
                Zvv.append(lZvv) 
 
        lr2 = []
        tmp_r2 = []
        Sijmj = self.Local.Sijmj 
        Sijim = self.Local.Sijim
        Sijmi = self.Local.Sijmi
        Sijmn = self.Local.Sijmn
        for i in range(no):
            ii = i*no + i
            for j in range(no):
                ij = i*no + j 
                jj = j*no + j
            
                r2 = np.zeros(dim[ij],dim[ij])
     
                #first term
                r2 = lpertbar.Avvoo[ij] - 0.5 *omega *X2[ij] 
  
                #second term
                r2 = r2 + contract('e, abe ->ab', X1[i], hbar.Hvvvo_im[ij])

                #fifth term
                r2 = r2 + contract('eb,ae->ab', t2[ij], Zvv[ij])
    
                #sixth term 
                r2 = r2 + contract('eb, ae->ab', X2[ij], hbar.Hvv[ij]) 

                #ninth term 
                r2 = r2 + 0.5 * contract('ef,abef->ab', X2[ij], hbar.Hvvvv[ij])
                   
                for m in range(no): 
                    ijm = ij*no + m 
                    mj = m*no + j 
                    im = i*no + m
                    mi = m*no + i 
                    mij = mi*no + j

                    #third term
                    r2 = r2 - contract('a,b->ab', X1[m] @ self.Local.Sijmm[ijm].T, hbar.Hovoo_ij[ijm]) 
 
                    #fourth term
                    r2 = r2 + Zoo[m,i] * self.Local.Sijmj[ijm] @ t2[mj] @ self.Local.Sijmj[ijm].T 

                    #seventh term 
                    r2 = r2 - ((Sijmj[ijm] @ X2[mj] @Sijmj[ijm].T) * hbar.Hoo[m,i]) 

                    #tenth term
                    #replaced hbar.Hovov_im to hbar.Hovov_mj[mij] 
                    r2 = r2 - contract('eb,ae->ab', X2[im] @ Sijim[ijm].T, hbar.Hovov_mj[mij])   

                    #eleventh term
                    #replaced hbar.Hovvo_im to hbar.Hovvo_mj[mij]
                    r2 = r2 - contract('ea,be->ab', X2[im] @ Sijim[ijm].T, hbar.Hovvo_mj[mij]) 

                    #twelveth term
                    #replacing hbar.Hmvvj_mi to hbar.Hovvo_mj[mij]             
                    r2 = r2 + 2.0 * contract('ea, be->ab', X2[mi] @ Sijmi[ijm].T, hbar.Hovvo_mj[mij])

                    #thirteenth term
                    #replaced hbar.Hovov_im to hbar.Hovov_mj[mij] 
                    r2 = r2 - contract('ea, be->ab', X2[mi] @ Sijmi[ijm].T, hbar.Hovov_mj[mij]) 

                    for n in range(no):
                        mn = m*no +n 
                        ijmn = ijm*no +n

                        #eight term 
                        r2 = r2 + (0.5 * (Sijmn[ijmn] @ X2[mn] @ Sijmn[ijmn].T) * hbar.Hoooo[m,n,i,j]) 
                tmp_r2.append(r2)

        for ij in range(no*no):
            i = ij // no 
            j = ij % no 
            ji = j*no + i 
   
            lr2.append(tmp_r2[ij].copy() + tmp_r2[ji].copy().transpose())            
       
        lX2_end = process_time()
        self.lX2_t += lX2_end - lX2_start 
        return lr2    

    def in_lY1(self, lpertbar, X1, X2):
        lY1_start = process_time()
        contract = self.contract
        no = self.ccwfn.no
        v = self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2
        hbar = self.hbar
        ERI = self.H.ERI
        L = self.H.L
        Sijmn = self.Local.Sijmn
        QL = self.Local.QL
        mu = lpertbar.pert
        ERIoovv = self.Local.ERIoovv

        in_Y1 = []
        for i in range(no): 
            ii = i * no + i 

            # <O|A_bar|phi^a_i> good
            r_Y1 = 2.0 * lpertbar.Aov[ii][i].copy()

            #collecting Gvv terms here
            for m in range(no): 
                for n in range(no):
                    nn = n*no + n
                    mn = m*no + n 
                    iimn = ii*(no*no) + mn
       
                    Gvv = -1.0 * contract('ab,eb -> ea', QL[ii].T @ L[m,n,v,v] @ QL[mn], X2[mn])            
                    r_Y1 = r_Y1 + contract('e, ea ->a', Sijmn[iimn].T @ l1[i], Gvv) 

            for n in range(no): 
                nn = n*no + n
                for m in range(no): 
                    for _o in range(no):
                        mo = m*no + _o
                        nnmo = nn*(no*no) + mo
  
                        Gvv = -1.0 * contract('bc,fc -> fb', Sijmn[nnmo] @ l2[mo], t2[mo]) 
                        tmp = contract('b, fb ->f', X1[n], Gvv)  
                        r_Y1 = r_Y1 + contract('af, f -> a', QL[ii].T @ L[i,n,v,v] @ QL[mo], tmp) 

            for m in range(no): 
                mm = m*no + m 
                for n in range(no):
                    for _o in range(no):
                        _no = n*no + _o 
                        iino = ii*(no*no) + _no                      

                        Gvv = -1.0 * contract('ac,fc -> fa', Sijmn[iino] @ l2[_no], t2[_no]) 
                        tmp = contract('e, fa -> efa', X1[m], Gvv) 
                        r_Y1 = r_Y1 + contract('ef, efa -> a', QL[mm].T @ L[m,i,v,v] @ QL[_no], tmp)             

            for m in range(no):
                for n in range(no):
                    mn = m*no + n
                    iimn = ii*(no*no) + mn

                    Gvv = -1.0 * contract('ab, eb -> ea', Sijmn[iimn] @ l2[mn], X2[mn])
                    r_Y1 = r_Y1 + contract('e, ea -> a', hbar.Hov[mn][i], Gvv)

            for m in range(no):
                for n in range(no):
                    mn = m*no + n
                    imn = i*(no*no) + mn

                    Gvv = -1.0 * contract('eb, gb -> ge', X2[mn], l2[mn]) 
                    r_Y1 = r_Y1 + contract('gea, ge -> a', -2.0 * hbar.Hvovv_imn[imn] + hbar.Hvovv_imns[imn].swapaxes(1,2), Gvv)            

            #Goo terms 
            for n in range(no):
                for _o in range(no):
                    _no = n*no + _o
                    io = i*no + _o 
                    iono = io*(no*no) + _no 
                    Goo = contract('ab, ab->', Sijmn[iono] @ t2[_no] @ Sijmn[iono].T, l2[io])

                    for m in range(no):
                        mm = m*no + m
                        tmp_X = X1[m] * Goo
                        r_Y1 = r_Y1 - contract('e, ea -> a', tmp_X, QL[mm].T @ L[m,n,v,v] @ QL[ii])
            
            #check this one again - Expression 8 G_in
            for n in range(no):
                for m in range(no): 
                    mm = m*no + m
                    X_tmp = contract('e,ea ->a', X1[m], QL[mm].T @ L[m,n,v,v] @ QL[ii]) 
                    for _o in range(no):
                        _no = n*no + _o
                        io = i*no + _o
                        iono = io*(no*no) + _no
                        Goo = contract('ab, ab ->', Sijmn[iono] @ t2[_no] @ Sijmn[iono].T, l2[io]) 
                        #r_Y1 = r_Y1 - (Goo * X_tmp)  

            for j in range(no):
                jj = j*no +j 
                for n in range(no):
                    for m in range(no):
                        nm = n*no + m 
                        jm = j*no + m
                        nmjm = nm*(no*no) + jm

                        Goo = contract('ab, ab->', Sijmn[nmjm].T @ t2[nm] @ Sijmn[nmjm], l2[jm]) 
                        tmp = X1[j] * Goo
                        r_Y1 = r_Y1 - contract('f, af ->a', tmp, QL[ii].T @ L[i,n,v,v] @ QL[jj])

            for m in range(no):
                for n in range(no): 
                    mn = m*no +n 
                    _in = i*no + n 
                    mnin = mn*(no*no) + _in

                    Goo = contract('ab, ab ->', X2[mn], Sijmn[mnin] @ l2[_in] @ Sijmn[mnin].T)
                    r_Y1 = r_Y1 - (Goo * hbar.Hov[ii][m]) 

            for m in range(no):
                for _o in range(no):
                    oo = _o*no + _o
                    for n in range(no):
                        mn = m*no + n
                        on = _o*no + n
                        mnon = mn*(no*no) + on 

                        Goo = contract('ab, ab ->', X2[mn], Sijmn[mnon] @ l2[on] @ Sijmn[mnon].T ) 
                        r_Y1 = r_Y1 + ((-2.0 * hbar.Hooov[ii][m,i,_o] + hbar.Hooov[ii][i,m,_o]) * Goo) 

            # <O|L1(0)|A_bar|phi^a_i> good
            for m in range(no):
                mm = m*no + m 
                iimm = ii*(no*no) + mm

                r_Y1 = r_Y1 - (lpertbar.Aoo[i,m] * l1[m] @ Sijmn[iimm].T)
           
            r_Y1 = r_Y1 + contract('e, ea -> a', l1[i], lpertbar.Avv[ii]) 
     
            #<O|L2(0)|A_bar|phi^a_i> 
            for m in range(no):
                im = i*no + m
                mi = m*no + i
                mm = m*no + m 
                iimm = ii*(no*no) + mm
                miim = mi*(no*no) + im
                immm = im*(no*no) + mm
                iim = ii*no + m
                mmi = mm*no + i 

                Avvvo = 0
                #for m sum in Avvvo becomes n since m is being used for the og terms
                for n in range(no):
                    nm = n*no + m 
                    nmim = nm*(no*no) + im 
                    Avvvo = Avvvo - contract('fe,a -> fea', Sijmn[nmim].T @ t2[nm] @ Sijmn[nmim], mu[n,v].copy() @ QL[ii]) 
                r_Y1 = r_Y1 + contract('fe, fea -> a', l2[im], Avvvo) 
                
                for n in range(no):
                    nm = n*no + m 
                    mn = m*no + n 
                    mnii = nm*(no*no) + ii 
                    nmmn = nm*(no*no) + mn

                    Aovoo = contract('fe, f->e', t2[nm] @ Sijmn[nmmn], mu[i,v] @ QL[nm]) 
                    r_Y1 = r_Y1 -0.5 * contract('e, ea -> a', Aovoo, l2[mn] @ Sijmn[mnii])

                    Aovoo = contract('fe, f->e', t2[mn], mu[i,v] @ QL[mn]) 
                    r_Y1 = r_Y1 -0.5 * contract('e, ae -> a', Aovoo, Sijmn[mnii].T @ l2[mn]) 

                # <O|[Hbar(0), X1]|phi^a_i>
                Loovv = QL[ii].T @ L[i,m,v,v] @ QL[mm]
                r_Y1 = r_Y1 + 2.0 * contract('ae, e ->a', Loovv, X1[m]) 

                # <O|L1(0)|[Hbar(0), X1]|phi^a_i>          
                tmp = -1.0 * contract('a, e -> ae', hbar.Hov[ii][m], l1[i] @ Sijmn[iimm]) 
                tmp = tmp - contract('a, e -> ae', Sijmn[iimm] @ l1[m], hbar.Hov[mm][i]) 
               
                for n in range(no):
                    nn = n* no + n 
                    nnmm = nn*(no*no) + mm
                    nnii = nn*(no*no) + ii
  
                    tmp = tmp + contract('a,e -> ae', -2.0 * hbar.Hooov[ii][m,i,n] + hbar.Hooov[ii][i,m,n], l1[n] @ Sijmn[nnmm]) 
                    tmp = tmp + contract('e,a -> ae', -2.0 * hbar.Hooov[mm][i,m,n] + hbar.Hooov[mm][m,i,n], l1[n] @ Sijmn[nnii])   
                    
                tmp = tmp + contract('fae, f -> ae', 2.0 * hbar.Hamef[iim] - hbar.Hamfe[iim].swapaxes(1,2), l1[i])
            
                tmp = tmp + contract('fea, f -> ae', 2.0 * hbar.Hamef[mmi]- hbar.Hamfe[mmi].swapaxes(1,2), l1[m]) 
                r_Y1 = r_Y1 + contract('ae, e -> a', tmp, X1[m]) 

                # <O|L1(0)|[Hbar(0), X2]|phi^a_i>
                for n in range(no): 
                    nn = n*no + n 
                    mn = m*no + n 
                    nm = n*no + m 
                    ni = n*no + i 
                    _in = i*no + n    
                    imn = im*no + n 
                    _min = mi*no + n 
                    nimm = ni*(no*no) + mm
                    nnmm = nn*(no*no) + mm
                    nmii = nm*(no*no) + ii
                    inmm = _in*(no*no) + mm
                    nmmm = nm*(no*no) + mm
                    nnmn = nn*(no*no) + mn
                    iini = ii*(no*no)+ ni
                    iinn = ii*(no*no) + nn
                    iimn = ii*(no*no) + mn

                    tmp = 2.0 * contract('ef, f -> e', X2[mn], l1[n] @ Sijmn[nnmn]) 
                    tmp =  tmp - contract('fe, f -> e', X2[mn], l1[n] @ Sijmn[nnmn])
                    Loovv = QL[ii].T @ L[i,m,v,v] @ QL[mn] 
                    r_Y1 = r_Y1 + contract('ae, e -> a', Loovv, tmp) 
                
                    Goo = contract('ab, ab ->', X2[nm], self.Local.Loovv[nm][i,m]) 
                    r_Y1 = r_Y1 - (Goo * l1[n] @ Sijmn[iinn].T) 
                    
                    # <O|L2(0)|[Hbar(0), X1]|phi^a_i>
                    tmp1 = -1.0 * contract('ef, fa -> ea', Sijmn[nimm].T @ l2[ni] , hbar.Hovov_ni[imn]) 
                
                    tmp1 = tmp1 -  contract('fe, af -> ea', hbar.Hovov_ni[_min] , Sijmn[nmii].T @ l2[nm]) 
                    
                    tmp1 = tmp1 - contract('ef,fa -> ea', Sijmn[inmm].T @ l2[_in], hbar.Hovvo_ni[imn]) 

                    tmp1 = tmp1 - contract('fe,fa -> ea', hbar.Hovvo_ni[_min], l2[nm] @ Sijmn[nmii]) 
                    
                    for _o in range(no):
                        oo = _o*no + _o
                        on = _o*no + n
                        _no = n*no + _o 
                        nomm = _no*(no*no) + mm
                        noii = _no*(no*no) + ii 
                        onmm = on*(no*no) + mm
                        onii = on*(no*no) + ii 
                        
                        tmp1 = tmp1 + 0.5 * hbar.Hoooo[i,m,n,_o] * (Sijmn[onmm].T @ l2[on] @ Sijmn[onii])

                        tmp1 = tmp1 + 0.5 * hbar.Hoooo[m,i,n,_o] * (Sijmn[nomm].T @ l2[_no] @ Sijmn[noii]) 

                    r_Y1 = r_Y1 + contract('ea,e ->a', tmp1, X1[m])       
                                           
                tmp1 = 0.5 * contract('fg, fgae -> ea', l2[im], hbar.Hvvvv_im[im]) 
 
                tmp1 = tmp1 + 0.5 * contract('gf, fgea -> ea', l2[im], hbar.Hvvvv_im[mi]) 
            
                r_Y1 = r_Y1 + contract('ea,e ->a', tmp1, X1[m])

                for n in range(no):
                    mn = m*no + n 
                    ni = n*no + i 
                    immn = im*(no*no) + mn
                    mimn = mi*(no*no) + mn
                    iimn = ii*(no*no) + mn 
                    mnni = mn*(no*no) + ni 
                    nm = n*no + m 
                    nmi = nm*no + i 
                    nim = ni*no + m
                    imn = i*(no*no) + mn
                    inm = i*(no*no) + nm

                    #g_im e_mn
                    tmp = contract('fg,ef->ge', Sijmn[immn].T @ l2[im], X2[mn]) 
                    r_Y1 = r_Y1 - contract('ge, gea -> a', tmp, hbar.Hgnea[imn])  
                     
                    #g_mi e_mn
                    tmp = contract('fg,ef->ge', Sijmn[mimn].T @ l2[mi], X2[mn])        
                    r_Y1 = r_Y1 - contract('ge, gae -> a', tmp, hbar.Hgnae[imn]) 
            
                    #g_mn a_ii e_mn f_mn
                    #v^4
                    tmp = contract('ga,ef->gaef', l2[mn] @ Sijmn[iimn].T , X2[mn])
                    r_Y1 = r_Y1 - contract('gef, gaef -> a', hbar.Hvovv_ij[mn][:,i], tmp)

                    #g_ni e_mn f_mn a_ii
                    tmp = contract('gae, ef -> gaf', 2.0 * hbar.Hgnae[inm] - hbar.Hgnea[inm].swapaxes(1,2), X2[mn]) 
                    r_Y1 = r_Y1 + contract('fg, gaf -> a', Sijmn[mnni] @ l2[ni], tmp) 

                    for _o in range(no): 
                        oi = _o*no + i 
                        oo = _o*no + _o
                        mo = m*no + _o 
                        on = _o*no + n
                        _no = n*no + _o
                        oimn = oi*(no*no) + mn
                        momn = mo*(no*no) + mn                        
                        moii = mo*(no*no) + ii 
                        onmn = on*(no*no) + mn
                        noii = _no*(no*no) + ii
                        onii = on*(no*no) + ii
                        nomn = _no*(no*no) + mn
 
                        tmp = contract('ef, ef ->', Sijmn[oimn].T @ l2[oi] @ Sijmn[oimn], X2[mn])
                        r_Y1 = r_Y1 + (tmp * hbar.Hooov[ii][m,n,_o]) 
    
                        tmp = contract('fa, ef -> ae', Sijmn[momn].T @ l2[mo] @ Sijmn[moii], X2[mn])
                        r_Y1 = r_Y1 + contract('e, ae -> a', hbar.Hooov[mn][i,n,_o], tmp) 

                        #a_ii f_mn
                        tmp = contract('ea, ef -> af', Sijmn[onmn].T @ l2[on] @ Sijmn[onii], X2[mn]) 
                        r_Y1 = r_Y1 + contract('f, af -> a', hbar.Hooov[mn][m,i,_o], tmp) 

                        tmp = contract('e, ef -> f', -2.0 * hbar.Hooov[mn][i,m,_o] + hbar.Hooov[mn][m,i,_o], X2[mn]) 
                        r_Y1 = r_Y1 + contract('f, fa -> a', tmp, Sijmn[nomn].T @ l2[_no] @ Sijmn[noii]) 
            in_Y1.append(r_Y1)

        lY1_end = process_time()
        self.lY1_t += lY1_end - lY1_start
        return in_Y1

    def lr_Y1(self, lpertbar, omega):
        lY1_start = process_time()
        contract = self.contract 
        hbar = self.hbar
        no = self.ccwfn.no
        o = self.ccwfn.o
        v = self.ccwfn.v
        F = self.H.F
        ERI = self.H.ERI
        L = self.H.L
        QL = self.Local.QL
        Sijmn = self.Local.Sijmn
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2
        Y1 = self.Y1
        Y2 = self.Y2

        #imhomogenous terms
        r_Y1 = self.im_Y1.copy()
        
        for i in range(no): 
            ii = i*no + i 

            r_Y1[i] = r_Y1[i] + (omega * Y1[i])
            r_Y1[i] = r_Y1[i] + contract('e, ea -> a', Y1[i], hbar.Hvv[ii]) 
            r_lY1 = 0

            #collecting Gvv terms here
            for m in range(no):
                for n in range(no):
                    mn = m*no + n
                    imn = i*(no*no) + mn

                    Gvv = -1.0 * contract('fb, eb -> ef', t2[mn], Y2[mn])
                    r_lY1 = r_lY1 + contract('efa, ef -> a', -2.0 * hbar.Hvovv_imn[imn] + hbar.Hvovv_imns[imn].swapaxes(1,2), Gvv)

            for m in range(no):
                for _o in range(no):
                    mo = m*no + _o
                    for n in range(no):
                        nn = n*no + n
                        _no = n*no + _o
                        mono = mo*(no*no) + _no    

                        Goo = contract('ab, ab ->', Sijmn[mono].T @ t2[mo] @ Sijmn[mono], Y2[_no])
                        r_lY1 = r_lY1 + ((-2.0 * hbar.Hooov[ii][m,i,n] + hbar.Hooov[ii][i,m,n]) * Goo)
 
            for m in range(no): 
                mm = m*no + m
                im = i*no + m 
                iimm = ii*(no*no) + mm
                mmim = mm*(no*no) + im 
                
                r_lY1 = r_lY1 - (hbar.Hoo[i,m] * Y1[m] @ Sijmn[iimm].T) 

                r_lY1 = r_lY1 + contract('ea,e -> a', 2.0 * hbar.Hovvo_mm[im] - hbar.Hovov_mm[im], Y1[m]) 
                   
                ##e_im f_im a_ii
                r_lY1 = r_lY1 + contract('ef, efa -> a', Y2[im], hbar.Hvvvo_im[im]) 
 
                for n in range(no): 
                    mn = m*no + n
                    iimn = ii*(no*no) + mn
                    mmmn = mm*(no*no) + mn
                    imn = im*no + n
                    mni = mn*no + i 

                    #replacing hbar.Hovoo_mn[imn] to hbar.Hovoo_ij[mni]
                    r_lY1 = r_lY1 - contract('e, ae -> a', hbar.Hovoo_ij[mni], Sijmn[iimn] @ Y2[mn])  
            r_Y1[i] = r_Y1[i] + r_lY1

        lY1_end = process_time()
        self.lY1_t += lY1_end - lY1_start   
        return r_Y1 
   
    def in_lY2(self, lpertbar, X1, X2):
        lY2_start = process_time()
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        Y1 = self.Y1
        Y2 = self.Y2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L
        ERI = self.H.ERI

        in_Y2 = []
 
        QL = self.Local.QL  
        Sijii = self.Local.Sijii
        Sijjj = self.Local.Sijjj
        Sijmj = self.Local.Sijmj
        Sijmm = self.Local.Sijmm
        Sijim = self.Local.Sijim
        Sijmn = self.Local.Sijmn

        for i in range(no):
            ii = i*no + i 
            for j in range(no):
                ij = i*no + j
                jj = j*no + j
                
                # <O|L1(0)|A_bar|phi^ab_ij>, Eqn 162
                r_Y2  = 2.0 * contract('a,b->ab', l1[i] @ Sijii[ij].T, lpertbar.Aov[ij][j].copy())
                r_Y2 = r_Y2 - contract('a,b->ab', l1[j] @ Sijjj[ij].T, lpertbar.Aov[ij][i].copy())

                # <O|L2(0)|A_bar|phi^ab_ij>, Eqn 163
                r_Y2 += contract('eb,ea->ab', l2[ij], lpertbar.Avv[ij])
               
                #collecting Gvv terms here
                for m in range(no):
                    for n in range(no):
                        mn = m*no + n
                        ijmn = ij*(no*no) + mn

                        Gvv = -1.0 * contract('fe, ae ->af', X2[mn], QL[ij].T @ L[m,n,v,v] @ QL[mn])
                        r_Y2 = r_Y2 + contract('fb, af -> ab', Sijmn[ijmn].T @ l2[ij], Gvv)                            

                for m in range(no):
                    for n in range(no):
                        mn = m*no + n
                        ijmn = ij*(no*no) + mn 

                        Gvv = -1.0 * contract('ef, bf -> be', X2[mn], Sijmn[ijmn] @ l2[mn])
                        r_Y2 = r_Y2 + contract('ae, be -> ab', QL[ij].T @ L[i,j,v,v] @ QL[mn], Gvv)             
       
                #last Goo here
                for m in range(no): 
                    for n in range(no): 
                        mn = m*no + n 
                        jn = j*no + n 
                        mnjn = mn*(no*no) + jn

                        Goo = contract('ab, ab ->', Sijmn[mnjn] @ l2[jn] @ Sijmn[mnjn].T, X2[mn])
                        r_Y2 = r_Y2 - (self.Local.Loovv[ij][i,m] * Goo)             

                for m in range(no):
                    mj = m*no + j
                    ijm = ij*no + m

                    tmp = Sijmj[ijm] @ l2[mj] @ Sijmj[ijm].T 
                    r_Y2 = r_Y2 - lpertbar.Aoo[i,m] * tmp

                # <O|L1(0)|[Hbar(0), X1]|phi^ab_ij>, Eqn 164
                for m in range(no):
                    ijm = ij*no + m
                    mm = m*no + m
                    iim = ii*no + m

                    tmp = contract('e,a->ea', X1[m], (l1[j] @ Sijjj[ij].T))
                    tmp1 = contract('eb, eE, bB ->EB', L[m,i,v,v], QL[mm], QL[ij])  
                    r_Y2 = r_Y2 - contract('eb, ea-> ab', tmp1, tmp) 
        
                    tmp = contract('e,b->eb', X1[m], (l1[m] @ Sijmm[ijm].T))  
                    tmp1 = contract('ae, aA, eE ->AE', L[i,j,v,v], QL[ij], QL[mm]) 
                    r_Y2 = r_Y2 - contract('ae, eb-> ab', tmp1, tmp) 

                    tmp = contract('e,e->', X1[m], (l1[i] @ Sijmm[iim]))   
                    r_Y2 = r_Y2 - tmp * self.Local.Loovv[ij][j,m].swapaxes(0,1) 

                    tmp = 2.0 * contract('e,b ->eb', X1[m], (l1[j] @ Sijjj[ij].T))
                    tmp1 = contract('ae, aA, eE ->AE', L[i,m,v,v], QL[ij], QL[mm])
                    r_Y2 = r_Y2 + contract('ae, eb-> ab', tmp1, tmp)
           
                # <O|L2(0)|[Hbar(0), X1]|phi^ab_ij>, Eqn 165
                for m in range(no):
                    mm = m*no + m
                    ijm = ij*no + m
                    jm = j*no + m
                    im = i*no + m 
 
                    tmp = contract('e,a-> ea',  X1[m], hbar.Hov[ij][m]) 
                    r_Y2 = r_Y2 - contract('eb,ea->ab', Sijmm[ijm].T @ l2[ij], tmp)
                    
                    tmp = contract('e,e->', X1[m], hbar.Hov[mm][i]) 
                    r_Y2 = r_Y2 - tmp * (Sijmj[ijm] @ l2[jm] @ Sijmj[ijm].T).swapaxes(0,1) 
 
                    #may need to double-check this one
                    tmp = contract('e,ef->f', X1[m], Sijmm[ijm].T @ l2[ij]) 
                    r_Y2 = r_Y2 - contract('f, fba -> ab', tmp, hbar.Hvovv_ij[ij][:,m,:,:])

                    tmp = contract('e,bf->ebf', X1[m], Sijim[ijm] @ l2[im]) 
                    r_Y2 = r_Y2 - contract('ebf, fea->ab', tmp, hbar.Hfjea[ijm]) 
 
                    tmp = contract('e,fa->efa', X1[m], l2[jm] @ Sijmj[ijm].T)
                    r_Y2 = r_Y2 - contract('fbe, efa->ab', hbar.Hfibe[ijm], tmp) 

                    tmp = contract('e, fae -> fa', X1[m], 2.0 * hbar.Hfmae[ijm] - hbar.Hfmea[ijm].swapaxes(1,2))
                    r_Y2 = r_Y2 + contract('fb,fa->ab', l2[ij], tmp)

                    tmp = contract('e, fea -> fa', X1[m], 2.0 * hbar.Hfieb[ijm] - hbar.Hfibe[ijm].swapaxes(1,2))
                    r_Y2  = r_Y2 + contract('fa,bf->ab', tmp, Sijmj[ijm] @ l2[jm]) 
 
                    for n in range(no):
                        ijmn = ijm*no + n 
                        _in = i*no + n
                        inm = _in * no + m
                        ijn = ij*no + n
                        ni = n*no + i
                        nim = ni*no + m
                        nm = n*no + m
                        ijnm = ij*(no*no) + nm
                        nj = n*no + j
                        njm = nj*no + m 
                        jn = j*no + n

                        imn = im*no + n

                        tmp = contract('e,a -> ea', X1[m], hbar.Hjmna[ijmn])
                        tmp1 = Sijmm[inm].T @ l2[_in] @ Sijim[ijn].T  
                        r_Y2 = r_Y2 + contract('eb,ea->ab', tmp1, tmp) 

                        tmp = contract('e,a -> ea', X1[m], hbar.Hmjna[ijmn])
                        tmp1 = Sijmm[nim].T @ l2[ni] @ Sijim[ijn].T  
                        r_Y2 = r_Y2 + contract('eb,ea->ab', tmp1, tmp) 

                        tmp = Sijmn[ijnm] @ l2[nm] @ Sijmn[ijnm].T
                        tmp = contract('e,ba->eba', X1[m], tmp) 
                        r_Y2 = r_Y2 + contract('eba,e->ab', tmp, hbar.Hjine[ijmn])

                        tmp = contract('e,a->ea', X1[m], 2.0*hbar.Hmine[ijmn] - hbar.Himne[ijmn])
                        tmp1 = Sijmm[njm].T @ l2[nj] @ Sijmj[ijn].T  
                        r_Y2 = r_Y2 - contract('ea, eb->ab', tmp, tmp1)     
 
                        tmp = contract('e,e->', X1[m], 2.0*hbar.Himne_mm[imn] - hbar.Hmine_mm[imn])
                        tmp1 = Sijmj[ijn] @ l2[jn] @ Sijmj[ijn].T
                        r_Y2 = r_Y2 - tmp * tmp1.swapaxes(0,1)

                #<O|L2(0)|[Hbar(0), X2]|phi^ab_ij>, Eqn 174
                Gin = np.zeros((no, no))
                for m in range(no):
                    ijm = ij*no + m
                    mi = m*no + i
                    im = i*no + m
                    mj = m*no + j 
                    for n in range(no):
                        mn = m*no +n
                        ijmn = ijm*(no) + n
                        imn = i*(no*no) + mn
                        min = mi*no + n 
                        mni = mn*no + i
                        nm = n*no + m
                        inm = i*(no*no) + nm
                        mjn = mj*no + n
                        jn = j*no + j
                        ijn = i*(no*no) + jn
                        ni = n*no + i 
                        nim = ni*no + m 
                        nj = n*no + j 
                        njm = nj*no + m
                        mnni = mn*(no*no) + ni
                        ijni = ij*(no*no) + ni
                        ijnj = ij*(no*no) + nj
                        mnnj = mn*(no*no) + nj

                        tmp = Sijmn[ijmn].T  @ l2[ij] @ Sijmn[ijmn]
                        tmp = 0.5 * contract('ef,ef->', tmp, X2[mn]) 
                        r_Y2 = r_Y2 + tmp * self.Local.ERIoovv[ij][m,n]
        
                        tmp = Sijmn[ijmn] @ l2[mn] @ Sijmn[ijmn].T
                        tmp1 = 0.5 * contract('fe,ef->', self.Local.ERIoovv[mn][i,j], X2[mn])
                        r_Y2 = r_Y2 + tmp1 * tmp.swapaxes(0,1)

                        tmp = Sijim[min].T @ l2[mi] @ Sijim[ijm].T    
                        tmp = contract('fb, ef-> be', tmp, X2[mn]) 
                        r_Y2 = r_Y2 + contract('be, ae->ab', tmp, QL[ij].T @ ERI[j,n,v,v] @ QL[mn]) 
                        
                        tmp = Sijim[min].T @ l2[im] @ Sijim[ijm].T
                        tmp = contract('fb, ef-> be', tmp, X2[mn])
                        r_Y2 = r_Y2 + contract('be, ae->ab', tmp, QL[ij].T @ ERI[n,j,v,v] @ QL[mn])                        
                         
                        tmp = Sijim[mjn].T @ l2[mj] @ Sijmj[ijm].T
                        tmp = contract('fb, ef-> be', tmp, X2[mn])
                        r_Y2 = r_Y2 - contract('be, ae->ab', tmp, QL[ij].T @ L[i,n,v,v] @ QL[mn])
 
                        # Expression 5, Term 10 
                        tmp = Sijmn[mnni] @ l2[ni] @ Sijmn[ijni].T 
                        tmp = contract('fb, ef-> be', tmp, X2[mn]) 
                        r_Y2 = r_Y2 - contract('be,ea->ab', tmp, QL[mn].T @ L[m,j,v,v] @ QL[ij]) 

                        # Expression 5, Term 11
                        tmp = Sijmn[mnnj] @ l2[nj] @ Sijmn[ijnj].T
                        tmp = 2.0 * contract('fb, ef-> be', tmp, X2[mn]) 
                        r_Y2 = r_Y2 + contract('ae, be-> ab', QL[ij].T @ L[i,m,v,v] @ QL[mn], tmp)  
 
                        #Goo term for Term 6   
                        Gin[i,n] += contract('ef,ef->', QL[nm].T @ L[i,m,v,v] @ QL[nm], X2[nm]) 
 
                for n in range(no):
                    ijn = ij*no + n
                    jn = j*no + n  
 
                    #Term 6
                    tmp = Sijmj[ijn] @ l2[jn] @ Sijmj[ijn].T
                    r_Y2 = r_Y2 - Gin[i,n] * tmp.swapaxes(0,1)
  
                in_Y2.append(r_Y2) 
        lY2_end = process_time()
        self.lY2_t += lY2_end - lY2_start
        return in_Y2
        
    def lr_Y2(self, lpertbar, omega):
        lY2_start = process_time()
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        Y1 = self.Y1
        Y2 = self.Y2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.lccwfn.t2
        hbar = self.hbar
        L = self.H.L
        ERI = self.H.ERI

        in_Y2 = []

        QL = self.Local.QL 
        Sijii = self.Local.Sijii
        Sijjj = self.Local.Sijjj
        Sijmj = self.Local.Sijmj
        Sijmm = self.Local.Sijmm
        Sijim = self.Local.Sijim
        Sijmn = self.Local.Sijmn

        tmp_Y2 = []
        lr_Y2 = []

        #build Goo and Gvv here
        Goo = self.cclambda.build_lGoo(t2, Y2)
        Gvv = self.cclambda.build_lGvv(t2, Y2)
 
        for i in range(no):
            for j in range(no):
                ij = i*no + j 
           
                #first term
                r_Y2 = self.im_Y2[ij].copy()

                #second term
                r_Y2 = r_Y2 + 0.5 * omega * self.Y2[ij].copy()

                #third term
                tmp1 = 2.0 * Sijii[ij] @ Y1[i]  
                r_Y2 = r_Y2 + contract('a,b->ab', tmp1, hbar.Hov[ij][j])
  
                #fourth term
                tmp = Sijjj[ij] @ Y1[j] 
                r_Y2 = r_Y2 - contract('a,b->ab', tmp, hbar.Hov[ij][i]) 

                #fifth term 
                r_Y2 = r_Y2 + contract('eb, ea -> ab', Y2[ij], hbar.Hvv[ij]) 

                #eigth term 
                r_Y2 = r_Y2 + 0.5 * contract('ef,efab->ab', Y2[ij], hbar.Hvvvv[ij])
 
                #ninth term 
                r_Y2 = r_Y2 + 2.0 * contract('e,eab->ab', Y1[i], hbar.Hvovv_ii[ij][:,j,:,:]) 
                
                #tenth term 
                r_Y2 = r_Y2 - contract('e,eba->ab', Y1[i], hbar.Hvovv_ii[ij][:,j,:,:])

                for m in range(no):
                    mi = m*no + i
                    mj = m*no + j 
                    ijm = ij*no + m
 
                    #sixth term
                    tmp = Sijmj[ijm] @ Y2[mj] @ Sijmj[ijm].T 
                    r_Y2 = r_Y2 - hbar.Hoo[i,m] * tmp  

                    #eleventh term and twelve term  
                    r_Y2 = r_Y2 - contract('b,a->ab', Sijmm[ijm] @ Y1[m], 2.0 * hbar.Hjiov[ij][m] - hbar.Hijov[ij][m]) 
        
                    #thirteenth term and fourteenth term  
                    r_Y2 = r_Y2 + contract('ea,eb -> ab', 2.0 * hbar.Hovvo_mj[ijm] - hbar.Hovov_mj[ijm], Y2[mj] @ Sijmj[ijm].T)  
 
                    #fifteenth term
                    tmp = Sijim[ijm] @ Y2[mi]
                    r_Y2 = r_Y2 - contract('be, ea->ab', Sijim[ijm] @ Y2[mi], hbar.Hovov_mi[ijm])  

                    #sixteenth term
                    #hbar.Hovvo_mi[ijm] can be changed to hbar.Hovvo_mj[mij] 
                    r_Y2 = r_Y2 - contract('eb, ea -> ab',  Y2[mi] @ Sijim[ijm].T, hbar.Hovvo_mi[ijm])  
                    
                    #eighteenth term
                    r_Y2 = r_Y2 - Goo[m,i] * self.Local.Loovv[ij][m,j] 

                    for n in range(no):
                        mn = m*no + n
                        ijmn = ij*(no*no) + mn 
 
                        #seventh term
                        tmp = Sijmn[ijmn] @ Y2[mn] @ Sijmn[ijmn].T 
                        r_Y2 = r_Y2 + 0.5 * hbar.Hoooo[i,j,m,n] * tmp 
                         
                        #seventeenth term
                        tmp = QL[mn].T @ L[i,j,v,v] @ QL[ij]
                        r_Y2 = r_Y2 + contract('eb,ae->ab', tmp, Sijmn[ijmn] @ Gvv[mn])   

                tmp_Y2.append(r_Y2)

        for ij in range(no*no):
            i = ij // no
            j = ij % no
            ji = j*no + i

            lr_Y2.append(tmp_Y2[ij].copy() + tmp_Y2[ji].copy().transpose())
        lY2_end = process_time()
        self.lY2_t += lY2_end - lY2_start
        return lr_Y2

    def local_pseudoresponse(self, lpertbar, X1, X2):
        pseudoresponse_start = process_time()
        contract = self.ccwfn.contract
        no = self.no
        Avo = lpertbar.Avo.copy()
        Avvoo = lpertbar.Avvoo.copy()
        polar1 = 0
        polar2 = 0
        for i in range(no):
            ii = i*no +i 
            polar1 += 2.0 * contract('a,a->', Avo[ii].copy(), X1[i].copy())
            for j in range(no):
                ij = i*no + j  
                polar2 += 2.0 * contract('ab,ab->', Avvoo[ij], (2.0*X2[ij] - X2[ij].transpose()))
        pseudoresponse_end = process_time()
        self.pseudoresponse_t += pseudoresponse_end - pseudoresponse_start
        return -2.0*(polar1 + polar2)
        
class lpertbar(object):
    def __init__(self, pert, ccwfn, lccwfn):
        o = ccwfn.o
        v = ccwfn.v
        no = ccwfn.no
        t1 = lccwfn.t1
        t2 = lccwfn.t2
        contract = ccwfn.contract
        QL = ccwfn.Local.QL 

        #saving H.mu[axis] here for on the fly generation of pertbar in the in_Y1 eqns
        self.pert = pert
        self.Aov = []
        self.Avv = []
        self.Avo = []
        self.Aovoo = []
        lAvvoo = []
        self.Avvoo = []
        self.Avvvo = []

        self.Avvvj_ii = []
        
        self.Aoo = pert[o,o].copy()
        for i in range(no):
            ii = i*no + i
            for m in range(no):
                self.Aoo[m,i] += contract('e,e->',t1[i], (pert[m,v].copy() @ QL[ii]))

        norm = 0 
        for ij in range(no*no):
            i = ij // no
            j = ij % no
            ii = i*no + i
            ji = j*no + i

            #Aov
            self.Aov.append(pert[o,v].copy() @ QL[ij])

            #Avv
            tmp = QL[ij].T @ pert[v,v].copy() @ QL[ij]

            Sijmm = ccwfn.Local.Sijmm
            for m in range(no):
                mm = m*no + m
                ijm = ij*no + m
                tmp -= contract('a,e->ae', t1[m] @ Sijmm[ijm].T , pert[m,v].copy() @ QL[ij])
            self.Avv.append(tmp)

            #Avo 
            tmp = QL[ij].T @ pert[v,i].copy()
            tmp += t1[i] @ (QL[ij].T @ pert[v,v].copy() @ QL[ii]).T
            
            Sijmi = ccwfn.Local.Sijmi
            for m in range(no):
                mi = m*no + i
                ijm = ij*no + m 
                tmp -= (t1[m] @ Sijmm[ijm].T) * pert[m,i].copy()  
                tmp1 = (2.0*t2[mi] - t2[mi].swapaxes(0,1)) @ Sijmi[ijm].T
                tmp += contract('ea,e->a', tmp1, pert[m,v].copy() @ QL[mi]) 
                tmp -= contract('e,a,e->a', t1[i], t1[m] @ Sijmm[ijm].T, pert[m,v].copy() @ QL[ii]) 
            self.Avo.append(tmp)

            #Aovoo -> Aov_{ij}ij 
            tmp = contract('eb,me->mb',t2[ij], pert[o,v].copy() @ QL[ij])   
            self.Aovoo.append(tmp)
            
            #Avvvo -> Avvvi
            tmp = 0  
            for m in range(no):
                mi = m*no + i
                ijm = ij*no + m 
                tmp -= contract('ab,e->abe', Sijmi[ijm] @ t2[mi] @ Sijmi[ijm].T, pert[m,v] @ QL[ij])      
            self.Avvvo.append(tmp) 
      
            #Avvv_{ii}j 
            tmp = 0
            Sijmj = ccwfn.Local.Sijmj
            for m in range(no):
                mj = m*no + j
                ijm = ij*no + m
                tmp -= contract('ab,e->abe', Sijmj[ijm] @ t2[mj] @ Sijmj[ijm].T, pert[m,v] @ QL[ii])
            self.Avvvj_ii.append(tmp)             


            #Avvoo -> Aoovv -> Aijv_{ij} V_{ij}
        for i in range(no):
            for j in range(no):
                ij = i*no + j
                ji = j*no + i
                tmp = contract('eb,ae->ab', t2[ij], self.Avv[ij]) 
                Sijmj = ccwfn.Local.Sijmj
                for m in range(no):
                    mj = m*no + j
                    mi = m*no + i 
                    ijm = ij*no + m
                    jim = ji*no + m
                    Sjimi = QL[ji].T @ QL[mi]
                    tmp -= (Sijmj[ijm]  @ t2[mj] @ Sijmj[ijm].T) * self.Aoo[m,i].copy()  
                lAvvoo.append(tmp)    

        norm = 0 
        for i in range(no):
            for j in range(no):
                ij = i*no + j 
                ji = j*no + i 
                self.Avvoo.append(0.5 * (lAvvoo[ij].copy() + lAvvoo[ji].copy().transpose()))
                norm += np.linalg.norm( 0.5 * (lAvvoo[ij].copy() + lAvvoo[ji].copy().transpose()))  
                
