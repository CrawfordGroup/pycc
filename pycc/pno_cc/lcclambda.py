"""
cclambda.py: Lambda-amplitude Solver
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import numpy as np
import time
from time import process_time
from opt_einsum import contract
from ..utils import helper_diis

class lcclambda(object):
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
    solve_llambda()
        Solves the local CC Lambda amplitude equations (PAO, PNO, PNO++) 
    solve_lambda()
        Solves the CC Lambda amplitude equations
    residuals()
        Computes the L1 and L2 residuals for a given set of amplitudes and Fock operator
    
    Notes
    -----
    
    For the local implementation: 
    Eqns can be found in LocalCCSD.pdf 

    To do:
    (1) need DIIS extrapolation
    (2) time table?
    """
    def __init__(self, ccwfn, hbar):
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

        self.lccwfn = ccwfn.lccwfn
        self.no = ccwfn.no
        self.local = ccwfn.local
        self.model = ccwfn.model         
        self.Local = ccwfn.Local          

        l1 = []
        l2 = []
        for i in range(self.no):
            l1.append(2.0 * self.lccwfn.t1[i])

            for j in range(self.no):
                ij = i*self.no + j
                l2.append(2.0 * (2.0 * self.lccwfn.t2[ij] - self.lccwfn.t2[ij].swapaxes(0, 1)))     
        
        self.l1 = l1
        self.l2 = l2

    def solve_llambda(self, e_conv=1e-7, r_conv=1e-7, maxiter=200, max_diis=8, start_diis=1):
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
            lCC pseudoenergy

        """
        contract = self.ccwfn.contract
        lambda_tstart = time.time()

        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccwfn.nv
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2
        l1 = self.l1
        l2 = self.l2
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        L = self.ccwfn.H.L

        # need to make the naming scheme more readable
        Hov = self.hbar.Hov
        Hvv = self.hbar.Hvv 
        Hoo = self.hbar.Hoo
        Hoooo = self.hbar.Hoooo
        Hvvvv = self.hbar.Hvvvv
        Hvovv_ii = self.hbar.Hvovv_ii 
        Hvovv_imn = self.hbar.Hvovv_imn
        Hvovv_imns = self.hbar.Hvovv_imns
        Hmine = self.hbar.Hmine
        Himne = self.hbar.Himne
        Hjiov = self.hbar.Hjiov
        Hijov = self.hbar.Hijov
        Hovvo_mi = self.hbar.Hovvo_mi 
        Hovvo_mj = self.hbar.Hovvo_mj
        Hovvo_mm = self.hbar.Hovvo_mm 
        Hovov_mi = self.hbar.Hovov_mi
        Hovov_mj = self.hbar.Hovov_mj
        Hovov_mm = self.hbar.Hovov_mm  
        Hvvvo_im = self.hbar.Hvvvo_im 
        Hovoo_mn = self.hbar.Hovoo_mn 

        #initialize variables for timing each function
        self.lr1_t = 0
        self.lr2_t = 0
        self.pseudoenergy_t = 0

        lecc = self.local_pseudoenergy(self.Local.ERIoovv, l2)

        print("\nLCC Iter %3d: LCC PseudoE = %.15f  dE = % .5E" % (0, lecc, -lecc))

        #diis = helper_diis(l1, l2, max_diis, self.ccwfn.precision)

        contract = self.contract
 
        for niter in range(1, maxiter+1):
            lecc_last = lecc

            l1 = self.l1
            l2 = self.l2

            Goo = self.build_lGoo(t2, l2)
            Gvv = self.build_lGvv(t2, l2)  
            r1 = self.lr_L1(o, v, l1, l2, Hov, Hvv, Hoo, Hovvo_mm, Hovov_mm, Hvvvo_im, Hovoo_mn, Hvovv_imn, Hvovv_imns, Hmine, Himne, Gvv, Goo)
            r2 = self.lr_L2(o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo_mj, Hovvo_mi, Hovov_mj, Hovov_mi, Hvovv_ii, Hjiov, Hijov, Gvv, Goo)

            rms = 0
            rms_l1 = 0
            rms_l2 = 0

            for i in range(self.no):

                ii = i*self.no + i

                self.l1[i] -= r1[i]/(self.Local.eps[ii].reshape(-1,) - F[i,i]) 

                rms_l1 += contract('Z,Z->',r1[i],r1[i])

                for j in range(self.no):
                    ij = i*self.no + j

                    self.l2[ij] -= r2[ij]/(self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - F[i,i] - F[j,j])
                    
                    rms_l2 += contract('ZY,ZY->',r2[ij],r2[ij])

            rms = np.sqrt(rms_l1 + rms_l2)
            lecc = self.local_pseudoenergy(self.Local.ERIoovv, l2)
            ediff = lecc - lecc_last
            print("lLCC Iter %3d: lLCC PseudoE = %.15f  dE = % .5E  rms = % .5E" % (niter, lecc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nLambda-CC has converged in %.3f seconds.\n" % (time.time() - lambda_tstart))
                print('Time table for intermediates')
                print("lr1_t = %6.6f" % self.lr1_t)
                print("lr2_t = %6.6f" % self.lr2_t)
                print("Pseudoenergy_t = %6.6f" % self.pseudoenergy_t)
                return lecc

    def build_lGoo(self, t2, l2):
        #Eqn 79
        contract = self.contract
        QL = self.Local.QL
        Sijmj = self.Local.Sijmj
        Goo = np.zeros((self.no,self.no))
        for i in range(self.no):
            for j in range(self.no): 
                ij = i*self.no + j
                
                for m in range(self.no):
                    mj = m*self.no + j
                    ijm = ij*self.no + m 

                    tmp = Sijmj[ijm] @ t2[mj] 
                    tmp = tmp @ Sijmj[ijm].T 
                    Goo[m,i] += contract('ab,ab->',tmp,l2[ij])
        return Goo 

    def build_lGvv(self, t2, l2):
        #Eqn 78
        contract = self.contract
        lGvv = []
        for ij in range(self.no*self.no): 
            Gvv = -1.0 * contract('eb,ab->ae', t2[ij], l2[ij]) 
            lGvv.append(Gvv) 
        return lGvv

    def lr_L1(self, o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hvovvs, Hmine, Himne, Gvv, Goo):
        lr1_start = process_time() 

        #Eqn 77
        contract = self.contract
        QL = self.Local.QL
        Sijmm = self.Local.Sijmm
        Sijmn = self.Local.Sijmn 
        hbar = self.hbar
        lr_l1 = []
        if self.ccwfn.model == 'CCD':
            for i in range(self.no):
                lr_l1.append(np.zeros_like(l1[i]))
        else: 
            for i in range(self.no):
                ii = i*self.no + i
                r_l1 = 2.0 * Hov[ii][i].copy()
       
                r_l1 = r_l1 + contract('e,ea->a',l1[i], Hvv[ii])
                    
                for m in range(self.no):
                    mm = m*self.no + m
                    im = i*self.no + m
                    iimm = ii*(self.no) + m 
           
                    tmp = Sijmm[iimm] @ l1[m]
                    r_l1 = r_l1 - tmp * Hoo[i,m]

                    r_l1 = r_l1 + contract('ef,efa->a', l2[im], Hvvvo[im])

                    r_l1 = r_l1 + contract('e,ea->a', l1[m], (2* Hovvo[im] - Hovov[im])) 
                     
                    for n in range(self.no):
                        mn = m*self.no + n
                        nm = n*self.no +m 
                        imn = im*self.no + n
                        iimn = ii*(self.no**2) + mn                   

                        tmp = Sijmn[iimn] @ l2[mn]
                        r_l1 = r_l1 - contract('ae,e->a', tmp, Hovoo[imn])
                                    
                        r_l1 = r_l1 - 2.0 * contract('ef,efa->a', Gvv[mn], Hvovv[imn])  
          
                        r_l1 = r_l1 + contract('ef,eaf->a', Gvv[mn], Hvovvs[imn])

                        #commenting out Hmine and using Hooov_test to check its viability
                        r_l1 = r_l1 - 2.0 * Goo[m,n] * hbar.Hooov[ii][m,i,n,:] #Hmine[iimn]
                        
                        r_l1 = r_l1 + Goo[m,n] * Himne[iimn]

                lr_l1.append(r_l1)
        lr1_end = process_time()
        self.lr1_t += lr1_end - lr1_start         
        return lr_l1

    def lr_L2(self, o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo_mj, Hovvo_mi, Hovov_mj, Hovov_mi, Hvovv, Hjiov, Hijov, Gvv, Goo):
        lr2_start = process_time()
        #Eqn 80 
        contract = self.contract
        QL = self.Local.QL
        Sijii = self.Local.Sijii
        Sijjj = self.Local.Sijjj
        Sijmm = self.Local.Sijmm
        Sijmj = self.Local.Sijmj
        Sijmi = self.Local.Sijmi
        Sijmn = self.Local.Sijmn
        lr_l2 = []
        nlr_l2 = []
        if self.ccwfn.model == 'CCD':
            for ij in range(self.no*self.no):
                i = ij // self.no 
                j = ij % self.no 
                ji = j * self.no + i 
                r_l2 = self.Local.Loovv[ij][i,j].copy()
 
                r_l2 = r_l2 + contract('eb,ea->ab', l2[ij], Hvv[ij])
                r_l2 = r_l2 + 0.5 * contract('ef,efab->ab', l2[ij], Hvvvv[ij])       
           
                for m in range(self.no):
                    mj = m*self.no + j
                    mi = m*self.no + i 
                    ijm = ij*self.no + m           
                    jim = ji*self.no + m 

                    tmp = Sijmj[ijm] @ l2[mj]
                    tmp = tmp @ Sijmj[ijm].T 
                    r_l2 = r_l2 - tmp * Hoo[i,m]
          
                    tmp = l2[mj] @ Sijmj[ijm].T 
                    r_l2 = r_l2 + contract('eb,ea->ab', tmp, 2.0 * Hovvo_mj[ijm] - Hovov_mj[ijm])               
                     
                    tmp = Sijmi[ijm] @ l2[mi] 
                    #trying out Hovov_mj[jim] instead
                    r_l2 = r_l2 - contract('be,ea->ab', tmp, Hovov_mj[jim])  
                    
                    tmp = l2[mi] @ Sijmi[ijm].T 
                    r_l2 = r_l2 - contract('eb,ea->ab', tmp, Hovvo_mi[ijm])
                  
                    r_l2 = r_l2 - Goo[m,i] * self.Local.Loovv[ij][m,j]

                    for n in range(self.no):
                        mn = m*self.no + n 
                        ijmn = ij*(self.no**2) + mn

                        tmp = Sijmn[ijmn] @ l2[mn] 
                        tmp = tmp @ Sijmn[ijmn].T 
                        r_l2 = r_l2 + 0.5 * tmp * Hoooo[i,j,m,n]

                        tmp = Sijmn[ijmn] @ Gvv[mn] 
                        tmp1 = contract('eb, eE, bB->EB', L[i,j,v,v], QL[mn], QL[ij]) 
                        r_l2 = r_l2 + contract('ae,eb->ab', tmp, tmp1)

                nlr_l2.append(r_l2)
        else: 
            for ij in range(self.no*self.no):
                i = ij // self.no 
                j = ij % self.no 
                ii = i*self.no + i
                jj = j*self.no + j 
                
                r_l2 = self.Local.Loovv[ij][i,j].copy()

                tmp = Sijii[ij] @ l1[i]
                r_l2 = r_l2 + 2.0 * contract('a,b->ab', tmp, Hov[ij][j])
              
                tmp = Sijjj[ij] @ l1[j]
                r_l2 = r_l2 - contract('a,b->ab', tmp, Hov[ij][i])

                r_l2 = r_l2 + 2.0 * contract('e,eab->ab', l1[i], Hvovv[ij][:,j,:,:])

                r_l2 = r_l2 - contract('e,eba ->ab', l1[i], Hvovv[ij][:,j,:,:]) 

                r_l2 = r_l2 + contract('eb,ea->ab',l2[ij], Hvv[ij])

                r_l2 = r_l2 + 0.5 * contract('ef,efab->ab', l2[ij], Hvvvv[ij])
 
                for m in range(self.no):
                    mm = m*self.no + m 
                    mj = m*self.no + j
                    mi = m*self.no + i
                    ijm = ij*self.no + m
                   
                    tmp = Sijmm[ijm] @ l1[m] 
                    r_l2 = r_l2 - 2.0 * contract('b,a->ab', tmp, Hjiov[ij][m])

                    r_l2 = r_l2 + contract('b,a->ab', tmp, Hijov[ij][m]) 

                    tmp = Sijmj[ijm] @ l2[mj]
                    tmp = tmp @ Sijmj[ijm].T
                    r_l2 = r_l2 - tmp * Hoo[i,m]

                    tmp = l2[mj] @ Sijmj[ijm].T
                    r_l2 = r_l2 + contract('eb,ea->ab', tmp, 2.0 * Hovvo_mj[ijm] - Hovov_mj[ijm])

                    tmp = Sijmi[ijm] @ l2[mi]
                    #trying out Hovov_mj[jim] instead
                    r_l2 = r_l2 - contract('be,ea->ab', tmp, Hovov_mi[ijm])

                    tmp = l2[mi] @ Sijmi[ijm].T
                    r_l2 = r_l2 - contract('eb,ea->ab', tmp, Hovvo_mi[ijm]) 

                    r_l2 = r_l2 - Goo[m,i] * self.Local.Loovv[ij][m,j] 
                   
                    for n in range(self.no):
                        mn = m*self.no + n
                        ijmn = ij*(self.no**2) + mn

                        tmp = Sijmn[ijmn] @ l2[mn]
                        tmp = tmp @ Sijmn[ijmn].T
                        r_l2 = r_l2 + 0.5 * tmp * Hoooo[i,j,m,n]

                        tmp = Sijmn[ijmn] @ Gvv[mn]
                        tmp1 = contract('eb, eE -> Eb', L[i,j,v,v], QL[mn])
                        tmp1 = contract('Eb, bB ->EB', tmp1, QL[ij])
                        r_l2 = r_l2 + contract('ae,eb->ab', tmp, tmp1)

                nlr_l2.append(r_l2)
    
        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j
                ji = j*self.no + i

                lr_l2.append(nlr_l2[ij].copy() + nlr_l2[ji].copy().transpose())
   
        lr2_end = process_time()
        self.lr2_t += lr2_end - lr2_start
        return lr_l2

    def local_pseudoenergy(self, ERIoovv, l2):
        pseudoenergy_start = process_time()
        contract = self.contract
        ecc = 0
        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j

                ecc_ij = contract('ab,ab->',l2[ij],ERIoovv[ij][i,j])
                ecc = ecc + ecc_ij
        pseudoenergy_end = process_time()
        self.pseudoenergy_t += pseudoenergy_end - pseudoenergy_start
        return 0.5 * ecc

