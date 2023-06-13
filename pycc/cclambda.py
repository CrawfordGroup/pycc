"""
cclambda.py: Lambda-amplitude Solver
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import numpy as np
import time
from opt_einsum import contract
from .utils import helper_diis
import torch
from .cctriples import t3c_ijk, l3_ijk, l3_ijk_alt

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

        if ccwfn.local is not None and ccwfn.filter is not True:
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

        else:
            self.l1 = 2.0 * self.ccwfn.t1
            self.l2 = 2.0 * (2.0 * self.ccwfn.t2 - self.ccwfn.t2.swapaxes(2, 3))


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
        #Eqn 77
        contract = self.contract
        QL = self.Local.QL
        Sijmm = self.Local.Sijmm
        Sijmn = self.Local.Sijmn 
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

                        r_l1 = r_l1 - 2.0 * Goo[m,n] * Hmine[imn]
                        
                        r_l1 = r_l1 + Goo[m,n] * Himne[imn]

                lr_l1.append(r_l1)
                 
        return lr_l1

    def lr_L2(self, o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo_mj, Hovvo_mi, Hovov_mj, Hovov_mi, Hvovv, Hjiov, Hijov, Gvv, Goo):
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

                r_l2 = self.Local.Loovv[ij][i,j].copy()
 
                r_l2 = r_l2 + contract('eb,ea->ab', l2[ij], Hvv[ij])
                r_l2 = r_l2 + 0.5 * contract('ef,efab->ab', l2[ij], Hvvvv[ij])       
           
                for m in range(self.no):
                    mj = m*self.no + j
                    mi = m*self.no + i 
                    ijm = ij*self.no + m           
          
                    tmp = Sijmj[ijm] @ l2[mj]
                    tmp = tmp @ Sijmj[ijm].T 
                    r_l2 = r_l2 - tmp * Hoo[i,m]
          
                    tmp = l2[mj] @ Sijmj[ijm].T 
                    r_l2 = r_l2 + contract('eb,ea->ab', tmp, 2.0 * Hovvo_mj[ijm] - Hovov_mj[ijm])               
                     
                    tmp = Sijmi[ijm] @ l2[mi] 
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
        #self.r2_t.stop()
        return lr_l2

    def local_pseudoenergy(self, ERIoovv, l2):
        contract = self.contract
        ecc = 0
        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j

                ecc_ij = contract('ab,ab->',l2[ij],ERIoovv[ij][i,j])
                ecc = ecc + ecc_ij
        return 0.5 * ecc

    def solve_lambda(self, e_conv=1e-7, r_conv=1e-7, maxiter=200, max_diis=8, start_diis=1):
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
        L = self.ccwfn.H.L

        Hov = self.hbar.Hov
        Hvv = self.hbar.Hvv
        Hoo = self.hbar.Hoo
        Hoooo = self.hbar.Hoooo
        Hvvvv = self.hbar.Hvvvv
        Hvovv = self.hbar.Hvovv
        Hooov = self.hbar.Hooov
        Hovvo = self.hbar.Hovvo
        Hovov = self.hbar.Hovov
        Hvvvo = self.hbar.Hvvvo
        Hovoo = self.hbar.Hovoo

        lecc = self.pseudoenergy(o, v, ERI, l2)

        print("\nLCC Iter %3d: LCC PseudoE = %.15f  dE = % .5E" % (0, lecc, -lecc))

        diis = helper_diis(l1, l2, max_diis, self.ccwfn.precision)
 
        contract = self.contract

        if self.ccwfn.model == 'CC3':
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

            # Building intermediates in t3l1
            if isinstance(t1, torch.Tensor):                
                Zmndi = torch.zeros_like(t2[:,:,:,:no])
                Zmdfa = torch.zeros_like(t2)
                Zmdfa = torch.nn.functional.pad(Zmdfa, (0, 0, 0, 0, 0, nv-no))
            else:
                Zmndi = np.zeros_like(t2[:,:,:,:no])
                Zmdfa = np.zeros_like(t2)
                Zmdfa = np.pad(Zmdfa, ((0,0), (0,nv-no), (0,0), (0,0)))
            for m in range(no):
                for n in range(no):
                    for l in range(no):
                        t3_lmn = t3c_ijk(o, v, l, m, n, t2, Wvvvo, Wovoo, F, contract, WithDenom=True)
                        Zmndi[m,n] += contract('def,ief->di', t3_lmn, ERI[o,l,v,v])
                        Zmndi[m,n] -= contract('fed,ief->di', t3_lmn, L[o,l,v,v])
                        Zmdfa[m] += contract('def,ea->dfa', t3_lmn, ERI[n,l,v,v])
                        Zmdfa[m] -= contract('dfe,ea->dfa', t3_lmn, L[n,l,v,v])

        for niter in range(1, maxiter+1):
            lecc_last = lecc

            l1 = self.l1
            l2 = self.l2

            Goo = self.build_Goo(t2, l2)
            Gvv = self.build_Gvv(t2, l2)
            r1 = self.r_L1(o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)
            r2 = self.r_L2(o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)
   
            if self.ccwfn.model == 'CC3':                                       
                if isinstance(t1, torch.Tensor):
                    Y1 = torch.zeros_like(l1)
                    Y2 = torch.zeros_like(l2)
                    # Z intermediates in CC3 l1, l2 equations
                    # t3l1
                    Znf = torch.zeros_like(l1)
                    #l3l1+l3l2
                    Zbide = torch.zeros_like(l2)
                    Zbide = torch.nn.functional.pad(Zbide, (0, 0, 0, 0, 0, 0, 0, nv-no))
                    Zblad_1 = torch.zeros_like(l2)
                    Zblad_1 = torch.nn.functional.pad(Zblad_1, (0, 0, 0, 0, 0, 0, 0, nv-no))
                    Zblad_2 = torch.zeros_like(l2)
                    Zblad_2 = torch.nn.functional.pad(Zblad_2, (0, 0, 0, 0, 0, 0, 0, nv-no))
                    Zjlma = torch.zeros_like(l2[:,:,:no,:])
                    Zjlid_1 = torch.zeros_like(l2[:,:,:no,:])
                    Zjlid_2 = torch.zeros_like(l2[:,:,:no,:])
                else:
                    Y1 = np.zeros_like(l1)
                    Y2 = np.zeros_like(l2)     
                    # Z intermediates in CC3 l1, l2 equations           
                    # t3l1
                    Znf = np.zeros_like(l1)
                    #l3l1+l3l2
                    Zbide = np.zeros_like(l2)
                    Zbide = np.pad(Zbide, ((0,nv-no), (0,0), (0,0), (0,0)))
                    Zblad_1 = np.zeros_like(l2)
                    Zblad_1 = np.pad(Zblad_1, ((0,nv-no), (0,0), (0,0), (0,0)))
                    Zblad_2 = np.zeros_like(l2)
                    Zblad_2 = np.pad(Zblad_2, ((0,nv-no), (0,0), (0,0), (0,0)))                    
                    Zjlma = np.zeros_like(l2[:,:,:no,:])
                    Zjlid_1 = np.zeros_like(l2[:,:,:no,:])
                    Zjlid_2 = np.zeros_like(l2[:,:,:no,:])
                # t3l1
                for l in range(no):
                    for m in range(no):
                        for n in range(no):
                            t3_lmn = t3c_ijk(o, v, l, m, n, t2, Wvvvo, Wovoo, F, contract, WithDenom=True)        
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
                              
                r1 += Y1
                r2 += Y2 + Y2.swapaxes(0,1).swapaxes(2,3) 
                                                 
            if self.ccwfn.local is not None:
                inc1, inc2 = self.ccwfn.Local.filter_amps(r1, r2)
                self.l1 += inc1
                self.l2 += inc2
                rms = contract('ia,ia->', inc1, inc1)
                rms += contract('ijab,ijab->', inc2, inc2)
                if isinstance(l1, torch.Tensor):
                    rms = torch.sqrt(rms)
                else: 
                    rms = np.sqrt(rms)
            else:
                self.l1 += r1/Dia
                self.l2 += r2/Dijab
                rms = contract('ia,ia->', r1/Dia, r1/Dia)
                rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                if isinstance(l1, torch.Tensor):
                    rms = torch.sqrt(rms)
                else:
                    rms = np.sqrt(rms)

            lecc = self.pseudoenergy(o, v, ERI, self.l2)
            ediff = lecc - lecc_last
            print("LCC Iter %3d: LCC PseudoE = %.15f  dE = % .5E  rms = % .5E" % (niter, lecc, ediff, rms))
            
            if isinstance(self.l1, torch.Tensor):
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

        if isinstance(r1, torch.Tensor):
            del Goo, Gvv, Hoo, Hvv, Hov, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov

        if (isinstance(r1, torch.Tensor)) & (self.ccwfn.model == 'CC3'):
            del Zmndi, Zmdfa, Znf, Zbide, Zjlma, Zblad_1, Zblad_2, Zjlid_1, Zjlid_2, Fov, Woooo, Wovoo, Wooov, Wvovv, Wvvvo, Wovov, Wovvo, Wvvvv
           
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

            # Building intermediates in t3l1
            if isinstance(t1, torch.Tensor):                
                Zmndi = torch.zeros_like(t2[:,:,:,:no])
                Zmdfa = torch.zeros_like(t2)
                Zmdfa = torch.nn.functional.pad(Zmdfa, (0, 0, 0, 0, 0, nv-no))
            else:
                Zmndi = np.zeros_like(t2[:,:,:,:no])
                Zmdfa = np.zeros_like(t2)
                Zmdfa = np.pad(Zmdfa, ((0,0), (0,nv-no), (0,0), (0,0)))
            for m in range(no):
                for n in range(no):
                    for l in range(no):
                        t3_lmn = t3c_ijk(o, v, l, m, n, t2, Wvvvo, Wovoo, F, contract, WithDenom=True)
                        Zmndi[m,n] += contract('def,ief->di', t3_lmn, ERI[o,l,v,v])
                        Zmndi[m,n] -= contract('fed,ief->di', t3_lmn, L[o,l,v,v])
                        Zmdfa[m] += contract('def,ea->dfa', t3_lmn, ERI[n,l,v,v])
                        Zmdfa[m] -= contract('dfe,ea->dfa', t3_lmn, L[n,l,v,v])                                                
            if isinstance(t1, torch.Tensor):
                Y1 = torch.zeros_like(l1)
                Y2 = torch.zeros_like(l2)
                # Z intermediates in CC3 l1, l2 equations
                # t3l1
                Znf = torch.zeros_like(l1)
                #l3l1+l3l2
                Zbide = torch.zeros_like(l2)
                Zbide = torch.nn.functional.pad(Zbide, (0, 0, 0, 0, 0, 0, 0, nv-no))
                Zblad_1 = torch.zeros_like(l2)
                Zblad_1 = torch.nn.functional.pad(Zblad_1, (0, 0, 0, 0, 0, 0, 0, nv-no))
                Zblad_2 = torch.zeros_like(l2)
                Zblad_2 = torch.nn.functional.pad(Zblad_2, (0, 0, 0, 0, 0, 0, 0, nv-no))
                Zjlma = torch.zeros_like(l2[:,:,:no,:])
                Zjlid_1 = torch.zeros_like(l2[:,:,:no,:])
                Zjlid_2 = torch.zeros_like(l2[:,:,:no,:])
            else:
                Y1 = np.zeros_like(l1)
                Y2 = np.zeros_like(l2)     
                # Z intermediates in CC3 l1, l2 equations           
                # t3l1
                Znf = np.zeros_like(l1)
                #l3l1+l3l2
                Zbide = np.zeros_like(l2)
                Zbide = np.pad(Zbide, ((0,nv-no), (0,0), (0,0), (0,0)))
                Zblad_1 = np.zeros_like(l2)
                Zblad_1 = np.pad(Zblad_1, ((0,nv-no), (0,0), (0,0), (0,0)))
                Zblad_2 = np.zeros_like(l2)
                Zblad_2 = np.pad(Zblad_2, ((0,nv-no), (0,0), (0,0), (0,0)))                    
                Zjlma = np.zeros_like(l2[:,:,:no,:])
                Zjlid_1 = np.zeros_like(l2[:,:,:no,:])
                Zjlid_2 = np.zeros_like(l2[:,:,:no,:])            
            # t3l1
            for l in range(no):
                for m in range(no):
                    for n in range(no):
                        t3_lmn = t3c_ijk(o, v, l, m, n, t2, Wvvvo, Wovoo, F, contract, WithDenom=True)        
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
                         
            r1 += Y1
            r2 += Y2 + Y2.swapaxes(0,1).swapaxes(2,3) 

            if isinstance(r1, torch.Tensor):
                del Zmndi, Zmdfa, Znf, Zbide, Zjlma, Zblad_1, Zblad_2, Zjlid_1, Zjlid_2, Fov, Woooo, Wovoo, Wooov, Wvovv, Wvvvo, Wovov, Wovvo, Wvvvv
       
        if isinstance(r1, torch.Tensor):
            del Goo, Gvv, Hoo, Hvv, Hov, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov
                                             
        return r1, r2

    def build_Goo(self, t2, l2):
        contract = self.contract
        return contract('mjab,ijab->mi', t2, l2)


    def build_Gvv(self, t2, l2):
        contract = self.contract 
        tmp = -1.0 * contract('ijeb,ijab->ae', t2, l2)
        #for ij in range(self.ccwfn.no**2):
            #print("Gvv", ij, contract('aA,bB,ab->AB', self.ccwfn.Local.Q[ij] @ self.ccwfn.Local.L[ij], self.ccwfn.Local.Q[ij] @ self.ccwfn.Local.L[ij], tmp))
        return -1.0 * contract('ijeb,ijab->ae', t2, l2)


    def r_L1(self, o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
        contract = self.contract 
        if self.ccwfn.model == 'CCD':
            if isinstance(l1, torch.Tensor):
                r_l1 = torch.zeros_like(l1)
            else:
                r_l1 = np.zeros_like(l1)
        else:
            if isinstance(l1, torch.Tensor):
                r_l1 = 2.0 * Hov.clone()
            else: 
                r_l1 = 2.0 * Hov.copy() #r_l1

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
        contract = self.contract 
        if self.ccwfn.model == 'CCD':
            if isinstance(l1, torch.Tensor):
                r_l2 = L[o,o,v,v].clone().to(self.ccwfn.device1)
            else:
                r_l2 = L[o,o,v,v].copy()

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
            if isinstance(l1, torch.Tensor):
                r_l2 = L[o,o,v,v].clone().to(self.ccwfn.device1)
            else:
                r_l2 = L[o,o,v,v].copy()

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
        contract = self.contract
        if isinstance(t1, torch.Tensor):
            W = ERI[o,v,o,v].clone().to(self.ccwfn.device1)
        else:
            W = ERI[o,v,o,v].copy()
        W = W + contract('mbfe,jf->mbje', ERI[o,v,v,v], t1)
        W = W - contract('mnje,nb->mbje', ERI[o,o,o,v], t1)
        W = W - contract('mnfe,jf,nb->mbje', ERI[o,o,v,v], t1, t1)
        return W

    def build_cc3_Wmbej(self, o, v, ERI, t1):
        contract = self.contract
        if isinstance(t1, torch.Tensor):
            W = ERI[o,v,v,o].clone().to(self.ccwfn.device1)
        else:
            W = ERI[o,v,v,o].copy()
        W = W + contract('mbef,jf->mbej', ERI[o,v,v,v], t1)
        W = W - contract('mnej,nb->mbej', ERI[o,o,v,o], t1)
        W = W - contract('mnef,jf,nb->mbej', ERI[o,o,v,v], t1, t1)
        return W

    def build_cc3_Wabef(self, o, v, ERI, t1):
        contract = self.contract
        if isinstance(t1, torch.Tensor):
            W = ERI[v,v,v,v].clone().to(self.ccwfn.device1)
        else:
            W = ERI[v,v,v,v].copy()
        tmp = contract('mbef,ma->abef', ERI[o,v,v,v], t1)
        W = W - tmp - tmp.swapaxes(0,1).swapaxes(2,3)
        W = W + contract('mnef,ma,nb->abef', ERI[o,o,v,v], t1, t1)
        return W
                                         
    def pseudoenergy(self, o, v, ERI, l2):
        contract = self.contract 
        return 0.5 * contract('ijab,ijab->',ERI[o,o,v,v], l2)
