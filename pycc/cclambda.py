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
    solve_lambda()
        Solves the CC Lambda amplitude equations
    residuals()
        Computes the L1 and L2 residuals for a given set of amplitudes and Fock operator
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

        self.l1 = 2.0 * self.ccwfn.t1
        self.l2 = 2.0 * (2.0 * self.ccwfn.t2 - self.ccwfn.t2.swapaxes(2, 3))

    def solve_lambda(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1):
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
                r_l1 = 2.0 * Hov.copy()

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
