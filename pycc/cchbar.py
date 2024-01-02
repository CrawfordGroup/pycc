"""
cchbar.py: Builds the similarity-transformed Hamiltonian (one- and two-body terms only).
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import time
import numpy as np
import torch

class cchbar(object):
    """
    An RHF-CC Similarity-Transformed Hamiltonian object.

    Attributes
    ----------
    Hov : NumPy array
        The occupied-virtual block of the one-body component HBAR.
    Hvv : NumPy array
        The virtual-virtual block of the one-body component HBAR.
    Hoo : NumPy array
        The occupied-occupied block of the one-body component HBAR.
    Hoooo : NumPy array
        The occ,occ,occ,occ block of the two-body component HBAR.
    Hvvvv : NumPy array
        The vir,vir,vir,vir block of the two-body component HBAR.
    Hvovv : NumPy array
        The vir,occ,vir,vir block of the two-body component HBAR.
    Hooov : NumPy array
        The occ,occ,occ,vir block of the two-body component HBAR.
    Hovvo : NumPy array
        The occ,vir,vir,occ block of the two-body component HBAR.
    Hovov : NumPy array
        The occ,vir,occ,vir block of the two-body component HBAR.
    Hvvvo : NumPy array
        The vir,vir,vir,occ block of the two-body component HBAR.
    Hovoo : NumPy array
        The occ,vir,occ,occ block of the two-body component HBAR.

    Notes
    -----
    For the local implementation: 
    Eqns can be found in LocalCCSD.pdf

    """
    def __init__(self, ccwfn):
        """
        Parameters
        ----------
        ccwfn : PyCC ccwfn object
            amplitudes instantiated to defaults or converged

        Returns
        -------
        None
        """

        time_init = time.time()
  
        self.ccwfn = ccwfn
        
        self.contract = self.ccwfn.contract
 
        o = ccwfn.o
        v = ccwfn.v
        F = ccwfn.H.F
        ERI = ccwfn.H.ERI
        L = ccwfn.H.L
        t1 = ccwfn.t1
        t2 = ccwfn.t2
        
        if ccwfn.local is None or ccwfn.filter is True:
            
            self.Hov = self.build_Hov(o, v, F, L, t1)
            self.Hvv = self.build_Hvv(o, v, F, L, t1, t2)
            self.Hoo = self.build_Hoo(o, v, F, L, t1, t2)
            self.Hoooo = self.build_Hoooo(o, v, ERI, t1, t2)
            self.Hvvvv = self.build_Hvvvv(o, v, ERI, t1, t2)
            self.Hvovv = self.build_Hvovv(o, v, ERI, t1)
            self.Hooov = self.build_Hooov(o, v, ERI, t1)
            self.Hovvo = self.build_Hovvo(o, v, ERI, L, t1, t2)
            self.Hovov = self.build_Hovov(o, v, ERI, t1, t2)
            self.Hvvvo = self.build_Hvvvo(o, v, ERI, L, self.Hov, self.Hvvvv, t1, t2)
            self.Hovoo = self.build_Hovoo(o, v, ERI, L, self.Hov, self.Hoooo, t1, t2)
 
            if isinstance(t1, torch.Tensor):
                print("Hov norm = %20.15f" % torch.linalg.norm(self.Hov))
                print("Hvv norm = %20.15f" % torch.linalg.norm(self.Hvv))
                print("Hoo norm = %20.15f" % torch.linalg.norm(self.Hoo))
                print("Hoooo norm = %20.15f" % torch.linalg.norm(self.Hoooo))
                print("Hvvvv norm = %20.15f" % torch.linalg.norm(self.Hvvvv))
            else:
                print("Hov norm = %20.15f" % np.linalg.norm(self.Hov))
                print("Hvv norm = %20.15f" % np.linalg.norm(self.Hvv))
                print("Hoo norm = %20.15f" % np.linalg.norm(self.Hoo))
                print("Hoooo norm = %20.15f" % np.linalg.norm(self.Hoooo))
                print("Hvvvv norm = %20.15f" % np.linalg.norm(self.Hvvvv))
        
        elif ccwfn.filter is not True:    
            self.Local = ccwfn.Local
            self.no = ccwfn.no
            self.nv = ccwfn.nv
            self.lccwfn = ccwfn.lccwfn
        
            self.Hov = self.build_lHov(o, v, ccwfn.no, self.Local.Fov, L, self.Local.QL, self.lccwfn.t1)
            self.Hvv = self.build_lHvv(o, v, ccwfn.no, F, L, self.Local.Fvv, self.Local.Fov, self.Local.Loovv, self.Local.QL, 
            self.lccwfn.t1,self.lccwfn.t2) 
            self.Hoo = self.build_lHoo(o ,v, ccwfn.no, F, L, self.Local.Fov, self.Local.Looov, self.Local.Loovv, 
            self.Local.QL, self.lccwfn.t1,self.lccwfn.t2) 
            self.Hoooo = self.build_lHoooo(o, v, ccwfn.no, ERI, self.Local.ERIoovv, self.Local.ERIooov, 
            self.Local.QL, self.lccwfn.t1, self.lccwfn.t2) 
            self.Hvvvv, self.Hvvvv_im = self.build_lHvvvv( o, v, ccwfn.no, ERI, self.Local.ERIoovv, self.Local.ERIvovv, self.Local.ERIvvvv, 
            self.Local.QL, self.lccwfn.t1, self.lccwfn.t2) 
            self.Hvovv_ii, self.Hvovv_imn, self.Hvovv_imns = self.build_lHvovv(o,v,ccwfn.no, ERI, self.Local.ERIvovv, self.Local.ERIoovv, self.Local.QL, self.lccwfn.t1)
            self.Hjiov, self.Hijov, self.Hmine, self.Himne = self.build_lHooov(o, v, ccwfn.no, ERI, self.Local.ERIooov, self.Local.QL, self.lccwfn.t1) 
            self.Hovvo_mi, self.Hovvo_mj, self.Hovvo_mm  = self.build_lHovvo(o, v, ccwfn.no, ERI, L, self.Local.ERIovvo, self.Local.QL, 
            self.lccwfn.t1, self.lccwfn.t2)
            self.Hovov_mi, self.Hovov_mj, self.Hovov_mm  = self.build_lHovov(o, v, ccwfn.no, ERI, self.Local.ERIovov, self.Local.ERIooov, self.Local.QL,
            self.lccwfn.t1,self.lccwfn.t2)
            self.Hvvvo_im = self.build_lHvvvo(o, v, ccwfn.no, ERI, L, self.Local.ERIvvvo, self.Local.ERIoovo, self.Local.ERIvoov, self.Local.ERIvovo, 
            self.Local.ERIoovv, self.Local.Loovv, self.Local.QL,self.lccwfn.t1, self.lccwfn.t2, self.Hov, self.Hvvvv, self.Hvvvv_im)
            self.Hovoo_mn = self.build_lHovoo(o, v, ccwfn.no, ERI, L, self.Local.ERIovoo, self.Local.ERIovvv, self.Local.ERIooov,
            self.Local.ERIovov, self.Local.ERIvoov, self.Local.Looov, self.Local.QL, self.lccwfn.t1, self.lccwfn.t2, self.Hov, self.Hoooo)

        print("\nHBAR constructed in %.3f seconds.\n" % (time.time() - time_init))

    """
    For GPU implementation:
    2-index tensors are stored on GPU
    4-index tensors are stored on CPU
    """
    def build_lHov(self, o, v, no, Fov, L, QL, t1):
        #Eqn 82
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            lHov = Fov.copy()
        else:
            lHov = []
            for ij in range(no*no):

                Hov = Fov[ij].copy()

                for n in range(no):
                    nn = n*no + n      

                    tmp = contract('eE, mef-> mEf', QL[ij], L[o,n,v,v])
                    tmp = contract('fF, mEf-> mEF', QL[nn], tmp)
                    Hov = Hov + contract('F,mEF->mE',t1[n], tmp)
                lHov.append(Hov)     
        return lHov

    def build_Hov(self, o, v, F, L, t1):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(F, torch.Tensor):
                Hov = F[o,v].clone()
            else:
                Hov = F[o,v].copy()
            
        else:
            if isinstance(F, torch.Tensor):
                Hov = F[o,v].clone()
            else:
                Hov = F[o,v].copy()
            Hov = Hov + contract('nf,mnef->me', t1, L[o,o,v,v])
        return Hov

    def build_lHvv(self,o, v, no, F, L, Fvv, Fov, Loovv, QL, t1, t2): 
        contract = self.contract
        Sijmn = self.Local.Sijmn
        Sijmm = self.Local.Sijmm
        if self.ccwfn.model == 'CCD':
            lHvv = []

            for ij in range(no*no):
                i = ij // no
                j = ij % no

                Hvv = Fvv[ij].copy()

                for mn in range(no*no):
                    m = mn // no
                    n = mn % no
                    ijmn = ij*(no**2) + mn

                    tmp = QL[mn].T @ L[m,n,v,v]
                    tmp = tmp @ QL[ij]
                    tmp1 = t2[mn] @ Sijmn[ijmn].T 
                    Hvv = Hvv - tmp1.T @ tmp                             
                lHvv.append(Hvv) 
        else:
            lHvv = []

            #Hv_ii v_ij - not needed for local lambda but may be needed for other eqns
            #for ij in range(no*no):
                #i = ij // no 
                #j = ij % no 
                #ii = i*no + i 

                #tmp = contract('ab,aA->Ab', F[v,v], QL[ii])
                #Hvv_ii = contract('Ab, bB->AB', tmp, QL[ij])  
                #Hvv_ii = contract('ab,aA,bB->AB', F[v,v], QL[ii], QL[ij])
                
                #for m in range(no):
                    #mm = m*no + m

                    #Siimm = QL[ii].T @ QL[mm]
                    #tmp = t1[m] @ Siimm.T
                    #Hvv_ii = Hvv_ii - contract('e,a->ae', Fov[ij][m], tmp)

                    #tmp1 = contract('aef,aA,eE,fF->AEF',L[v,m,v,v],QL[ii],QL[ij],QL[mm])
                    #tmp1 = contract('aef,aA->Aef',L[v,m,v,v],QL[ii])
                    #tmp1 = contract('Aef,eE-> AEf', tmp1, QL[ij]) 
                    #tmp1 = contract('AEf,fF-> AEF', tmp1, QL[mm])  
                    #Hvv_ii = Hvv_ii + contract('F,aeF->ae',t1[m],tmp1)

                    #for n in range(no):
                        #mn = m*no + n
                        #nn = n*no + n

                        #Siimn = QL[ii].T @ QL[mn]
                        #tmp2 = QL[mn].T @ L[m,n,v,v]
                        #tmp2 = tmp2 @ QL[ij]
                        #tmp3 = t2[mn] @ Siimn.T
                        #Hvv_ii = Hvv_ii - tmp3.T @ tmp2

                        #Siinn = QL[ii].T @ QL[nn]
                        #tmp4 = QL[mm].T @ L[m,n,v,v]
                        #tmp4 = tmp4 @ QL[ij]
                        #tmp5 = t1[n] @ Siinn.T
                        #Hvv_ii = Hvv_ii - contract('F,A,Fe->Ae',t1[m], tmp5, tmp4)
                #lHvv_ii.append(Hvv_ii) 

            #Hv_ijv_ij - Eqn 84
            for ij in range(no*no):
                 
                Hvv = Fvv[ij].copy()

                for m in range(no):
                    mm = m*no + m
                    ijm = ij*no + m
 
                    tmp = t1[m] @ Sijmm[ijm].T 
                    Hvv = Hvv - contract('e,a->ae', Fov[ij][m], tmp)
                    
                    tmp = contract('aef, aA-> Aef', L[v,m,v,v], QL[ij])
                    tmp = contract('Aef, eE-> AEf', tmp, QL[ij])
                    tmp = contract('AEf, fF-> AEF', tmp, QL[mm])
                    Hvv = Hvv + contract('F,aeF->ae', t1[m], tmp)
                    
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n
                        ijmn = ij*(no**2) + mn
                        ijn = ij*no + n

                        tmp = QL[mn].T @ L[m,n,v,v] 
                        tmp = tmp @ QL[ij]
                        tmp1 = t2[mn] @ Sijmn[ijmn].T
                        Hvv = Hvv - tmp1.T @ tmp
                        
                        tmp = QL[mm].T @ L[m,n,v,v] 
                        tmp = tmp @ QL[ij]
                        
                        #Sijnn <- Sijmm
                        tmp1 = t1[n] @ Sijmm[ijn].T 
                        Hvv = Hvv - contract('F,A,Fe->Ae',t1[m], tmp1, tmp)
                lHvv.append(Hvv)               
        return lHvv

    def build_Hvv(self, o, v, F, L, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(F, torch.Tensor):
                Hvv = F[v,v].clone()
            else:
                Hvv = F[v,v].copy()
            Hvv = Hvv - contract('mnfa,mnfe->ae', t2, L[o,o,v,v])
        else:
            if isinstance(F, torch.Tensor):
                Hvv = F[v,v].clone()
            else:
                Hvv = F[v,v].copy()
            Hvv = Hvv - contract('me,ma->ae', F[o,v], t1)
            Hvv = Hvv + contract('mf,amef->ae', t1, L[v,o,v,v])
            Hvv = Hvv - contract('mnfa,mnfe->ae', self.ccwfn.build_tau(t1, t2), L[o,o,v,v])
        return Hvv

    def build_lHoo(self, o ,v, no, F, L, Fov, Looov, Loovv, QL, t1,t2):  
        #Eqn 85
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hoo = F[o,o].copy() 

            for _in in range(no*no):
                i = _in // no
                n = _in % no

                Hoo[:,i] = Hoo[:,i] + contract('ef,mef->m',t2[_in],Loovv[_in][:,n])
        else:
            Hoo = F[o,o].copy()

            for i in range(no):
                ii = i*no + i 

                Hoo[:,i] = Hoo[:,i] + t1[i] @ Fov[ii].T      
                
                for n in range(no):
                    nn = n*no + n 
                    _in = i*no + n

                    Hoo[:,i] = Hoo[:,i] + contract('e,me-> m', t1[n], Looov[nn][:,n,i])
                    
                    Hoo[:,i] = Hoo[:,i] + contract('ef,mef->m', t2[_in], Loovv[_in][:,n])
                    
                    tmp = contract('eE, mef -> mEf', QL[ii], L[o,n,v,v]) 
                    tmp = contract('fF, mEf -> mEF', QL[nn], tmp) 
                    Hoo[:,i] = Hoo[:,i] + contract('e,f,mef->m', t1[i], t1[n], tmp)
        return Hoo
  
    def build_Hoo(self, o, v, F, L, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(F, torch.Tensor):
                Hoo = F[o,o].clone()
            else:
                Hoo = F[o,o].copy()
            Hoo = Hoo + contract('inef,mnef->mi', t2, L[o,o,v,v])
        else:
            if isinstance(F, torch.Tensor):
                Hoo = F[o,o].clone()
            else:
                Hoo = F[o,o].copy()
            Hoo = Hoo + contract('ie,me->mi', t1, F[o,v])
            Hoo = Hoo + contract('ne,mnie->mi', t1, L[o,o,o,v])
            Hoo = Hoo + contract('inef,mnef->mi', self.ccwfn.build_tau(t1, t2), L[o,o,v,v])
        return Hoo

    def build_lHoooo(self, o, v, no, ERI, ERIoovv, ERIooov, QL, t1, t2):  
        #Eqn 86
        contract = self.contract 
        if self.ccwfn.model == 'CCD':
            Hoooo = ERI[o,o,o,o].copy()
 
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                                           
                Hoooo[:,:,i,j] = Hoooo[:,:,i,j] + contract('ef,mnef->mn', t2[ij], ERIoovv[ij])
            lHoooo = Hoooo
        else:
            Hoooo = ERI[o,o,o,o].copy()
       
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i
                jj = j*no + j

                tmp = contract('e,mne->mn',t1[j], ERIooov[jj][:,:,i])
                tmp1 = contract('e,mne->nm', t1[i], ERIooov[ii][:,:,j]) 
                Hoooo[:,:,i,j] = Hoooo[:,:,i,j] + tmp + tmp1
                                  
                Hoooo[:,:,i,j] = Hoooo[:,:,i,j] + contract('ef,mnef->mn', t2[ij], ERIoovv[ij])
                
                tmp = contract('eE,mnef -> mnEf', QL[ii], ERI[o,o,v,v])
                tmp = contract('fF, mnEf -> mnEF', QL[jj], tmp)
                Hoooo[:,:,i,j] = Hoooo[:,:,i,j] + contract('e,f,mnef->mn', t1[i], t1[j], tmp)
            lHoooo = Hoooo  
        return lHoooo  

    def build_Hoooo(self, o, v, ERI, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(t1, torch.Tensor):
                Hoooo = ERI[o,o,o,o].clone().to(self.ccwfn.device1)
            else: 
                Hoooo = ERI[o,o,o,o].copy()
            Hoooo = Hoooo + contract('ijef,mnef->mnij', t2, ERI[o,o,v,v])
        else:
            if isinstance(ERI, torch.Tensor):
                Hoooo = ERI[o,o,o,o].clone().to(self.ccwfn.device1)
            else:
                Hoooo = ERI[o,o,o,o].copy()
            tmp = contract('je,mnie->mnij', t1, ERI[o,o,o,v])
            Hoooo = Hoooo + (tmp + tmp.swapaxes(0,1).swapaxes(2,3))
            if self.ccwfn.model == 'CC2':
                Hoooo = Hoooo + contract('jf,mnif->mnij', t1, contract('ie,mnef->mnif', t1, ERI[o,o,v,v]))
            else:
                Hoooo = Hoooo + contract('ijef,mnef->mnij', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v]) 
        return Hoooo

    def build_lHvvvv(self, o, v, no, ERI, ERIoovv, ERIvovv, ERIvvvv, QL, t1, t2):
        contract = self.contract
        Sijmn = self.Local.Sijmn
        Sijmm = self.Local.Sijmm
        if self.ccwfn.model == 'CCD':  
            lHvvvv = []
            lHvvvv_im = []
            for ij in range(no*no):

                Hvvvv = ERIvvvv[ij].copy()
                
                for mn in range(no*no):
                    m = mn // no 
                    n = mn % no 
                    ijmn = ij*(no**2) + mn

                    tmp = Sijmn[ijmn] @ t2[mn]
                    tmp = tmp @ Sijmn[ijmn].T 
                    Hvvvv = Hvvvv + contract('ab,ef->abef',tmp, ERIoovv[ij][m,n]) 
                lHvvvv.append(Hvvvv)
        else: 
            lHvvvv = []
            lHvvvv_im = []
            
            #lHvvvv_im -> needed for Hvvvo_ijm - Eqn 89
            for i in range(no): 
                ii = i*no + i 

                for m in range(no):
                    im = i*no + m
                    mm = m*no +m 
 
                    Hvvvv_im = contract('abcd, aA -> Abcd', ERI[v,v,v,v], QL[im]) 
                    Hvvvv_im = contract('Abcd, bB -> ABcd', Hvvvv_im, QL[im]) 
                    Hvvvv_im = contract('ABcd, cC -> ABCd', Hvvvv_im, QL[ii]) 
                    Hvvvv_im = contract('ABCd, dD -> ABCD', Hvvvv_im, QL[mm])
 
                    for f in range(no):
                        ff = f*no + f                                       
                        imf = im*no + f
                           
                        #Simff
                        tmp = t1[f] @ Sijmm[imf].T
                        tmp1 = contract('aef,aA -> Aef', ERI[v,f,v,v], QL[im]) 
                        tmp2 = contract('Aef,eE -> AEf', tmp1, QL[ii]) 
                        tmp2 = contract('AEf, fF -> AEF', tmp2, QL[mm])
                        tmp2 = contract('b,aef->abef',tmp, tmp2)
                        
                        tmp1 = contract('Aef,eE->AEf', tmp1, QL[mm])
                        tmp1 = contract('AEf, fF -> AEF', tmp1, QL[ii])  
                        tmp1 = contract('b,aef->abef',tmp, tmp1)
                        Hvvvv_im = Hvvvv_im - (tmp2 + tmp1.swapaxes(0,1).swapaxes(2,3))
                        
                        for n in range(no): 
                            fn = f*no + n 
                            nn = n*no + n 
                            imfn = im*(no**2) + fn
                            imn = im*no + n

                            #Simfn 
                            tmp = Sijmn[imfn] @ t2[fn]
                            tmp = tmp @ Sijmn[imfn].T
                            tmp1 = contract('ef,eE->Ef', ERI[f,n,v,v], QL[ii])
                            tmp1 = contract('Ef, fF -> EF', tmp1, QL[mm]) 
                            Hvvvv_im = Hvvvv_im + contract('ab,ef->abef',tmp, tmp1)

                            #Simff
                            tmp = t1[f] @ Sijmm[imf].T
                            #Simnn
                            tmp2 = t1[n] @ Sijmm[imn].T
                            Hvvvv_im = Hvvvv_im + contract('a,b,ef->abef',tmp, tmp2, tmp1)
                    lHvvvv_im.append(Hvvvv_im)

            #Hv_ij v_ij v_ij v_ij - Eqn 92
            for ij in range(no*no):
                Hvvvv = ERIvvvv[ij].copy()

                for m in range(no):
                    mm = m*no + m
                    ijm = ij*no + m 

                    tmp = t1[m] @ Sijmm[ijm].T 
                    tmp = contract('b,aef->abef',tmp, ERIvovv[ij][:,m,:,:]) 
                    Hvvvv = Hvvvv - (tmp + tmp.swapaxes(0,1).swapaxes(2,3))
 
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n                       
                        ijmn = ij*(no**2) + mn
                        ijn = ij*no + n

                        tmp = Sijmn[ijmn] @ t2[mn]
                        tmp = tmp @ Sijmn[ijmn].T 
                        Hvvvv = Hvvvv + contract('ab,ef->abef',tmp, ERIoovv[ij][m,n])
                             
                        tmp = t1[m] @ Sijmm[ijm].T
                        #Sijnn 
                        tmp1 = t1[n] @ Sijmm[ijn].T 
                        Hvvvv = Hvvvv + contract('a,b,ef->abef',tmp, tmp1, ERIoovv[ij][m,n]) 
                lHvvvv.append(Hvvvv)  
        return lHvvvv, lHvvvv_im            

    def build_Hvvvv(self, o, v, ERI, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(ERI, torch.Tensor):
                Hvvvv = ERI[v,v,v,v].clone().to(self.ccwfn.device1)
            else:
                Hvvvv = ERI[v,v,v,v].copy()
            Hvvvv = Hvvvv + contract('mnab,mnef->abef', t2, ERI[o,o,v,v])           
        else:
            if isinstance(ERI, torch.Tensor):
                Hvvvv = ERI[v,v,v,v].clone().to(self.ccwfn.device1)
            else:
                Hvvvv = ERI[v,v,v,v].copy()
            tmp = contract('mb,amef->abef', t1, ERI[v,o,v,v])
            Hvvvv = Hvvvv - (tmp + tmp.swapaxes(0,1).swapaxes(2,3))
            if self.ccwfn.model == 'CC2':
                Hvvvv = Hvvvv + contract('nb,anef->abef', t1, contract('ma,mnef->anef', t1, ERI[o,o,v,v]))
            else:
                Hvvvv = Hvvvv + contract('mnab,mnef->abef', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
        return Hvvvv

    def build_lHvovv(self,o,v,no, ERI, ERIvovv, ERIoovv, QL, t1): 
        contract = self.contract 
        Sijmn = self.Local.Sijmn
        Sijmm = self.Local.Sijmm
        if self.ccwfn.model == 'CCD':
            lHvovv = ERIvovv.copy()
            lHvovv_ii = []
            lHvovv_imn = [] 
            lHvovv_imns = [] 
        else:
            lHvovv = []
            lHvovv_ii = []
            lHvovv_imn = []
            lHvovv_imns = []

            #Hv_mn i v_ii v_mn -> Hvovv_imns - Eqn 95  
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n

                        tmp = contract('afe, aA -> Afe', ERI[v,i,v,v], QL[mn]) 
                        tmp = contract('Afe, fF -> AFe', tmp, QL[ii]) 
                        Hvovv_imns = contract('AFe, eE -> AFE', tmp, QL[mn]) 

                        for k in range(no):
                            kk = k*no + k 
                            mnk = mn*no + k

                            #Smnkk
                            tmp = t1[k] @ Sijmm[mnk].T
                            tmp1 = contract('fe,fF->Fe', ERI[k,i,v,v], QL[ii]) 
                            tmp1 = contract('Fe, eE -> FE', tmp1, QL[mn]) 
                            Hvovv_imns = Hvovv_imns - contract('a,fe->afe', tmp, tmp1)                       
                        lHvovv_imns.append(Hvovv_imns)

            #Hv_mn i v_mn v_ii - Eqn 96 
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    for n in range(no): 
                        mn = m*no + n
                        nn = n*no + n

                        Hvovv_imn = contract('aef,aA-> Aef', ERI[v,i,v,v], QL[mn]) 
                        Hvovv_imn = contract('Aef, eE -> AEf', Hvovv_imn, QL[mn]) 
                        Hvovv_imn = contract('AEf, fF -> AEF', Hvovv_imn, QL[ii]) 
                        
                        for k in range(no):
                            kk = k*no + k 
                            mnk = mn*no + k

                            #Smnkk 
                            tmp = t1[k] @ Sijmm[mnk].T  
                            tmp1 = contract('ef,eE ->Ef', ERI[k,i,v,v], QL[mn]) 
                            tmp1 = contract('Ef, fF -> EF', tmp1, QL[ii])
                            Hvovv_imn = Hvovv_imn - contract('a,ef->aef', tmp, tmp1)     
                        lHvovv_imn.append(Hvovv_imn)
 
            # Hv_ii o v_ij v_ij - Eqn 97 
            for ij in range(no*no):
                i = ij // no
                j = ij % no 
                ii = i*no + i 

                Hvovv_ii = contract('ajbc, aA ->Ajbc', ERI[v,o,v,v], QL[ii])
                Hvovv_ii = contract('Ajbc, bB -> AjBc', Hvovv_ii, QL[ij])
                Hvovv_ii = contract('AjBc, cC -> AjBC', Hvovv_ii, QL[ij]) 

                for n in range(no): 
                    nn = n*no + n
                    iin = ii*no + n

                    #Siinn
                    tmp = t1[n] @ Sijmm[iin].T
                    Hvovv_ii = Hvovv_ii - contract('a,mef->amef',tmp, ERIoovv[ij][n,:])
                lHvovv_ii.append(Hvovv_ii) 

            #Hvovv_ij - not needed for lambda but may be needed for other eqns 
            #for ij in range(no*no):
                
                #Hvovv = ERIvovv[ij].copy()
                #for n in range(no):
                    #nn = n*no + n
                    
                    #Sijnn = QL[ij].T @ QL[nn] 
                    #tmp = t1[n] @ Sijnn.T 
                    #Hvovv_1 -= contract('a,mef->amef',tmp, ERIoovv[ij][n,:])
                #lHvovv.append( Hvovv + Hvovv_1)       
        return lHvovv_ii, lHvovv_imn, lHvovv_imns
        
    def build_Hvovv(self, o, v, ERI, t1):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(ERI, torch.Tensor):
                Hvovv = ERI[v,o,v,v].clone().to(self.ccwfn.device1)
            else:
                Hvovv = ERI[v,o,v,v].copy()
        else:
            if isinstance(ERI, torch.Tensor):
                Hvovv = ERI[v,o,v,v].clone().to(self.ccwfn.device1)
            else:
                Hvovv = ERI[v,o,v,v].copy()
            Hvovv = Hvovv - contract('na,nmef->amef', t1, ERI[o,o,v,v])
        return Hvovv

    def build_lHooov(self, o, v, no, ERI, ERIooov, QL, t1):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            lHooov = ERIooov.copy()
            lHijov = []
            lHjiov = []
            lHmine = []
            lHimne = []
        else:
            lHooov = []
            lHijov = []
            lHjiov = []

            lHmine = []
            lHimne = []
                        
            #Hmine and Himne - Eqn 98 and 99  
            for i in range(no): 
                ii = i*no + i
                for m in range(no): 
                    for n in range(no):
                        nn = n*no + n                       
                        
                        Hmine = ERIooov[ii][m,i,n].copy()
                        
                        tmp = contract('ef,eE ->Ef', ERI[i,m,v,v], QL[ii])
                        tmp = contract('Ef, fF -> EF', tmp, QL[nn]) 
                        Hmine = Hmine + contract('f,ef->e', t1[n], tmp) 

                        lHmine.append(Hmine) 

                        Himne = ERIooov[ii][i,m,n].copy()    
   
                        tmp = contract('ef, eE- >Ef', ERI[m,i,v,v], QL[ii]) 
                        tmp = contract('Ef, fF -> EF', tmp, QL[nn]) 
                        Himne = Himne + contract('f,ef->e', t1[n], tmp) 

                        lHimne.append(Himne)       
 
            #Hijov and Hjiov - Eqn 100 and 101
            for ij in range(no*no):
                i = ij // no
                j = ij % no 
                ii = i*no + i

                Hijov = ERIooov[ij][i,j,:,:].copy()
                Hjiov = ERIooov[ij][j,i,:,:].copy()
                
                for m in range(no):
                    mm = m*no + m
           
                    tmp = contract('eE, ef ->Ef', QL[ij], ERI[i,j,v,v]) 
                    tmp = contract('fF, Ef -> EF', QL[mm], tmp)
                    Hjiov[m] = Hjiov[m] + contract('f,ef->e',t1[m], tmp)

                    tmp = contract('eE, ef ->Ef', QL[ij], ERI[j,i,v,v])
                    tmp = contract('fF, Ef -> EF', QL[mm], tmp)

                    Hijov[m] = Hijov[m] + contract('f,ef->e',t1[m], tmp)
                lHijov.append(Hijov) 
                lHjiov.append(Hjiov)  
 
            #Hooov_ij - not needed for lambda but may be needed for other eqns
            #for ij in range(no*no):
                #i = ij // no
                #ii = i*no + i 
       
                #Hooov = ERIooov[ij].copy()
                
                #tmp = contract('eE, nmef ->nmEf', QL[ij], ERI[o,o,v,v]) 
                #tmp = contract('fF, nmEf -> nmEF', QL[ii], tmp)
                #Hooov[:,:,i,:] = contract('f,nmef->mne',t1[i], tmp)         
                #lHooov.append(Hooov) 
        return lHjiov, lHijov, lHmine, lHimne   
        
    def build_Hooov(self, o, v, ERI, t1):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(ERI, torch.Tensor):
                Hooov = ERI[o,o,o,v].clone().to(self.ccwfn.device1)
            else:
                Hooov = ERI[o,o,o,v].copy() 
        else:
            if isinstance(ERI, torch.Tensor):
                Hooov = ERI[o,o,o,v].clone().to(self.ccwfn.device1)
            else:
                Hooov = ERI[o,o,o,v].copy()
            Hooov = Hooov + contract('if,nmef->mnie', t1, ERI[o,o,v,v])
        return Hooov

    def build_lHovvo(self, o, v, no, ERI, L, ERIovvo, QL, t1, t2):
        contract = self.contract
        Sijim = self.Local.Sijim
        Sijmm = self.Local.Sijmm
        Sijii = self.Local.Sijii 
        if self.ccwfn.model == 'CCD':
            lHovvo_mi = []
            lHovvo_mj = []
            lHovvo_mm = []
 
            #Hiv_mj v_ij o
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mj = m*no + j

                    Hovvo_mj = contract('bB, be -> Be', QL[mj], ERI[i,v,v,m]) 
                    Hovvo_mj = contract('eE, Be -> BE', QL[ij], Hovvo_mj) 
                    
                    for n in range(no):
                        mn = m*no + n
                        nm = n*no + m
                        mjn = mj*no + n

                        #Smjmn <- Sijim
                        tmp = t2[mn] @ Sijim[mjn].T
                        tmp1 = QL[ij].T @ ERI[i,n,v,v]
                        tmp1 = tmp1 @ QL[mn]
                        Hovvo_mj = Hovvo_mj - tmp.T @ tmp1.T

                        #Smjnm <- Sijmi 
                        tmp = t2[nm] @ Sijim[mjn].T
                        tmp1 = QL[ij].T @ L[i,n,v,v]
                        tmp1 = tmp1 @ QL[nm]
                        Hovvo_mj = Hovvo_mj + tmp.T @ tmp1.T
                    lHovvo_mj.append(Hovvo_mj)

            #Hiv_mi v_ij o
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mi = m*no + i

                    Hovvo_mi = contract('bB, eE, be -> BE', QL[mi], QL[ij], ERI[j,v,v,m])

                    for n in range(no):
                        mn = m*no + n
                        nm = n*no + m
                        min = mi*no + n

                        #Smimn <- Sijim
                        tmp = t2[mn] @ Sijim[min].T
                        tmp1 = QL[ij].T @ ERI[j,n,v,v]
                        tmp1 = tmp1 @ QL[mn]
                        Hovvo_mi = Hovvo_mi - tmp.T @ tmp1.T

                        #Sminm <- Sijmi
                        tmp = t2[nm] @ Sijim[min].T
                        tmp1 = QL[ij].T @ L[j,n,v,v]
                        tmp1 = tmp1 @ QL[nm]
                        Hovvo_mi = Hovvo_mi + tmp.T @ tmp1.T
                    lHovvo_mi.append(Hovvo_mi)

            #Hovvo_ij - not needed for lambda but may be needed for other eqns
            #for ij in range(no*no):
                #i = ij // no
                #j = ij % no

                #Hovvo = ERIovvo[ij].copy()

                #for mn in range(no*no):
                    #m = mn // no
                    #n = mn % no 
                    #jn = j*no + n
                    #nj = n*no + j
 
                    #Sijjn = QL[ij].T @ QL[jn]
                    #tmp = t2[jn] @ Sijjn.T 
                    #tmp1 = QL[ij].T @ ERI[m,n,v,v]
                    #tmp1 = tmp1 @ QL[jn]
                    #Hovvo[m,:,:,j] = Hovvo[m,:,:,j] - tmp.T @ tmp1.T
                    
                    #Sijnj = QL[ij].T @ QL[nj] 
                    #tmp = t2[nj] @ Sijnj.T 
                    #tmp1 = QL[ij].T @ L[m,n,v,v] 
                    #tmp1 = tmp1 @ QL[nj] 
                    #Hovvo[m,:,:,j] = Hovvo[m,:,:,j] + tmp.T @ tmp1.T
                #lHovvo.append(Hovvo)
        else:
            lHovvo_mi = []
            lHovvo_mj = []
            lHovvo_mm = []
            
            #Hi v_mm v_ii o - Eqn 102 
            for i in range(no): 
                ii = i*no + i 
                for m in range(no):
                    mm = m*no + m

                    Hovvo_mm = contract('be, bB-> Be', ERI[i,v,v,m], QL[mm]) 
                    Hovvo_mm = contract('Be, eE-> BE', Hovvo_mm, QL[ii]) 
      
                    tmp = contract('bef, bB-> Bef', ERI[i,v,v,v], QL[mm])
                    tmp = contract('Bef, eE-> BEf', tmp, QL[ii])
                    tmp = contract('BEf, fF-> BEF', tmp, QL[mm])
                    Hovvo_mm = Hovvo_mm + contract('f,bef->be', t1[m], tmp)
                    
                    for n in range(no):
                        nn = n*no + n
                        mn = m*no + n
                        nm = n*no + m
                        mmn = mm*no + n

                        #Smmnn <- Siijj
                        tmp = Sijmm[mmn] @ t1[n]
                        tmp1 = contract('e, eE->E', ERI[i,n,v,m], QL[ii])
                        Hovvo_mm = Hovvo_mm - contract('b,e->be', tmp, tmp1)
                        
                        #Smmmn <- Siiij 
                        tmp1 = t2[mn] @ Sijii[mn] 
                        #tmp2 = contract('ef, eE,fF->EF', ERI[i,n,v,v], QL[ii], QL[mn]) 
                        tmp2 = contract('ef, eE->Ef', ERI[i,n,v,v], QL[ii])
                        tmp3 = contract('Ef, fF->EF', tmp2, QL[mn])
                        Hovvo_mm = Hovvo_mm - contract('fb,ef->be', tmp1, tmp3) 
 
                        #tmp1 = contract('ef, eE,fF->EF', ERI[i,n,v,v], QL[ii], QL[mm])
                        tmp2 = contract('Ef, fF->EF', tmp2, QL[mm])
                        Hovvo_mm = Hovvo_mm - contract('f,b,ef->be', t1[m], tmp, tmp2) 

                        #Smmnm <- Siiij  
                        tmp = t2[nm] @ Sijii[mn] 
                        #tmp1 = contract('ef,eE,fF->EF', L[i,n,v,v], QL[ii], QL[nm]) 
                        tmp1 = contract('ef, eE->Ef', L[i,n,v,v], QL[ii])
                        tmp1 = contract('Ef, fF->EF', tmp1, QL[nm])
                        Hovvo_mm = Hovvo_mm + contract('fb,ef->be', tmp, tmp1)                            
                    lHovvo_mm.append(Hovvo_mm)

            #Hj v_mi v_ij o - Eqn 103
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                jj =  j*no + j

                for m in range(no):
                    mi = m*no + i
                    mm = m*no + m

                    #Hovvo_mi = contract('bB, eE, be -> BE', QL[mi], QL[ij], ERI[j,v,v,m])
                    Hovvo_mi = contract('bB, be -> Be', QL[mi], ERI[j,v,v,m])
                    Hovvo_mi = contract('eE, Be -> BE', QL[ij], Hovvo_mi)

                    #tmp = contract('abc,aA,bB,cC->ABC',ERI[j,v,v,v], QL[mi], QL[ij], QL[mm])
                    tmp = contract('abc, aA ->Abc', ERI[j,v,v,v], QL[mi])
                    tmp = contract('Abc, bB ->ABc', tmp, QL[ij])
                    tmp = contract('ABc, cC->ABC', tmp, QL[mm])
                    Hovvo_mi = Hovvo_mi + contract('f,bef->be', t1[m], tmp)

                    for n in range(no):
                        mn = m*no + n
                        nm = n*no + m
                        nn = n*no + n
                        min = mi*no + n 

                        #Sminn <- Sijmm
                        tmp = Sijmm[min] @ t1[n]
                        tmp1 = contract ('e,eE->E', ERI[j,n,v,m], QL[ij])
                        Hovvo_mi = Hovvo_mi - contract('b,e->be', tmp, tmp1)

                        #Smimn <- Sijim 
                        tmp1 = t2[mn] @ Sijim[min].T
                        tmp2 = QL[ij].T @ ERI[j,n,v,v]
                        tmp3 = tmp2 @ QL[mn]
                        Hovvo_mi = Hovvo_mi - tmp1.T @ tmp3.T

                        tmp1 = tmp2 @ QL[mm]
                        Hovvo_mi = Hovvo_mi - contract('f,b,ef->be', t1[m], tmp, tmp1)

                        #Sminm <- Sijmi  
                        tmp = t2[nm] @ Sijim[min].T
                        tmp1 = QL[ij].T @ L[j,n,v,v]
                        tmp1 = tmp1 @ QL[nm]
                        Hovvo_mi = Hovvo_mi + tmp.T @ tmp1.T
                    lHovvo_mi.append(Hovvo_mi)

            #Hi v_mj v_ij o - Eqn 104
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                jj =  j*no + j
 
                for m in range(no):
                    mj = m*no + j
                    mm = m*no + m

                    #Hovvo_mj = contract('bB, eE, be -> BE', QL[mj], QL[ij], ERI[i,v,v,m])
                    Hovvo_mj = contract('bB, be -> Be', QL[mj], ERI[i,v,v,m]) 
                    Hovvo_mj = contract('eE, Be -> BE', QL[ij], Hovvo_mj)

                    #tmp = contract('abc,aA,bB,cC->ABC',ERI[i,v,v,v], QL[mj], QL[ij], QL[mm])
                    tmp = contract('abc, aA->Abc', ERI[i,v,v,v], QL[mj])
                    tmp = contract('Abc, bB->ABc', tmp, QL[ij])
                    tmp = contract('ABc, cC->ABC', tmp, QL[mm])
                    Hovvo_mj = Hovvo_mj + contract('f,bef->be', t1[m], tmp)

                    for n in range(no):
                        mn = m*no + n
                        nm = n*no + m
                        nn = n*no + n
                        mjn = mj*no + n

                        #Smjnn <- Sijmm 
                        tmp = Sijmm[mjn] @ t1[n] 
                        tmp1 = contract ('e,eE->E', ERI[i,n,v,m], QL[ij])
                        Hovvo_mj = Hovvo_mj - contract('b,e->be', tmp, tmp1) 

                        #Smjmn <- Sijim
                        tmp1 = t2[mn] @ Sijim[mjn].T
                        tmp2 = QL[ij].T @ ERI[i,n,v,v]
                        tmp3 = tmp2 @ QL[mn]
                        Hovvo_mj = Hovvo_mj - tmp1.T @ tmp3.T

                        tmp1 = tmp2 @ QL[mm] 
                        Hovvo_mj = Hovvo_mj - contract('f,b,ef->be', t1[m], tmp, tmp1) 

                        #Smjnm <- Sijmi
                        tmp = t2[nm] @ Sijim[mjn].T
                        tmp1 = QL[ij].T @ L[i,n,v,v]
                        tmp1 = tmp1 @ QL[nm]
                        Hovvo_mj = Hovvo_mj + tmp.T @ tmp1.T
                    lHovvo_mj.append(Hovvo_mj)

            #for ij in range(no*no):
                #i = ij // no
                #j = ij % no
                #jj = j*no + j
 
                #Hovvo = ERIovvo[ij].copy()
                    
                #for m in range(no):

                    #tmp = contract('abc,aA->Abc',ERI[m,v,v,v], QL[ij])
                    #tmp = contract('Abc,bB->ABc',tmp, QL[ij])
                    #tmp = contract('ABc,cC->ABC',tmp, QL[jj])                      
                    #Hovvo[m,:,:,j] = Hovvo + contract('f,bef->be',t1[j], tmp) 
                    
                    #for n in range(no):
                        #nn = n*no + n
                        #jn = j*no + n
                        #nj = n*no + j
 
                        #Sijnn = QL[ij].T @ QL[nn]
                        #tmp = t1[n] @ Sijnn.T
                        #tmp1 = QL[ij].T @ ERI[m,n,v,j]     
                        #Hovvo[m,:,:,j] = Hovvo[m,:,:,j] - contract('b,e->be', tmp, tmp1)

                        #Sijjn = QL[ij].T @ QL[jn]
                        #tmp1 = t2[jn] @ Sijjn.T
                        #tmp2 = QL[ij].T @ ERI[m,n,v,v]
                        #tmp3 = tmp2 @ QL[jn] 
                        #Hovvo[m,:,:,j] = Hovvo[m,:,:,j] - tmp1.T @ tmp3.T 
                        
                        #tmp1 = tmp2 @ QL[jj]
                        #Hovvo[m,:,:,j] = Hovvo[m,:,:,j] - contract('f,b,ef->be',t1[j], tmp, tmp1)
 
                        #Sijnj = QL[ij].T @ QL[nj]
                        #tmp = t2[nj] @ Sijnj.T
                        #tmp1 = QL[ij].T @ L[m,n,v,v]
                        #tmp1 = tmp1 @ QL[nj] 
                        #Hovvo[m,:,:,j] = Hovvo[m,:,:,j] + tmp.T @ tmp1.T   
                #lHovvo.append(Hovvo)            
        return lHovvo_mi, lHovvo_mj, lHovvo_mm  

    def build_Hovvo(self, o, v, ERI, L, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(ERI, torch.Tensor):
                Hovvo = ERI[o,v,v,o].clone().to(self.ccwfn.device1)
            else:
                Hovvo = ERI[o,v,v,o].copy()
            # clean up
            Hovvo = Hovvo - contract('jnfb,mnef->mbej', t2, ERI[o,o,v,v])
            Hovvo = Hovvo + contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        else:
            if isinstance(ERI, torch.Tensor):
                Hovvo = ERI[o,v,v,o].clone().to(self.ccwfn.device1)
            else:
                Hovvo = ERI[o,v,v,o].copy()
            Hovvo = Hovvo + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
            Hovvo = Hovvo - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
            if self.ccwfn.model != 'CC2':
                Hovvo = Hovvo - contract('jnfb,mnef->mbej', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v]) #self.ccwfn.build_tau(t1, t2)
                Hovvo = Hovvo + contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        return Hovvo

    def build_lHovov(self, o, v, no, ERI, ERIovov, ERIooov, QL, t1, t2):
        Sijii = self.Local.Sijii
        Sijmm = self.Local.Sijmm
        Sijim = self.Local.Sijim
        Sijmn = self.Local.Sijmn 
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            #lHovov = [] 
            lHovov_mi = []
            lHovov_mj = []
            lHovov_mm = []

            #Hovov_ij - not needed for lambda but may be needed fro other eqns
            #for ij in range(no*no):
                #i = ij // no 
                #j = ij % no
   
                #Hovov = np.zeros((no,self.Local.dim[ij], self.Local.dim[ij]))
                #for m in range(no):
                    
                    #Hovov[m] += Hovov[m] + ERIovov[ij][m,:,j,:].copy()
                    
                    #for n in range(no):
                        #jn = j*no + n

                        #Sijjn = QL[ij].T @ QL[jn]
                        #tmp = t2[jn] @ Sijjn.T
                        #tmp1 = QL[ij].T @ ERI[n,m,v,v]
                        #tmp1 = tmp1 @ QL[jn]
                        #Hovov[m] -= Hovov[m] - tmp.T @ tmp2.T

                #lHovov.append(Hovov) 
            
            #Hiv_mj ov_ij
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mj = m*no + j

                    #Hovov_mj = contract('be, bB, eE-> BE', ERI[i,v,m,v], QL[mj], QL[ij])
                    Hovov_mj = contract('be, bB-> Be', ERI[i,v,m,v], QL[mj])      
                    Hovov_mj = contract('Be, eE-> BE', Hovov_mj, QL[ij])   
   
                    for n in range(no):
                        mn = m*no + n
                        mjn = mj*no + n

                        #Smjmn <- Sijim
                        tmp = t2[mn] @ Sijim[mjn].T
                        tmp1 = QL[ij].T @ ERI[n,i,v,v]
                        tmp1 = tmp1 @ QL[mn]
                        Hovov_mj = Hovov_mj - tmp.T @ tmp1.T

                    lHovov_mj.append(Hovov_mj)

            #Hjv_mi ov_ij
            for ij in range(no*no):
                i = ij // no 
                j = ij % no

                for m in range(no):
                    mi = m*no + i 

                    #Hovov_mi = contract('be, bB, eE-> BE', ERI[j,v,m,v], QL[mi], QL[ij])
                    Hovov_mi = contract('be, bB-> Be', ERI[j,v,m,v], QL[mi])
                    Hovov_mi = contract('Be, eE-> BE', Hovov_mi, QL[ij])    
                   
                    for n in range(no):
                        mn = m*no + n 
                        min = mi*no + n

                        #Smimn <- Sijim
                        tmp = t2[mn] @ Sijim[min].T
                        tmp1 = QL[ij].T @ ERI[n,j,v,v]
                        tmp1 = tmp1 @ QL[mn]
                        Hovov_mi = Hovov_mi - tmp.T @ tmp1.T
                    lHovov_mi.append(Hovov_mi)     
        else:  
            lHovov = []
            lHovov_mi = []
            lHovov_mj = []
            lHovov_mm = []
    
            #Hiv_mm ov_ii - Eqn 105   
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    mm = m* no + m
                    
                    #Hovov_mm = Hovov_mm + contract('be,bB,eE->BE', ERI[i,v,m,v], QL[mm], QL[ii])
                    Hovov_mm = contract('be, bB->Be', ERI[i,v,m,v], QL[mm])
                    Hovov_mm = contract('Be, eE->BE', Hovov_mm, QL[ii])                
 
                    tmp = contract('bef, bB, eE,fF->BEF', ERI[v,i,v,v], QL[mm], QL[ii], QL[mm])
                    Hovov_mm = Hovov_mm + contract('f,bef->be', t1[m], tmp)
                        
                    for n in range(no):
                        nn = n*no + n
                        mn = m*no + n
                        nm = n*no + m
                        mmn = mm*no + n
                        mmmn = mm*(no**2) + mn

                        #Smmnn <- Siimm
                        tmp1 = Sijmm[mmn] @ t1[n]
                        tst = contract('e, eE->E', ERI[i,n,m,v], QL[ii])
                        Hovov_mm = Hovov_mm - contract('b,e->be', tmp1, tst)
                        
                        #Smmmn <- Siiim
                        tmp2 = t2[mn] @ Sijii[mn] 
                        tmp3 = contract('ef, eE,fF->EF', ERI[n,i,v,v], QL[ii], QL[mn])
                        Hovov_mm = Hovov_mm - contract('fb,ef->be', tmp2, tmp3) 
                        
                        tmp3 = contract('ef, eE,fF->EF', ERI[n,i,v,v], QL[ii], QL[mm])
                        Hovov_mm = Hovov_mm - contract('f,b,ef->be', t1[m], tmp1, tmp3) 
                    lHovov_mm.append(Hovov_mm)

            #Hiv_mj ov_ij - Eqn 106 
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mm = m*no + m
                    mj = m*no + j

                    #Hovov_mj = contract('be, bB, eE-> BE', ERI[i,v,m,v], QL[mj], QL[ij])
                    Hovov_mj = contract('be, bB-> Be', ERI[i,v,m,v], QL[mj])            
                    Hovov_mj = contract('Be, eE-> BE', Hovov_mj, QL[ij])     
                 
                    #tmp = contract('bef, bB, eE, fF-> BEF', ERI[v,i,v,v], QL[mj], QL[ij], QL[mm])
                    tmp = contract('bef, bB-> Bef', ERI[v,i,v,v], QL[mj])
                    tmp = contract('Bef, eE-> BEf', tmp, QL[ij])
                    tmp = contract('BEf, fF-> BEF', tmp, QL[mm]) 
                    Hovov_mj = Hovov_mj + contract('f,bef->be', t1[m], tmp)

                    for n in range(no):
                        nn = n*no + n
                        mn = m*no + n
                        mjn = mj*no + n

                        #Smjnn <- Sijmm
                        tmp = Sijmm[mjn] @ t1[n]
                        tmp1 = contract('e,eE->E', ERI[i,n,m,v], QL[ij]) 
                        Hovov_mj = Hovov_mj - contract('b,e->be', tmp, tmp1)       

                        #Smjmn <- Sijim
                        tmp1 = t2[mn] @ Sijim[mjn].T
                        tmp2 = QL[ij].T @ ERI[n,i,v,v]
                        tmp3 = tmp2 @ QL[mn]
                        Hovov_mj =  Hovov_mj - tmp1.T @ tmp3.T

                        tmp1 = tmp2 @ QL[mm]
                        Hovov_mj = Hovov_mj - contract('f,b,ef->be',t1[m], tmp, tmp1) 
                    lHovov_mj.append(Hovov_mj)

            #Hjv_mi ov_ij - Eqn 107 
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mm = m*no + m
                    mi = m*no + i

                    #Hovov_mi = contract('be, bB, eE-> BE', ERI[j,v,m,v], QL[mi], QL[ij])
                    Hovov_mi = contract('be, bB-> Be', ERI[j,v,m,v], QL[mi])
                    Hovov_mi = contract('Be, eE-> BE', Hovov_mi, QL[ij])

                    #tmp = contract('bef, bB, eE, fF-> BEF', ERI[v,j,v,v], QL[mi], QL[ij], QL[mm])
                    tmp = contract('bef, bB-> Bef', ERI[v,j,v,v], QL[mi])
                    tmp = contract('Bef, eE-> BEf', tmp, QL[ij])
                    tmp = contract('BEf, fF-> BEF', tmp, QL[mm])
                    Hovov_mi = Hovov_mi + contract('f,bef->be', t1[m], tmp)

                    for n in range(no):
                        nn = n*no + n
                        mn = m*no + n
                        min = mi*no + n

                        #Sminn <- Sijmm
                        tmp = Sijmm[min] @ t1[n]
                        tmp1 = contract('e,eE->E', ERI[j,n,m,v], QL[ij])
                        Hovov_mi = Hovov_mi - contract('b,e->be', tmp, tmp1)

                        #Smimn <- Sijim
                        tmp1 = t2[mn] @ Sijim[min].T
                        tmp2 = QL[ij].T @ ERI[n,j,v,v]
                        tmp3 = tmp2 @ QL[mn]
                        Hovov_mi = Hovov_mi - tmp1.T @ tmp3.T

                        tmp1 = tmp2 @ QL[mm]
                        Hovov_mi = Hovov_mi - contract('f,b,ef->be',t1[m], tmp, tmp1)
                    lHovov_mi.append(Hovov_mi)

            #Hovov_ij - not needed for lambda but may be needed for other eqns            
            #for ij in range(no*no):
                #j = ij % no 
                #jj = j*no + j
   
                #Hovov = ERIovov[ij].copy()
               
                #for m in range(no):
        
                    #tmp = contract('bB,bef-> Bef', QL[ij], ERI[v,m,v,v])
                    #tmp = contract('eE,Bef->BEf', QL[ij], tmp)
                    #tmp = contract('fF,BEf->BEF',QL[jj], tmp)
                    #Hovov[m,:,j,:] = Hovov[m,:,j,:] + contract('f,bef->be',t1[j], tmp)

                    #for n in range(no):
                        #nn = n*no + n
                        #jn = j*no + n
   
                        #Sijnn = QL[ij].T @ QL[nn]  
                        #tmp = t1[n] @ Sijnn.T
                        #Hovov[m,:,j,:] = Hovov[m,:,j,:] - contract('b,e->be', tmp, ERIooov[ij][m,n,j,:]) 

                        #Sijjn = QL[ij].T @ QL[jn]
                        #tmp1 = t2[jn] @ Sijjn.T
                        #tmp2 = QL[ij].T @ ERI[n,m,v,v]
                        #tmp3 = tmp2 @ QL[jn]
                        #Hovov[m,:,j,:] = Hovov[m,:,j,:] - tmp1.T @ tmp3.T

                        #tmp1 = tmp2 @ QL[jj]
                        #Hovov[m,:,j,:] = Hovov[m,:,j,:] - contract('f,b,ef->be',t1[j], tmp, tmp1)

                #lHovov.append(Hovov) 
        return lHovov_mi, lHovov_mj, lHovov_mm 

    def build_Hovov(self, o, v, ERI, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(ERI, torch.Tensor):
                Hovov = ERI[o,v,o,v].clone().to(self.ccwfn.device1)
            else:
                Hovov = ERI[o,v,o,v].copy()
            Hovov = Hovov - contract('jnfb,nmef->mbje', t2, ERI[o,o,v,v])
        else:
            if isinstance(ERI, torch.Tensor):
                Hovov = ERI[o,v,o,v].clone().to(self.ccwfn.device1)
            else:
                Hovov = ERI[o,v,o,v].copy()
            Hovov = Hovov + contract('jf,bmef->mbje', t1, ERI[v,o,v,v])
            Hovov = Hovov - contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
            if self.ccwfn.model != 'CC2':
                Hovov = Hovov - contract('jnfb,nmef->mbje', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
        return Hovov

    def build_lHvvvo(self, o, v, no, ERI, L, ERIvvvo, ERIoovo, ERIvoov, ERIvovo, ERIoovv, Loovv, QL, t1, t2, Hov, Hvvvv, Hvvvv_im):  
        Sijmj = self.Local.Sijmj
        Sijmm = self.Local.Sijmm
        Sijmn = self.Local.Sijmn
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            lHvvvo_im = []
            
            #Hvvvo_ij - not needed for lambda but may be needed for other eqns
            #for ij in range(no*no):
                #i = ij // no
                #j = ij % no

                #Hvvvo = ERIvvvo[ij].copy()
          
                #for m in range(no):
                    #mi = m*no + i 
                    #im = i*no + m
                    #mm = m*no + m
  
                    #Sijmi = QL[ij].T @ QL[mi] 
                    #tmp = t2[mi] @ Sijmi.T  
                    #tmp1 = Sijmi @ tmp 
                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] - contract('e,ab->abe',Hov[ij][m],tmp1)
                
                    #tmp1 = contract('aef,aA->Aef', L[v,m,v,v], QL[ij])
                    #tmp1 = contract('Aef,eE->AEf',tmp1, QL[ij])
                    #tmp1 = contract('AEf,fF->AEF',tmp1, QL[mi])
                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] + contract('fb,aef->abe',tmp, tmp1)

                    #Sijim = QL[ij].T @ QL[im]
                    #tmp = t2[im] @ Sijim.T 
                    #tmp1 = contract('bfe,bB->Bfe',ERI[v,m,v,v], QL[ij])
                    #tmp2 = contract('Bfe,fF->BFe',tmp1, QL[im])
                    #tmp2 = contract('BFe,eE->BFE',tmp2, QL[ij])
                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] - contract('fa,bfe->abe', tmp, tmp2)

                    #tmp1 = contract('Aef,eE->AEf',tmp1, QL[ij])
                    #tmp1 = contract('AEf,fF->AEF',tmp1, QL[mi]) 
                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] - contract('fb,aef->abe', tmp, tmp1)
                
                    #for n in range(no): 
                        #mn = m*no + n     
                        #nn = n*no + n
                    
                        #Sijmn = QL[ij].T @ QL[mn] 
                        #tmp = Sijmn @ t2[mn]
                        #tmp = tmp @ Sijmn.T 
                        #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] + contract('ab,e->abe',tmp, ERIoovo[ij][m,n,:,i]) 
              
                        #Sijmm = QL[ij].T @ QL[mm]
                        #Sijnn = QL[ij].T @ QL[nn] 
                        #tmp = t1[m] @ Sijmm.T 
                        #tmp1 = t1[n] @ Sijnn.T 
                        #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] + contract('a,b,e->abe',tmp,tmp1, ERIoovo[ij][m,n,:,i])

                #lHvvvo.append( Hvvvo)
        else:
            #lHvvvo = []
            #ltmp_ij = []
            #ltmp1_ij = []

            lHvvvo_im = []
            ltmp_im = []
            ltmp1_im = []

            #terms for Hv_im v_im v_ii o - Eqn 109 and Eqn 110
            for i in range(no): 
                ii = i*no + i 
                for m in range(no):
                    im = i*no + m
                       
                    for n in range(no): 
                        mn = m*no + n
                        nn = n*no + n
                        
                        #tmp = contract('ae,aA,eE->AE', ERI[v,n,v,m], QL[im], QL[ii]) 
                        tmp_im = contract('ae, aA-> Ae', ERI[v,n,v,m], QL[im])
                        tmp_im = contract('Ae, eE->AE', tmp_im, QL[ii])

                        #tmp1 = contract('be,bB,eE->BE', ERI[v,n,m,v], QL[im], QL[ii])                         
                        tmp1_im = contract('be, bB-> Be', ERI[v,n,m,v], QL[im]) 
                        tmp1_im = contract('Be, eE-> BE', tmp1_im, QL[ii]) 

                        for k in range(no): 
                            mk = m *no + k
                            km = k*no + m
                            imk = im*no + k

                            #Simmk <- Sijjm
                            tmp = t2[mk] @ Sijmj[imk].T 
                            #tmp1 = contract('fe,fF,eE->FE', ERI[n,k,v,v], QL[mk], QL[ii]) 
                            tmp1 = contract('fe, fF-> Fe', ERI[n,k,v,v], QL[mk])
                            tmp1 = contract('Fe, eE-> FE', tmp1, QL[ii])
                            tmp_im = tmp_im - contract('fa,fe->ae', tmp, tmp1)  

                            #tmp1 = contract('ef, eE-> Ef', ERI[n,k,v,v], QL[ii], QL[mk])
                            tmp1 = contract('ef, eE-> Ef', ERI[n,k,v,v], QL[ii])
                            tmp1 = contract('Ef, fF-> EF', tmp1, QL[mk]) 
                            tmp1_im = tmp1_im - contract('fb,ef->be', tmp, tmp1) 

                            #Simkm <- Sijmj
                            tmp = t2[km] @ Sijmj[imk].T
                            #tmp1 = contract('ef,eE,fF->EF', L[n,k,v,v], QL[ii], QL[mk]) 
                            tmp1 = contract('ef, eE-> Ef', L[n,k,v,v], QL[ii])
                            tmp1 = contract('Ef, fF-> EF', tmp1, QL[mk])
                            tmp1_im = tmp1_im + contract('fb,ef->be', tmp, tmp1)       

                        ltmp_im.append(tmp_im)
                        ltmp1_im.append(tmp1_im)        

            #Hv_im v_im v_ii o - Eqn 108
            for i in range(no): 
                ii = i*no + i
                for m in range(no):
                    im = i*no + m 
                    mi = m*no + i
                    mm = m*no + m               

                    #Hvvvo_im = contract('abe, aA, bB, eE->ABE', ERI[v,v,v,m], QL[im], QL[im], QL[ii])   
                    Hvvvo_im = contract('abe, aA-> Abe', ERI[v,v,v,m], QL[im])
                    Hvvvo_im = contract('Abe, bB-> ABe', Hvvvo_im, QL[im])                    
                    Hvvvo_im = contract('ABe, eE-> ABE', Hvvvo_im, QL[ii])

                    Hvvvo_im = Hvvvo_im + contract('f,abef->abe', t1[m], Hvvvv_im[im]) 
                    
                    for n in range(no): 
                        nm = n*no + m 
                        nn = n*no + n
                        mn = m*no + n
                        imn = im*no + n 

                        #Simnm <- Sijmj
                        tmp = t2[nm] @ Sijmj[imn].T
                        tmp1 = Sijmj[imn] @ tmp 
                        Hvvvo_im = Hvvvo_im - contract('e, ab->abe', Hov[ii][n], tmp1) 

                        #tmp1 = contract('bfe,bB,fF,eE->BFE', ERI[v,n,v,v], QL[im], QL[mn], QL[ii]) 
                        tmp1 = contract('bfe, bB-> Bfe', ERI[v,n,v,v], QL[im])
                        tmp1 = contract('Bfe, fF-> BFe', tmp1, QL[mn])
                        tmp1 = contract('BFe, eE-> BFE', tmp1, QL[ii])
                        #Simmn <- Sijjm
                        tmp2 = t2[mn] @ Sijmj[imn].T 
                        Hvvvo_im = Hvvvo_im - contract('fa,bfe->abe', tmp2, tmp1) 
                        
                        #tmp1 = contract('aef,aA,eE,fF->AEF', ERI[v,n,v,v], QL[im], QL[ii], QL[mn])
                        tmp1 = contract('aef, aA-> Aef', ERI[v,n,v,v], QL[im])
                        tmp1 = contract('Aef, eE-> AEf', tmp1, QL[ii])
                        tmp1 = contract('AEf, fF-> AEF', tmp1, QL[mn])
                        Hvvvo_im = Hvvvo_im - contract('fb,aef->abe', tmp2, tmp1) 
                        
                        #tmp1 = contract('aef,aA,eE,fF->AEF', L[v,n,v,v], QL[im], QL[ii], QL[nm]) 
                        tmp1 = contract('aef, aA-> Aef', L[v,n,v,v], QL[im])
                        tmp1 = contract('Aef, eE-> AEf', tmp1, QL[ii])
                        tmp1 = contract('AEf, fF-> AEF', tmp1, QL[nm])
                        Hvvvo_im = Hvvvo_im + contract('fb,aef->abe',tmp, tmp1)
        
                        #Simnn <- Sijmm 
                        tmp = Sijmm[imn] @ t1[n] 
                        Hvvvo_im = Hvvvo_im - contract('b,ae->abe', tmp, ltmp_im[imn])

                        Hvvvo_im = Hvvvo_im - contract('a,be->abe', tmp, ltmp1_im[imn]) 

                        for k in range(no): 
                            kn = k*no + n 
                            kk = k*no + k
                            nn = n*no + n
                            imnk = imn*no + k
                            imk = im*no + k

                            #Simkn <- Sijmn 
                            tmp = Sijmn[imnk] @ t2[kn]
                            tmp = tmp @ Sijmn[imnk].T 
                            tmp1 = QL[ii].T @ ERI[k,n,v,m] 
                            Hvvvo_im = Hvvvo_im + contract('ab,e->abe',tmp, tmp1) 

                            #Simkk <- Sijmm 
                            #Simnn <- Sijmm  
                            tmp = Sijmm[imk] @ t1[k]
                            tmp2 = Sijmm[imn] @ t1[n]
                            Hvvvo_im = Hvvvo_im + contract('a,b,e->abe', tmp, tmp2, tmp1) 
       
                    lHvvvo_im.append( Hvvvo_im)

            #Hvvvo_ij - not needed for lambda but may be needed for other eqns 
            #for i in range(no):
                #for j in range(no):
                     #ij = i*no + j

                     #for m in range(no):
                         
                         #tmp_ij = ERIvovo[ij][:,m,:,i].copy()
                         #tmp1_ij = ERIvoov[ij][:,m,i,:].copy()
                         
                         #for n in range(no):
                             #_in = i*no + n
                             #ni = n*no + i
 
                             #Sijin = QL[ij].T @ QL[_in]
                             #tmp = t2[_in] @ Sijin.T 
                             #tmp1 = contract('ab, aA-> Ab', ERI[m,n,v,v], QL[_in])
                             #tmp1 = contract('Ab, bB-> AB', tmp1, QL[ij])

                             #tmp_ij = tmp_ij - contract('fa, fe-> ae', tmp, tmp1)

                             #tmp1 = contract('ab, aA-> Ab', ERI[m,n,v,v], QL[ij])
                             #tmp1 = contract('Ab, bB-> AB', tmp1, QL[_in])
                             #tmp1_ij = tmp1_ij - contract('fb, ef-> be', tmp, tmp1)

                             #Sijni = QL[ij].T @ QL[ni]
                             #tmp = t2[ni] @ Sijni.T
                             #tmp1 = contract('ab, aA-> Ab', L[m,n,v,v], QL[ij])
                             #tmp1 = contract('Ab, bB-> AB', tmp1, QL[ni])
                             #tmp1_ij = tmp1_ij + contract('fb, ef-> be', tmp, tmp1)

                         #ltmp_ij.append(tmp_ij)
                         #ltmp1_ij.append(tmp_ij)
 
            #for ij in range(no*no):
                #i = ij // no
                #j = ij % no
                #ii = i*no + i

                #Hvvvo = ERIvvvo[ij].copy()

                #Sijii = QL[ij].T @ QL[ii]
                #tmp = t1[i] @ Sijii.T
                #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] + contract('f, abef-> abe', tmp, Hvvvv[ij])

                #for m in range(no):
                    #mi = m*no + i
                    #im = i*no + m
                    #mm = m*no + m
                    #ijm = ij*no + m 

                    #Sijmi = QL[ij].T @ QL[mi]
                    #tmp = t2[mi] @ Sijmi.T
                    #tmp1 = Sijmi @ tmp

                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] - contract('e,ab->abe',Hov[ij][m],tmp1)
                  
                    #tmp1 = contract('aef, aA-> Aef', L[v,m,v,v], QL[ij])
                    #tmp1 = contract('Aef, eE-> AEf', tmp1, QL[ij])
                    #tmp1 = contract('AEf, fF-> AEF', tmp1, QL[mi])
                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] + contract('fb, aef-> abe', tmp, tmp1)

                    #Sijim = QL[ij].T @ QL[im]
                    #tmp = t2[im] @ Sijim.T
                    #tmp1 = contract('bfe, bB-> Bfe', ERI[v,m,v,v], QL[ij])
                    #tmp2 = contract('Bfe, fF-> BFe', tmp1, QL[im])
                    #tmp2 = contract('BFe, eE-> BFE', tmp2, QL[ij])
                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] - contract('fa, bfe-> abe', tmp, tmp2)

                    #tmp1 = contract('Aef, eE-> AEf', tmp1, QL[ij])
                    #tmp1 = contract('AEf, fF-> AEF',tmp1, QL[mi])
                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] - contract('fb, aef-> abe', tmp, tmp1)

                    #Sijmm = QL[ij].T @ QL[mm]
                    #tmp = t1[m] @ Sijmm.T 
                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] - contract('b, ae -> abe', tmp, ltmp_ij[ijm])                  

                    #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] - contract('a,be->abe', tmp , ltmp1_ij[ijm])

                    #for n in range(no):
                        #mn = m*no + n
                        #nn = n*no + n
                        #_in = i*no + n
                        #ni = n*no + i              
   
                        #Sijmn = QL[ij].T @ QL[mn]
                        #tmp = Sijmn @ t2[mn]
                        #tmp = tmp @ Sijmn.T
                        #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] + contract('ab, e-> abe', tmp, ERIoovo[ij][m,n,:,i])

                        #Sijmm = QL[ij].T @ QL[mm]
                        #Sijnn = QL[ij].T @ QL[nn]
                        #tmp = t1[m] @ Sijmm.T
                        #tmp1 = t1[n] @ Sijnn.T
                        #Hvvvo[:,:,:,i] = Hvvvo[:,:,:,i] + contract('a,b,e->abe',tmp,tmp1, ERIoovo[ij][m,n,:,i])
  
                #lHvvvo.append( Hvvvo)
        return lHvvvo_im

    def build_Hvvvo(self, o, v, ERI, L, Hov, Hvvvv, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(ERI, torch.Tensor):
                Hvvvo = ERI[v,v,v,o].clone().to(self.ccwfn.device1)
            else:
                Hvvvo = ERI[v,v,v,o].copy()
            Hvvvo = Hvvvo - contract('me,miab->abei', Hov, t2)
            Hvvvo = Hvvvo + contract('mnab,mnei->abei', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,o])
            Hvvvo = Hvvvo - contract('imfa,bmfe->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo - contract('imfb,amef->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo + contract('mifb,amef->abei', t2, L[v,o,v,v])
        elif self.ccwfn.model == 'CC2':
            if isinstance(ERI, torch.Tensor):
                Hvvvo = ERI[v,v,v,o].clone().to(self.ccwfn.device1)  
            else:
                Hvvvo = ERI[v,v,v,o].copy()
            Hvvvo = Hvvvo - contract('me,miab->abei', self.ccwfn.H.F[o,v], t2)
            Hvvvo = Hvvvo + contract('if,abef->abei', t1, Hvvvv)
            Hvvvo = Hvvvo + contract('nb,anei->abei', t1, contract('ma,mnei->anei', t1, ERI[o,o,v,o]))
            Hvvvo = Hvvvo - contract('mb,amei->abei', t1, ERI[v,o,v,o])
            Hvvvo = Hvvvo - contract('ma,bmie->abei', t1, ERI[v,o,o,v])
        else:
            if isinstance(ERI, torch.Tensor):
                Hvvvo = ERI[v,v,v,o].clone().to(self.ccwfn.device1)
            else:
                Hvvvo = ERI[v,v,v,o].copy()
            Hvvvo = Hvvvo - contract('me,miab->abei', Hov, t2)
            Hvvvo = Hvvvo + contract('if,abef->abei', t1, Hvvvv)
            Hvvvo = Hvvvo + contract('mnab,mnei->abei', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,o])
            Hvvvo = Hvvvo - contract('imfa,bmfe->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo - contract('imfb,amef->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo + contract('mifb,amef->abei', t2, L[v,o,v,v])   
            if isinstance(ERI, torch.Tensor):
                tmp = ERI[v,o,v,o].clone().to(self.ccwfn.device1)
            else: 
                tmp = ERI[v,o,v,o].copy()
            tmp = tmp - contract('infa,mnfe->amei', t2, ERI[o,o,v,v])                
            Hvvvo = Hvvvo - contract('mb,amei->abei', t1, tmp)
            if isinstance(ERI, torch.Tensor):
                tmp = ERI[v,o,o,v].clone().to(self.ccwfn.device1)
            else:
                tmp = ERI[v,o,o,v].copy()
            tmp = tmp - contract('infb,mnef->bmie', t2, ERI[o,o,v,v])
            tmp = tmp + contract('nifb,mnef->bmie', t2, L[o,o,v,v])
            Hvvvo = Hvvvo - contract('ma,bmie->abei', t1, tmp)
            if isinstance(tmp, torch.Tensor):
                del tmp           
        return Hvvvo

    def build_lHovoo(self, o, v, no, ERI, L, ERIovoo, ERIovvv, ERIooov, ERIovov, ERIvoov, Looov, QL, t1, t2, Hov, Hoooo):
        Sijim = self.Local.Sijim
        Sijmj = self.Local.Sijmj
        Sijmm = self.Local.Sijmm
        contract = self.contract
        if self.ccwfn.model =='CCD':
           lHovoo_mn = []
           
           #Hovoo_ij - not needed for lambda but may be needed for other eqns
           #for ij in range(no*no):
               #i = ij // no
               #j = ij % no
 
               #Hovoo = ERIovoo[ij].copy()
 
               #for m in range(no):
               
                   #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] + Hov[ij][m] @ t2[ij]
                   
                   #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] + contract('ef, bef-> b', t2[ij], ERIovvv[ij][m]) 
 
                   #for n in range(no):
                       #_in = i*no + n
                       #jn = j*no + n
                       #nj = n*no + j
                       
                       #Sijin = QL[ij].T @ QL[_in]
                       #tmp = t2[_in] @ Sijin.T 
                       #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] - tmp.T @ ERIooov[_in][n,m,j]

                       #Sijjn = QL[ij].T @ QL[jn]
                       #tmp = t2[jn] @ Sijjn.T 
                       #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] - tmp.T @ ERIooov[jn][m,n,i] 
                       
                       #Sijnj = QL[ij].T @ QL[nj]
                       #tmp = t2[nj] @ Sijnj.T
                       #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] + tmp.T @ Looov[nj][m,n,i]  
               #lHovoo.append(Hovoo)
        else:
            #lHovoo = []
            #ltmp_ij = []
            #ltmp1_ij = []

            lHovoo_mn = []

            #Hov_mn oo - Eqn 111 
            for i in range(no):
                for m in range(no):
                    mm = m*no + m
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n

                        Hovoo_mn = contract('b, bB-> B', ERI[i,v,m,n], QL[mn])
                        
                        Hovoo_mn = Hovoo_mn + contract('e, eb-> b', Hov[mn][i], t2[mn]) 
                        
                        Hovoo_mn = Hovoo_mn + contract('ef, bef-> b', t2[mn], ERIovvv[mn][i])
                        
                        #tmp = contract('bef,bB, eE,fF->BEF', ERI[i,v,v,v], QL[mn], QL[mm], QL[nn])
                        tmp = contract('bef, bB-> Bef', ERI[i,v,v,v], QL[mn])
                        tmp = contract('Bef, eE-> BEf', tmp, QL[mm])
                        tmp = contract('BEf, fF-> BEF', tmp, QL[nn])
                        Hovoo_mn = Hovoo_mn + contract('e, f, bef-> b', t1[m], t1[n], tmp)
 
                        #tmp_mn = contract('be,bB,eE->BE', ERI[i,v,m,v], QL[mn], QL[nn])
                        tmp_mn = contract('be, bB->Be', ERI[i,v,m,v], QL[mn])
                        tmp_mn = contract('Be, eE-> BE', tmp_mn, QL[nn])                        

                        for k in range(no):
                            mk = m*no + k
                            mnk = mn*no + k

                            #Smnmk <- Sijim 
                            tmp = t2[mk] @ Sijim[mnk].T 
                            #tmp1 = contract('fe,fF,eE->FE', ERI[i,k,v,v], QL[mk], QL[nn])
                            tmp1 = contract('fe, fF-> Fe', ERI[i,k,v,v], QL[mk])
                            tmp1 = contract('Fe, eE-> FE', tmp1, QL[nn])
                            tmp_mn = tmp_mn - contract('fb, fe-> be', tmp, tmp1)
                        
                        Hovoo_mn = Hovoo_mn + contract('e, be-> b', t1[n], tmp_mn)

                        #tmp1_mn = contract('be,bB,eE->BE', ERI[v,i,n,v], QL[mn], QL[mm])
                        tmp_mn = contract('be, bB-> Be', ERI[v,i,n,v], QL[mn])
                        tmp_mn = contract('Be, eE-> BE', tmp_mn, QL[mm])
                        
                        for k in range(no):
                            kn = k*no + n
                            nk = n*no + k
                            mnk = mn*no + k

                            #Smnnk <- Sijjm
                            tmp = t2[nk] @ Sijmj[mnk].T
                            #tmp1 = contract('ef,eE,fF->EF', ERI[i,k,v,v], QL[mm], QL[nk])
                            tmp1 = contract('ef, eE-> Ef', ERI[i,k,v,v], QL[mm])
                            tmp1 = contract('Ef, fF-> EF', tmp1, QL[nk])
                            tmp_mn = tmp_mn - contract('fb, ef-> be', tmp, tmp1)  

                            #Smnnk <- Sijmj
                            tmp = t2[kn] @ Sijmj[mnk].T
                            tmp1 = contract('ef, eE-> Ef', L[i,k,v,v], QL[mm])
                            tmp1 = contract('Ef, fF-> EF', tmp1, QL[kn])
                            tmp_mn = tmp_mn + contract('fb, ef-> be', tmp, tmp1) 

                        Hovoo_mn = Hovoo_mn + contract('e, be-> b', t1[m], tmp_mn)    

                        for k in range(no):
                            kk = k*no + k 
                            mk = m*no + k
                            nk = n*no + k
                            kn = k*no + n
                            mnk = mn*no + k

                            #Smnkk <- Sijmm
                            tmp = Sijmm[mnk] @ t1[k] 
                            Hovoo_mn = Hovoo_mn - (tmp * Hoooo[i,k,m,n])
                            
                            #Smnmk <- Sijim 
                            tmp = t2[mk] @ Sijim[mnk].T 
                            Hovoo_mn = Hovoo_mn - contract('eb, e-> b', tmp, ERIooov[mk][k,i,n]) 

                            #Smnnk <- Sijjm
                            tmp = t2[nk] @ Sijmj[mnk].T 
                            Hovoo_mn = Hovoo_mn - contract('eb, e-> b', tmp, ERIooov[nk][i,k,m])            
                            
                            #Smnkn <- Sijmj
                            tmp = t2[kn] @ Sijmj[mnk].T 
                            Hovoo_mn = Hovoo_mn + contract('eb, e-> b', tmp, Looov[kn][i,k,m]) 
            
                        lHovoo_mn.append( Hovoo_mn)
            
            #Hovoo_ij - not needed for lambda but may be needed for other eqns            
            #for i in range(no):
                #ii = i*no + i 
                #for j in range(no):
                    #ij = i*no + j 
                    #jj = j*no + j
       
                    #for m in range(no):
                        
                        #tmp_ij = contract('aA,bB,ab->AB', QL[ij], QL[jj], ERI[m,v,i,v])
                        #tmp_ij = contract('aA, ab-> Ab', QL[ij], ERI[m,v,i,v])
                        #tmp_ij = contract('bB, Ab-> AB', QL[jj], tmp_ij)

                        #tmp1_ij = contract('aA,bB,ab->AB', QL[ij], QL[ii], ERI[v,m,j,v])
                        #tmp1_ij = contract('aA, ab-> Ab', QL[ij], ERI[v,m,j,v])
                        #tmp1_ij = contract('bB, Ab->AB', QL[ii], tmp1_ij)

                        #for n in range(no):
                            #_in = i*no + n
                            #jn = j*no + n
                            #nj = n*no + j

                            #Sijin = QL[ij].T @ QL[_in]
                            #tmp = t2[_in] @ Sijin.T

                            #tmp1 = contract('ab, aA-> Ab', ERI[m,n,v,v], QL[_in])
                            #tmp1 = contract('Ab, bB-> AB', tmp1, QL[jj])

                            #tmp_ij = tmp_ij - contract('fa, fe-> ae', tmp, tmp1)

                            #Sijjn = QL[ij].T @ QL[jn]
                            #tmp = t2[jn] @ Sijjn.T
                            #tmp1 = contract('ab, aA-> Ab', ERI[m,n,v,v], QL[ii])
                            #tmp1 = contract('Ab, bB-> AB', tmp1, QL[jn])
                            #tmp1_ij = tmp1_ij - contract('fb, ef-> be', tmp, tmp1)

                            #Sijnj = QL[ij].T @ QL[nj]
                            #tmp = t2[nj] @ Sijnj.T
                            #tmp1 = contract('ab, aA-> Ab', L[m,n,v,v], QL[ii])
                            #tmp1 = contract('Ab, bB-> AB', tmp1, QL[nj])
                            #tmp1_ij = tmp1_ij + contract('fb, ef-> be', tmp, tmp1)

                        #ltmp_ij.append(tmp_ij)
                        #ltmp1_ij.append(tmp1_ij)       

            #for ij in range(no*no):
                #i = ij // no
                #j = ij % no
                #ii = i*no + i 
                #jj = j*no + j

                #Hovoo = ERIovoo[ij].copy()
 
                #for m in range(no):
                    #ijm = ij*no + m
 
                    #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] + Hov[ij][m] @ t2[ij]

                    #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] + contract('ef, bef-> b', t2[ij], ERIovvv[ij][m])
 
                    #tmp = contract('abc, aA-> Abc', ERI[m,v,v,v], QL[ij])
                    #tmp = contract('Abc, bB-> ABc', tmp, QL[ii])
                    #tmp = contract('ABc, cC-> ABC', tmp, QL[jj])
                    #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] + contract('e, f, bef-> b', t1[i], t1[j], tmp)
  
                    #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] + contract('e, be-> b', t1[j], ltmp_ij[ijm]) 
  
                    #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] + contract('e, be-> b', t1[i], ltmp1_ij[ijm])
                    
                    #for n in range(no):
                        #_in = i*no + n
                        #jn = j*no + n
                        #nj = n*no + j
                        #nn = n*no + n

                        #Sijin = QL[ij].T @ QL[_in]
                        #tmp = t2[_in] @ Sijin.T
                        #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] - tmp.T @ ERIooov[_in][n,m,j]

                        #Sijjn = QL[ij].T @ QL[jn]
                        #tmp = t2[jn] @ Sijjn.T
                        #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] - tmp.T @ ERIooov[jn][m,n,i]

                        #Sijnj = QL[ij].T @ QL[nj]
                        #tmp = t2[nj] @ Sijnj.T
                        #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] + tmp.T @ Looov[nj][m,n,i]
  
                        #Sijnn = QL[ij].T @ QL[nn]
                        #tmp = t1[n] @ Sijnn.T
                        #Hovoo[m,:,i,j] = Hovoo[m,:,i,j] - t1[n] @ Sijnn.T * Hoooo[m,n,i,j]

                #lHovoo.append(Hovoo)      
        return lHovoo_mn

    def build_Hovoo(self, o, v, ERI, L, Hov, Hoooo, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(ERI, torch.Tensor):
                Hovoo = ERI[o,v,o,o].clone().to(self.ccwfn.device1)
            else:
                Hovoo = ERI[o,v,o,o].copy()
            Hovoo = Hovoo + contract('me,ijeb->mbij', Hov, t2)
            Hovoo = Hovoo + contract('ijef,mbef->mbij', t2, ERI[o,v,v,v])
            Hovoo = Hovoo - contract('ineb,nmje->mbij', t2, ERI[o,o,o,v])
            Hovoo = Hovoo - contract('jneb,mnie->mbij', t2, ERI[o,o,o,v])
            Hovoo = Hovoo + contract('njeb,mnie->mbij', t2, L[o,o,o,v])

        elif self.ccwfn.model == 'CC2':
            if isinstance(ERI, torch.Tensor):
                Hovoo = ERI[o,v,o,o].clone().to(self.ccwfn.device1)
            else:
                Hovoo = ERI[o,v,o,o].copy()
            Hovoo = Hovoo + contract('me,ijeb->mbij', self.ccwfn.H.F[o,v], t2)
            Hovoo = Hovoo - contract('nb,mnij->mbij', t1, Hoooo)
            Hovoo = Hovoo + contract('jf,mbif->mbij', t1, contract('ie,mbef->mbif', t1, ERI[o,v,v,v]))
            Hovoo = Hovoo + contract('je,mbie->mbij', t1, ERI[o,v,o,v])
            Hovoo = Hovoo + contract('ie,bmje->mbij', t1, ERI[v,o,o,v])     
  
        else:
            if isinstance(ERI, torch.Tensor):
                Hovoo = ERI[o,v,o,o].clone().to(self.ccwfn.device1)
            else:
                Hovoo = ERI[o,v,o,o].copy()
            Hovoo = Hovoo + contract('me,ijeb->mbij', Hov, t2)
            Hovoo = Hovoo - contract('nb,mnij->mbij', t1, Hoooo)
            Hovoo = Hovoo + contract('ijef,mbef->mbij', self.ccwfn.build_tau(t1, t2), ERI[o,v,v,v])
            Hovoo = Hovoo - contract('ineb,nmje->mbij', t2, ERI[o,o,o,v])
            Hovoo = Hovoo - contract('jneb,mnie->mbij', t2, ERI[o,o,o,v])
            Hovoo = Hovoo + contract('njeb,mnie->mbij', t2, L[o,o,o,v])
            if isinstance(ERI, torch.Tensor):
                tmp = ERI[o,v,o,v].clone().to(self.ccwfn.device1)
            else:
                tmp = ERI[o,v,o,v].copy()
            tmp = tmp - contract('infb,mnfe->mbie', t2, ERI[o,o,v,v])
            Hovoo = Hovoo + contract('je,mbie->mbij', t1, tmp)
            if isinstance(ERI, torch.Tensor):
                tmp = ERI[v,o,o,v].clone().to(self.ccwfn.device1)
            else:
                tmp = ERI[v,o,o,v].copy()
            tmp = tmp - contract('jnfb,mnef->bmje', t2, ERI[o,o,v,v])
            tmp = tmp + contract('njfb,mnef->bmje', t2, L[o,o,v,v])
            Hovoo = Hovoo + contract('ie,bmje->mbij', t1, tmp)
            if isinstance(tmp, torch.Tensor):
                del tmp
        return Hovoo
