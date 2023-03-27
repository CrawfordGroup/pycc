"""
cchbar.py: Builds the similarity-transformed Hamiltonian (one- and two-body terms only).
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import time
import numpy as np
import torch
from debug import Debug

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

        if ccwfn.local is not None:
            #print("Here first")
            self.Local = ccwfn.Local
            self.no = ccwfn.no
            self.nv = ccwfn.nv
            self.Debug = Debug(self.no,self.nv)
        
        if ccwfn.filter is True:    
            #print("I am here")     
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
                    
        elif ccwfn.filter is not True:    
            #print("Here'second")
            self.lccwfn = ccwfn.lccwfn

            self.Hov = self.build_lHov(o, v, ccwfn.no, self.Local.Fov, L, self.Local.QL, self.lccwfn.t1)
            self.Hvv = self.build_lHvv(o, v, ccwfn.no, L, self.Local.Fvv, self.Local.Fov, self.Local.Loovv, self.Local.QL, 
            self.lccwfn.t1,self.lccwfn.t2) 
            self.Hoo = self.build_lHoo(o ,v, ccwfn.no, F, L, self.Local.Fov, self.Local.Looov, self.Local.Loovv, 
            self.Local.QL, self.lccwfn.t1,self.lccwfn.t2) 
            self.Hoooo = self.build_lHoooo(o, v, ccwfn.no, ERI, self.Local.ERIoovv, self.Local.ERIooov, 
            self.Local.QL, self.lccwfn.t1, self.lccwfn.t2) 
            self.Hvvvv = self.build_lHvvvv( o, v, ccwfn.no, self.Local.ERIoovv, self.Local.ERIvovv, self.Local.ERIvvvv, 
            self.Local.QL, self.lccwfn.t1, self.lccwfn.t2) 
            self.Hvovv = self.build_lHvovv(o,v,ccwfn.no, self.Local.ERIvovv, self.Local.ERIoovv, self.Local.QL, self.lccwfn.t1)
            self.Hooov = self.build_lHooov(o, v, ccwfn.no, ERI, self.Local.ERIooov, self.Local.QL, self.lccwfn.t1) 
            self.Hovvo = self.build_lHovvo(o, v, ccwfn.no, ERI, L, self.Local.ERIovvo, self.Local.QL, 
            self.lccwfn.t1, self.lccwfn.t2)
            self.Hovov = self.build_lHovov(o, v, ccwfn.no, ERI, self.Local.ERIovov, self.Local.ERIooov, self.Local.QL,
            self.lccwfn.t1,self.lccwfn.t2) 
            self.Hvvvo = self.build_lHvvvo(o, v, ccwfn.no, ERI, L, self.Local.ERIvvvo, self.Local.ERIoovo, self.Local.ERIvoov, self.Local.ERIvovo, 
            self.Local.ERIoovv, self.Local.Loovv, self.Local.QL,self.lccwfn.t1, self.lccwfn.t2, self.Hov, self.Hvvvv)
            self.Hovoo = self.build_lHovoo(o, v, ccwfn.no, ERI, L, self.Local.ERIovoo, self.Local.ERIovvv, self.Local.ERIooov,
            self.Local.ERIovov, self.Local.ERIvoov, self.Local.Looov, self.Local.QL, self.lccwfn.t1, self.lccwfn.t2, self.Hov, self.Hoooo)

        #if isinstance(t1, torch.Tensor):
            #print("Hov norm = %20.15f" % torch.linalg.norm(self.Hov))
            #print("Hvv norm = %20.15f" % torch.linalg.norm(self.Hvv))
            #print("Hoo norm = %20.15f" % torch.linalg.norm(self.Hoo))
            #print("Hoooo norm = %20.15f" % torch.linalg.norm(self.Hoooo))
            #print("Hvvvv norm = %20.15f" % torch.linalg.norm(self.Hvvvv))
        #else:
            #print("Hov norm = %20.15f" % np.linalg.norm(self.Hov))
            #print("Hvv norm = %20.15f" % np.linalg.norm(self.Hvv))
            #print("Hoo norm = %20.15f" % np.linalg.norm(self.Hoo))
            #print("Hoooo norm = %20.15f" % np.linalg.norm(self.Hoooo))
            #print("Hvvvv norm = %20.15f" % np.linalg.norm(self.Hvvvv))

        print("\nHBAR constructed in %.3f seconds.\n" % (time.time() - time_init))

    """
    For GPU implementation:
    2-index tensors are stored on GPU
    4-index tensors are stored on CPU
    """
    def build_lHov(self, o, v, no, Fov, L, QL, t1):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            lHov = Fov.copy()
        else:
            lHov = []
            for ij in range(no*no):

                Hov = Fov[ij].copy()

                for n in range(no):
                    nn = n*no + n      

                    tmp = contract('eE,fF,mef->mEF',QL[ij],QL[nn],L[o,n,v,v])
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

    def build_lHvv(self,o, v, no, L, Fvv, Fov, Loovv, QL, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            lHvv = []
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                Hvv = Fvv[ij].copy()

                Hvv_1 = np.zeros_like(Hvv)
                for mn in range(no*no):
                    m = mn // no
                    n = mn % no

                    tmp = QL[mn].T @ L[m,n,v,v]
                    tmp1 = tmp @ QL[ij]
    
                    Sijmn = QL[ij].T @ QL[mn]
                    tmp2 = t2[mn] @ Sijmn.T 
                    Hvv_1 = tmp2.T @ tmp1                             

                lHvv.append(Hvv+ Hvv_1)
        else:
            lHvv = []
            for ij in range(no*no):
                 
                Hvv = Fvv[ij].copy()

                Hvv_1 = np.zeros_like(Hvv) 
                Hvv_2 = np.zeros_like(Hvv)
                Hvv_3 = np.zeros_like(Hvv)
                Hvv_4 = np.zeros_like(Hvv) 
                for m in range(no):
                    mm = m*no + m

                    Sijmm = QL[ij].T @ QL[mm] 
                    tmp = t1[m] @ Sijmm.T 
                    Hvv_1 -= contract('e,a->ae', Fov[ij][m], tmp)
                    
                    tmp1 = contract('aef,aA,eE,fF->AEF',L[v,m,v,v],QL[ij],QL[ij],QL[mm]) 
                    Hvv_2 += contract('F,aeF->ae',t1[m],tmp1)
                    
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n

                        Sijmn = QL[ij].T @ QL[mn]
                        tmp2 = QL[mn].T @ L[m,n,v,v] 
                        tmp3 = tmp2 @ QL[ij]
                        tmp4 = t2[mn] @ Sijmn.T
                        Hvv_3 -= tmp4.T @ tmp3
                        
                        Sijnn = QL[ij].T @ QL[nn]
                        tmp5 = QL[mm].T @ L[m,n,v,v] 
                        tmp6 = tmp5 @ QL[ij]
                         
                        tmp7 = t1[n] @ Sijnn.T 
                        Hvv_4 -= contract('F,A,Fe->Ae',t1[m], tmp7, tmp6)
                lHvv.append(Hvv + Hvv_1 + Hvv_2 + Hvv_3 + Hvv_4)              
        return Hvv

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
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hoo = F[o,o].copy()
            
            Hoo_1 = np.zeros_like(Hoo) 
            for _in in range(no*no):
                i = _in // no
                n = _in % no
                Hoo_1[:,i] += contract('ef,mef->m',t2[_in],Loovv[_in][:,n])
            lHoo = Hoo + Hoo_1
        else:
            Hoo = F[o,o].copy()

            Hoo_1 = np.zeros_like(Hoo)
            Hoo_2 = np.zeros_like(Hoo)
            Hoo_3 = np.zeros_like(Hoo)
            Hoo_4 = np.zeros_like(Hoo)
            for i in range(no):
                ii = i*no + i 

                Hoo_1[:,i] += t1[i] @ Fov[ii].T      
                
                for n in range(no):
                    nn = n*no + n 
                    _in = i*no + n

                    Hoo_2[:,i] += contract('e,me-> m', t1[n], Looov[nn][:,n,i])
                    
                    Hoo_3[:,i] += contract('ef,mef->m',t2[_in],Loovv[_in][:,n])
                    
                    tmp = contract('eE,fF,mef->mEF',QL[ii], QL[nn], L[o,n,v,v]) 
                    Hoo_4[:,i] += contract('e,f,mef->m',t1[i], t1[n], tmp)

            lHoo = Hoo + Hoo_1 + Hoo_2 + Hoo_3 + Hoo_4
        return lHoo
  
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
        contract = self.contract 
        if self.ccwfn.model == 'CCD':
            Hoooo = ERI[o,o,o,o].copy()
 
            Hoooo_1 = np.zeros_like(Hoooo) 
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                                           
                Hoooo_1[:,:,i,j] += contract('ef,mnef->mn',t2[ij], ERIoovv[ij])
            lHoooo = Hoooo + Hoooo_1 
        else:
            Hoooo = ERI[o,o,o,o].copy()
 
            Hoooo_1 = np.zeros_like(Hoooo)
            tmp = np.zeros_like(Hoooo) 
            Hoooo_2 = np.zeros_like(Hoooo)
            Hoooo_3 = np.zeros_like(Hoooo)
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i
                jj = j*no + j

                tmp[:,:,i,j] = contract('e,mne->mn',t1[j], ERIooov[jj][:,:,i])
                Hoooo_1 = tmp + tmp.swapaxes(0,1).swapaxes(2,3)
                                  
                Hoooo_2[:,:,i,j] += contract('ef,mnef->mn',t2[ij], ERIoovv[ij])
                
                tmp1 = contract('eE,fF,mnef->mnEF',QL[ii], QL[jj], ERI[o,o,v,v])
                Hoooo_3[:,:,i,j] += contract('e,f,mnef->mn',t1[i],t1[j],tmp1)
            lHoooo = Hoooo + Hoooo_1 + Hoooo_2 + Hoooo_3
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

    def build_lHvvvv(self, o, v, no, ERIoovv, ERIvovv, ERIvvvv, QL, t1, t2):
        contract = self.contract
        if self.ccwfn.model == 'CCD':  
            lHvvvv = []
            for ij in range(no*no):

                Hvvvv = ERIvvvv[ij].copy()
                
                Hvvvv_1 = np.zeros_like(Hvvvv)
                for mn in range(no*no):
                    m = mn // no 
                    n = mn % no 

                    Sijmn = QL[ij].T @ QL[mn]
                    tmp = Sijmn @ t2[mn]
                    tmp1 = tmp @ Sijmn.T 
                    Hvvvv_1 += contract('ab,ef->abef',tmp1, ERIoovv[ij][m,n]) 
                Hvvvv.append(Hvvvv + Hvvvv_1)
                #if ij == 1:  
                    #print("hvvvv_ij", Hvvvv_ij[ij])              
                #if self.Local.dim[ij] > 0:
                    #Hvvv_ij.append(Hvvvv_ij[ij][:,:,:,-1]) 
                    #Hv_ij.append(Hvvvv_ij[ij][-1,-1,-1,:])
                #else:
                    #Hvvv_ij.append(np.empty((self.Local.dim[ij],self.Local.dim[ij],self.Local.dim[ij])))
                    #Hv_ij.append(np.empty((self.Local.dim[ij])))
            
            #ij = 1
            #print("compare")
            #print("tot",Hvvvv_ij[ij])
            #print("concat", contract('abe,f->abef',Hvvv_ij[ij], Hv_ij[ij])) #  + Hvvvv_1[:,:,:,-1], Hvvvv[-1,-1,-1,:] + Hvvvv_1[-1,-1,-1,:]))
            #print("Hvvvv_ij[ij][:,:,:,-1]", Hvvvv_ij[ij][:,:,:,-1]) 
            #print("Hvvv_ij[0]", Hvvv_ij[0])
        else: 
            lHvvvv = []
            for ij in range(no*no):
                Hvvvv = ERIvvvv[ij].copy()
                
                tmp1 = np.zeros_like(Hvvvv)
                Hvvvv_1 = np.zeros_like(Hvvvv)
                Hvvvv_2 = np.zeros_like(Hvvvv)
                Hvvvv_3 = np.zeros_like(Hvvvv) 
                for m in range(no):
                    mm = m*no + m

                    Sijmm = QL[ij].T @ QL[mm]
                    tmp = t1[m] @ Sijmm.T 
                    tmp1 = contract('b,aef->abef',tmp, ERIvovv[ij][:,m,:,:]) 
                    Hvvvv_1 -= tmp1  + tmp1.swapaxes(0,1).swapaxes(2,3)
 
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n                       
 
                        Sijmn = QL[ij].T @ QL[mn]
                        tmp2 = Sijmn @ t2[mn]
                        tmp3 = tmp2 @ Sijmn.T 
                        Hvvvv_2 += contract('ab,ef->abef',tmp3, ERIoovv[ij][m,n])
         
                        Sijnn = QL[ij].T @ QL[nn]                    
                        tmp4 = t1[m] @ Sijmm.T 
                        tmp5 = t1[n] @ Sijnn.T 
                        Hvvvv_3 += contract('a,b,ef->abef',tmp4, tmp5, ERIoovv[ij][m,n])
                lHvvvv.append(Hvvvv + Hvvvv_1 + Hvvvv_2 + Hvvvv_3) 
        return lHvvvv            

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
            QL = self.Local.Q[1] @ self.Local.L[1]
            tmp3 = contract('abcd,aA->Abcd',Hvvvv, QL)
            tmp4 = contract('Abcd,bB->ABcd',tmp3, QL)
            tmp5 = contract('ABcd,cC->ABCd',tmp4, QL)
            #print("cool", contract('ABCd,dD->ABCD',tmp5, QL))
        return Hvvvv

    def build_lHvovv(self,o,v,no, ERIvovv, ERIoovv, QL, t1): 
        contract = self.contract 
        if self.ccwfn.model == 'CCD':
            lHvovv = ERIvovv.copy()
        else:
            lHvovv = []
            for ij in range(no*no):
                
                Hvovv = ERIvovv[ij].copy()
                Hvovv_1 = np.zeros_like(Hvovv)  
                for n in range(no):
                    nn = n*no + n
                    
                    Sijnn = QL[ij].T @ QL[nn] 
                    tmp = t1[n] @ Sijnn.T 
                    Hvovv_1 -= contract('a,mef->amef',tmp, ERIoovv[ij][n,:])
                lHvovv.append( Hvovv + Hvovv_1)       
        return lHvovv
        
    def build_Hvovv(self, o, v, ERI, t1):
        Q = self.Local.Q
        Lij = self.Local.L 
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
        else:
            lHooov = []
            for ij in range(no*no):
                i = ij // no
                ii = i*no + i 
       
                Hooov = ERIooov[ij].copy()
                Hooov_1 = np.zeros_like(Hooov)
                tmp = contract('eE,fF,nmef->nmEF',QL[ij], QL[ii], ERI[o,o,v,v])
                Hooov_1[:,:,i,:] = contract('f,nmef->mne',t1[i], tmp)         

                lHooov.append(Hooov + Hooov_1) 
        return lHooov   
        
    def build_Hooov(self, o, v, ERI, t1):
        Q = self.Local.Q
        Lij = self.Local.L
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
        if self.ccwfn.model == 'CCD':
            lHovvo = [] 
            for ij in range(no*no):
                j = ij % no
                Hovvo = ERIovvo[ij].copy()
                
                Hovvo_1 = np.zeros_like(Hovvo)
                Hovvo_2 = np.zeros_like(Hovvo)
                for mn in range(no*no):
                    m = mn // no
                    n = mn % no 
                    jn = j*no + n
                    nj = n*no + j
 
                    Sijjn = QL[ij].T @ QL[jn]
                    tmp = t2[jn] @ Sijjn.T 
                    tmp1 = QL[ij].T @ ERI[m,n,v,v]
                    tmp2 = tmp1 @ QL[jn]
                    Hovvo_1[m,:,:,j] -= tmp.T @ tmp2.T
                    
                    Sijnj = QL[ij].T @ QL[nj] 
                    tmp3 = t2[nj] @ Sijnj.T 
                    tmp4 = QL[ij].T @ L[m,n,v,v] 
                    tmp5 = tmp4 @ QL[nj] 
                    Hovvo_2[m,:,:,j] += tmp3.T @ tmp5.T
                lHovvo.append(Hovvo + Hovvo_1 + Hovvo_2)
        else:
            lHovvo = []
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                jj = j*no + j
 
                Hovvo = ERIovvo[ij].copy()
                    
                Hovvo_1 = np.zeros_like(Hovvo)
                Hovvo_2 = np.zeros_like(Hovvo)
                Hovvo_3 = np.zeros_like(Hovvo)
                Hovvo_4 = np.zeros_like(Hovvo) 
                Hovvo_5 = np.zeros_like(Hovvo)
                for m in range(no):

                    tmp1 = contract('abc,aA->Abc',ERI[m,v,v,v], QL[ij])
                    tmp2 = contract('Abc,bB->ABc',tmp1, QL[ij])
                    tmp3 = contract('ABc,cC->ABC',tmp2, QL[jj])                      
                    Hovvo_1[m,:,:,j] += contract('f,bef->be',t1[j], tmp3) 
                    
                    for n in range(no):
                        nn = n*no + n
                        jn = j*no + n
                        nj = n*no + j
 
                        Sijnn = QL[ij].T @ QL[nn]
                        tmp4 = t1[n] @ Sijnn.T
                        tmp5 = QL[ij].T @ ERI[m,n,v,j]     
                        Hovvo_2[m,:,:,j] -= contract('b,e->be', tmp4, tmp5)

                        Sijjn = QL[ij].T @ QL[jn]
                        tmp6 = t2[jn] @ Sijjn.T
                        tmp7 = QL[ij].T @ ERI[m,n,v,v]
                        tmp8 = tmp7 @ QL[jn] 
                        Hovvo_3[m,:,:,j] -= tmp6.T @ tmp8.T 
                        
                        tmp12 = tmp7 @ QL[jj]
                        Hovvo_4[m,:,:,j] -= contract('f,b,ef->be',t1[j], tmp4, tmp12)
 
                        Sijnj = QL[ij].T @ QL[nj]
                        tmp9 = t2[nj] @ Sijnj.T
                        tmp10 = QL[ij].T @ L[m,n,v,v]
                        tmp11 = tmp10 @ QL[nj] 
                        Hovvo_5[m,:,:,j] += tmp9.T @ tmp11.T   
                lHovvo.append(Hovvo + Hovvo_1 + Hovvo_2 + Hovvo_3 + Hovvo_4 + Hovvo_5)            
        return lHovvo    


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
                Hovvo = Hovvo - contract('jnfb,mnef->mbej', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
                Hovvo = Hovvo + contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        return Hovvo

    def build_lHovov(self, o, v, no, ERI, ERIovov, ERIooov, QL, t1, t2): 
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            lHovov = [] 
            for ij in range(no*no):
                j = ij % no
 
                Hovov = ERIovov[ij].copy()
                
                Hovov_1 = np.zeros_like(Hovov) 
                for nm in range(no*no):
                    n = nm // no
                    m = nm % no
                    jn = j*no + n
                    
                    Sijjn = QL[ij].T @ QL[jn]
                    tmp = t2[jn] @ Sijjn.T 
                    tmp1 = QL[ij].T @ ERI[n,m,v,v] 
                    tmp2 = tmp1 @ QL[jn]
                    Hovov_1[m,:,j,:] -=  tmp.T @ tmp2.T 
                lHovov.append(Hovov + Hovov_1) 
        else:  
            lHovov = []
            for ij in range(no*no):
                j = ij % no 
                jj = j*no + j
   
                Hovov = ERIovov[ij].copy()
               
                Hovov_1 = np.zeros_like(Hovov) 
                Hovov_2 = np.zeros_like(Hovov)
                Hovov_3 = np.zeros_like(Hovov)
                Hovov_4 = np.zeros_like(Hovov)
                for m in range(no):
        
                    tmp = contract('bB,bef-> Bef', QL[ij], ERI[v,m,v,v])
                    tmp1 = contract('eE,Bef->BEf', QL[ij], tmp)
                    tmp2 = contract('fF,BEf->BEF',QL[jj], tmp1)
                    Hovov_1[m,:,j,:] += contract('f,bef->be',t1[j], tmp2)

                    for n in range(no):
                        nn = n*no + n
                        jn = j*no + n
   
                        Sijnn = QL[ij].T @ QL[nn]  
                        tmp3 = t1[n] @ Sijnn.T
                        Hovov_2[m,:,j,:] -=  contract('b,e->be', tmp3, ERIooov[ij][m,n,j,:]) 

                        Sijjn = QL[ij].T @ QL[jn]
                        tmp4 = t2[jn] @ Sijjn.T
                        tmp5 = QL[ij].T @ ERI[n,m,v,v]
                        tmp6 = tmp5 @ QL[jn]
                        Hovov_3[m,:,j,:] -= tmp4.T @ tmp6.T

                        tmp7 = tmp5 @ QL[jj]
                        Hovov_4[m,:,j,:] -= contract('f,b,ef->be',t1[j], tmp3, tmp7)
                lHovov.append(Hovov + Hovov_1 + Hovov_2 + Hovov_3 + Hovov_4 )               
        return lHovov 

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

    def gen_lHvvvv(self,o,v,no, ERI, QL, t1_ii, t2_ij, ii, ij):
        contract = self.contract

        tmp = contract('abcd,aA->Abcd',ERI[v,v,v,v], QL[ij])
        tmp1 = contract('Abcd,bB->ABcd',tmp, QL[ij])
        tmp2 = contract('ABcd,cC->ABCd',tmp1, QL[ij])
        Hvvvv = contract('ABCd,dD->ABCD',tmp2, QL[ii])

        tmp3 = np.zeros_like(Hvvvv)
        Hvvvv_1 = np.zeros_like(Hvvvv)
        Hvvvv_2 = np.zeros_like(Hvvvv)
        Hvvvv_3 = np.zeros_like(Hvvvv)
        for m in range(no):
            mm = m*no + m

            Sijmm = QL[ij].T @ QL[mm]
            tmp = t1_ii[m] @ Sijmm.T

            tmp4 = contract('aibc,aA->Aibc',ERI[v,o,v,v], QL[ij])
            tmp5 = contract('Aibc,bB->AiBc',tmp4, QL[ij])
            tmp6 = contract('AiBc,cC->AiBC',tmp5, QL[ii])
            tmp3 = contract('b,aef->abef',tmp, tmp6[:,m,:,:])
            Hvvvv_1 -= tmp3 # + tmp3.swapaxes(0,1).swapaxes(2,3)

            for n in range(no):
                mn = m*no + n
                nn = n*no + n

                Sijmn = QL[ij].T @ QL[mn]
                tmp7 = Sijmn @ t2_ij[mn]
                tmp8 = tmp7 @ Sijmn.T

                tmp9 = contract('ijab,aA->ijAb',ERI[o,o,v,v], QL[ij])
                tmp10 = contract('ijAb,bB->ijAB',tmp9,QL[ii])
                Hvvvv_2 += contract('ab,ef->abef',tmp8, tmp10[m,n])

                Sijnn = QL[ij].T @ QL[nn]
                tmp4 = t1_ii[m] @ Sijmm.T
                tmp5 = t1_ii[n] @ Sijnn.T
                Hvvvv_3 += contract('a,b,ef->abef',tmp4, tmp5, tmp10[m,n])
        lHvvvv = Hvvvv + Hvvvv_1 + Hvvvv_2 + Hvvvv_3
        print(lHvvvv[6][:,:,:,:])
        return lHvvvv

    def build_lHvvvo(self, o, v, no, ERI, L, ERIvvvo, ERIoovo, ERIvoov, ERIvovo, ERIoovv, Loovv, QL, t1, t2, Hov, Hvvvv):  
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            lHvvvo = []
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                Hvvvo = ERIvvvo[ij].copy()
          
                Hvvvo_1 = np.zeros_like(Hvvvo)  
                Hvvvo_2 = np.zeros_like(Hvvvo)
                Hvvvo_3 = np.zeros_like(Hvvvo)
                Hvvvo_4 = np.zeros_like(Hvvvo)
                Hvvvo_5 = np.zeros_like(Hvvvo)
                Hvvvo_6 = np.zeros_like(Hvvvo)
                for m in range(no):
                    mi = m*no + i 
                    im = i*no + m
                    mm = m*no + m
  
                    Sijmi = QL[ij].T @ QL[mi] 
                    tmp = t2[mi] @ Sijmi.T  
                    tmp1 = Sijmi @ tmp 
                    Hvvvo_1[:,:,:,i] -= contract('e,ab->abe',Hov[ij][m],tmp1)
                
                    tmp2 = contract('aef,aA->Aef', L[v,m,v,v], QL[ij])
                    tmp3 = contract('Aef,eE->AEf',tmp2, QL[ij])
                    tmp4 = contract('AEf,fF->AEF',tmp3, QL[mi])
                    Hvvvo_6[:,:,:,i] += contract('fb,aef->abe',tmp, tmp4)

                    Sijim = QL[ij].T @ QL[im]
                    tmp5 = t2[im] @ Sijim.T 
                    tmp6 = contract('bfe,bB->Bfe',ERI[v,m,v,v], QL[ij])
                    tmp7 = contract('Bfe,fF->BFe',tmp6, QL[im])
                    tmp8 = contract('BFe,eE->BFE',tmp7, QL[ij])
                    Hvvvo_4[:,:,:,i] -= contract('fa,bfe->abe', tmp5, tmp8)

                    tmp9 = contract('Aef,eE->AEf',tmp6, QL[ij])
                    tmp10 = contract('AEf,fF->AEF',tmp9, QL[mi]) 
                    Hvvvo_5[:,:,:,i] -= contract('fb,aef->abe', tmp5, tmp10)
                
                    for n in range(no): 
                        mn = m*no + n     
                        nn = n*no + n
                    
                        Sijmn = QL[ij].T @ QL[mn] 
                        tmp11 = Sijmn @ t2[mn]
                        tmp12 = tmp11 @ Sijmn.T 
                        Hvvvo_2[:,:,:,i] += contract('ab,e->abe',tmp12, ERIoovo[ij][m,n,:,i]) 
              
                        Sijmm = QL[ij].T @ QL[mm]
                        Sijnn = QL[ij].T @ QL[nn] 
                        tmp13 = t1[m] @ Sijmm.T 
                        tmp14 = t1[n] @ Sijnn.T 
                        Hvvvo_3[:,:,:,i] += contract('a,b,e->abe',tmp13,tmp14, ERIoovo[ij][m,n,:,i])
                lHvvvo.append( Hvvvo + Hvvvo_1 + Hvvvo_2 + Hvvvo_3 + Hvvvo_4 + Hvvvo_5 + Hvvvo_6)
        else:
            lHvvvo = []
            tmp_ij = []
            tmp1_ij = []
            for i in range(no):
                for j in range(no):
                     ij = i*no + j
                     tmp1 = np.zeros((self.Local.dim[ij],no, self.Local.dim[ij], no))
                     tmp2 = np.zeros((self.Local.dim[ij],no, no,self.Local.dim[ij]))
                     for m in range(no):
                         tmp1[:,m,:,i] += ERIvovo[ij][:,m,:,i].copy()
                         tmp2[:,m,i,:] += ERIvoov[ij][:,m,i,:].copy()
                         
                         for n in range(no):
                             _in = i*no + n
                             ni = n*no + i
 
                             Sijin = QL[ij].T @ QL[_in]
                             tmp3 = t2[_in] @ Sijin.T 

                             rtmp = contract('ab,aA->Ab', ERI[m,n,v,v], QL[_in])
                             rtmp1 = contract('Ab,bB->AB',rtmp,QL[ij])

                             tmp1[:,m,:,i] -= contract('fa,fe->ae',tmp3, rtmp1)

                             rtmp2 = contract('ab,aA->Ab', ERI[m,n,v,v], QL[ij])
                             rtmp3 = contract('Ab,bB->AB',rtmp2,QL[_in])
                             tmp2[:,m,i,:] -= contract('fb,ef->be',tmp3, rtmp3)

                             Sijni = QL[ij].T @ QL[ni]
                             tmp4 = t2[ni] @ Sijni.T
                             rtmp4 = contract('ab,aA->Ab', L[m,n,v,v], QL[ij])
                             rtmp5 = contract('Ab,bB->AB',rtmp4,QL[ni])
                             tmp2[:,m,i,:] += contract('fb,ef->be', tmp4, rtmp5)

                     tmp_ij.append(tmp1)
                     tmp1_ij.append(tmp2)
 
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i

                Hvvvo = ERIvvvo[ij].copy()

                #got this to work :)
                Sijii = QL[ij].T @ QL[ii]
                Hvvvo_7 = np.zeros_like(Hvvvo)
                Hvvvo_7[:,:,:,i] += contract('f,abef->abe', t1[i] @ Sijii.T, Hvvvv[ij]) #self.gen_lHvvvv(o,v,no, ERI, QL, t1, t2, ii, ij) 

                Hvvvo_1 = np.zeros_like(Hvvvo)
                Hvvvo_2 = np.zeros_like(Hvvvo)
                Hvvvo_3 = np.zeros_like(Hvvvo)
                Hvvvo_4 = np.zeros_like(Hvvvo)
                Hvvvo_5 = np.zeros_like(Hvvvo)
                Hvvvo_6 = np.zeros_like(Hvvvo)
                Hvvvo_8 = np.zeros_like(Hvvvo)
                Hvvvo_9 = np.zeros_like(Hvvvo)
                for m in range(no):
                    mi = m*no + i
                    im = i*no + m
                    mm = m*no + m

                    Sijmi = QL[ij].T @ QL[mi]
                    tmp = t2[mi] @ Sijmi.T
                    tmp1 = Sijmi @ tmp

                    Hvvvo_1[:,:,:,i] -= contract('e,ab->abe',Hov[ij][m],tmp1)
                  
                    tmp2 = contract('aef,aA->Aef', L[v,m,v,v], QL[ij])
                    tmp3 = contract('Aef,eE->AEf',tmp2, QL[ij])
                    tmp4 = contract('AEf,fF->AEF',tmp3, QL[mi])
                    Hvvvo_6[:,:,:,i] += contract('fb,aef->abe',tmp, tmp4)

                    Sijim = QL[ij].T @ QL[im]
                    tmp5 = t2[im] @ Sijim.T
                    tmp6 = contract('bfe,bB->Bfe',ERI[v,m,v,v], QL[ij])
                    tmp7 = contract('Bfe,fF->BFe',tmp6, QL[im])
                    tmp8 = contract('BFe,eE->BFE',tmp7, QL[ij])
                    Hvvvo_4[:,:,:,i] -= contract('fa,bfe->abe', tmp5, tmp8)

                    tmp9 = contract('Aef,eE->AEf',tmp6, QL[ij])
                    tmp10 = contract('AEf,fF->AEF',tmp9, QL[mi])
                    Hvvvo_5[:,:,:,i] -= contract('fb,aef->abe', tmp5, tmp10)

                    Sijmm = QL[ij].T @ QL[mm]
                    tmp17 = t1[m] @ Sijmm.T 
                    Hvvvo_8[:,:,:,i] -= contract('b,ae->abe',tmp17, tmp_ij[ij][:,m,:,i])                  
                    
                    Hvvvo_9[:,:,:,i] -= contract('a,be->abe', tmp17 , tmp1_ij[ij][:,m,i,:])

                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n
                        _in = i*no + n
                        ni = n*no + i              
   
                        Sijmn = QL[ij].T @ QL[mn]
                        tmp11 = Sijmn @ t2[mn]
                        tmp12 = tmp11 @ Sijmn.T
                        Hvvvo_2[:,:,:,i] += contract('ab,e->abe',tmp12, ERIoovo[ij][m,n,:,i])

                        Sijmm = QL[ij].T @ QL[mm]
                        Sijnn = QL[ij].T @ QL[nn]
                        tmp13 = t1[m] @ Sijmm.T
                        tmp14 = t1[n] @ Sijnn.T
                        Hvvvo_3[:,:,:,i] += contract('a,b,e->abe',tmp13,tmp14, ERIoovo[ij][m,n,:,i])
  
                lHvvvo.append( Hvvvo + Hvvvo_1 + Hvvvo_2 + Hvvvo_3 + Hvvvo_4 + Hvvvo_5 + Hvvvo_6 + Hvvvo_8 + Hvvvo_9 + Hvvvo_7)
            print("all lHvvvo", lHvvvo[6][:,:,:,1])          
        return lHvvvo

    def gen_Hvvvv(self, o, v, ERI, t1, t2): 
        contract = self.contract
        Hvvvv = ERI[v,v,v,v].copy()
        tmp = contract('mb,amef->abef', t1, ERI[v,o,v,v])
        Hvvvv = Hvvvv - (tmp + tmp.swapaxes(0,1).swapaxes(2,3))
        Hvvvv = Hvvvv + contract('mnab,mnef->abef', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
        QL = self.Local.Q[6] @ self.Local.L[6]
        tmp3 = contract('abcd,aA->Abcd',Hvvvv, QL)
        tmp4 = contract('Abcd,bB->ABcd',tmp3, QL)
        tmp5 = contract('ABcd,cC->ABCd',tmp4, QL)
        print("Hvvvv[6]", contract('ABCd,dD->ABCD',tmp5, QL)) 
        return Hvvvv
 
    def build_Hvvvo(self, o, v, ERI, L, Hov, Hvvvv, t1, t2):
        Q = self.Local.Q
        Lij = self.Local.L
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(ERI, torch.Tensor):
                Hvvvo = ERI[v,v,v,o].clone().to(self.ccwfn.device1)
            else:
                Hvvvo = ERI[v,v,v,o].copy()
            Hvvvo = Hvvvo - contract('me,miab->abei', Hov, t2)
            Hvvvo = Hvvvo + contract('mnab,mnei->abei',self.ccwfn.build_tau(t1, t2), ERI[o,o,v,o])
            Hvvvo = Hvvvo - contract('imfa,bmfe->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo - contract('imfb,amef->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo + contract('mifb,amef->abei', t2, L[v,o,v,v])
            #QL = Q[0] @ Lij[0]
            #print("Hvvvo_ij", 0)
            #tmp = contract('aA,bB,cC,abci->ABCi',QL,QL,QL,Hvvvo)
            #print(tmp[:,:,:,0])
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
            Hvvvo = Hvvvo + contract('if,abef->abei', t1, Hvvvv) #self.gen_Hvvvv(o,v,ERI,t1,t2))
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
            QL = self.Local.Q[6] @ self.Local.L[6]
            tmp7 = contract('abci,aA->Abci',Hvvvo, QL)
            tmp8 = contract('Abci,bB->ABci',tmp7, QL)
            print("all Hvvvo", contract('ABci,cC->ABCi',tmp8, QL)[:,:,:,1]) 

            tmp9 = contract('abci,aA->Abci',contract('if,abef->abei', t1, Hvvvv), QL)
            tmp10 = contract('Abci,bB->ABci',tmp9, QL)
            print("Hvvvo_7", contract('ABci,cC->ABCi',tmp10, QL)[:,:,:,1])           
        return Hvvvo

    def build_lHovoo(self, o, v, no, ERI, L, ERIovoo, ERIovvv, ERIooov, ERIovov, ERIvoov, Looov, QL, t1, t2, Hov, Hoooo):
        contract = self.contract
        if self.ccwfn.model =='CCD':
           lHovoo = []
           for ij in range(no*no):
               i = ij // no
               j = ij % no
 
               Hovoo = ERIovoo[ij].copy()
 
               Hovoo_1 = np.zeros_like(Hovoo)
               Hovoo_2 = np.zeros_like(Hovoo)
               Hovoo_3 = np.zeros_like(Hovoo)
               Hovoo_4 = np.zeros_like(Hovoo)
               Hovoo_5 = np.zeros_like(Hovoo)
               for m in range(no):
               
                   Hovoo_1[m,:,i,j] += Hov[ij][m] @ t2[ij]
                   
                   Hovoo_2[m,:,i,j] += contract('ef,bef->b',t2[ij], ERIovvv[ij][m]) 
 
                   for n in range(no):
                       _in = i*no + n
                       jn = j*no + n
                       nj = n*no + j
                       
                       Sijin = QL[ij].T @ QL[_in]
                       tmp = t2[_in] @ Sijin.T 
                       Hovoo_3[m,:,i,j] -= tmp.T @ ERIooov[_in][n,m,j]

                       Sijjn = QL[ij].T @ QL[jn]
                       tmp1 = t2[jn] @ Sijjn.T 
                       Hovoo_4[m,:,i,j] -= tmp1.T @ ERIooov[jn][m,n,i] 
                       
                       Sijnj = QL[ij].T @ QL[nj]
                       tmp2 = t2[nj] @ Sijnj.T
                       Hovoo_5[m,:,i,j] += tmp2.T @ Looov[nj][m,n,i]  
               lHovoo.append(Hovoo + Hovoo_1 + Hovoo_2 + Hovoo_3 + Hovoo_4 + Hovoo_5)
        else:
            lHovoo = []
            tmp_ij = []
            tmp1_ij = []
            for i in range(no):
                ii = i*no + i 
                for j in range(no):
                    ij = i*no + j 
                    jj = j*no + j
       
                    tmp = np.zeros((no,self.Local.dim[ij], no,self.Local.dim[jj]))
                    tmp1 = np.zeros((self.Local.dim[ij], no, no, self.Local.dim[ii]))
                    for m in range(no):
                        
                        tmp[m,:,i,:] += contract('aA,bB,ab->AB', QL[ij], QL[jj], ERI[m,v,i,v])
                        tmp1[:,m,j,:] += contract('aA,bB,ab->AB', QL[ij], QL[ii], ERI[v,m,j,v])
                        for n in range(no):
                            _in = i*no + n
                            jn = j*no + n
                            nj = n*no + j

                            Sijin = QL[ij].T @ QL[_in]
                            tmp3 = t2[_in] @ Sijin.T

                            rtmp = contract('ab,aA->Ab', ERI[m,n,v,v], QL[_in])
                            rtmp1 = contract('Ab,bB->AB',rtmp,QL[jj])

                            tmp[m,:,i,:] -= contract('fa,fe->ae',tmp3, rtmp1)

                            Sijjn = QL[ij].T @ QL[jn]
                            tmp4 = t2[jn] @ Sijjn.T
                            rtmp2 = contract('ab,aA->Ab', ERI[m,n,v,v], QL[ii])
                            rtmp3 = contract('Ab,bB->AB',rtmp2,QL[jn])
                            tmp1[:,m,j,:] -= contract('fb,ef->be',tmp4, rtmp3)

                            Sijnj = QL[ij].T @ QL[nj]
                            tmp5 = t2[nj] @ Sijnj.T
                            rtmp4 = contract('ab,aA->Ab', L[m,n,v,v], QL[ii])
                            rtmp5 = contract('Ab,bB->AB',rtmp4,QL[nj])
                            tmp1[:,m,j,:] += contract('fb,ef->be', tmp5, rtmp5)

                    tmp_ij.append(tmp)
                    tmp1_ij.append(tmp1)       

            for ij in range(no*no):
                i = ij // no
                j = ij % no

                ii = i*no + i 
                jj = j*no + j

                Hovoo = ERIovoo[ij].copy()
 
                Hovoo_1 = np.zeros_like(Hovoo)
                Hovoo_2 = np.zeros_like(Hovoo)
                Hovoo_3 = np.zeros_like(Hovoo)
                Hovoo_4 = np.zeros_like(Hovoo)
                Hovoo_5 = np.zeros_like(Hovoo)
                Hovoo_6 = np.zeros_like(Hovoo)
                Hovoo_7 = np.zeros_like(Hovoo)
                Hovoo_8 = np.zeros_like(Hovoo)
                Hovoo_9 = np.zeros_like(Hovoo)
                for m in range(no):

                    Hovoo_1[m,:,i,j] += Hov[ij][m] @ t2[ij]

                    Hovoo_2[m,:,i,j] += contract('ef,bef->b',t2[ij], ERIovvv[ij][m])
 
                    rtmp_1 = contract('abc,aA->Abc',ERI[m,v,v,v], QL[ij])
                    rtmp_2 = contract('Abc,bB->ABc',rtmp_1, QL[ii])
                    rtmp_3 = contract('ABc,cC->ABC',rtmp_2, QL[jj])
                    Hovoo_6[m,:,i,j] += contract('e,f,bef->b', t1[i], t1[j], rtmp_3)
  
                    Hovoo_7[m,:,i,j] += contract('e,be->b',t1[j], tmp_ij[ij][m,:,i,:])   

                    Hovoo_8[m,:,i,j] += contract('e,be->b',t1[i], tmp1_ij[ij][:,m,i,:])
                    for n in range(no):
                        _in = i*no + n
                        jn = j*no + n
                        nj = n*no + j
                        nn = n*no + n

                        Sijin = QL[ij].T @ QL[_in]
                        tmp = t2[_in] @ Sijin.T
                        Hovoo_3[m,:,i,j] -= tmp.T @ ERIooov[_in][n,m,j]

                        Sijjn = QL[ij].T @ QL[jn]
                        tmp1 = t2[jn] @ Sijjn.T
                        Hovoo_4[m,:,i,j] -= tmp1.T @ ERIooov[jn][m,n,i]

                        Sijnj = QL[ij].T @ QL[nj]
                        tmp2 = t2[nj] @ Sijnj.T
                        Hovoo_5[m,:,i,j] += tmp2.T @ Looov[nj][m,n,i]
  
                        Sijnn = QL[ij].T @ QL[nn]
                        Hovoo_9[m,:,i,j] -= (t1[n] @ Sijnn.T) * Hoooo[m,n,i,j]

                lHovoo.append(Hovoo + Hovoo_1 + Hovoo_2 + Hovoo_3 + Hovoo_4 + Hovoo_5 + Hovoo_6 + Hovoo_9 + Hovoo_7 + Hovoo_8) 
            #print("hovoo",lHovoo[1][:,:,0,1])       
        return lHovoo

    def build_Hovoo(self, o, v, ERI, L, Hov, Hoooo, t1, t2):
        contract = self.contract
        Q = self.Local.Q
        Lij = self.Local.L 
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
            #print("Hovoo_ij", 0)
            #print(Hovoo[:,:,0,0] @ Q[0] @ Lij[0])
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
            Hovoo = Hovoo + contract('ijef,mbef->mbij', self.ccwfn.build_tau(t1, t2), ERI[o,v,v,v]) # self.ccwfn.build_tau(t1, t2)
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
            #print("hovoo", contract('bB,mbij->mBij',self.Local.Q[1] @ self.Local.L[1], Hovoo)[:,:,0,1])
            if isinstance(tmp, torch.Tensor):
                del tmp
        return Hovoo
