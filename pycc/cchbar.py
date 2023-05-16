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
            self.Hvovv, self.Hvovv_ii, self.Hvovv_imn, self.Hvovv_imns = self.build_lHvovv(o,v,ccwfn.no, ERI, self.Local.ERIvovv, self.Local.ERIoovv, self.Local.QL, self.lccwfn.t1)
            self.Hooov, self.Hjiov, self.Hijov, self.Hmine, self.Himne = self.build_lHooov(o, v, ccwfn.no, ERI, self.Local.ERIooov, self.Local.QL, self.lccwfn.t1) 
            self.Hovvo, self.Hovvo_mi, self.Hovvo_mj, self.Hovvo_mm  = self.build_lHovvo(o, v, ccwfn.no, ERI, L, self.Local.ERIovvo, self.Local.QL, 
            self.lccwfn.t1, self.lccwfn.t2) #self.Hovvo_mi, self.Hovvo_mj <- this is built in CCD but not CCSD at the moment 
            self.Hovov, self.Hovov_mi, self.Hovov_mj, self.Hovov_mm  = self.build_lHovov(o, v, ccwfn.no, ERI, self.Local.ERIovov, self.Local.ERIooov, self.Local.QL,
            self.lccwfn.t1,self.lccwfn.t2) #self.Hovov_mi, self.Hovov_mj <- this is built in CCD but not CCSd at the moment
            self.Hvvvo, self.Hvvvo_ijm = self.build_lHvvvo(o, v, ccwfn.no, ERI, L, self.Local.ERIvvvo, self.Local.ERIoovo, self.Local.ERIvoov, self.Local.ERIvovo, 
            self.Local.ERIoovv, self.Local.Loovv, self.Local.QL,self.lccwfn.t1, self.lccwfn.t2, self.Hov, self.Hvvvv, self.Hvvvv_im)
            self.Hovoo, self.Hovoo_mn = self.build_lHovoo(o, v, ccwfn.no, ERI, L, self.Local.ERIovoo, self.Local.ERIovvv, self.Local.ERIooov,
            self.Local.ERIovov, self.Local.ERIvoov, self.Local.Looov, self.Local.QL, self.lccwfn.t1, self.lccwfn.t2, self.Hov, self.Hoooo)

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

                    #tmp = contract('eE,fF,mef->mEF',QL[ij],QL[nn],L[o,n,v,v])
                    tmp = contract('eE, mef->mEf',QL[ij], L[o,n,v,v])
                    tmp = contract('fF,mEf->mEF', QL[nn], tmp)
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
        if self.ccwfn.model == 'CCD':
            lHvv = []
            #lHvv_ii = []
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                Hvv = Fvv[ij].copy()

                for mn in range(no*no):
                    m = mn // no
                    n = mn % no

                    tmp = QL[mn].T @ L[m,n,v,v]
                    tmp = tmp @ QL[ij]
                    Sijmn = QL[ij].T @ QL[mn]
                    tmp1 = t2[mn] @ Sijmn.T 
                    Hvv = Hvv - tmp1.T @ tmp                             

                lHvv.append(Hvv) 
        else:
            lHvv = []
            #lHvv_ii = []

            #lHvv_ii - not needed for local lambda but may be needed for other eqns
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

            #lHvv
            for ij in range(no*no):
                 
                Hvv = Fvv[ij].copy()

                for m in range(no):
                    mm = m*no + m

                    Sijmm = QL[ij].T @ QL[mm] 
                    tmp = t1[m] @ Sijmm.T 
                    Hvv = Hvv - contract('e,a->ae', Fov[ij][m], tmp)
                    
                    
                    #tmp1 = contract('aef,aA,eE,fF->AEF',L[v,m,v,v],QL[ij],QL[ij],QL[mm]) 
                    tmp1 = contract('aef,aA->Aef',L[v,m,v,v],QL[ij])
                    tmp1 = contract('Aef,eE-> AEf', tmp1, QL[ij])
                    tmp1 = contract('AEf,fF-> AEF', tmp1, QL[mm])
                    Hvv = Hvv + contract('F,aeF->ae',t1[m],tmp1)
                    
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n

                        Sijmn = QL[ij].T @ QL[mn]
                        tmp2 = QL[mn].T @ L[m,n,v,v] 
                        tmp2 = tmp2 @ QL[ij]
                        tmp3 = t2[mn] @ Sijmn.T
                        Hvv = Hvv - tmp3.T @ tmp2
                        
                        Sijnn = QL[ij].T @ QL[nn]
                        tmp4 = QL[mm].T @ L[m,n,v,v] 
                        tmp4 = tmp4 @ QL[ij]
                         
                        tmp5 = t1[n] @ Sijnn.T 
                        Hvv = Hvv - contract('F,A,Fe->Ae',t1[m], tmp5, tmp4)
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
                    
                    Hoo[:,i] = Hoo[:,i] + contract('ef,mef->m',t2[_in],Loovv[_in][:,n])
                    
                    #tmp = contract('eE,fF,mef->mEF',QL[ii], QL[nn], L[o,n,v,v]) 
                    tmp = contract('eE, mef -> mEf', QL[ii], L[o,n,v,v]) 
                    tmp = contract('fF, mEf -> mEF', QL[nn], tmp) 
                    Hoo[:,i] = Hoo[:,i] + contract('e,f,mef->m',t1[i], t1[n], tmp)
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
        contract = self.contract 
        if self.ccwfn.model == 'CCD':
            Hoooo = ERI[o,o,o,o].copy()
 
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                                           
                Hoooo[:,:,i,j] = Hoooo[:,:,i,j] + contract('ef,mnef->mn',t2[ij], ERIoovv[ij])
            lHoooo = Hoooo + Hoooo_1 
        else:
            Hoooo = ERI[o,o,o,o].copy()
  
            tmp = np.zeros_like(Hoooo)           
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i
                jj = j*no + j

                #something is weird here
                tmp[:,:,i,j] = contract('e,mne->mn',t1[j], ERIooov[jj][:,:,i])
                Hoooo_1 = tmp + tmp.swapaxes(0,1).swapaxes(2,3)
                                  
                Hoooo[:,:,i,j] = Hoooo[:,:,i,j] + contract('ef,mnef->mn',t2[ij], ERIoovv[ij])
                
                #tmp1 = contract('eE,fF,mnef->mnEF',QL[ii], QL[jj], ERI[o,o,v,v])
                tmp1 = contract('eE,mnef -> mnEf', QL[ii], ERI[o,o,v,v])
                tmp1 = contract('fF, mnEf -> mnEF', QL[jj], tmp1)
                Hoooo[:,:,i,j] = Hoooo[:,:,i,j] + contract('e,f,mnef->mn',t1[i],t1[j],tmp1)
            lHoooo = Hoooo + Hoooo_1 
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
        if self.ccwfn.model == 'CCD':  
            lHvvvv = []
            lHvvvv_im = []
            for ij in range(no*no):

                Hvvvv = ERIvvvv[ij].copy()
                
                for mn in range(no*no):
                    m = mn // no 
                    n = mn % no 

                    Sijmn = QL[ij].T @ QL[mn]
                    tmp = Sijmn @ t2[mn]
                    tmp = tmp @ Sijmn.T 
                    Hvvvv = Hvvvv + contract('ab,ef->abef',tmp1, ERIoovv[ij][m,n]) 
                lHvvvv.append(Hvvvv)
        else: 
            lHvvvv = []
            lHvvvv_im = []

            #lHvvvv_im  -> needed for Hvvvo_ijm
            for i in range(no): 
                ii = i*no + i 

                for m in range(no):
                    im = i*no + m
                    mm = m*no +m 

                    
                    #Hvvvv_im = contract('abcd, aA, bB, cC, dD-> ABCD', ERI[v,v,v,v], QL[im], QL[im], QL[ii], QL[mm]) 
                    tmp = contract('abcd, aA -> Abcd', ERI[v,v,v,v], QL[im]) 
                    tmp = contract('Abcd, bB -> ABcd', tmp, QL[im]) 
                    tmp = contract('ABcd, cC -> ABCd', tmp, QL[ii]) 
                    Hvvvv_im = contract('ABCd, dD -> ABCD', tmp, QL[mm])
 
                    for f in range(no):
                        ff = f*no + f                                       
                    
                        Simff = QL[im].T @ QL[ff]
                        tmp = t1[f] @ Simff.T
                        #tmp1 = contract('aef,aA,eE,fF->AEF', ERI[v,f,v,v], QL[im], QL[ii], QL[mm])   
                        tmp1 = contract('aef,aA -> Aef', ERI[v,f,v,v], QL[im]) 
                        tmp2 = contract('Aef,eE -> AEf', tmp1, QL[ii]) 
                        tmp2 = contract('AEf, fF -> AEF', tmp2, QL[mm])
                        tmp3 = contract('b,aef->abef',tmp, tmp2)
                        
                        #tmp4 = contract('Aef,eE,fF->AEF', tmp1, QL[mm] , QL[ii])
                        tmp4 = contract('Aef,eE->AEf', tmp1, QL[mm])
                        tmp4 = contract('AEf, fF -> AEF', tmp4, QL[ii])  
                        tmp5 = contract('b,aef->abef',tmp, tmp4)
                        Hvvvv_im = Hvvvv_im - (tmp3 + tmp5.swapaxes(0,1).swapaxes(2,3))
                        
                        for n in range(no): 
                            fn = f*no + n 
                            nn = n*no + n 
                    
                            Simfn = QL[im].T @ QL[fn]
                            tmp6 = Simfn @ t2[fn]
                            tmp6 = tmp6 @ Simfn.T
                            #tst = contract('ef,eE, fF->EF', ERI[f,n,v,v], QL[ii], QL[mm])
                            tmp7 = contract('ef,eE->Ef', ERI[f,n,v,v], QL[ii])
                            tmp7 = contract('Ef, fF -> EF', tmp7, QL[mm]) 
                            Hvvvv_im = Hvvvv_im + contract('ab,ef->abef',tmp6, tmp7)

                            Simnn = QL[im].T @ QL[nn]
                            tmp8 = t1[f] @ Simff.T
                            tmp9 = t1[n] @ Simnn.T
                            Hvvvv_im = Hvvvv_im + contract('a,b,ef->abef',tmp8, tmp9, tmp7)
                    lHvvvv_im.append(Hvvvv_im)

            #lHvvvv
            for ij in range(no*no):
                Hvvvv = ERIvvvv[ij].copy()

                for m in range(no):
                    mm = m*no + m

                    Sijmm = QL[ij].T @ QL[mm]
                    tmp = t1[m] @ Sijmm.T 
                    tmp1 = contract('b,aef->abef',tmp, ERIvovv[ij][:,m,:,:]) 
                    Hvvvv = Hvvvv - (tmp1  + tmp1.swapaxes(0,1).swapaxes(2,3))
 
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n                       
 
                        Sijmn = QL[ij].T @ QL[mn]
                        tmp2 = Sijmn @ t2[mn]
                        tmp3 = tmp2 @ Sijmn.T 
                        Hvvvv = Hvvvv + contract('ab,ef->abef',tmp3, ERIoovv[ij][m,n])
         
                        Sijnn = QL[ij].T @ QL[nn]                    
                        tmp4 = t1[m] @ Sijmm.T 
                        tmp5 = t1[n] @ Sijnn.T 
                        Hvvvv = Hvvvv + contract('a,b,ef->abef',tmp4, tmp5, ERIoovv[ij][m,n])
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

            #lHvovv_imn swap
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n
                        Hvovv_imns = np.zeros((self.Local.dim[mn], self.Local.dim[ii], self.Local.dim[mn]))
                        Hvovv_imns = Hvovv_imns + contract('afe, aA,fF,eE-> AFE', ERI[v,i,v,v], QL[mn], QL[ii], QL[mn])

                        for k in range(no):
                            kk = k*no + k 
                            Smnkk = QL[mn].T @ QL[kk]
                            tmp = t1[k] @ Smnkk.T
                            tmp1 = contract('fe,fF,eE->FE', ERI[k,i,v,v], QL[ii], QL[mn])
                            Hvovv_imns = Hvovv_imns - contract('a,fe->afe', tmp, tmp1)
                       
                        lHvovv_imns.append(Hvovv_imns)
            #Hvovv_imn 
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    for n in range(no): 
                        mn = m*no + n
                        nn = n*no + n
                        Hvovv_imn = np.zeros((self.Local.dim[mn], self.Local.dim[mn], self.Local.dim[ii]))
                        Hvovv_imn = Hvovv_imn + contract('aef, aA,eE,fF-> AEF', ERI[v,i,v,v], QL[mn], QL[mn], QL[ii]) 

                        for k in range(no):
                            kk = k*no + k 
                            Smnkk = QL[mn].T @ QL[kk]
                            tmp = t1[k] @ Smnkk.T  
                            tmp1 = contract('ef,eE,fF->EF', ERI[k,i,v,v], QL[mn], QL[ii])
                            Hvovv_imn = Hvovv_imn - contract('a,ef->aef', tmp, tmp1) 
                        
                        lHvovv_imn.append(Hvovv_imn)
 
            # H v_ii o v_ij v_ij
            for ij in range(no*no):
                i = ij // no
                j = ij % no 
                ii = i*no + i 

                Hvovv_ii = contract('ajbc,aA, bB, cC->AjBC', ERI[v,o,v,v], QL[ii], QL[ij], QL[ij])
                Hvovv1_ii = np.zeros_like(Hvovv_ii) 

                for n in range(no): 
                    nn = n*no + n
                   
                    Siinn = QL[ii].T @ QL[nn]
                    tmp = t1[n] @ Siinn.T
                    Hvovv1_ii -= contract('a,mef->amef',tmp, ERIoovv[ij][n,:])
                lHvovv_ii.append( Hvovv_ii + Hvovv1_ii) 

            for ij in range(no*no):
                
                Hvovv = ERIvovv[ij].copy()
                Hvovv_1 = np.zeros_like(Hvovv)  
                for n in range(no):
                    nn = n*no + n
                    
                    Sijnn = QL[ij].T @ QL[nn] 
                    tmp = t1[n] @ Sijnn.T 
                    Hvovv_1 -= contract('a,mef->amef',tmp, ERIoovv[ij][n,:])
                lHvovv.append( Hvovv + Hvovv_1)       
        return lHvovv, lHvovv_ii, lHvovv_imn, lHvovv_imns
        
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
        else:
            lHooov = []
            lHijov = []
            lHjiov = []

            lHmine = []
            lHimne = []
           
            for i in range(no): 
                ii = i* self.no + i
                for m in range(no): 
                    for n in range(no):
                        nn = n*no + n
                        Hmine = np.zeros((self.Local.dim[ii]))                         
                        Hmine = Hmine + ERIooov[ii][m,i,n].copy()
                        
                        tmp = contract('ef,eE,fF->EF', ERI[i,m,v,v], QL[ii], QL[nn]) 
                        Hmine = Hmine + contract('f,ef->e', t1[n], tmp) 

                        lHmine.append(Hmine) 

                        Himne = np.zeros((self.Local.dim[ii])) 
                        Himne = Himne + ERIooov[ii][i,m,n].copy()    
   
                        tmp = contract('ef,eE,fF->EF', ERI[m,i,v,v], QL[ii], QL[nn]) 
                        Himne = Himne + contract('f,ef->e', t1[n], tmp) 

                        lHimne.append(Himne)       
 
            #Hijov 
            for ij in range(no*no):
                i = ij // no
                j = ij % no 
                ii = i*no + i

                Hjiov = ERIooov[ij][j,i,:,:].copy()
                Hjiov_1 = np.zeros_like(Hjiov)
                for m in range(no):
                    mm = m*no + m
           
                    tmp = contract('eE,fF,ef->EF',QL[ij], QL[mm], ERI[i,j,v,v])
                    Hjiov_1[m] += contract('f,ef->e',t1[m], tmp)

                lHjiov.append(Hjiov + Hjiov_1)  
 
            #Hijov
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i

                Hijov = ERIooov[ij][i,j,:,:].copy()
                Hijov_1 = np.zeros_like(Hijov)
                for m in range(no):
                    mm = m*no + m

                    tmp = contract('eE,fF,ef->EF',QL[ij], QL[mm], ERI[j,i,v,v])
                    Hijov_1[m] += contract('f,ef->e',t1[m], tmp)

                lHijov.append(Hijov + Hijov_1)

            for ij in range(no*no):
                i = ij // no
                ii = i*no + i 
       
                Hooov = ERIooov[ij].copy()
                Hooov_1 = np.zeros_like(Hooov)
                tmp = contract('eE,fF,nmef->nmEF',QL[ij], QL[ii], ERI[o,o,v,v])
                Hooov_1[:,:,i,:] = contract('f,nmef->mne',t1[i], tmp)         

                lHooov.append(Hooov + Hooov_1) 
        return lHooov, lHjiov, lHijov, lHmine, lHimne   
        
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
        if self.ccwfn.model == 'CCD':
            lHovvo = [] 
            lHovvo_mi = []
            lHovvo_mj = []
            
            #Hovvo_mj 
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mj = m*no + j

                    Hovvo_mj = np.zeros((self.Local.dim[mj],self.Local.dim[ij]))
                    Hovvo_mj = contract('bB, eE, be -> BE', QL[mj], QL[ij], ERI[i,v,v,m]) # , QL[mi], QL[ij]) ERIovvo[ij][j,:,:,m].copy()

                    Hovvo1_mj = np.zeros_like(Hovvo_mj)
                    Hovvo2_mj  = np.zeros_like(Hovvo_mj)
                    for n in range(no):
                        mn = m*no + n
                        nm = n*no + m

                        Smjmn = QL[mj].T @ QL[mn]
                        tmp = t2[mn] @ Smjmn.T
                        tmp1 = QL[ij].T @ ERI[i,n,v,v]
                        tmp2 = tmp1 @ QL[mn]
                        Hovvo1_mj -= tmp.T @ tmp2.T

                        Smjnm = QL[mj].T @ QL[nm]
                        tmp3 = t2[nm] @ Smjnm.T
                        tmp4 = QL[ij].T @ L[i,n,v,v]
                        tmp5 = tmp4 @ QL[nm]
                        Hovvo2_mj += tmp3.T @ tmp5.T

                    lHovvo_mj.append(Hovvo_mj + Hovvo1_mj + Hovvo2_mj)

            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mi = m*no + i

                    Hovvo_mi = np.zeros((self.Local.dim[mi],self.Local.dim[ij]))
                    Hovvo_mi = contract('bB, eE, be -> BE', QL[mi], QL[ij], ERI[j,v,v,m]) # , QL[mi], QL[ij]) ERIovvo[ij][j,:,:,m].copy()

                    Hovvo1_mi = np.zeros_like(Hovvo_mi)
                    Hovvo2_mi  = np.zeros_like(Hovvo_mi)
                    for n in range(no):
                        mn = m*no + n
                        nm = n*no + m

                        Smimn = QL[mi].T @ QL[mn]
                        tmp = t2[mn] @ Smimn.T
                        tmp1 = QL[ij].T @ ERI[j,n,v,v]
                        tmp2 = tmp1 @ QL[mn]
                        Hovvo1_mi -= tmp.T @ tmp2.T

                        Sminm = QL[mi].T @ QL[nm]
                        tmp3 = t2[nm] @ Sminm.T
                        tmp4 = QL[ij].T @ L[j,n,v,v]
                        tmp5 = tmp4 @ QL[nm]
                        Hovvo2_mi += tmp3.T @ tmp5.T

                    lHovvo_mi.append(Hovvo_mi + Hovvo1_mi + Hovvo2_mi)
                    #print("lHovvo_mi", ij, mi, Hovvo_mi + Hovvo1_mi + Hovvo2_mi)

            for ij in range(no*no):
                i = ij // no
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
            #print("Hovvo", lHovvo[6][1,:,:,1])
        else:
            lHovvo = []
            lHovvo_mi = []
            lHovvo_mj = []
            lHovvo_mm = []
            
            #Hovvo_mm 
            for i in range(no): 
                ii = i*no + i 
                for m in range(no):
                    mm = m*no + m
                    Hovvo_mm = np.zeros((self.Local.dim[mm], self.Local.dim[ii]))
                    Hovvo_mm = Hovvo_mm + contract('be, bB, eE-> BE', ERI[i,v,v,m], QL[mm], QL[ii] ) 
                    
                    tmp = contract('bef, bB, eE,fF->BEF', ERI[i,v,v,v], QL[mm], QL[ii], QL[mm]) 
                    Hovvo_mm = Hovvo_mm + contract('f,bef->be', t1[m], tmp)
                    
                    for n in range(no):
                        nn = n*no + n
                        mn = m*no + n
                        nm = n*no + m
 
                        Smmnn = QL[mm].T @ QL[nn]
                        tmp1 = Smmnn @ t1[n]
                        tst = contract('e, eE->E', ERI[i,n,v,m], QL[ii])
                        Hovvo_mm = Hovvo_mm - contract('b,e->be', tmp1, tst)
                        
                        Smmmn = QL[mm].T @ QL[mn]
                        tmp2 = t2[mn] @ Smmmn.T 
                        tmp3 = contract('ef, eE,fF->EF', ERI[i,n,v,v], QL[ii], QL[mn]) 
                        Hovvo_mm = Hovvo_mm - contract('fb,ef->be', tmp2, tmp3) 
 
                        tmp3 = contract('ef, eE,fF->EF', ERI[i,n,v,v], QL[ii], QL[mm])
                        Hovvo_mm = Hovvo_mm - contract('f,b,ef->be', t1[m], tmp1, tmp3) 

                        Smmnm = QL[mm].T @ QL[nm]
                        tmp2 = t2[nm] @ Smmnm.T 
                        tmp4 = contract('ef,eE,fF->EF', L[i,n,v,v], QL[ii], QL[nm]) 
                        Hovvo_mm = Hovvo_mm + contract('fb,ef->be', tmp2, tmp4)                            
                    lHovvo_mm.append(Hovvo_mm)

            #Hovvo_mi 
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                jj =  j*no + j

                for m in range(no):
                    mi = m*no + i
                    mm = m*no + m

                    Hovvo_mi = np.zeros((self.Local.dim[mi],self.Local.dim[ij]))
                    Hovvo_mi = contract('bB, eE, be -> BE', QL[mi], QL[ij], ERI[j,v,v,m]) # , QL[mi], QL[ij]) ERIovvo[ij][j,:,:,m].copy()

                    Hovvo1_mi = np.zeros_like(Hovvo_mi)

                    tmp = contract('abc,aA,bB,cC->ABC',ERI[j,v,v,v], QL[mi], QL[ij], QL[mm])
                    Hovvo1_mi += contract('f,bef->be', t1[m], tmp)

                    Hovvo2_mi  = np.zeros_like(Hovvo_mi)
                    Hovvo3_mi = np.zeros_like(Hovvo_mi)
                    Hovvo4_mi  = np.zeros_like(Hovvo_mi)
                    Hovvo5_mi = np.zeros_like(Hovvo_mi)
                    for n in range(no):
                        mn = m*no + n
                        nm = n*no + m
                        nn = n*no + n

                        Sminn = QL[mi].T @ QL[nn]
                        tmp1 = Sminn @ t1[n]
                        tmp2 = contract ('e,eE->E', ERI[j,n,v,m], QL[ij])
                        Hovvo2_mi -= contract('b,e->be', tmp1, tmp2)

                        Smimn = QL[mi].T @ QL[mn]
                        tmp3 = t2[mn] @ Smimn.T
                        tmp4 = QL[ij].T @ ERI[j,n,v,v]
                        tmp5 = tmp4 @ QL[mn]
                        Hovvo3_mi -= tmp3.T @ tmp5.T

                        tmp6 = tmp4 @ QL[mm]
                        Hovvo4_mi -= contract('f,b,ef->be', t1[m], tmp1, tmp6)

                        Sminm = QL[mi].T @ QL[nm]
                        tmp3 = t2[nm] @ Sminm.T
                        tmp4 = QL[ij].T @ L[j,n,v,v]
                        tmp5 = tmp4 @ QL[nm]
                        Hovvo5_mi += tmp3.T @ tmp5.T

                    lHovvo_mi.append(Hovvo_mi + Hovvo1_mi + Hovvo2_mi + Hovvo3_mi + Hovvo4_mi + Hovvo5_mi) #            Hovvo1_mj + Hovvo2_mj + Hovvo3_mj + Hovvo4_mj + Hovvo5_mj)

            #Hovvo_mj
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                jj =  j*no + j
 
                for m in range(no):
                    mj = m*no + j
                    mm = m*no + m

                    Hovvo_mj = np.zeros((self.Local.dim[mj],self.Local.dim[ij]))
                    Hovvo_mj = contract('bB, eE, be -> BE', QL[mj], QL[ij], ERI[i,v,v,m]) # , QL[mi], QL[ij]) ERIovvo[ij][j,:,:,m].copy()

                    Hovvo1_mj = np.zeros_like(Hovvo_mj)

                    tmp = contract('abc,aA,bB,cC->ABC',ERI[i,v,v,v], QL[mj], QL[ij], QL[mm])
                    Hovvo1_mj += contract('f,bef->be', t1[m], tmp)

                    Hovvo2_mj  = np.zeros_like(Hovvo_mj)
                    Hovvo3_mj = np.zeros_like(Hovvo_mj)
                    Hovvo4_mj  = np.zeros_like(Hovvo_mj)
                    Hovvo5_mj = np.zeros_like(Hovvo_mj)
                    for n in range(no):
                        mn = m*no + n
                        nm = n*no + m
                        nn = n*no + n

                        Smjnn = QL[mj].T @ QL[nn]
                        tmp1 = Smjnn @ t1[n] 
                        tmp2 = contract ('e,eE->E', ERI[i,n,v,m], QL[ij])
                        Hovvo2_mj -= contract('b,e->be', tmp1, tmp2) 

                        Smjmn = QL[mj].T @ QL[mn]
                        tmp3 = t2[mn] @ Smjmn.T
                        tmp4 = QL[ij].T @ ERI[i,n,v,v]
                        tmp5 = tmp4 @ QL[mn]
                        Hovvo3_mj -= tmp3.T @ tmp5.T

                        tmp6 = tmp4 @ QL[mm] 
                        Hovvo4_mj -= contract('f,b,ef->be', t1[m], tmp1, tmp6) 

                        Smjnm = QL[mj].T @ QL[nm]
                        tmp3 = t2[nm] @ Smjnm.T
                        tmp4 = QL[ij].T @ L[i,n,v,v]
                        tmp5 = tmp4 @ QL[nm]
                        Hovvo5_mj += tmp3.T @ tmp5.T

                    lHovvo_mj.append(Hovvo_mj + Hovvo1_mj + Hovvo2_mj + Hovvo3_mj + Hovvo4_mj + Hovvo5_mj) #            Hovvo1_mj + Hovvo2_mj + Hovvo3_mj + Hovvo4_mj + Hovvo5_mj)

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
        return lHovvo, lHovvo_mi, lHovvo_mj, lHovvo_mm  

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
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            lHovov = [] 
            lHovov_mi = []
            lHovov_mj = []
            #for ij in range(no*no):
                #i = ij // no 
                #j = ij % no
   
                # this way seems to be getting the right answer 
                #Hovov = np.zeros((self.no,self.Local.dim[ij], self.Local.dim[ij]))
                #Hovov_1 = np.zeros_like(Hovov)
                #for m in range(no):
                    #Hovov = np.zeros((self.Local.dim[ij], self.Local.dim[ij]))
                    #Hovov[m] += ERIovov[ij][m,:,j,:].copy()
                    
                    #Hovov_1 = np.zeros_like(Hovov)
                    #for n in range(no):
                        #jn = j*no + n

                        #Sijjn = QL[ij].T @ QL[jn]
                        #tmp = t2[jn] @ Sijjn.T
                        #tmp1 = QL[ij].T @ ERI[n,m,v,v]
                        #tmp2 = tmp1 @ QL[jn]
                        #Hovov_1[m] -=  tmp.T @ tmp2.T
                #lHovov.append(Hovov + Hovov_1) 
            
            #Hovov_mj
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mj = m*no + j
                    Hovov_mj = np.zeros((self.Local.dim[mj], self.Local.dim[ij]))
                    Hovov_mj = contract('be, bB, eE-> BE', ERI[i,v,m,v], QL[mj], QL[ij])         #ERIovov[ij].copy()

                    Hovov1_mj = np.zeros_like(Hovov_mj)
                    for n in range(no):
                        mn = m*no + n

                        Smjmn = QL[mj].T @ QL[mn]
                        tmp = t2[mn] @ Smjmn.T
                        tmp1 = QL[ij].T @ ERI[n,i,v,v]
                        tmp2 = tmp1 @ QL[mn]
                        Hovov1_mj -=  tmp.T @ tmp2.T
                    lHovov_mj.append(Hovov_mj + Hovov1_mj)
            for ij in range(no*no):
                i = ij // no 
                j = ij % no

                #this is for lHovov_im 
                for m in range(no):
                    mi = m*no + i 
                    Hovov_mi = np.zeros((self.Local.dim[mi], self.Local.dim[ij]))
                    Hovov_mi = contract('be, bB, eE-> BE', ERI[j,v,m,v], QL[mi], QL[ij])         #ERIovov[ij].copy()
                   
                    Hovov1_mi = np.zeros_like(Hovov_mi)
                    for n in range(no):
                        mn = m*no + n 
                        
                        Smimn = QL[mi].T @ QL[mn]
                        tmp = t2[mn] @ Smimn.T
                        tmp1 = QL[ij].T @ ERI[n,j,v,v]
                        tmp2 = tmp1 @ QL[mn]
                        Hovov1_mi -=  tmp.T @ tmp2.T
                    lHovov_mi.append(Hovov_mi + Hovov1_mi)  
                    #print("Hovov", ij, mi, Hovov_mi + Hovov1_mi) 
            for ij in range(no*no):
                i = ij // no 
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
            #print("Hovov", lHovov[6][1,:,1,:])    
        else:  
            lHovov = []
            lHovov_mi = []
            lHovov_mj = []
            lHovov_mm = []
    
            #Hovov_mm 
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    mm = m* no + m
                    
                    Hovov_mm = np.zeros((self.Local.dim[mm], self.Local.dim[ii]))
                    Hovov_mm = Hovov_mm + contract('be,bB,eE->BE', ERI[i,v,m,v], QL[mm], QL[ii])
                 
                    tmp = contract('bef, bB, eE,fF->BEF', ERI[v,i,v,v], QL[mm], QL[ii], QL[mm])
                    Hovov_mm = Hovov_mm + contract('f,bef->be', t1[m], tmp)
                        
                    for n in range(no):
                        nn = n*no + n
                        mn = m*no + n
                        nm = n*no + m
                        
                        Smmnn = QL[mm].T @ QL[nn]
                        tmp1 = Smmnn @ t1[n]
                        tst = contract('e, eE->E', ERI[i,n,m,v], QL[ii])
                        Hovov_mm = Hovov_mm - contract('b,e->be', tmp1, tst)
                        
                        Smmmn = QL[mm].T @ QL[mn]
                        tmp2 = t2[mn] @ Smmmn.T 
                        tmp3 = contract('ef, eE,fF->EF', ERI[n,i,v,v], QL[ii], QL[mn])
                        Hovov_mm = Hovov_mm - contract('fb,ef->be', tmp2, tmp3) 
                        
                        tmp3 = contract('ef, eE,fF->EF', ERI[n,i,v,v], QL[ii], QL[mm])
                        Hovov_mm = Hovov_mm - contract('f,b,ef->be', t1[m], tmp1, tmp3)
                        
                    lHovov_mm.append(Hovov_mm)
            #Hovov_mj
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mm = m*no + m
                    mj = m*no + j

                    Hovov_mj = np.zeros((self.Local.dim[mj], self.Local.dim[ij]))
                    Hovov_mj = contract('be, bB, eE-> BE', ERI[i,v,m,v], QL[mj], QL[ij])         #ERIovov[ij].copy()

                    Hovov1_mj = np.zeros_like(Hovov_mj)
                 
                    tmp = contract('bef, bB, eE, fF-> BEF', ERI[v,i,v,v], QL[mj], QL[ij], QL[mm])
                    Hovov1_mj = contract('f,bef->be', t1[m], tmp)

                    Hovov2_mj = np.zeros_like(Hovov_mj) 
                    Hovov3_mj = np.zeros_like(Hovov_mj) 
                    Hovov4_mj = np.zeros_like(Hovov_mj)
                    for n in range(no):
                        nn = n*no + n
                        mn = m*no + n

                        Smjnn = QL[mj].T @ QL[nn]
                        tmp1 = contract('e,eE->E', ERI[i,n,m,v], QL[ij]) 
                        Hovov2_mj -= contract('b,e->be', Smjnn @ t1[n], tmp1)       

                        Smjmn = QL[mj].T @ QL[mn]
                        tmp2 = t2[mn] @ Smjmn.T
                        tmp3 = QL[ij].T @ ERI[n,i,v,v]
                        tmp4 = tmp3 @ QL[mn]
                        Hovov3_mj -=  tmp2.T @ tmp4.T

                        tmp5 = tmp3 @ QL[mm]
                        Hovov4_mj -= contract('f,b,ef->be',t1[m], Smjnn @ t1[n], tmp5) 

                    lHovov_mj.append(Hovov_mj + Hovov1_mj + Hovov2_mj + Hovov3_mj + Hovov4_mj)

            #Hovov_mi
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mm = m*no + m
                    mi = m*no + i

                    Hovov_mi = np.zeros((self.Local.dim[mi], self.Local.dim[ij]))
                    Hovov_mi = contract('be, bB, eE-> BE', ERI[j,v,m,v], QL[mi], QL[ij])         #ERIovov[ij].copy()

                    Hovov1_mi = np.zeros_like(Hovov_mi)

                    tmp = contract('bef, bB, eE, fF-> BEF', ERI[v,j,v,v], QL[mi], QL[ij], QL[mm])
                    Hovov1_mi = contract('f,bef->be', t1[m], tmp)

                    Hovov2_mi = np.zeros_like(Hovov_mi)
                    Hovov3_mi = np.zeros_like(Hovov_mi)
                    Hovov4_mi = np.zeros_like(Hovov_mi)
                    for n in range(no):
                        nn = n*no + n
                        mn = m*no + n

                        Sminn = QL[mi].T @ QL[nn]
                        tmp1 = contract('e,eE->E', ERI[j,n,m,v], QL[ij])
                        Hovov2_mi -= contract('b,e->be', Sminn @ t1[n], tmp1)

                        Smimn = QL[mi].T @ QL[mn]
                        tmp2 = t2[mn] @ Smimn.T
                        tmp3 = QL[ij].T @ ERI[n,j,v,v]
                        tmp4 = tmp3 @ QL[mn]
                        Hovov3_mi -=  tmp2.T @ tmp4.T

                        tmp5 = tmp3 @ QL[mm]
                        Hovov4_mi -= contract('f,b,ef->be',t1[m], Sminn @ t1[n], tmp5)

                    lHovov_mi.append(Hovov_mi + Hovov1_mi + Hovov2_mi + Hovov3_mi + Hovov4_mi)
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
        return lHovov, lHovov_mi, lHovov_mj, lHovov_mm 

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
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            lHvvvo = []
            lHvvvo_ijm = []
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

            lHvvvo_ijm = []
            tmp_ijm = []
            tmp1_ijm = []

            for i in range(no): 
                ii = i*no + i 
                for m in range(no):
                    im = i*no + m
                       
                    for n in range(no): 
                        mn = m*no + n
                        nn = n*no + n
                        
                        tmp = np.zeros((self.Local.dim[im], self.Local.dim[ii]))
                        tmp1 = np.zeros_like(tmp)  

                        tmp += contract('ae,aA,eE->AE', ERI[v,n,v,m], QL[im], QL[ii]) 

                        tmp_0 = np.zeros_like(tmp)
                        tmp_1 = np.zeros_like(tmp)
                        tmp_2 = np.zeros_like(tmp)
                        tmp_0 += contract('be,bB,eE->BE', ERI[v,n,m,v], QL[im], QL[ii])                         
                        for k in range(no): 
                            mk = m *no + k
                            km = k*no + m

                            Simmk = QL[im].T @ QL[mk]
                            tst = t2[mk] @ Simmk.T 
                            tst1 = contract('fe,fF,eE->FE', ERI[n,k,v,v], QL[mk], QL[ii]) 
                            tmp1 -= contract('fa,fe->ae', tst, tst1)  

                            tst2 = contract('ef,eE,fF->EF', ERI[n,k,v,v], QL[ii], QL[mk])
                            tmp_1 -= contract('fb,ef->be', tst, tst2) 

                            Simkm = QL[im].T @ QL[km]
                            tst3 = t2[km] @ Simkm.T
                            tst4 = contract('ef,eE,fF->EF', L[n,k,v,v], QL[ii], QL[mk]) 
                            tmp_2 += contract('fb,ef->be', tst3, tst4)       

                        tmp_ijm.append(tmp + tmp1)
                        tmp1_ijm.append(tmp_0 + tmp_1+ tmp_2)        

            #lHvvvo_ijm 
            for i in range(no): 
                ii = i*no + i
                for m in range(no):
                    im = i*no + m 
                    mi = m*no + i
                    mm = m*no + m               

                    Hvvvo_ijm = np.zeros((self.Local.dim[im], self.Local.dim[im], self.Local.dim[ii])) 
                    Hvvvo_ijm = contract('abe, aA, bB, eE->ABE', ERI[v,v,v,m], QL[im], QL[im], QL[ii])   

                    Hvvvo2_ijm = np.zeros_like(Hvvvo_ijm)
                    #Siimm = QL[ii].T @ QL[mm]
                    #tmp = Siimm @ t1[m]  
                    Hvvvo2_ijm = contract('f,abef->abe', t1[m], Hvvvv_im[im]) 
 
                    Hvvvo1_ijm = np.zeros_like(Hvvvo_ijm) 
                    Hvvvo3_ijm = np.zeros_like(Hvvvo_ijm) 
                    Hvvvo4_ijm = np.zeros_like(Hvvvo_ijm) 
                    Hvvvo5_ijm = np.zeros_like(Hvvvo_ijm) 
                    Hvvvo6_ijm = np.zeros_like(Hvvvo_ijm) 
                    Hvvvo7_ijm = np.zeros_like(Hvvvo_ijm)
                    Hvvvo8_ijm = np.zeros_like(Hvvvo_ijm)
                    Hvvvo9_ijm = np.zeros_like(Hvvvo_ijm)
                    for n in range(no): 
                        nm = n*no + m 
                        nn = n*no + n
                        mn = m*no + n
                        imn = im*no + n 

                        Simnm = QL[im].T @ QL[nm] 
                        tmp = t2[nm] @ Simnm.T
                        tmp1 = Simnm @ tmp 
                        Hvvvo1_ijm -=  contract('e, ab->abe', Hov[ii][n], tmp1) 

                        tmp2 = contract('bfe,bB,fF,eE->BFE', ERI[v,n,v,v], QL[im], QL[mn], QL[ii]) 
                        Simmn = QL[im].T @ QL[mn]
                        tmp3 = t2[mn] @ Simmn.T 
                        Hvvvo5_ijm -= contract('fa,bfe->abe', tmp3, tmp2) 
                        
                        tmp5 = contract('aef,aA,eE,fF->AEF', ERI[v,n,v,v], QL[im], QL[ii], QL[mn])
                        Hvvvo6_ijm -= contract('fb,aef->abe', tmp3, tmp5) 
                        
                        tmp4 = contract('aef,aA,eE,fF->AEF', L[v,n,v,v], QL[im], QL[ii], QL[nm]) 
                        Hvvvo7_ijm += contract('fb,aef->abe',tmp, tmp4)
        
                        Simnn = QL[im].T @ QL[nn] 
                        tmp6 = Simnn @ t1[n] 
                        cool = contract('ae,aA,eE->AE', ERI[v,n,v,m] , QL[im], QL[ii])
                        
                        Simmn = QL[im].T @ QL[mn]
                        tmp8 = t2[mn] @ Simmn.T 

                        Hvvvo8_ijm -= contract('b,ae->abe', tmp6, tmp_ijm[imn])

                        Hvvvo9_ijm -= contract('a,be->abe', tmp6, tmp1_ijm[imn]) 
                        for k in range(no): 
                            kn = k*no + n 
                            kk = k*no + k
                            nn = n*no + n
                                                  

                            Simkn = QL[im].T @ QL[kn] 
                            tmp = Simkn @ t2[kn] @ Simkn.T 
                            tmp1 = QL[ii].T @ ERI[k,n,v,m] 
                            Hvvvo3_ijm += contract('ab,e->abe',tmp, tmp1) 

                            Simkk = QL[im].T @ QL[kk]
                            Simnn = QL[im].T @ QL[nn] 
                            tmp = Simkk @ t1[k]
                            tmp2 = Simnn @ t1[n]
                            Hvvvo4_ijm += contract('a,b,e->abe', tmp, tmp2, tmp1) 
       
                    lHvvvo_ijm.append( Hvvvo_ijm + Hvvvo1_ijm + Hvvvo2_ijm + Hvvvo3_ijm + Hvvvo4_ijm + Hvvvo5_ijm + Hvvvo6_ijm + Hvvvo7_ijm + Hvvvo8_ijm + Hvvvo9_ijm)       

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
            #print("all lHvvvo", lHvvvo[6][:,:,:,1])          
        return lHvvvo, lHvvvo_ijm

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
        return Hvvvo

    def build_lHovoo(self, o, v, no, ERI, L, ERIovoo, ERIovvv, ERIooov, ERIovov, ERIvoov, Looov, QL, t1, t2, Hov, Hoooo):
        contract = self.contract
        if self.ccwfn.model =='CCD':
           lHovoo = []
           lHovoo_mn = []
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

            lHovoo_mn = []
            #tmp_mn = []
            #tmp1_mn = [] 

            for i in range(no):
                for m in range(no):
                    mm = m*no + m
                    for n in range(no):
                        mn = m*no + n
                        nn = n*no + n

                        Hovoo_mn = np.zeros((self.Local.dim[mn]))
                        Hovoo_mn = contract('b,bB->B', ERI[i,v,m,n], QL[mn])
                        
                        Hovoo_mn = Hovoo_mn + contract('e,eb->b', Hov[mn][i], t2[mn]) 
                        
                        Hovoo_mn = Hovoo_mn + contract('ef,bef->b', t2[mn], ERIovvv[mn][i])
                        
                        tmp = contract('bef,bB, eE,fF->BEF', ERI[i,v,v,v], QL[mn], QL[mm], QL[nn])
                        Hovoo_mn = Hovoo_mn + contract('e,f,bef->b', t1[m], t1[n], tmp)
 
                        tmp_mn = contract('be,bB,eE->BE', ERI[i,v,m,v], QL[mn], QL[nn])
                        
                        for k in range(no):
                            mk = m*no + k
                            
                            Smnmk = QL[mn].T @ QL[mk] 
                            tmp2 = t2[mk] @ Smnmk.T 
                            tmp5 = contract('fe,fF,eE->FE', ERI[i,k,v,v], QL[mk], QL[nn])
                            tmp_mn = tmp_mn - contract('fb,fe->be', tmp2, tmp5)
                        
                        Hovoo_mn = Hovoo_mn + contract('e,be->b', t1[n], tmp_mn)

                        tmp1_mn = contract('be,bB,eE->BE', ERI[v,i,n,v], QL[mn], QL[mm])
                        for k in range(no):
                            kn = k*no + n
                            nk = n*no + k
                            
                            Smnnk = QL[mn].T @ QL[nk]
                            tmp3 = t2[nk] @ Smnnk.T
                            tmp5 = contract('ef,eE,fF->EF', ERI[i,k,v,v], QL[mm], QL[nk])
                            tmp1_mn = tmp1_mn - contract('fb,ef->be', tmp3, tmp5)  

                            Smnkn = QL[mn].T @ QL[kn]
                            tmp4 = t2[kn] @ Smnkn.T
                            tmp6 = contract('ef,eE,fF->EF', L[i,k,v,v], QL[mm], QL[kn])
                            tmp1_mn = tmp1_mn + contract('fb,ef->be', tmp4, tmp6) 
                        Hovoo_mn = Hovoo_mn + contract('e,be->b', t1[m], tmp1_mn)    
                        for k in range(no):
                            kk = k*no + k 
                            mk = m*no + k
                            nk = n*no + k
                            kn = k*no + n

                            Smnkk = QL[mn].T @ QL[kk]
                            tmp1 = Smnkk @ t1[k] 
                            Hovoo_mn = Hovoo_mn - (tmp1 * Hoooo[i,k,m,n])
                            
                            Smnmk = QL[mn].T @ QL[mk] 
                            tmp2 = t2[mk] @ Smnmk.T 
                            Hovoo_mn = Hovoo_mn - contract('eb,e->b', tmp2, ERIooov[mk][k,i,n]) 

                            Smnnk = QL[mn].T @ QL[nk]
                            tmp3 = t2[nk] @ Smnnk.T 
                            Hovoo_mn = Hovoo_mn - contract('eb,e->b', tmp3, ERIooov[nk][i,k,m])            
                            
                            Smnkn = QL[mn].T @ QL[kn]
                            tmp4 = t2[kn] @ Smnkn.T 
                            Hovoo_mn = Hovoo_mn + contract('eb,e->b', tmp4, Looov[kn][i,k,m]) 
        
                            #tmp5 = contract('fe,fF,eE->FE', ERI[i,k,v,v], QL[mk], QL[nn])
                            #tmp_mn = tmp_mn - contract('fb,fe->be', tmp2, tmp5)
                            #Hovoo_mn = Hovoo_mn + contract('e,be->b', t1[n], tmp_mn)     
                        lHovoo_mn.append( Hovoo_mn)
                         
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
        return lHovoo, lHovoo_mn

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
            if isinstance(tmp, torch.Tensor):
                del tmp
        return Hovoo
