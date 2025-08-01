import time
import numpy as np
import torch


class lcchbar(object):
    """
    An LPNO RHF-CC Similarity-Transformed Hamiltonian object.

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
        F = ccwfn.H.F
        L = ccwfn.H.L
        ERI = ccwfn.H.ERI
        self.Local = ccwfn.Local
        o = self.o = ccwfn.o
        v = self.v = ccwfn.v
        self.no = ccwfn.no
        self.nv = ccwfn.nv
        self.lccwfn = ccwfn.lccwfn
     
        t1 = self.lccwfn.t1 
        t2 = self.lccwfn.t2 


        #build one-body Hbar 
        self.Hov = self.build_lHov(o, v, L, t1)
        self.Hvv = self.build_lHvv(o, v, F, L, t1, t2)
        self.Hoo = self.build_lHoo(o ,v, F, L, t1, t2)

        #build two-body Hbar
        self.Hoooo = self.build_lHoooo(o, v, ERI, t1, t2)
        self.Hvvvv, self.Hvvvv_im, self.Hvvvv_ij = self.build_lHvvvv( o, v, ERI, t1, t2)
        
        ( self.Hvovv_ij, self.Hvovv_ii, self.Hvovv_imn, self.Hvovv_imns, 
        self.Hamef, self.Hamfe, self.Hfibe, self.Hfieb, self.Hgnea, self.Hgnae, 
        self.Halbc, self.Halcb, self.Hfjea, self.Hfmae, self.Hfmea ) = self.build_lHvovv(o,v, ERI, t1)
        
        ( self.Hjiov, self.Hijov, self.Hmine, self.Himne, self.Hmine_mm, self.Himne_mm, self.Hmnie, self.Hnmie, 
        self.Hjmna, self.Hmjna, self.Hjine, self.Hooov ) = self.build_lHooov(o, v, ERI, t1)
        
        self.Hovvo_mi, self.Hovvo_mj, self.Hovvo_mm, self.Hovvo_ni = self.build_lHovvo(o, v, ERI, L, t1, t2)
        self.Hovov_mi, self.Hovov_mj, self.Hovov_mm, self.Hovov_ni = self.build_lHovov(o, v, ERI, t1, t2)
        self.Hvvvo_im  = self.build_lHvvvo(o, v, ERI, L, t1, t2, self.Hov, self.Hvvvv, self.Hvvvv_im)
        self.Hovoo_mn, self.Hovoo_ij = self.build_lHovoo(o, v, ERI, L, t1, t2, self.Hov, self.Hoooo)

        print("\nLPNO-HBAR constructed in %.3f seconds.\n" % (time.time() - time_init))

    def build_lHov(self, o, v, L, t1):
        contract = self.contract
        no = self.no 
        Fov =  self.Local.Fov 
        QL = self.Local.QL 
   
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

    def build_lHvv(self,o, v, F, L, t1, t2):
        contract = self.contract
        Sijmn = self.Local.Sijmn
        Sijmm = self.Local.Sijmm
        no = self.no 
        Fvv = self.Local.Fvv
        Fov = self.Local.Fov 
        QL = self.Local.QL 
       

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

    def build_lHoo(self, o ,v, F, L, t1, t2):
        contract = self.contract
        no = self.no 
        Fov = self.Local.Fov 
        Looov = self.Local.Looov 
        Loovv = self.Local.Loovv
        QL = self.Local.QL 
       
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

    def build_lHoooo(self, o, v, ERI, t1, t2):
        contract = self.contract
        no = self.no 
        ERIoovv = self.Local.ERIoovv 
        ERIooov = self.Local.ERIooov 
        QL = self.Local.QL 

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

    def build_lHvvvv(self, o, v, ERI, t1, t2):
        contract = self.contract
        Sijmn = self.Local.Sijmn
        Sijmm = self.Local.Sijmm
        no = self.no 
        ERIvvvv = self.Local.ERIvvvv
        ERIoovv = self.Local.ERIoovv
        ERIvovv = self.Local.ERIvovv 
        QL = self.Local.QL 
  
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

            #Hv_ij v_ij v_ii v_jj - Eqn 148
            lHvvvv_ij = []
            for i in range(no):
                ii = i*no + i
                for j in range(no):
                    ij = i*no + j
                    jj = j*no + j

                    #first term
                    Hvvvv_ij = contract('abcd, aA -> Abcd', ERI[v,v,v,v], QL[ij])
                    Hvvvv_ij = contract('Abcd, bB -> ABcd', Hvvvv_ij, QL[ij])
                    Hvvvv_ij = contract('ABcd, cC -> ABCd', Hvvvv_ij, QL[ii])
                    Hvvvv_ij = contract('ABCd, dD -> ABCD', Hvvvv_ij, QL[jj])

                    for m in range(no):
                        mm = m*no + m
                        ijm = ij*no + m

                        #second term
                        tmp = t1[m] @ Sijmm[ijm].T
                        tmp1 = contract('aef,aA -> Aef', ERI[v,m,v,v], QL[ij])
                        tmp2 = contract('Aef,eE -> AEf', tmp1, QL[ii])
                        tmp2 = contract('AEf, fF -> AEF', tmp2, QL[jj])
                        tmp2 = contract('b,aef->abef',tmp, tmp2)

                        #third term
                        tmp1 = contract('Aef,eE->AEf', tmp1, QL[jj])
                        tmp1 = contract('AEf, fF -> AEF', tmp1, QL[ii])
                        tmp1 = contract('b,aef->abef',tmp, tmp1)
                        Hvvvv_ij = Hvvvv_ij - (tmp2 + tmp1.swapaxes(0,1).swapaxes(2,3))

                        for n in range(no):
                            mn = m*no + n
                            nn = n*no + n
                            ijmn = ij*(no**2) + mn
                            ijn = ij*no + n

                            #fourth term
                            tmp = Sijmn[ijmn] @ t2[mn]
                            tmp = tmp @ Sijmn[ijmn].T
                            tmp1 = contract('ef,eE->Ef', ERI[m,n,v,v], QL[ii])
                            tmp1 = contract('Ef, fF -> EF', tmp1, QL[jj])
                            Hvvvv_ij = Hvvvv_ij + contract('ab,ef->abef',tmp, tmp1)

                            #fifth term
                            tmp = t1[m] @ Sijmm[ijm].T
                            tmp2 = t1[n] @ Sijmm[ijn].T
                            Hvvvv_ij = Hvvvv_ij + contract('a,b,ef->abef',tmp, tmp2, tmp1)
                    lHvvvv_ij.append(Hvvvv_ij)

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
        return lHvvvv, lHvvvv_im, lHvvvv_ij

    def build_lHvovv(self, o, v, ERI, t1):
        contract = self.contract
        Sijmn = self.Local.Sijmn
        Sijmm = self.Local.Sijmm
        no = self.no 
        ERIvovv = self.Local.ERIvovv 
        ERIoovv = self.Local.ERIoovv
        QL = self.Local.QL   

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
            lHamef = []
            lHamfe = []
            lHfjea = []

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

            #H a_ij m e_ij f_mm - Eqn 160
            #H a_ij m f_mm e_ij - Eqn 161
            Sijii = self.Local.Sijii
            for i in range(no):
                ii = i*no + i
                for j in range(no):
                    ij = i*no + j
                    for m in range(no):
                        mm = m*no + m
                        ijm = ij*no + m

                        #first term
                        Hamef = contract('aef,aA, eE, fF -> AEF', ERI[v,m,v,v], QL[ij], QL[ij], QL[mm])
                        Hamfe = contract('afe,aA, fF, eE -> AFE', ERI[v,m,v,v], QL[ij], QL[mm], QL[ij])
                        for n in range(no):
                            ijn = ij*no + n

                            tmp = t1[n] @ Sijmm[ijn].T
                            tmp1 = contract('ef,eE ->Ef', ERI[n,m,v,v], QL[ij])
                            tmp1 = contract('Ef, fF -> EF', tmp1, QL[mm])
                            Hamef = Hamef - contract('a,ef->aef', tmp, tmp1)

                            tmp1 = contract('fe,fF ->Fe', ERI[n,m,v,v], QL[mm])
                            tmp1 = contract('Fe, eE -> FE', tmp1, QL[ij])
                            Hamfe = Hamfe - contract('a,fe->afe', tmp, tmp1)

                        lHamef.append(Hamef)
                        lHamfe.append(Hamfe)

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

            #Hvovv_ij - part of Eqn 165 
            for ij in range(no*no):

                Hvovv = ERIvovv[ij].copy()
                for n in range(no):
                    nn = n*no + n

                    Sijnn = QL[ij].T @ QL[nn]
                    tmp = t1[n] @ Sijnn.T
                    Hvovv -= contract('a,mef->amef',tmp, ERIoovv[ij][n,:])
                lHvovv.append(Hvovv)

            #Hf{im}je{mm}a{ij} - part of Eqn 165, eqn 166 
            Sijim = self.Local.Sijim
            Sijmm = self.Local.Sijmm
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    im = i*no + m
                    iim = ii*no + m
                    ijm = ij*no + m
                    imm = im*no + m

                    Hfjea = contract('fea, fF, eE, aA -> FEA', ERI[v,j,v,v], QL[im], QL[mm], QL[ij])

                    for n in range(no):
                        imn = im * no + n

                        tmp = t1[n] @ Sijmm[imn].T
                        Hfjea -= contract('f,ea->fea', tmp, QL[mm].T @ ERI[n,j,v,v] @ QL[ij])

                    lHfjea.append(Hfjea)

            #Hf{jm}ib{ij}e{mm} - part of Eqn 165, eqn 167
            Sijmm = self.Local.Sijmm
            Sijmj = self.Local.Sijmj
            lHfibe = []
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    im = i*no + m
                    iim = ii*no + m
                    ijm = ij*no + m
                    imm = im*no + m
                    jm = j*no + m

                    Hfibe = contract('fbe, fF, bB, eE -> FBE', ERI[v,i,v,v], QL[jm], QL[ij], QL[mm])

                    for n in range(no):
                        imn = im * no + n
                        jmn = jm * no + n

                        tmp = t1[n] @ Sijmm[jmn].T
                        Hfibe -= contract('f,be->fbe', tmp, QL[ij].T @ ERI[n,i,v,v] @ QL[mm])

                    lHfibe.append(Hfibe)

            #Hf{jm}ie{mm}b{ij} - part of Eqn 165, eqn 170
            Sijmm = self.Local.Sijmm
            Sijmj = self.Local.Sijmj
            lHfieb = []
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    im = i*no + m
                    iim = ii*no + m
                    ijm = ij*no + m
                    imm = im*no + m
                    jm = j*no + m

                    Hfieb = contract('feb, fF, eE, bB -> FEB', ERI[v,i,v,v], QL[jm], QL[mm], QL[ij])

                    for n in range(no):
                        imn = im * no + n
                        jmn = jm * no + n

                        tmp = t1[n] @ Sijmm[jmn].T
                        Hfieb -= contract('f,eb->feb', tmp, QL[mm].T @ ERI[n,i,v,v] @ QL[ij])

                    lHfieb.append(Hfieb)

            #Hf{ij}ma{ij}e{mm} - part of Eqn 165, eqn 168
            Sijmm = self.Local.Sijmm
            Sijmj = self.Local.Sijmj
            lHfmae = []
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    im = i*no + m
                    iim = ii*no + m
                    ijm = ij*no + m
                    imm = im*no + m
                    jm = j*no + m

                    Hfmae = contract('fae, fF, aA, eE -> FAE', ERI[v,m,v,v], QL[ij], QL[ij], QL[mm])

                    for n in range(no):
                        imn = im * no + n
                        jmn = jm * no + n
                        ijn = ij * no + n

                        tmp = t1[n] @ Sijmm[ijn].T
                        Hfmae -= contract('f,ae->fae', tmp, QL[ij].T @ ERI[n,m,v,v] @ QL[mm])

                    lHfmae.append(Hfmae)

            #Hf{ij}me{mm}a{ij} - part of Eqn 165, eqn 169
            Sijmm = self.Local.Sijmm
            Sijmj = self.Local.Sijmj
            lHfmea = []
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    im = i*no + m
                    iim = ii*no + m
                    ijm = ij*no + m
                    imm = im*no + m
                    jm = j*no + m

                    Hfmea = contract('fea, fF, eE, aA -> FEA', ERI[v,m,v,v], QL[ij], QL[mm], QL[ij])

                    for n in range(no):
                        imn = im * no + n
                        jmn = jm * no + n
                        ijn = ij * no + n

                        tmp = t1[n] @ Sijmm[ijn].T
                        Hfmea -= contract('f,ea->fea', tmp, QL[mm].T @ ERI[n,m,v,v] @ QL[ij])

                    lHfmea.append(Hfmea)
            lHgnea = []
            lHgnae = []
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    im = i*no + m
                    mi = m*no +i
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

                        Hvovv = contract('gea, gG -> Gea', ERI[v,n,v,v], QL[im])
                        Hvovv = contract('Gea, eE -> GEa', Hvovv, QL[mn])
                        Hvovv = contract('GEa, aA -> GEA', Hvovv, QL[ii])

                        for _o in range(no):
                            oo = _o*no + _o
                            imoo = im*(no*no) + oo

                            Hvovv = Hvovv - contract('g, ea -> gea', Sijmn[imoo] @ t1[_o], QL[mn].T @ ERI[_o,n,v,v] @ QL[ii])

                        lHgnea.append(Hvovv)

                        #g_mi e_mn
                        Hvovv = contract('gae, gG -> Gae', ERI[v,n,v,v], QL[mi])
                        Hvovv = contract('Gae, aA -> GAe', Hvovv, QL[ii])
                        Hvovv = contract('GAe, eE -> GAE', Hvovv, QL[mn])

                        for _o in range(no):
                            oo = _o*no + _o
                            mioo = mi*(no*no) + oo

                            Hvovv = Hvovv - contract('g, ae -> gae', Sijmn[mioo] @ t1[_o], QL[ii].T @ ERI[_o,n,v,v] @ QL[mn])

                        lHgnae.append(Hvovv)

            lHalbc = []
            lHalcb = []
            for j in range(no):
                jj = j*no + j

                for k in range(no):
                    kk = k*no + k
                    jk = j*no + k

                    for l in range(no):
                        ll = l*no + l
                        jl = j*no + l
                        kl = k*no + l
                        jkll = jk*(no*no) + ll
                        jlll = jl*(no*no) + ll
                        jlkk = jl*(no*no) + kk
                        jjkl = jj*(no*no) + kl

                        Hvovv = contract('aA, abc -> Abc', QL[jk], ERI[v,l,v,v])
                        Hvovv = contract('bB, Abc -> ABc', QL[jj] , Hvovv)
                        Hvovv = contract('cC, ABc -> ABC', QL[kk], Hvovv)
                        for n in range(no):
                            nn = n*no + n
                            jknn = jk*(no*no) + nn

                            Hvovv = Hvovv - contract('a, bc ->abc', (Sijmn[jknn] @ t1[n]), QL[jj].T @ ERI[n,l,v,v] @ QL[kk])

                        lHalbc.append(Hvovv)
                        Hvovv = contract('aA, abc -> Abc', QL[jk], ERI[v,l,v,v])
                        Hvovv = contract('bB, Abc -> ABc', QL[kk] , Hvovv)
                        Hvovv = contract('cC, ABc -> ABC', QL[jj], Hvovv)
                        for n in range(no):
                            nn = n*no + n
                            jknn = jk*(no*no) + nn

                            Hvovv = Hvovv - contract('a, bc ->abc', (Sijmn[jknn] @ t1[n]), QL[kk].T @ ERI[n,l,v,v] @ QL[jj])

                        lHalcb.append(Hvovv)

        return lHvovv, lHvovv_ii, lHvovv_imn, lHvovv_imns, lHamef, lHamfe, lHfibe, lHfieb, lHgnea, lHgnae, lHalbc, lHalcb, lHfjea, lHfmae, lHfmea
    def build_lHooov(self, o, v, ERI, t1):
        contract = self.contract
        no = self.no 
        ERIooov = self.Local.ERIooov 
        QL = self.Local.QL 

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

            lHmine_mm = []
            lHimne_mm = []

            lHmnie = []
            lHnmie = []

            lHjmna = []
            lHmjna = []

            lHjine = []

            for i in range(no):
                for j in range(no):
                    ij = i*no + j
                    Hooov = ERIooov[ij].copy()
                    for k in range(no):
                        kk = k*no + k
                        for n in range(no):
                            nn = n*no + n

                            tmp = contract('mef, eE -> mEf', ERI[n,o,v,v].copy(), QL[ij])
                            tmp = contract('mEf, fF -> mEF', tmp, QL[kk])
                            Hooov[:,n,k,:] = Hooov[:,n,k,:] + contract('f, mef -> me', t1[k], tmp)
                    lHooov.append(Hooov)

            #Hmine and Himne - Eqn 98 and 99  
            for i in range(no):
                ii = i*no + i
                for j in range(no):
                    ij = i*no + j
                    for m in range(no):
                        for n in range(no):
                            nn = n*no + n

                            Hmine = ERIooov[ij][m,i,n].copy()

                            tmp = contract('ef,eE ->Ef', ERI[i,m,v,v], QL[ij])
                            tmp = contract('Ef, fF -> EF', tmp, QL[nn])
                            Hmine = Hmine + contract('f,ef->e', t1[n], tmp)

                            lHmine.append(Hmine)

                            Himne = ERIooov[ij][i,m,n].copy()

                            tmp = contract('ef, eE- >Ef', ERI[m,i,v,v], QL[ij])
                            tmp = contract('Ef, fF -> EF', tmp, QL[nn])
                            Himne = Himne + contract('f,ef->e', t1[n], tmp)

                            lHimne.append(Himne)

            #Hmine{mm} and Himne{mm} - Eqn 98 and 99 
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    mm = m*no + m
                    for n in range(no):
                        nn = n*no + n

                        Hmine_mm = ERIooov[mm][m,i,n].copy()

                        tmp = contract('ef,eE ->Ef', ERI[i,m,v,v], QL[mm])
                        tmp = contract('Ef, fF -> EF', tmp, QL[nn])
                        Hmine_mm = Hmine_mm + contract('f,ef->e', t1[n], tmp)

                        lHmine_mm.append(Hmine_mm)

                        Himne_mm = ERIooov[mm][i,m,n].copy()

                        tmp = contract('ef, eE- > Ef', ERI[m,i,v,v], QL[mm])
                        tmp = contract('Ef, fF -> EF', tmp, QL[nn])
                        Himne_mm = Himne_mm + contract('f,ef->e', t1[n], tmp)

                        lHimne_mm.append(Himne_mm)

            #Hmnie and Hnmie - Eqn 158 and 159 
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    for n in range(no):
                        nn = n*no + n

                        Hmnie = ERIooov[nn][m,n,i].copy()

                        tmp = contract('ef,eE ->Ef', ERI[n,m,v,v], QL[nn])
                        tmp = contract('Ef, fF -> EF', tmp, QL[ii])
                        Hmnie = Hmnie + contract('f,ef->e', t1[i], tmp)

                        lHmnie.append(Hmnie)

                        Hnmie = ERIooov[nn][n,m,i].copy()

                        tmp = contract('ef, eE- >Ef', ERI[m,n,v,v], QL[nn])
                        tmp = contract('Ef, fF -> EF', tmp, QL[ii])
                        Hnmie = Hnmie + contract('f,ef->e', t1[i], tmp)

                        lHnmie.append(Hnmie)

            #Hjmna{ij} and Hmjna{ij} - Eqn 171 and 172 
            for i in range(no):
                ii = i*no + i
                for j in range(no):
                    ij = i*no + j
                    for m in range(no):
                        for n in range(no):
                            nn = n*no + n

                            Hjmna = ERIooov[ij][j,m,n].copy()

                            tmp = contract('af,aA ->Af', ERI[m,j,v,v], QL[ij])
                            tmp = contract('Af, fF -> AF', tmp, QL[nn])
                            Hjmna = Hjmna + contract('f,af->a', t1[n], tmp)

                            lHjmna.append(Hjmna)

                            Hmjna = ERIooov[ij][m,j,n].copy()

                            tmp = contract('af, aA- >Af', ERI[j,m,v,v], QL[ij])
                            tmp = contract('Af, fF -> AF', tmp, QL[nn])
                            Hmjna = Hmjna + contract('f,af->a', t1[n], tmp)

                            lHmjna.append(Hmjna)

            #Hjine{mm} - Eqn 173
            for i in range(no):
                ii = i*no + i
                for j in range(no):
                    ij = i*no + j
                    for m in range(no):
                        mm = m*no + m
                        for n in range(no):
                            nn = n*no + n

                            Hjine = ERIooov[mm][j,i,n].copy()

                            tmp = contract('af,aA ->Af', ERI[i,j,v,v], QL[mm])
                            tmp = contract('Af, fF -> AF', tmp, QL[nn])
                            Hjine = Hjine + contract('f,af->a', t1[n], tmp)

                            lHjine.append(Hjine)

            #Hijov and Hjiov - Eqn 100 and 101
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i

                Hijov = ERIooov[ij][i,j,:,:].copy()
                Hjiov = ERIooov[ij][j,i,:,:].copy()

                for m in range(no):
                    mm = m*no + m

                    tmp = contract('eE, ef -> Ef', QL[ij], ERI[i,j,v,v])
                    tmp = contract('fF, Ef -> EF', QL[mm], tmp)
                    Hjiov[m] = Hjiov[m] + contract('f,ef->e',t1[m], tmp)

                    tmp = contract('eE, ef ->Ef', QL[ij], ERI[j,i,v,v])
                    tmp = contract('fF, Ef -> EF', QL[mm], tmp)

                    Hijov[m] = Hijov[m] + contract('f,ef->e',t1[m], tmp)
                lHijov.append(Hijov)
                lHjiov.append(Hjiov)

        return lHjiov, lHijov, lHmine, lHimne, lHmine_mm, lHimne_mm, lHmnie, lHnmie, lHjmna, lHmjna, lHjine, lHooov

    def build_lHovvo(self, o, v, ERI, L, t1, t2):
        contract = self.contract
        Sijim = self.Local.Sijim
        Sijmm = self.Local.Sijmm
        Sijii = self.Local.Sijii
        Sijmn = self.Local.Sijmn
        no = self.no 
        QL = self.Local.QL 


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
        else:
            lHovvo_mi = []
            lHovvo_im = []
            lHovvo_mj = []
            lHovvo_mm = []
            lHmvvj_mi = []

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
                        tmp2 = contract('ef, eE->Ef', ERI[i,n,v,v], QL[ii])
                        tmp3 = contract('Ef, fF->EF', tmp2, QL[mn])
                        Hovvo_mm = Hovvo_mm - contract('fb,ef->be', tmp1, tmp3)

                        tmp2 = contract('Ef, fF->EF', tmp2, QL[mm])
                        Hovvo_mm = Hovvo_mm - contract('f,b,ef->be', t1[m], tmp, tmp2)

                        #Smmnm <- Siiij  
                        tmp = t2[nm] @ Sijii[mn]
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

                    Hovvo_mi = contract('bB, be -> Be', QL[mi], ERI[j,v,v,m])
                    Hovvo_mi = contract('eE, Be -> BE', QL[ij], Hovvo_mi)

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

            #Hm v_ij v_im j - Eqn 157
            #Hm v_ij v_mi j - Eqn 158
            Sijmj = self.Local.Sijmj
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                ii = i*no + i
                jj =  j*no + j

                for m in range(no):
                    im = i*no + m
                    mi = m*no + i

                    #first term
                    Hovvo_im = contract('bB, be -> Be', QL[ij], ERI[m,v,v,j])
                    Hmvvj_mi = contract('eE, Be -> BE', QL[mi], Hovvo_im)
                    Hovvo_im = contract('eE, Be -> BE', QL[im], Hovvo_im)

                    #second term 
                    tmp = contract('bef, bB ->Bef', ERI[m,v,v,v], QL[ij])
                    tmp_mi = contract('Bef, eE -> BEf', tmp, QL[mi])
                    tmp = contract('Bef, eE ->BEf', tmp, QL[im])
                    tmp_mi = contract('BEf, fF-> BEF', tmp_mi, QL[jj])
                    tmp = contract('BEf, fF-> BEF', tmp, QL[jj])
                    Hovvo_im = Hovvo_im + contract('f,bef->be', t1[j], tmp)
                    Hmvvj_mi = Hmvvj_mi + contract('f,bef->be', t1[j], tmp_mi)

                    for n in range(no):
                        mn = m*no + n
                        nm = n*no + m
                        nn = n*no + n
                        imn = im*no + n
                        ijn = ij*no + n
                        _in = i*no + n
                        ni = n*no + i
                        jn = j*no +n
                        nj = n*no + j

                        #third term 
                        #Simnn <- Sijmm
                        tmp = Sijmm[ijn] @ t1[n]
                        tmp1 = contract ('e,eE->E', ERI[m,n,v,j], QL[im])
                        tmp1_mi = contract ('e,eE->E', ERI[m,n,v,j], QL[mi])
                        Hovvo_im = Hovvo_im - contract('b,e->be', tmp, tmp1)
                        Hmvvj_mi = Hmvvj_mi - contract('b,e->be', tmp, tmp1_mi)

                        #fourth term
                        #Sijin <- Sijim
                        tmp1 = t2[jn] @ Sijmj[ijn].T
                        tmp2 = QL[im].T @ ERI[m,n,v,v]
                        tmp2_mi = QL[mi].T @ ERI[m,n,v,v]
                        tmp3 = tmp2 @ QL[jn]
                        tmp3_mi = tmp2_mi @ QL[jn]
                        Hovvo_im = Hovvo_im - tmp1.T @ tmp3.T
                        Hmvvj_mi = Hmvvj_mi - tmp1.T @ tmp3_mi.T

                        #fifth term
                        tmp1 = tmp2 @ QL[jj]
                        tmp1_mi = tmp2_mi @ QL[jj]
                        Hovvo_im = Hovvo_im - contract('f,b,ef->be', t1[j], tmp, tmp1)
                        Hmvvj_mi = Hmvvj_mi - contract('f,b,ef->be', t1[j], tmp, tmp1_mi)

                        #sixth term
                        #Sminm <- Sijnj
                        tmp = t2[nj] @ Sijmj[ijn].T
                        tmp1 = QL[im].T @ L[m,n,v,v]
                        tmp1_mi = QL[mi].T @ L[m,n,v,v]
                        tmp1 = tmp1 @ QL[nj]
                        tmp1_mi = tmp1_mi @ QL[nj]
                        Hovvo_im = Hovvo_im + tmp.T @ tmp1.T
                        Hmvvj_mi = Hmvvj_mi + tmp.T @ tmp1_mi.T
                    lHovvo_im.append(Hovvo_im)
                    lHmvvj_mi.append(Hmvvj_mi)

            #Hj v_mi v_ij o - Eqn 103
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                jj =  j*no + j

                for m in range(no):
                    mi = m*no + i
                    mm = m*no + m

                    Hovvo_mi = contract('bB, be -> Be', QL[mi], ERI[j,v,v,m])
                    Hovvo_mi = contract('eE, Be -> BE', QL[ij], Hovvo_mi)

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

                    Hovvo_mj = contract('bB, be -> Be', QL[mj], ERI[i,v,v,m])
                    Hovvo_mj = contract('eE, Be -> BE', QL[ij], Hovvo_mj)

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

            lHovvo_ni = []
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    for n in range(no):
                        _in = i*no + n
                        nn = n*no + n

                        Hovvo = QL[_in].T @ ERI[m,v,v,n] @ QL[ii]
                        ERIovvv = contract('fae, fF -> Fae', ERI[m,v,v,v], QL[_in])
                        ERIovvv = contract('Fae, aA -> FAe', ERIovvv, QL[ii])
                        ERIovvv = contract('FAe, eE -> FAE', ERIovvv, QL[nn])
                        Hovvo = Hovvo + contract('e, fae -> fa', t1[n], ERIovvv)

                        for _o in range(no):
                            oo = _o*no + _o
                            _no = n*no + _o
                            on = _o*no + n
                            inoo = _in*(no*no) + oo
                            inno = _in*(no*no) + _no
                            inon = _in*(no*no) + on

                            Hovvo = Hovvo - contract('f, a -> fa', Sijmn[inoo] @ t1[_o], ERI[m,_o, v, n] @ QL[ii])
                            Hovvo = Hovvo - contract('ef, ae -> fa', t2[_no] @ Sijmn[inno].T, QL[ii].T @ ERI[m,_o,v,v] @ QL[_no])
                            tmp = contract('f, ae -> fae', Sijmn[inoo] @ t1[_o], QL[ii].T @ ERI[m,_o,v,v] @ QL[nn])
                            Hovvo = Hovvo - contract('e, fae -> fa', t1[n], tmp)
                            Hovvo = Hovvo + contract('ef, ae -> fa', t2[on] @ Sijmn[inon].T , QL[ii].T @ L[m,_o,v,v] @ QL[on])
                        lHovvo_ni.append(Hovvo)

        return lHovvo_mi, lHovvo_mj, lHovvo_mm, lHovvo_ni

    def build_lHovov(self, o, v, ERI, t1, t2):
        Sijii = self.Local.Sijii
        Sijmm = self.Local.Sijmm
        Sijim = self.Local.Sijim
        Sijmn = self.Local.Sijmn
        contract = self.contract
        no = self.no 
        ERIovov = self.Local.ERIovov 
        ERIooov = self.Local.ERIooov 
        QL = self.Local.QL 

        if self.ccwfn.model == 'CCD':
            lHovov_mi = []
            lHovov_mj = []
            lHovov_mm = []

            #Hiv_mj ov_ij
            for ij in range(no*no):
                i = ij // no
                j = ij % no

                for m in range(no):
                    mj = m*no + j

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
            lHovov_im = []
            lHovov_mi = []
            lHovov_mj = []
            lHovov_mm = []

            #Hiv_mm ov_ii - Eqn 105   
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    mm = m* no + m

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

                    Hovov_mj = contract('be, bB-> Be', ERI[i,v,m,v], QL[mj])
                    Hovov_mj = contract('Be, eE-> BE', Hovov_mj, QL[ij])

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

                    Hovov_mi = contract('be, bB-> Be', ERI[j,v,m,v], QL[mi])
                    Hovov_mi = contract('Be, eE-> BE', Hovov_mi, QL[ij])

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

            #Hmv{ij}jv{im} -   
            Sijjn = self.Local.Sijjn
            for ij in range(no*no):
                i = ij // no
                j = ij % no
                jj = j*no + j
                for m in range(no):
                    mm = m*no + m
                    im = i*no + m

                    #first term
                    #Hovov_im = contract('ae, aA, eE-> AE', ERI[m,v,j,v], QL[ij], QL[im])
                    Hovov_im = contract('ae, aA-> Ae', ERI[m,v,j,v], QL[ij])
                    Hovov_im = contract('Ae, eE-> AE', Hovov_im, QL[im])

                    #second term 
                    #tmp = contract('aef, aA, eE, fF-> AEF', ERI[v,m,v,v], QL[ij], QL[im], QL[jj])
                    tmp = contract('aef, aA-> Aef', ERI[v,m,v,v], QL[ij])
                    tmp = contract('Aef, eE-> AEf', tmp, QL[im])
                    tmp = contract('AEf, fF-> AEF', tmp, QL[jj])
                    Hovov_im = Hovov_im + contract('f,aef->ae', t1[j], tmp)

                    for n in range(no):
                        nn = n*no + n
                        jn = j*no + n
                        ijn = ij*no + n

                        #third term
                        #Sijnn <- Sijmm
                        tmp = Sijmm[ijn] @ t1[n]
                        tmp1 = contract('e,eE->E', ERI[m,n,j,v], QL[im])
                        Hovov_im = Hovov_im - contract('a,e->ae', tmp, tmp1)

                        #fourth term
                        #Smimn <- Sijim
                        tmp1 = t2[jn] @ Sijjn[ijn].T
                        tmp2 = QL[im].T @ ERI[n,m,v,v]
                        tmp3 = tmp2 @ QL[jn]
                        Hovov_im = Hovov_im - tmp1.T @ tmp3.T

                        #fifth term
                        tmp1 = tmp2 @ QL[jj]
                        Hovov_im = Hovov_im - contract('f,a,ef->ae',t1[j], tmp, tmp1)
                    lHovov_im.append(Hovov_im)

            lHovov_ni = []
            for i in range(no):
                ii = i*no + i
                for m in range(no):
                    for n in range(no):
                        ni = n*no + i
                        nn = n*no + n

                        Hovov = QL[ni].T @ ERI[m,v,n,v] @ QL[ii]
                        ERIvovv = contract('fae, fF -> Fae', ERI[v,m,v,v], QL[ni])
                        ERIvovv = contract('Fae, aA -> FAe', ERIvovv, QL[ii])
                        ERIvovv = contract('FAe, eE -> FAE', ERIvovv, QL[nn])
                        Hovov = Hovov + contract('e, fae -> fa', t1[n], ERIvovv)

                        for _o in range(no):
                            oo = _o*no + _o
                            _no = n*no + _o
                            nioo = ni*(no*no) + oo
                            nino = ni*(no*no) + _no
                            Hovov = Hovov - contract('f, a-> fa', Sijmn[nioo] @ t1[_o], ERI[m,_o,n,v] @ QL[ii])
                            Hovov = Hovov - contract('ef, ae -> fa', t2[_no] @Sijmn[nino].T, QL[ii].T @ ERI[_o,m,v,v] @ QL[_no])
                            tmp = contract('f, ae -> fae', Sijmn[nioo] @ t1[_o], QL[ii].T @ ERI[_o,m,v,v] @ QL[nn])
                            Hovov = Hovov - contract('e, fae -> fa', t1[n], tmp)
                        lHovov_ni.append(Hovov)

        return lHovov_mi, lHovov_mj, lHovov_mm, lHovov_ni

    def build_lHvvvo(self, o, v, ERI, L, t1, t2, Hov, Hvvvv, Hvvvv_im):
        Sijmj = self.Local.Sijmj
        Sijmm = self.Local.Sijmm
        Sijmn = self.Local.Sijmn
        contract = self.contract
        no = self.no 
        ERIvvvo = self.Local.ERIvvvo 
        ERIoovo = self.Local.ERIoovo 
        ERIvoov = self.Local.ERIvoov
        ERIvovo = self.Local.ERIvovo
        ERIoovv = self.Local.ERIoovv
        Loovv = self.Local.Loovv
        QL = self.Local.QL 

        if self.ccwfn.model == 'CCD':
            lHvvvo_im = []

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

                    #first term
                    #Hvvvo_im = contract('abe, aA, bB, eE->ABE', ERI[v,v,v,m], QL[im], QL[im], QL[ii])   
                    Hvvvo_im = contract('abe, aA-> Abe', ERI[v,v,v,m], QL[im])
                    Hvvvo_im = contract('Abe, bB-> ABe', Hvvvo_im, QL[im])
                    Hvvvo_im = contract('ABe, eE-> ABE', Hvvvo_im, QL[ii])

                    #third term
                    Hvvvo_im = Hvvvo_im + contract('f,abef->abe', t1[m], Hvvvv_im[im])

                    for n in range(no):
                        nm = n*no + m
                        nn = n*no + n
                        mn = m*no + n
                        imn = im*no + n

                        #second term 
                        #Simnm <- Sijmj
                        tmp = t2[nm] @ Sijmj[imn].T
                        tmp1 = Sijmj[imn] @ tmp
                        Hvvvo_im = Hvvvo_im - contract('e, ab->abe', Hov[ii][n], tmp1)

                        #sixth
                        #tmp1 = contract('bfe,bB,fF,eE->BFE', ERI[v,n,v,v], QL[im], QL[mn], QL[ii]) 
                        tmp1 = contract('bfe, bB-> Bfe', ERI[v,n,v,v], QL[im])
                        tmp1 = contract('Bfe, fF-> BFe', tmp1, QL[mn])
                        tmp1 = contract('BFe, eE-> BFE', tmp1, QL[ii])
                        #Simmn <- Sijjm
                        tmp2 = t2[mn] @ Sijmj[imn].T
                        Hvvvo_im = Hvvvo_im - contract('fa,bfe->abe', tmp2, tmp1)

                        #seventh
                        #tmp1 = contract('aef,aA,eE,fF->AEF', ERI[v,n,v,v], QL[im], QL[ii], QL[mn])
                        tmp1 = contract('aef, aA-> Aef', ERI[v,n,v,v], QL[im])
                        tmp1 = contract('Aef, eE-> AEf', tmp1, QL[ii])
                        tmp1 = contract('AEf, fF-> AEF', tmp1, QL[mn])
                        Hvvvo_im = Hvvvo_im - contract('fb,aef->abe', tmp2, tmp1)

                        #eight
                        #tmp1 = contract('aef,aA,eE,fF->AEF', L[v,n,v,v], QL[im], QL[ii], QL[nm]) 
                        tmp1 = contract('aef, aA-> Aef', L[v,n,v,v], QL[im])
                        tmp1 = contract('Aef, eE-> AEf', tmp1, QL[ii])
                        tmp1 = contract('AEf, fF-> AEF', tmp1, QL[nm])
                        Hvvvo_im = Hvvvo_im + contract('fb,aef->abe',tmp, tmp1)

                        #Simnn <- Sijmm 
                        #ninth term
                        tmp = Sijmm[imn] @ t1[n]
                        Hvvvo_im = Hvvvo_im - contract('b,ae->abe', tmp, ltmp_im[imn])

                        #tenth term
                        Hvvvo_im = Hvvvo_im - contract('a,be->abe', tmp, ltmp1_im[imn])

                        for k in range(no):
                            kn = k*no + n
                            kk = k*no + k
                            nn = n*no + n
                            imnk = imn*no + k
                            imk = im*no + k

                            #fourth term
                            #Simkn <- Sijmn 
                            tmp = Sijmn[imnk] @ t2[kn]
                            tmp = tmp @ Sijmn[imnk].T
                            tmp1 = QL[ii].T @ ERI[k,n,v,m]
                            Hvvvo_im = Hvvvo_im + contract('ab,e->abe',tmp, tmp1)

                            #fifth term
                            #Simkk <- Sijmm 
                            #Simnn <- Sijmm  
                            tmp = Sijmm[imk] @ t1[k]
                            tmp2 = Sijmm[imn] @ t1[n]
                            Hvvvo_im = Hvvvo_im + contract('a,b,e->abe', tmp, tmp2, tmp1)

                    lHvvvo_im.append( Hvvvo_im)

            #terms for Hv_ij v_ij v_ii j - Eqn 151 and Eqn 152
            ltmp_ij = []
            ltmp1_ij = []
            for i in range(no):
                ii = i*no + i
                for j in range(no):
                    ij = i*no + j
                    for m in range(no):
                        jm = j*no + m
                        mj = m*no + j

                        #Eqn 151 first term
                        #tmp = contract('ae,aA,eE->AE', ERI[v,m,v,j], QL[ij], QL[ii])
                        tmp_ij = contract('ae, aA-> Ae', ERI[v,m,v,j], QL[ij])
                        tmp_ij = contract('Ae, eE->AE', tmp_ij, QL[ii])

                        #Eqn 152 first term 
                        #tmp1 = contract('be,bB,eE->BE', ERI[v,m,j,v], QL[ij], QL[ii])
                        tmp1_ij = contract('be, bB-> Be', ERI[v,m,j,v], QL[ij])
                        tmp1_ij = contract('Be, eE-> BE', tmp1_ij, QL[ii])

                        for n in range(no):
                            jn = j *no + n
                            nj = n*no + j
                            ijn = ij*no + n

                            #Eqn 151 second term
                            tmp = t2[jn] @ Sijmj[ijn].T
                            #tmp1 = contract('fe,fF,eE->FE', ERI[n,k,v,v], QL[jn], QL[ii])
                            tmp1 = contract('fe, fF-> Fe', ERI[m,n,v,v], QL[jn])
                            tmp1 = contract('Fe, eE-> FE', tmp1, QL[ii])
                            tmp_ij = tmp_ij - contract('fa,fe->ae', tmp, tmp1)

                            #Eqn 152 second term
                            #tmp1 = contract('ef, eE-> Ef', ERI[n,k,v,v], QL[ii], QL[jn])
                            tmp1 = contract('ef, eE-> Ef', ERI[m,n,v,v], QL[ii])
                            tmp1 = contract('Ef, fF-> EF', tmp1, QL[jn])
                            tmp1_ij = tmp1_ij - contract('fb,ef->be', tmp, tmp1)

                            #Eqn 152 third term
                            #Simkm <- Sijmj
                            tmp = t2[nj] @ Sijmj[ijn].T
                            #tmp1 = contract('ef,eE,fF->EF', L[n,k,v,v], QL[ii], QL[nj])
                            tmp1 = contract('ef, eE-> Ef', L[m,n,v,v], QL[ii])
                            tmp1 = contract('Ef, fF-> EF', tmp1, QL[nj])
                            tmp1_ij = tmp1_ij + contract('fb,ef->be', tmp, tmp1)

                        ltmp_ij.append(tmp_ij)
                        ltmp1_ij.append(tmp1_ij)

            #Hv_ij v_ij v_ii j - Eqn 108
            lHvvvo_ij = []
            for i in range(no):
                ii = i*no + i
                for j in range(no):
                    ij = i*no + j
                    ji = j*no + i
                    jj = j*no + j

                    #first term
                    #Hvvvo_ij = contract('abe, aA, bB, eE->ABE', ERI[v,v,v,j], QL[ij], QL[ij], QL[ii])
                    Hvvvo_ij = contract('abe, aA-> Abe', ERI[v,v,v,j], QL[ij])
                    Hvvvo_ij = contract('Abe, bB-> ABe', Hvvvo_ij, QL[ij])
                    Hvvvo_ij = contract('ABe, eE-> ABE', Hvvvo_ij, QL[ii])

                    #third term 
                    Hvvvo_ij = Hvvvo_ij + contract('f,abef->abe', t1[j], self.Hvvvv_ij[ij])

                    for m in range(no):
                        mj = m*no + j
                        mm = m*no + mm
                        jm = j*no + m
                        ijm = ij*no + m

                        #second term
                        tmp = t2[mj] @ Sijmj[ijm].T
                        tmp1 = Sijmj[ijm] @ tmp
                        Hvvvo_ij = Hvvvo_ij - contract('e, ab->abe', Hov[ii][m], tmp1)

                        #sixth term
                        #Sijmj same as Sijjm due to mj == jm
                        #tmp1 = contract('bfe,bB,fF,eE->BFE', ERI[v,m,v,v], QL[ij], QL[jm], QL[ii])
                        tmp1 = contract('bfe, bB-> Bfe', ERI[v,m,v,v], QL[ij])
                        tmp1 = contract('Bfe, fF-> BFe', tmp1, QL[jm])
                        tmp1 = contract('BFe, eE-> BFE', tmp1, QL[ii])
                        tmp2 = t2[jm] @ Sijmj[ijm].T
                        Hvvvo_ij = Hvvvo_ij - contract('fa,bfe->abe', tmp2, tmp1)

                        #seventh                
                        #tmp1 = contract('aef,aA,eE,fF->AEF', ERI[v,m,v,v], QL[ij], QL[ii], QL[jm])
                        tmp1 = contract('aef, aA-> Aef', ERI[v,m,v,v], QL[ij])
                        tmp1 = contract('Aef, eE-> AEf', tmp1, QL[ii])
                        tmp1 = contract('AEf, fF-> AEF', tmp1, QL[jm])
                        Hvvvo_ij = Hvvvo_ij - contract('fb,aef->abe', tmp2, tmp1)

                        #eight
                        #tmp1 = contract('aef,aA,eE,fF->AEF', L[v,m,v,v], QL[ij], QL[ii], QL[mj])
                        tmp1 = contract('aef, aA-> Aef', L[v,m,v,v], QL[ij])
                        tmp1 = contract('Aef, eE-> AEf', tmp1, QL[ii])
                        tmp1 = contract('AEf, fF-> AEF', tmp1, QL[mj])
                        Hvvvo_ij = Hvvvo_ij + contract('fb,aef->abe',tmp, tmp1)

                        #ninth
                        tmp = Sijmm[ijm] @ t1[m]
                        Hvvvo_ij = Hvvvo_ij - contract('b,ae->abe', tmp, ltmp_ij[ijm])

                        #tenth
                        Hvvvo_ij = Hvvvo_ij - contract('a,be->abe', tmp, ltmp1_ij[ijm])

                        for n in range(no):
                            nm = n*no + m
                            nn = n*no + n
                            ijmn = ijm*no + n
                            ijn = ij*no + n

                            #fourth
                            tmp = Sijmn[ijmn] @ t2[nm]
                            tmp = tmp @ Sijmn[ijmn].T
                            tmp1 = QL[ii].T @ ERI[n,m,v,j]
                            Hvvvo_ij = Hvvvo_ij + contract('ab,e->abe',tmp, tmp1)

                            #fifth
                            tmp = Sijmm[ijn] @ t1[n]
                            tmp2 = Sijmm[ijm] @ t1[m]
                            Hvvvo_ij = Hvvvo_ij + contract('a,b,e->abe', tmp, tmp2, tmp1)
                    lHvvvo_ij.append(Hvvvo_ij)

        return lHvvvo_im

    def build_lHovoo(self, o, v, ERI, L, t1, t2, Hov, Hoooo):
        Sijim = self.Local.Sijim
        Sijmj = self.Local.Sijmj
        Sijmm = self.Local.Sijmm
        contract = self.contract
        no = self.no  
        ERIovvv = self.Local.ERIovvv
        ERIooov = self.Local.ERIooov 
        Looov = self.Local.Looov 
        QL = self.Local.QL 

        if self.ccwfn.model =='CCD':
           lHovoo_mn = []

        else:
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

                        tmp = contract('bef, bB-> Bef', ERI[i,v,v,v], QL[mn])
                        tmp = contract('Bef, eE-> BEf', tmp, QL[mm])
                        tmp = contract('BEf, fF-> BEF', tmp, QL[nn])
                        Hovoo_mn = Hovoo_mn + contract('e, f, bef-> b', t1[m], t1[n], tmp)

                        tmp_mn = contract('be, bB->Be', ERI[i,v,m,v], QL[mn])
                        tmp_mn = contract('Be, eE-> BE', tmp_mn, QL[nn])

                        for k in range(no):
                            mk = m*no + k
                            mnk = mn*no + k

                            #Smnmk <- Sijim 
                            tmp = t2[mk] @ Sijim[mnk].T
                            tmp1 = contract('fe, fF-> Fe', ERI[i,k,v,v], QL[mk])
                            tmp1 = contract('Fe, eE-> FE', tmp1, QL[nn])
                            tmp_mn = tmp_mn - contract('fb, fe-> be', tmp, tmp1)

                        Hovoo_mn = Hovoo_mn + contract('e, be-> b', t1[n], tmp_mn)

                        tmp_mn = contract('be, bB-> Be', ERI[v,i,n,v], QL[mn])
                        tmp_mn = contract('Be, eE-> BE', tmp_mn, QL[mm])

                        for k in range(no):
                            kn = k*no + n
                            nk = n*no + k
                            mnk = mn*no + k

                            #Smnnk <- Sijjm
                            tmp = t2[nk] @ Sijmj[mnk].T
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

            lHovoo_ij = []
            #Hmv{ij}ij - Eqn 153            
            for i in range(no):
                ii = i*no + i
                for j in range(no):
                    ij = i*no + j
                    jj = j*no + j
                    for m in range(no):

                        #first term
                        Hovoo_ij = contract('b, bB-> B', ERI[m,v,i,j], QL[ij])

                        #second term
                        Hovoo_ij = Hovoo_ij + contract('c, cb-> b', Hov[ij][m], t2[ij])

                        #fourth term 
                        Hovoo_ij = Hovoo_ij + contract('cf, bcf-> b', t2[ij], ERIovvv[ij][m])

                        #fifth term 
                        tmp = contract('bef, bB-> Bef', ERI[m,v,v,v], QL[ij])
                        tmp = contract('Bef, eE-> BEf', tmp, QL[ii])
                        tmp = contract('BEf, fF-> BEF', tmp, QL[jj])
                        Hovoo_ij = Hovoo_ij + contract('c, f, bcf-> b', t1[i], t1[j], tmp)

                        #ninth term - X term
                        tmp_ij = contract('bc, bB->Bc', ERI[m,v,i,v], QL[ij])
                        tmp_ij = contract('Bc, cC-> BC', tmp_ij, QL[jj])

                        for n in range(no):
                            _in = i*no + n
                            ijn = ij*no + n

                            tmp = t2[_in] @ Sijim[ijn].T
                            tmp1 = contract('fc, fF-> Fc', ERI[m,n,v,v], QL[_in])
                            tmp1 = contract('Fc, cC-> FC', tmp1, QL[jj])
                            tmp_ij = tmp_ij - contract('fb, fc-> bc', tmp, tmp1)

                        #ninth term
                        Hovoo_ij = Hovoo_ij + contract('c, bc-> b', t1[j], tmp_ij)

                        #tenth term - X term
                        tmp_ij = contract('bc, bB-> Bc', ERI[v,m,j,v], QL[ij])
                        tmp_ij = contract('Bc, cC-> BC', tmp_ij, QL[ii])

                        for n in range(no):
                            jn = j*no + n
                            nj = n*no + j
                            ijn = ij*no + n

                            #Sijnj <- Sijmj
                            tmp = t2[jn] @ Sijmj[ijn].T
                            tmp1 = contract('cf, cC-> Cf', ERI[m,n,v,v], QL[ii])
                            tmp1 = contract('Cf, fF-> CF', tmp1, QL[jn])
                            tmp_ij = tmp_ij - contract('fb, cf-> bc', tmp, tmp1)

                            #Sijnj <- Sijmj
                            tmp = t2[nj] @ Sijmj[ijn].T
                            tmp1 = contract('cf, cC-> Cf', L[m,n,v,v], QL[ii])
                            tmp1 = contract('Cf, fF-> CF', tmp1, QL[nj])
                            tmp_ij = tmp_ij + contract('fb, cf-> bc', tmp, tmp1)

                        #tenth term
                        Hovoo_ij = Hovoo_ij + contract('c, bc-> b', t1[i], tmp_ij)

                        for n in range(no):
                            nn = n*no + n
                            _in = i*no + n
                            jn = j*no + n
                            nj = n*no + j
                            ijn = ij*no + n

                            #third term 
                            #Smnkk <- Sijmm
                            tmp = Sijmm[ijn] @ t1[n]
                            Hovoo_ij = Hovoo_ij - (tmp * Hoooo[m,n,i,j])

                            #sixth term
                            #Sijin <- Sijim
                            tmp = t2[_in] @ Sijim[ijn].T
                            Hovoo_ij = Hovoo_ij - contract('cb, c-> b', tmp, ERIooov[_in][n,m,j])

                            #seventh term
                            #Sijjn <- Sijjm
                            tmp = t2[jn] @ Sijmj[ijn].T
                            Hovoo_ij = Hovoo_ij - contract('cb, c-> b', tmp, ERIooov[jn][m,n,i])

                            #eigth term
                            #Sijnj <- Sijmj
                            tmp = t2[nj] @ Sijmj[ijn].T
                            Hovoo_ij = Hovoo_ij + contract('cb, c-> b', tmp, Looov[nj][m,n,i])
                        lHovoo_ij.append( Hovoo_ij)
        return lHovoo_mn, lHovoo_ij
