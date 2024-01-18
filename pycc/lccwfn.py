#import time
from time import process_time
import numpy as np
from opt_einsum import contract


class lccwfn(object):
    """
    A PNO-CC object.

    Attributes
    ----------
    o : Numpy slice
        occupied orbital subspace
    v : Numpy slice
        virtual orbital subspace
    no: int
        number of (active) occupied orbitals
    nv: int
        number of virtual orbitals
    H: PyCC Hamiltonian object
       Hamiltonian object with integrals, Fock matrices, and orbital coefficients
    Local: PyCC Local object
       Local object with transformation matrices, local Fock matrices and integrals, etc.

    Parameters
    ----------
    local: string
           type of local calculation ("PAO", "PNO", etc.)
    model: string
           type of CC calculation ("CCD","CCSD", etc.)
    eref: float
          reference energy (typically HF/SCF energy)

    Methods
    -------
    solve_lcc()
        Solves the local CC T amplitude equations
    local_residuals()
        Computes the local T1 and T2 residuals for a given set of amplitudes and Fock operator

    Notes
    -----
    To do:
    (1) need DIIS extrapolation
    (2) time table for each intermediate?
    """

    def __init__(self, o, v, no, nv, H, local, model, eref, Local):
        self.o = o
        self.v = v
        self.no = no
        self.nv = nv
        self.H = H
        self.local = local
        self.model = model
        self.eref = eref
        self.Local = Local
        self.QL = self.Local.QL
        self.dim = self.Local.dim
        self.eps = self.Local.eps
	
        t1 = []
        t2 = []

        for i in range(self.no):
            ii = i*self.no + i

            t1.append(np.zeros((self.Local.dim[ii])))

            for j in range(self.no):
                ij = i*self.no + j

                t2.append(-1* self.Local.ERIoovv[ij][i,j] / (self.eps[ij].reshape(1,-1) + self.eps[ij].reshape(-1,1)
                - self.H.F[i,i] - self.H.F[j,j]))

        self.t1 = t1
        self.t2 = t2

    def solve_lcc(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8,start_diis=1):
        """
        Parameters
        ----------

        e_conv : float
            convergence condition for correlation energy (default if 1e-7)
        r_conv : float
            convergence condition for wave function rmsd (default if 1e-7)
        maxiter : int
            maximum allowed number of iterations of the CC equations (default is 100)

        Returns
        -------
        elcc: float
            lCC correlation energy
        """
        lcc_tstart = process_time()

        #initialize variables for timing each function
        self.fae_t = 0
        self.fme_t = 0
        self.fmi_t = 0
        self.wmnij_t = 0
        self.zmbij_t = 0
        self.wmbej_t = 0
        self.wmbje_t = 0
        self.tau_t = 0
        self.r1_t = 0
        self.r2_t = 0
       	self.energy_t = 0
	
        #ldiis = helper_ldiis(self.t1, self.t2, max_diis)

        elcc = self.lcc_energy(self.Local.Fov,self.Local.Loovv,self.t1, self.t2)

        print("CC Iter %3d: lCC Ecorr = %.15f dE = % .5E MP2" % (0,elcc,-elcc))

        for niter in range(1, maxiter+1):

            elcc_last = elcc

            r1, r2 = self.local_residuals(self.t1, self.t2)

            rms = 0
            rms_t1 = 0
            rms_t2 = 0

            for i in range(self.no):

                ii = i*self.no + i

                self.t1[i] -= r1[i]/(self.Local.eps[ii].reshape(-1,) - self.H.F[i,i])

                rms_t1 += contract('Z,Z->',r1[i],r1[i])

                for j in range(self.no):
                    ij = i*self.no + j

                    self.t2[ij] -= r2[ij]/(self.eps[ij].reshape(1,-1) + self.eps[ij].reshape(-1,1)
                    - self.H.F[i,i] - self.H.F[j,j])

                    rms_t2 += contract('ZY,ZY->',r2[ij],r2[ij])

            rms = np.sqrt(rms_t1 + rms_t2)
            elcc = self.lcc_energy(self.Local.Fov,self.Local.Loovv,self.t1, self.t2)
            ediff = elcc - elcc_last
            print("lCC Iter %3d: lCC Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, elcc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nlCC has converged in %.3f seconds.\n" % (process_time() - lcc_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                print("E(%s) = %20.15f" % (self.local + "-" + self.model, elcc))
                print("E(TOT)  = %20.15f" % (elcc + self.eref))
                self.elcc = elcc
                print('Time table for intermediates')
                print("Fae = %6.6f" % self.fae_t)
                print("Fme = %6.6f" % self.fme_t)
                print("Fmi = %6.6f" % self.fmi_t)
                print("Wmnij = %6.6f" % self.wmnij_t)
                print("Zmbij = %6.6f" % self.zmbij_t)
                print("Wmbej = %6.6f" % self.wmbej_t)
                print("Wmbje = %6.6f" % self.wmbje_t)
                print("Tau_t = %6.6f" % self.tau_t)
                print("r1_t = %6.6f" % self.r1_t)
                print("r2_t = %6.6f" % self.r2_t)
                print("Energy_t = %6.6f" % self.energy_t)
                return elcc
      
            #ldiis.add_error_vector(self.t1,self.t2)
            #if niter >= start_diis:
                #self.t1, self.t2 = ldiis.extrapolate(self.t1, self.t2)

    def local_residuals(self, t1, t2):
        """
        Constructing the two- and four-index intermediates
        Then evaluating the singles and doubles residuals, storing them in a list of length occ (single) and length occ*occ (doubles)

        To do
        ------
        """
        o = self.o
        v = self.v
        F = self.H.F
        L = self.H.L
        ERI = self.H.ERI

        Fae = []
        Fme = []
        Wmbej = []
        Wmbje = []
        Wmbie = []
        Zmbij = []
        r1 = []
        r2 = []

        Fae = self.build_Fae(Fae, L, self.Local.Fvv, self.Local.Fov, self.Local.Sijmm, self.Local.Sijmn, t1, t2)
        Fmi = self.build_Fmi(o, F, L, self.Local.Fov, self.Local.Looov, self.Local.Loovv, t1, t2)
        Fme = self.build_Fme(Fme, L, self.Local.Fov, t1)
        Wmnij = self.build_Wmnij(o, ERI, self.Local.ERIooov, self.Local.ERIoovo, self.Local.ERIoovv, t1, t2)
        Zmbij = self.build_Zmbij(Zmbij, ERI, self.Local.ERIovvv, t1, t2)
        Wmbej = self.build_Wmbej(Wmbej, ERI, L, self.Local.ERIoovo, self.Local.Sijnn, self.Local.Sijnj, self.Local.Sijjn, t1, t2)
        Wmbje, Wmbie = self.build_Wmbje(Wmbje, Wmbie, ERI, self.Local.ERIooov, self.Local.Sijnn, self.Local.Sijin, self.Local.Sijjn, t1, t2)

        r1 = self.r_T1(r1, self.Local.Fov , ERI, L, self.Local.Loovo, self.Local.Sijmm, self.Local.Sijim, self.Local.Sijmn,  
        t1, t2, Fae, Fmi, Fme)
        r2 = self.r_T2(r2, ERI, self.Local.ERIoovv, self.Local.ERIvvvv, self.Local.ERIovoo, self.Local.Sijmm, self.Local.Sijim, 
        self.Local.Sijmj, self.Local.Sijnn, self.Local.Sijmn, t1, t2, Fae ,Fmi, Fme, Wmnij, Zmbij, Wmbej, Wmbje, Wmbie)

        return r1, r2   
    
    def build_Fae(self, Fae_ij, L, Fvv, Fov, Sijmm, Sijmn, t1, t2):
        fae_start = process_time()
        o = self.o
        v = self.v
        QL = self.QL
        
        if self.model == 'CCD':
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no

                Fae = Fvv[ij].copy()

                for m in range(self.no):
                    for n in range(self.no):
                        mn = m *self.no +n
                        ijmn = ij*(self.no**2) + mn

                        tmp = Sijmn[ijmn] @ t2[mn]
                        tmp1 = QL[ij].T @ L[m,n,v,v]
                        tmp1 = tmp1 @ QL[mn]
                        Fae -= tmp @ tmp1.T

                Fae_ij.append(Fae)
        else:     
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no

                Fae = Fvv[ij].copy()

                for m in range(self.no):
                    mm = m*self.no + m
                    ijm = ij*(self.no) + m

                    tmp = Sijmm[ijm] @ t1[m]
                    Fae -= 0.5* contract('e,a->ae',Fov[ij][m],tmp)

                    tmp1 = contract('abc,aA->Abc',L[m,v,v,v], QL[ij])
                    tmp1 = contract('Abc,bB->ABc',tmp1, QL[mm])
                    tmp1 = contract('ABc,cC->ABC',tmp1, QL[ij])
                    Fae += contract('F,aFe->ae',t1[m],tmp1)

                    for n in range(self.no):
                        mn = m *self.no +n
                        nn = n*self.no + n
                        ijmn = ij*(self.no**2) + mn

                        tmp2 = Sijmn[ijmn] @ t2[mn]
                        tmp3_0 = QL[ij].T @ L[m,n,v,v]
                        tmp3_1 = tmp3_0 @ QL[mn]
                        Fae -= tmp2 @ tmp3_1.T

                        tmp4 = tmp3_0 @ QL[nn]
                        Fae -= 0.5 *contract('a,F,eF->ae', tmp, t1[n], tmp4)

                Fae_ij.append(Fae)
        fae_end = process_time()
        self.fae_t += fae_end - fae_start
        return Fae_ij

    def build_Fmi(self, o, F, L, Fov, Looov, Loovv, t1, t2):
        fmi_start = process_time()
        v = self.v
        QL = self.QL

        Fmi = F[o,o].copy()

        if self.model == 'CCD':
            for j in range(self.no):
               for n in range(self.no):
                   jn = j*self.no + n

                   Fmi[:,j] += contract('EF,mEF->m',t2[jn],Loovv[jn][:,n,:,:])
        else:
            for j in range(self.no):
                jj = j*self.no +j
                for n in range(self.no):
                   jn = j*self.no + n
                   nn = n*self.no + n

                   Fmi[:,j] += 0.5 * contract('e,me->m', t1[j], Fov[jj])
                   Fmi[:,j] += contract('e,me->m',t1[n],Looov[nn][:,n,j])
                   Fmi[:,j] += contract('EF,mEF->m',t2[jn],Loovv[jn][:,n,:,:])

                   tmp = contract('mab,aA->mAb', L[o,n,v,v],QL[jj])
                   tmp = contract('mAb,bB->mAB', tmp, QL[nn])
                   Fmi[:,j] += 0.5 * contract('E,F,mEF->m',t1[j], t1[n], tmp)

        fmi_end = process_time()
        self.fmi_t += fmi_end - fmi_start
        return Fmi

    def build_Fme(self, Fme_ij, L, Fov, t1):
        fme_start = process_time()
        QL = self.QL
        v = self.v

        if self.model == 'CCD':
            return 
        else:
            for ij in range(self.no*self.no):

                Fme = Fov[ij].copy()

                for m in range(self.no):
                    for n in range(self.no):
                        nn = n*self.no + n

                        tmp = QL[ij].T @ L[m,n,v,v]
                        tmp = tmp @ QL[nn]
                        Fme[m] += t1[n] @ tmp.T

                Fme_ij.append(Fme)

        fme_end = process_time()
        self.fme_t += fme_end - fme_start
        return Fme_ij

    def build_Wmnij(self, o, ERI, ERIooov, ERIoovo, ERIoovv, t1, t2):
        wmnij_start = process_time()
        v = self.v
        QL = self.Local.QL

        Wmnij = ERI[o,o,o,o].copy()

        if self.model == 'CCD':
            for i in range(self.no):
                for j in range(self.no):
                    ij = i*self.no + j

                    Wmnij[:,:,i,j] += contract('ef,mnef->mn',t2[ij], ERIoovv[ij])
        else:
            for i in range(self.no):
                for j in range(self.no):
                    ij = i*self.no + j
                    ii = i*self.no + i
                    jj = j*self.no + j

                    Wmnij[:,:,i,j] += contract('E,mnE->mn', t1[j], ERIooov[jj][:,:,i,:])
                    Wmnij[:,:,i,j] += contract('E,mnE->mn', t1[i], ERIoovo[ii][:,:,:,j])
                    Wmnij[:,:,i,j] += contract('ef,mnef->mn',t2[ij], ERIoovv[ij])

                    tmp = contract('aA,mnab->mnAb', QL[ii], ERI[o,o,v,v])
                    tmp = contract('bB,mnAb->mnAB', QL[jj], tmp)
                    Wmnij[:,:,i,j] += contract('e,f,mnef->mn', t1[i], t1[j], tmp)

        wmnij_end = process_time()
        self.wmnij_t += wmnij_end - wmnij_start
        return Wmnij

    def build_Zmbij(self, Zmbij_ij, ERI, ERIovvv, t1, t2):
        zmbij_start = process_time()
        o = self.o
        v = self.v
        QL = self.QL

        if self.model == 'CCD':
            return
        else:
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                ii = i*self.no + i
                jj = j*self.no + j

                Zmbij = np.zeros((self.no,self.Local.dim[ij]))

                Zmbij = contract('mbef,ef->mb', ERIovvv[ij], t2[ij])

                tmp = contract('iabc,aA->iAbc',ERI[o,v,v,v], QL[ij])
                tmp = contract('iAbc,bB->iABc',tmp, QL[ii])
                tmp = contract('iABc,cC->iABC',tmp, QL[jj])
                Zmbij += contract('e,f,mbef->mb',t1[i], t1[j], tmp)

                Zmbij_ij.append(Zmbij)

        zmbij_end = process_time()
        self.zmbij_t += zmbij_end - zmbij_start
        return Zmbij_ij

    def build_Wmbej(self, Wmbej_ijim, ERI, L, ERIoovo, Sijnn, Sijnj, Sijjn, t1, t2):
        wmbej_start = process_time()
        v = self.v
        o = self.o
        QL = self.QL
        dim = self.dim

        if self.model == 'CCD':
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                for m in range(self.no):
                    im = i*self.no + m

                    Wmbej = np.zeros((dim[ij],dim[im]))

                    tmp = QL[ij].T @ ERI[m,v,v,j]
                    Wmbej = tmp @ QL[im]

                    for n in range(self.no):
                        jn = j*self.no + n
                        nj = n*self.no + j
                        ijn = ij*self.no + n

                        tmp = 0.5 * t2[jn] @ Sijjn[ijn].T
                        tmp1 = QL[im].T @ ERI[m,n,v,v]
                        tmp1 = tmp1 @ QL[jn]
                        Wmbej -= tmp.T @ tmp1.T

                        tmp2 = t2[nj] @ Sijnj[ijn].T
                        tmp3 = QL[im].T @ L[m,n,v,v]
                        tmp3 = tmp3 @ QL[nj]
                        Wmbej += 0.5 * tmp2.T @ tmp3.T

                    Wmbej_ijim.append(Wmbej)
        else:
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                jj = j*self.no + j

                for m in range(self.no):
                    im = i*self.no + m

                    Wmbej = np.zeros((dim[ij],dim[im]))

                    tmp = QL[ij].T @ self.H.ERI[m,v,v,j]
                    Wmbej = tmp @ QL[im]

                    tmp = contract('abc,aA->Abc',ERI[m,v,v,v], QL[ij])
                    tmp = contract('Abc,bB->ABc',tmp, QL[im])
                    tmp = contract('ABc,cC->ABC',tmp, QL[jj])
                    Wmbej += contract('F,beF->be', t1[j], tmp)

                    for n in range(self.no):
                        nn = n*self.no + n
                        jn = j*self.no + n
                        nj = n*self.no + j
                        ijn = ij*(self.no) + n

                        tmp1 = Sijnn[ijn] @ t1[n]
                        Wmbej -= contract('b,e->be', tmp1, ERIoovo[im][m,n,:,j])

                        tmp2 = 0.5 * t2[jn] @ Sijjn[ijn].T
                        tmp3_0 = QL[im].T @ ERI[m,n,v,v]
                        tmp3 = tmp3_0 @ QL[jn]
                        Wmbej -= tmp2.T @ tmp3.T

                        tmp4 = tmp3_0 @ QL[jj]
                        Wmbej -= contract('f,b,ef-> be',t1[j],tmp1,tmp4)

                        tmp5 = t2[nj] @ Sijnj[ijn].T
                        tmp6 = QL[im].T @ L[m,n,v,v]
                        tmp6 = tmp6 @ QL[nj]
                        Wmbej += 0.5 * tmp5.T @ tmp6.T

                    Wmbej_ijim.append(Wmbej)
        wmbej_end = process_time()
        self.wmbej_t += wmbej_end - wmbej_start
        return Wmbej_ijim

    def build_Wmbje(self, Wmbje_ijim, Wmbie_ijmj, ERI, ERIooov, Sijnn, Sijin, Sijjn, t1, t2):
        wmbje_start = process_time()
        o = self.o
        v = self.v
        QL = self.QL
        dim = self.dim

        if self.model == 'CCD':
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no

                for m in range(self.no):
                    im = i*self.no + m
                    mj = m*self.no + j

                    Wmbje = np.zeros(dim[ij],dim[im])
                    Wmbie = np.zeros(dim[ij],dim[mj])

                    tmp = QL[ij].T @ ERI[m,v,j,v]
                    tmp_mj = QL[ij].T @ ERI[m,v,i,v]
                    Wmbje = -1.0 * tmp @ QL[im]
                    Wmbie = -1.0 * tmp_mj @ QL[mj]

                    for n in range(self.no):
                        jn = j*self.no + n
                        _in = i*self.no + n
                        ijn = ij*self.no + n

                        tmp1 = 0.5* t2[jn] @ Sijjn[ijn].T
                        tmp2 = QL[jn].T @ ERI[m,n,v,v]
                        tmp2 = tmp2 @ QL[im]
                        Wmbje += tmp1.T @ tmp2

                        tmp1_mj = 0.5 * t2[_in] @ Sijin[ijn].T
                        tmp2_mj = QL[_in].T @ ERI[m,n,v,v]
                        tmp2_mj = tmp2_mj @ QL[mj]
                        Wmbie += tmp1_mj.T @ tmp2_mj

                    Wmbje_ijim.append(Wmbje)
                    Wmbie_ijmj.append(Wmbie)
        else:
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                ii = i*self.no + i
                jj = j*self.no + j

                for m in range(self.no):
                    im = i*self.no + m
                    mj = m*self.no + j

                    Wmbje = np.zeros(dim[ij],dim[im])
                    Wmbie = np.zeros(dim[ij],dim[mj])

                    tmp = QL[ij].T @ ERI[m,v,j,v]
                    tmp_mj = QL[ij].T @ ERI[m,v,i,v]
                    Wmbje = -1.0 * tmp @ QL[im]
                    Wmbie = -1.0 * tmp_mj @ QL[mj]

                    tmp1_0 = contract('abc,aA->Abc',ERI[m,v,v,v], QL[ij])
                    tmp1 = contract('Abc,bB->ABc',tmp1_0, QL[jj])
                    tmp1 = contract('ABc,cC->ABC',tmp1, QL[im])
                    Wmbje -=  contract('F,bFe->be', t1[j], tmp1)

                    tmp1_mj = contract('Abc,bB->ABc',tmp1_0, QL[ii])
                    tmp1_mj = contract('ABc,cC->ABC',tmp1_mj, QL[mj])
                    Wmbie -=  contract('F,bFe->be', t1[i], tmp1_mj)

                    for n in range(self.no):
                        nn = n*self.no + n
                        jn = j*self.no + n
                        _in = i*self.no + n
                        ijn = ij*self.no + n

                        tmp2 = Sijnn[ijn] @ t1[n]
                        Wmbje += contract('b,e->be',tmp2,ERIooov[im][m,n,j])

                        Wmbie += contract('b,e->be',tmp2,ERIooov[mj][m,n,i])

                        tmp3 = 0.5 * t2[jn] @ Sijjn[ijn].T
                        tmp4 = QL[jn].T @ ERI[m,n,v,v]
                        tmp4 = tmp4 @ QL[im]
                        Wmbje += tmp3.T @ tmp4

                        tmp5 = QL[jj].T @ ERI[m,n,v,v]
                        tmp5= tmp5 @ QL[im]
                        Wmbje += contract('f,b,fe->be',t1[j], tmp2, tmp5)

                        tmp2_mj = 0.5 * t2[_in] @ Sijin[ijn].T
                        tmp3_mj = QL[_in].T @ ERI[m,n,v,v]
                        tmp3_mj = tmp3_mj @ QL[mj]
                        Wmbie += tmp2_mj.T @ tmp3_mj

                        tmp4_mj = QL[ii].T @ ERI[m,n,v,v]
                        tmp4_mj = tmp4_mj @ QL[mj]
                        Wmbie += contract('f,b,fe->be',t1[i], tmp2, tmp4_mj)

                    Wmbje_ijim.append(Wmbje)
                    Wmbie_ijmj.append(Wmbie)
        wmbje_end = process_time()
        self.wmbje_t += wmbje_end - wmbje_start
        return Wmbje_ijim, Wmbie_ijmj

    def r_T1(self, r1_ii, Fov , ERI, L, Loovo, Sijmm, Sijim, Sijmn, t1, t2, Fae, Fmi, Fme):
        r1_start = process_time()
        v = self.v
        QL = self.QL

        if self.model == 'CCD':
            for i in range(self.no):
                r1_ii.append(np.zeros_like(t1[i]))

        else:
            for i in range(self.no):
                ii = i*self.no + i

                r1 = np.zeros(self.Local.dim[ii])

                r1 = Fov[ii][i].copy()
                r1 += contract('e,ae->a', t1[i], Fae[ii])

                for m in range(self.no):
                    mm = m*self.no + m
                    im = i*self.no + m
                    mi = m*self.no + i
                    iim = ii*(self.no) + m

                    tmp = Sijmm[iim] @ t1[m] 
                    r1 -= tmp * Fmi[m,i]

                    tmp1 = Sijim[iim] @ (2*t2[im] - t2[im].swapaxes(0,1))
                    r1 += contract('aE,E->a',tmp1, Fme[im][m])

                    tmp2 = contract('abc,aA->Abc',ERI[m,v,v,v], QL[ii])
                    tmp2 = contract('Abc,bB->ABc',tmp2, QL[mi])
                    tmp2 = contract('ABc,cC->ABC',tmp2, QL[mi])
                    r1 += contract('EF,aEF->a', (2.0*t2[mi] - t2[mi].swapaxes(0,1)), tmp2)

                for n in range(self.no):
                    nn = n*self.no + n

                    tmp3 = contract('ab,aA->Ab', L[n,v,v,i],QL[ii])
                    tmp3 = contract('Ab,bB->AB', tmp3, QL[nn])
                    r1 += contract('F,aF->a', t1[n], tmp3)

                for mn in range(self.no*self.no):
                    m = mn // self.no
                    n = mn % self.no
                    iimn =ii*(self.no**2) + mn 
             
                    tmp4 = Sijmn[iimn] @ t2[mn]
                    r1 -= contract('aE,E->a',tmp4,Loovo[mn][n,m,:,i])

                r1_ii.append(r1)
        
        r1_end = process_time()
        self.r1_t += r1_end - r1_start	
        return r1_ii

    def r_T2(self,r2_ij, ERI, ERIoovv, ERIvvvv, ERIovoo, Sijmm, Sijim, Sijmj, Sijnn, Sijmn, t1, t2, Fae,Fmi, Fme, Wmnij, Zmbij, Wmbej, Wmbje, Wmbie):
        r2_start = process_time()
        v = self.v
        QL = self.QL
        dim = self.dim

        nr2 = []
        if self.model == 'CCD':
            for ij in range(self.no*self.no):
                i = ij //self.no
                j = ij % self.no
                ii = i*self.no + i
                jj = j*self.no + j

                r2 = np.zeros(dim[ij],dim[ij])

                r2 = 0.5 * ERIoovv[ij][i,j]
                r2 += t2[ij] @ Fae[ij].T
                r2 += 0.5 * contract('ef,abef->ab',t2[ij],ERIvvvv[ij])

                for m in range(self.no):
                    mm = m *self.no + m
                    im = i*self.no + m
                    mj = m*self.no+ j
                    ijm = ij*self.no + m

                    tmp = Sijim[ijm] @ t2[im]
                    tmp = tmp @ Sijim[ijm].T
                    r2 -= tmp * Fmi[m,j]

                    tmp1 = Sijim[ijm] @ (t2[im] - t2[im].swapaxes(0,1))
                    r2 += tmp1 @ Wmbej[ijm].T

                    tmp2 = Sijim[ijm] @ t2[im]
                    tmp3 = Wmbej[ijm] + Wmbje[ijm]
                    r2 += tmp2 @ tmp3.T

                    tmp4 = Sijmj[ijm] @ t2[mj]
                    r2 += tmp4 @ Wmbie[ijm].T

                    for n in range(self.no):
                        mn = m*self.no + n
                        ijmn = ij*(self.no**2) + mn

                        tmp5 = Sijmn[ijmn] @ t2[mn]
                        tmp5 = tmp5 @ Sijmn[ijmn].T
                        r2 += 0.5 * tmp5 * Wmnij[m,n,i,j]

                nr2.append(r2)
        else:
            for ij in range(self.no*self.no):
                i = ij //self.no
                j = ij % self.no
                ii = i*self.no + i
                jj = j*self.no + j

                r2 = np.zeros(dim[ij],dim[ij])

                r2 = 0.5 * ERIoovv[ij][i,j].copy()
                r2 += t2[ij] @ Fae[ij].T
                r2 += 0.5 * contract('ef,abef->ab', t2[ij],ERIvvvv[ij])

                tmp = contract('abcd,aA->Abcd',self.H.ERI[v,v,v,v], QL[ij])
                tmp = contract('Abcd,bB->ABcd', tmp, QL[ij])
                tmp = contract('ABcd,cC->ABCd', tmp, QL[ii])
                tmp = contract('ABCd,dD->ABCD', tmp, QL[jj])
                r2 += 0.5 *contract('e,f,abef->ab',t1[i],t1[j], tmp)

                tmp1 = contract('abc,aA->Abc',ERI[v,v,v,j], QL[ij])
                tmp1 = contract('Abc,bB->ABc',tmp1, QL[ij])
                tmp1 = contract('ABc,cC->ABC',tmp1, QL[ii])
                r2 += contract('E,abE->ab', t1[i], tmp1)

                for m in range(self.no):
                    mm = m *self.no + m
                    im = i*self.no + m
                    mj = m*self.no + j
                    ijm = ij*self.no + m

                    tmp2_0 = Sijmm[ijm] @ t1[m]
                    tmp2 = contract('b,e->be', tmp2_0, Fme[ij][m])
                    r2 -= 0.5 * t2[ij] @ tmp2.T

                    tmp3 = Sijim[ijm] @ t2[im]
                    tmp3 = tmp3 @ Sijim[ijm].T
                    r2 -= tmp3 * Fmi[m,j]

                    tmp4 = contract('E,E->',t1[j], Fme[jj][m])
                    r2 -= 0.5 * tmp3 * tmp4

                    r2 -= contract('a,b->ab',tmp2_0,Zmbij[ij][m])

                    tmp5 = Sijim[ijm] @ (t2[im] - t2[im].swapaxes(0,1))
                    r2 += tmp5 @ Wmbej[ijm].T
   
                    tmp6 = Sijim[ijm] @ t2[im]
                    tmp7 = Wmbej[ijm] + Wmbje[ijm]
                    r2 += tmp6 @ tmp7.T

                    tmp8 = Sijmj[ijm] @ t2[mj]
                    r2 += tmp8 @ Wmbie[ijm].T

                    tmp9 = QL[ij].T @ ERI[m,v,v,j]
                    tmp9 = tmp9 @ QL[ii]
                    tmp10 = contract ('E,a->Ea', t1[i], tmp2_0)
                    r2 -= tmp10.T @ tmp9.T

                    tmp11 = QL[ij].T @ ERI[m,v,j,v]
                    tmp11 = tmp11 @ QL[ii]
                    r2 -= tmp11 @ tmp10

                    r2 -= contract('a,b->ab',tmp2_0, ERIovoo[ij][m,:,i,j])

                    for n in range(self.no):
                        mn = m*self.no + n
                        nn = n*self.no + n
                        ijmn = ij*(self.no**2) + mn
                        ijn = ij*self.no + n

                        tmp12 = Sijmn[ijmn] @ t2[mn]
                        tmp12 = tmp12 @ Sijmn[ijmn].T
                        tmp13 = Sijnn[ijn] @ t1[n]
                        r2 += 0.5 * tmp12 * Wmnij[m,n,i,j]
                        r2 += 0.5 * contract('a,b->ab',tmp2_0, tmp13) * Wmnij[m,n,i,j]

                nr2.append(r2) 
    
        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j
                ji = j*self.no + i

                r2_ij.append(nr2[ij].copy() + nr2[ji].copy().transpose())

        r2_end = process_time()
        self.r2_t += r2_end - r2_start
        return r2_ij

    def lcc_energy(self, Fov, Loovv, t1, t2):
        energy_start = process_time()
        QL = self.QL
        v = self.v
        ecc_ii = 0
        ecc_ij = 0
        ecc = 0
        
        if self.model == 'CCD':
            for i in range(self.no):
                for j in range(self.no):
                    ij = i*self.no + j

                    ecc_ij = contract('ab,ab->',t2[ij],Loovv[ij][i,j])
                    ecc += ecc_ij
        else:
            for i in range(self.no):
                ii = i*self.no + i

                ecc_ii = 2.0 *contract ('a,a->',Fov[ii][i], t1[i])
                ecc += ecc_ii

                for j in range(self.no):
                    ij = i*self.no + j
                    ii = i*self.no + i
                    jj = j*self.no + j

                    ecc_ij = contract('ab,ab->',t2[ij],Loovv[ij][i,j])
                    tmp2 = QL[ii].T @ self.H.L[i,j,v,v]
                    tmp2 = tmp2 @ QL[jj]
                    ecc_ij += contract('a,b,ab->',t1[i], t1[j], tmp2)
                    ecc += ecc_ij
        energy_end = process_time()
        self.energy_t += energy_end - energy_start
        return ecc

