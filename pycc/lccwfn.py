import time 
#from timer import Timer
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
    (3) generate and store overlap terms prior to the calculation of the residuals
    (4) remove redundant transformed integrals 
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

        self.t1 = np.zeros((self.no, self.nv))
        t1_ii = []
        t2_ij = [] 

        for i in range(self.no):
            ii = i*self.no + i
    
            t1_ii.append(self.QL[ii].T @ self.t1[i]) 
 
            for j in range(self.no):
                ij = i*self.no + j
                                
                t2_ij.append(-1* self.Local.ERIoovv_ij[ij][i,j] / (self.eps[ij].reshape(1,-1) + self.eps[ij].reshape(-1,1) 
                - self.H.F[i,i] - self.H.F[j,j]))   

        self.t1_ii = t1_ii    
        self.t2_ij = t2_ij 

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
        lcc_tstart = time.time()

        #initialize variables for timing each function
        #self.fae_t = Timer("Fae")
        #self.fme_t = Timer("Fme")
        #self.fmi_t = Timer("Fmi")
        #self.wmnij_t = Timer("Wmnij")
        #self.zmbij_t = Timer("Zmbij")
        #self.wmbej_t = Timer("Wmbej")
        #self.wmbje_t = Timer("Wmbje")
        #self.tau_t = Timer("tau")
        #self.r1_t = Timer("r1")
        #self.r2_t = Timer("r2")
        #self.energy_t = Timer("energy")

        #ldiis = helper_ldiis(self.t1_ii, self.t2_ij, max_diis) 
        
        elcc = self.lcc_energy(self.Local.Fov_ij,self.Local.Loovv_ij,self.t1_ii, self.t2_ij)
        print("CC Iter %3d: lCC Ecorr = %.15f dE = % .5E MP2" % (0,elcc,-elcc))

        for niter in range(1, maxiter+1):

            elcc_last = elcc

            r1_ii, r2_ij = self.local_residuals(self.t1_ii, self.t2_ij)

            rms = 0
            rms_t1 = 0
            rms_t2 = 0

            for i in range(self.no):
                ii = i*self.no + i  
                
                #need to change to reshape
                for a in range(self.Local.dim[ii]):
                    self.t1_ii[i][a] += r1_ii[i][a]/(self.H.F[i,i] - self.Local.eps[ii][a]) 

                rms_t1 += contract('Z,Z->',r1_ii[i],r1_ii[i])

                for j in range(self.no):
                    ij = i*self.no + j

                    self.t2_ij[ij] -= r2_ij[ij]/(self.eps[ij].reshape(1,-1) + self.eps[ij].reshape(-1,1)
                    - self.H.F[i,i] - self.H.F[j,j])

                    rms_t2 += contract('ZY,ZY->',r2_ij[ij],r2_ij[ij])

            rms = np.sqrt(rms_t1 + rms_t2)
            elcc = self.lcc_energy(self.Local.Fov_ij,self.Local.Loovv_ij,self.t1_ii, self.t2_ij)
            ediff = elcc - elcc_last
            print("lCC Iter %3d: lCC Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, elcc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nlCC has converged in %.3f seconds.\n" % (time.time() - lcc_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                print("E(%s) = %20.15f" % (self.local + "-" + self.model, elcc))
                print("E(TOT)  = %20.15f" % (elcc + self.eref))
                self.elcc = elcc
                #print(Timer.timers)
                return elcc

            #ldiis.add_error_vector(self.t1_ii,self.t2_ij)
            #if niter >= start_diis:
                #self.t1_ii, self.t2_ij = ldiis.extrapolate(self.t1_ii, self.t2_ij)

    def local_residuals(self, t1_ii, t2_ij):
        """
        Constructing the two- and four-index intermediates
        Then evaluating the singles and doubles residuals, storing them in a list of length occ (single) and length occ*occ (doubles)
        Naming scheme same as integrals where _ij is attach as a suffix to the intermediate name 
        ... special naming scheme for those involving two different pair space ij and im -> _ijm

        To do
        ------
        There are some arguments that isn't really being used and will be updated once things are good to go
        Listed here are intermediates with corresponding arguments that aren't needed since it requires "on the fly" generation
        Fae_ij Lovvv_ij
        Fme_ij (Fme_im) Loovv_ij
        Wmbej_ijim ERIovvv_ij, ERIoovv_ij, Loovv_ij
        Wmbje_ijim (Wmbie_ijmj) ERIovvv_ij, ERIoovv_ij
        r1_ii Lovvo_ij, ERIovvo_ij
        r2_ij ERIovvo_ij, ERIovov_ij, ERIvvvo_ij
        """
        o = self.o
        v = self.v
        F = self.H.F
        L = self.H.L
        ERI = self.H.ERI

        Fae_ij = []
        Fme_ij = []
        Fme_im = []      
 
        Wmbej_ijim = []
        Wmbje_ijim = []
        Wmbie_ijmj = []
        Zmbij_ij = []

        r1_ii = []
        r2_ij = []

        Fae_ij = self.build_lFae(Fae_ij, self.Local.Fvv_ij, self.Local.Fov_ij, 
        self.Local.Lovvv_ij, self.Local.Loovv_ij, t1_ii, t2_ij)
        lFmi = self.build_lFmi(o, F, self.Local.Fov_ij, self.Local.Looov_ij, self.Local.Loovv_ij, t1_ii, t2_ij)
        Fme_ij, Fme_im = self.build_lFme(Fme_ij, Fme_im, self.Local.Fov_ij, self.Local.Loovv_ij, t1_ii)
        lWmnij = self.build_lWmnij(o, ERI, self.Local.ERIooov_ij, self.Local.ERIoovo_ij, 
        self.Local.ERIoovv_ij, t1_ii, t2_ij)
        Zmbij = self.build_lZmbij(Zmbij_ij, self.Local.ERIovvv_ij, t1_ii, t2_ij)
        Wmbej_ijim = self.build_lWmbej(Wmbej_ijim, self.Local.ERIoovv_ij, self.Local.ERIovvo_ij, 
        self.Local.ERIovvv_ij,self.Local.ERIoovo_ij, self.Local.Loovv_ij, t1_ii, t2_ij)
        Wmbje_ijim, Wmbie_ijmj = self.build_lWmbje(Wmbje_ijim, Wmbie_ijmj, self.Local.ERIovov_ij, 
        self.Local.ERIovvv_ij,self.Local.ERIoovv_ij, self.Local.ERIooov_ij, t1_ii, t2_ij)

        #looks like I used Fme_ij for r1_ii and it works ...
        r1_ii = self.lr_T1(r1_ii, self.Local.Fov_ij , self.Local.ERIovvv_ij, self.Local.Lovvo_ij, self.Local.Loovo_ij, 
        t1_ii, t2_ij,Fae_ij, Fme_ij, lFmi)
        r2_ij = self.lr_T2(r2_ij, self.Local.ERIoovv_ij, self.Local.ERIvvvv_ij, self.Local.ERIovvo_ij, self.Local.ERIovoo_ij, 
        self.Local.ERIvvvo_ij,self.Local.ERIovov_ij, t1_ii, t2_ij, Fae_ij ,lFmi,Fme_ij,lWmnij, Zmbij_ij, Wmbej_ijim, Wmbje_ijim, Wmbie_ijmj)

        return r1_ii, r2_ij    

    def build_lFae(self, Fae_ij, Fvv_ij,Fov_ij, Lovvv_ij, Loovv_ij, t1_ii, t2_ij):
        #self.fae_t.start()
        o = self.o
        v = self.v
        QL = self.QL
        
        if self.model == 'CCD':
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no

                Fae = Fvv_ij[ij].copy()

                Fae_3 = np.zeros_like(Fae)
                for m in range(self.no):
                    for n in range(self.no):
                        mn = m *self.no +n
                        nn = n*self.no + n

                        Sijmn = QL[ij].T @ QL[mn]

                        tmp2 = Sijmn @ t2_ij[mn]
                        tmp3_0 = QL[ij].T @ self.H.L[m,n,v,v]
                        tmp3_1 = tmp3_0 @ QL[mn]
                        Fae_3 -= tmp2 @ tmp3_1.T

                Fae_ij.append(Fae + Fae_3)  
        else:      
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no

                Fae = Fvv_ij[ij].copy()

                Fae_1 = np.zeros_like(Fae)
                Fae_2 = np.zeros_like(Fae)
                Fae_3 = np.zeros_like(Fae)
                Fae_4 = np.zeros_like(Fae)
                for m in range(self.no):
                    mm = m*self.no + m

                    Sijmm = QL[ij].T @ QL[mm]
                    tmp = Sijmm @ t1_ii[m]

                    Fae_1 -= 0.5* contract('e,a->ae',Fov_ij[ij][m],tmp)

                    tmp1_0 = contract('abc,aA->Abc',self.H.L[m,v,v,v], QL[ij])
                    tmp1_1 = contract('Abc,bB->ABc',tmp1_0, QL[mm])
                    tmp1_2 = contract('ABc,cC->ABC',tmp1_1, QL[ij])

                    Fae_2 += contract('F,aFe->ae',t1_ii[m],tmp1_2)

                    for n in range(self.no):
                        mn = m *self.no +n
                        nn = n*self.no + n

                        Sijmn = QL[ij].T @ QL[mn]

                        tmp2 = Sijmn @ t2_ij[mn]
                        tmp3_0 = QL[ij].T @ self.H.L[m,n,v,v]
                        tmp3_1 = tmp3_0 @ QL[mn]
                        Fae_3 -= tmp2 @ tmp3_1.T

                        tmp4 = tmp3_0 @ QL[nn]
                        Fae_4 -= 0.5 *contract('a,F,eF->ae', tmp, t1_ii[n], tmp4)

                Fae_ij.append(Fae + Fae_1 + Fae_2 + Fae_3 + Fae_4)
        #self.fae_t.stop()
        return Fae_ij

    def build_lFmi(self, o, F, Fov_ij, Looov_ij, Loovv_ij, t1_ii, t2_ij):
        #self.fmi_t.start()
        v = self.v
        QL = self.QL

        Fmi = F[o,o].copy()

        if self.model == 'CCD':
            Fmi_3 = np.zeros_like(Fmi)
            for j in range(self.no):
               for n in range(self.no):
                   jn = j*self.no + n

                   Fmi_3[:,j] += contract('EF,mEF->m',t2_ij[jn],Loovv_ij[jn][:,n,:,:])

            Fmi_tot = Fmi + Fmi_3
        else:
            Fmi_1 = np.zeros_like(Fmi)
            Fmi_2 = np.zeros_like(Fmi)
            Fmi_3 = np.zeros_like(Fmi)
            Fmi_4 = np.zeros_like(Fmi)
            for j in range(self.no):
                jj = j*self.no +j
                for n in range(self.no):
                   jn = j*self.no + n
                   nn = n*self.no + n

                   Fmi_1[:,j] += 0.5 * contract('e,me->m', t1_ii[j], Fov_ij[jj])
                   Fmi_2[:,j] += contract('e,me->m',t1_ii[n],Looov_ij[nn][:,n,j])

                   Fmi_3[:,j] += contract('EF,mEF->m',t2_ij[jn],Loovv_ij[jn][:,n,:,:])

                   tmp = contract('mab,aA,bB->mAB', self.H.L[o,n,v,v].copy(),QL[jj],QL[nn])
                   Fmi_4[:,j] += 0.5 * contract('E,F,mEF->m',t1_ii[j], t1_ii[n], tmp.copy())

            Fmi_tot = Fmi + Fmi_1 + Fmi_2 + Fmi_3  + Fmi_4 #+ Fmi_1 + Fmi_2 #Fmi_1 + Fmi_2 + Fmi_3 + Fmi_4
        #self.fmi_t.stop()
        return Fmi_tot 

    def build_lFme(self, Fme_ij, Fme_totim, Fov_ij, Loovv_ij, t1_ii):
        #self.fme_t.start()
        QL = self.QL
        v = self.v
        
        if self.model == 'CCD':
            return 
        else:
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no

                Fme = np.zeros((self.no,self.Local.dim[ij]))
                Fme_1 = np.zeros_like(Fme)

                Fme = Fov_ij[ij].copy()

                for m in range(self.no):
                    im = i*self.no + m

                    Fme_im = np.zeros((self.no,self.Local.dim[im]))
                    Fme1_im = np.zeros_like(Fme_im)

                    Fme_im = Fov_ij[im].copy()

                    for n in range(self.no):
                        nn = n*self.no + n

                        tmp = QL[ij].T @ self.H.L[m,n,v,v]
                        tmp1 = tmp @ QL[nn]
                        Fme_1[m] += t1_ii[n] @ tmp1.T

                        tmp_im = QL[im].T @ self.H.L[m,n,v,v]
                        tmp1_im = tmp_im @ QL[nn]
                        Fme1_im[m] += t1_ii[n] @ tmp1_im.T

                    Fme_totim.append(Fme_im + Fme1_im)
                Fme_ij.append(Fme + Fme_1)
        #self.fme_t.stop()
        return Fme_ij, Fme_totim

    def build_lWmnij(self, o, ERI, ERIooov_ij, ERIoovo_ij, ERIoovv_ij, t1_ii, t2_ij):
        #self.wmnij_t.start()
        v = self.v 
        QL = self.Local.QL

        Wmnij = ERI[o,o,o,o].copy()

        if self.model == 'CCD':
            Wmnij_3 = np.zeros_like(Wmnij)
            for i in range(self.no):
                for j in range(self.no):
                    ij = i*self.no + j

                    Wmnij_3[:,:,i,j] += contract('ef,mnef->mn',t2_ij[ij], ERIoovv_ij[ij])

            Wmnij_tot = Wmnij + Wmnij_3
        else:
            Wmnij_1 = np.zeros_like(Wmnij)
            Wmnij_2 = np.zeros_like(Wmnij)
            Wmnij_3 = np.zeros_like(Wmnij)
            Wmnij_4 = np.zeros_like(Wmnij)
            for i in range(self.no):
                for j in range(self.no):
                    ij = i*self.no + j
                    ii = i*self.no + i
                    jj = j*self.no + j

                    Wmnij_1[:,:,i,j] += contract('E,mnE->mn', t1_ii[j], ERIooov_ij[jj][:,:,i,:])
                    Wmnij_2[:,:,i,j] += contract('E,mnE->mn', t1_ii[i], ERIoovo_ij[ii][:,:,:,j])

                    Wmnij_3[:,:,i,j] += contract('ef,mnef->mn',t2_ij[ij], ERIoovv_ij[ij])
                    tmp = contract('aA,bB,mnab->mnAB',QL[ii], QL[jj], self.H.ERI[o,o,v,v].copy())
                    Wmnij_4[:,:,i,j] += contract('e,f,mnef->mn', t1_ii[i], t1_ii[j], tmp.copy())

            Wmnij_tot = Wmnij + Wmnij_1 + Wmnij_2 + Wmnij_3 + Wmnij_4
        #self.wmnij_t.stop()
        return Wmnij_tot

    def build_lZmbij(self, Zmbij_ij, ERIovvv_ij, t1_ii, t2_ij):
        #self.zmbij_t.start()
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

                Zmbij = contract('mbef,ef->mb', ERIovvv_ij[ij], t2_ij[ij])
                tmp = contract('iabc,aA->iAbc',self.H.ERI[o,v,v,v], QL[ij])
                tmp1 = contract('iAbc,bB->iABc',tmp, QL[ii])
                tmp2 = contract('iABc,cC->iABC',tmp1, QL[jj])
                Zmbij = Zmbij.copy() + contract('e,f,mbef->mb',t1_ii[i], t1_ii[j], tmp2)

                Zmbij_ij.append(Zmbij)
        #self.zmbij_t.stop()
        return

    def build_lWmbej(self, Wmbej_ijim, ERIoovv_ij, ERIovvo_ij, ERIovvv_ij, ERIoovo_ij, Loovv_ij, t1_ii, t2_ij):
        #self.wmbej_t.start()
        v = self.v
        o = self.o
        QL = self.QL
        dim = self.dim

        if self.model == 'CCD':
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                jj = j*self.no + j

                for m in range(self.no):
                    im = i*self.no + m

                    Wmbej = np.zeros((dim[ij],dim[im]))

                    tmp = QL[ij].T @ self.H.ERI[m,v,v,j]
                    Wmbej = tmp @ QL[im]

                    Wmbej_1 = np.zeros_like(Wmbej)
                    Wmbej_2 = np.zeros_like(Wmbej)
                    Wmbej_3 = np.zeros_like(Wmbej)
                    Wmbej_4 = np.zeros_like(Wmbej)
                    for n in range(self.no):
                        nn = n*self.no + n
                        jn = j*self.no + n
                        nj = n*self.no + j

                        Sijjn = QL[ij].T @ QL[jn]

                        tmp5 = 0.5 * t2_ij[jn] @ Sijjn.T
                        tmp6_1 = QL[im].T @ self.H.ERI[m,n,v,v]
                        tmp6_2 = tmp6_1 @ QL[jn]
                        Wmbej_3 -= tmp5.T @ tmp6_2.T

                        Sijnj = QL[ij].T @ QL[nj]

                        tmp7 = t2_ij[nj] @ Sijnj.T
                        tmp8_1 = QL[im].T @ self.H.L[m,n,v,v]
                        tmp8_2 = tmp8_1 @ QL[nj]
                        Wmbej_4 += 0.5 * tmp7.T @ tmp8_2.T

                    Wmbej_ijim.append(Wmbej + Wmbej_3 + Wmbej_4)
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

                    Wmbej_1 = np.zeros_like(Wmbej)
                    Wmbej_2 = np.zeros_like(Wmbej)
                    Wmbej_3 = np.zeros_like(Wmbej)
                    Wmbej_4 = np.zeros_like(Wmbej)

                    tmp1 = contract('abc,aA->Abc',self.H.ERI[m,v,v,v], QL[ij])
                    tmp2 = contract('Abc,bB->ABc',tmp1, QL[im])
                    tmp3 = contract('ABc,cC->ABC',tmp2, QL[jj])
                    Wmbej_1 = contract('F,beF->be', t1_ii[j], tmp3)
                    for n in range(self.no):
                        nn = n*self.no + n
                        jn = j*self.no + n
                        nj = n*self.no + j

                        Sijnn = QL[ij].T @ QL[nn]

                        tmp4 = Sijnn @ t1_ii[n]
                        Wmbej_2 -= contract('b,e->be',tmp4,ERIoovo_ij[im][m,n,:,j])

                        Sijjn = QL[ij].T @ QL[jn]

                        tmp5 = 0.5 * t2_ij[jn] @ Sijjn.T
                        tmp6_1 = QL[im].T @ self.H.ERI[m,n,v,v].copy()
                        tmp6_2 = tmp6_1.copy() @ QL[jn]
                        Wmbej_3 -= tmp5.T @ tmp6_2.T
                        tmp6_3 =  tmp6_1.copy() @ QL[jj]
                        Wmbej_3 = Wmbej_3.copy() - contract('f,b,ef-> be',t1_ii[j],tmp4.copy(),tmp6_3.copy())

                        Sijnj = QL[ij].T @ QL[nj]

                        tmp7 = t2_ij[nj] @ Sijnj.T
                        tmp8_1 = QL[im].T @ self.H.L[m,n,v,v]
                        tmp8_2 = tmp8_1 @ QL[nj]
                        Wmbej_4 += 0.5 * tmp7.T @ tmp8_2.T

                    Wmbej_ijim.append(Wmbej + Wmbej_1 + Wmbej_2 + Wmbej_4 + Wmbej_3)
        #self.wmbej_t.stop()
        return Wmbej_ijim

    def build_lWmbje(self, Wmbje_ijim,Wmbie_ijmj,ERIovov_ij, ERIovvv_ij, ERIoovv_ij, ERIooov_ij, t1_ii, t2_ij):
        #self.wmbje_t.start()
        o = self.o
        v = self.v
        QL = self.QL
        dim = self.dim
   
        if self.model == 'CCD':
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

                    tmp_im = QL[ij].T @ self.H.ERI[m,v,j,v]
                    tmp_mj = QL[ij].T @ self.H.ERI[m,v,i,v]
                    Wmbje = -1.0 * tmp_im @ QL[im]
                    Wmbie = -1.0 * tmp_mj @ QL[mj]

                    Wmbje_3 = np.zeros_like(Wmbje)
                    Wmbie_3 = np.zeros_like(Wmbie)
                    for n in range(self.no):
                        nn = n*self.no + n
                        jn = j*self.no + n
                        _in = i*self.no + n

                        Sijjn = QL[ij].T @ QL[jn]
                        Sijin = QL[ij].T @ QL[_in]

                        tmp5 = 0.5* t2_ij[jn] @ Sijjn.T
                        tmp6_1 = QL[jn].T @ self.H.ERI[m,n,v,v]
                        tmp6_2 = tmp6_1 @ QL[im]
                        Wmbje_3 += tmp5.T @ tmp6_2

                        tmp5_mj = 0.5 * t2_ij[_in] @ Sijin.T
                        tmp6_1mj = QL[_in].T @ self.H.ERI[m,n,v,v]
                        tmp6_2mj = tmp6_1mj @ QL[mj]
                        Wmbie_3 += tmp5_mj.T @ tmp6_2mj

                    Wmbje_ijim.append(Wmbje + Wmbje_3)
                    Wmbie_ijmj.append(Wmbie + Wmbie_3)        
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

                    tmp_im = QL[ij].T @ self.H.ERI[m,v,j,v]
                    tmp_mj = QL[ij].T @ self.H.ERI[m,v,i,v]
                    Wmbje = -1.0 * tmp_im @ QL[im]
                    Wmbie = -1.0 * tmp_mj @ QL[mj]

                    Wmbje_1 = np.zeros_like(Wmbje)
                    Wmbje_2 = np.zeros_like(Wmbje)
                    Wmbje_3 = np.zeros_like(Wmbje)

                    Wmbie_1 = np.zeros_like(Wmbie)
                    Wmbie_2 = np.zeros_like(Wmbie)
                    Wmbie_3 = np.zeros_like(Wmbie)

                    tmp1 = contract('abc,aA->Abc',self.H.ERI[m,v,v,v], QL[ij])
                    tmp2 = contract('Abc,bB->ABc',tmp1, QL[jj])
                    tmp3 = contract('ABc,cC->ABC',tmp2, QL[im])
                    Wmbje_1 = -1.0 * contract('F,bFe->be', t1_ii[j], tmp3)

                    tmp2_mj = contract('Abc,bB->ABc',tmp1, QL[ii])
                    tmp3_mj = contract('ABc,cC->ABC',tmp2_mj, QL[mj])
                    Wmbie_1 = -1.0 * contract('F,bFe->be', t1_ii[i], tmp3_mj)

                    for n in range(self.no):
                        nn = n*self.no + n
                        jn = j*self.no + n
                        _in = i*self.no + n

                        Sijnn = QL[ij].T @ QL[nn]

                        tmp4 = Sijnn @ t1_ii[n]
                        Wmbje_2 += contract('b,e->be',tmp4,ERIooov_ij[im][m,n,j])

                        Wmbie_2 += contract('b,e->be',tmp4,ERIooov_ij[mj][m,n,i])

                        Sijjn = QL[ij].T @ QL[jn]
                        Sijin = QL[ij].T @ QL[_in]

                        #tmp5 = self.build_ltau(jn,t1_ii,t2_ij, 0.5, 1.0) @ Sijjn.T
                        tmp5 = 0.5 * t2_ij[jn] @ Sijjn.T
                        tmp6_1 = QL[jn].T @ self.H.ERI[m,n,v,v]
                        tmp6_2 = tmp6_1 @ QL[im]
                        Wmbje_3 += tmp5.T @ tmp6_2
                        tmp6_3 = QL[jj].T @ self.H.ERI[m,n,v,v]
                        tmp6_4 = tmp6_3 @ QL[im]
                        Wmbje_3 = Wmbje_3.copy() + contract('f,b,fe->be',t1_ii[j], tmp4, tmp6_4)

                        #tmp5_mj = self.build_ltau(_in,t1_ii,t2_ij, 0.5, 1.0) @ Sijin.T
                        tmp5_mj = 0.5*t2_ij[_in] @ Sijin.T
                        tmp6_1mj = QL[_in].T @ self.H.ERI[m,n,v,v]
                        tmp6_2mj = tmp6_1mj @ QL[mj]
                        Wmbie_3 += tmp5_mj.T @ tmp6_2mj
                        tmp6_3mj = QL[ii].T @ self.H.ERI[m,n,v,v]
                        tmp6_4mj = tmp6_3mj @ QL[mj]
                        Wmbie_3 = Wmbie_3.copy() + contract('f,b,fe->be',t1_ii[i], tmp4, tmp6_4mj)

                    Wmbje_ijim.append(Wmbje + Wmbje_1 + Wmbje_2 +  Wmbje_3)
                    Wmbie_ijmj.append(Wmbie + Wmbie_1 + Wmbie_2 + Wmbie_3)
        #self.wmbje_t.stop()
        return Wmbje_ijim, Wmbie_ijmj

    def lr_T1(self, r1_ii, Fov_ij , ERIovvv_ij, Lovvo_ij, Loovo_ij, t1_ii, t2_ij, Fae_ij , Fme_im, lFmi):
        #self.r1_t.start()
        v = self.v
        QL = self.QL

        if self.model == 'CCD':
            for i in range(self.no):
                r1_ii.append(np.zeros_like(t1_ii[i]))        
        else:
            for i in range(self.no):
                ii = i*self.no + i

                r1 = np.zeros(self.Local.dim[ii])
                r1_1 = np.zeros_like(r1)

                r1 = Fov_ij[ii][i]
                r1_1 = contract('e,ae->a', t1_ii[i], Fae_ij[ii])

                r1_2 = np.zeros_like(r1)
                r1_3 = np.zeros_like(r1)
                r1_5 = np.zeros_like(r1)

                for m in range(self.no):
                    mm = m*self.no + m

                    Siimm = QL[ii].T @ QL[mm]

                    tmp =  Siimm @ t1_ii[m]
                    r1_2 -= tmp * lFmi[m,i]

                    im = i*self.no + m
                    Siiim = QL[ii].T @ QL[im]

                    tmp1 = Siiim @ (2*t2_ij[im] - t2_ij[im].swapaxes(0,1))
                    r1_3 += contract('aE,E->a',tmp1, Fme_im[im][m])

                    mi = m*self.no + i

                    tmp2 = contract('abc,aA->Abc',self.H.ERI[m,v,v,v], QL[ii])
                    tmp3 = contract('Abc,bB->ABc',tmp2, QL[mi])
                    tmp4 = contract('ABc,cC->ABC',tmp3, QL[mi])
                    r1_5 += contract('EF,aEF->a', (2.0*t2_ij[mi] - t2_ij[mi].swapaxes(0,1)), tmp4)

                r1_4 = np.zeros_like(r1)
                for n in range(self.no):
                    nn = n*self.no + n

                    tmp5 = contract('ab,aA->Ab', self.H.L[n,v,v,i],QL[ii])
                    tmp5_1 = contract('Ab,bB->AB',tmp5, QL[nn])
                    r1_4 += contract('F,aF->a', t1_ii[n], tmp5_1)

                r1_6 = np.zeros_like(r1)
                for mn in range(self.no*self.no):
                    m = mn // self.no
                    n = mn % self.no

                    Siimn = QL[ii].T @ QL[mn]

                    tmp3 = Siimn @ t2_ij[mn]
                    r1_6 -= contract('aE,E->a',tmp3,Loovo_ij[mn][n,m,:,i])

                r1_ii.append(r1 + r1_1 + r1_2 + r1_3 +  r1_4 + r1_5 + r1_6) #+ r1_3 + r1_4 + r1_5 + r1_6)
        #self.r1_t.stop()
        return r1_ii

    def lr_T2(self,r2_ij,ERIoovv_ij, ERIvvvv_ij, ERIovvo_ij, ERIovoo_ij, ERIvvvo_ij, ERIovov_ij, t1_ii,
    t2_ij, Fae_ij,lFmi,Fme_ij, lWmnij, Zmbij_ij, Wmbej_ijim, Wmbje_ijim, Wmbie_ijmj):
        #self.r2_t.start()
        v = self.v
        QL = self.QL
        dim = self.dim

        nr2_ij = []
        if self.model == 'CCD':
            for ij in range(self.no*self.no):
                i = ij //self.no
                j = ij % self.no
                ii = i*self.no + i
                jj = j*self.no + j

                r2 = np.zeros(dim[ij],dim[ij])
                r2_1one = np.zeros_like(r2)
                r2_4 = np.zeros_like(r2)

                r2 = 0.5 * ERIoovv_ij[ij][i,j]
                r2_1one = t2_ij[ij] @ Fae_ij[ij].T
                r2_4 = 0.5 * contract('ef,abef->ab',t2_ij[ij],ERIvvvv_ij[ij])

                r2_2one = np.zeros_like(r2)
                r2_3 = np.zeros_like(r2)
                r2_6 = np.zeros_like(r2)
                r2_7 = np.zeros_like(r2)
                r2_8 = np.zeros_like(r2)
                for m in range(self.no):
                    mm = m *self.no + m
                    im = i*self.no + m
                    mj = m*self.no+ j 
                    ijm = ij*self.no + m

                    Sijmm = QL[ij].T @ QL[mm]
                    Sijim = QL[ij].T @ QL[im]
                    Sijmj = QL[ij].T @ QL[mj]

                    tmp = Sijmm @ t1_ii[m]

                    tmp2_1 = Sijim @ t2_ij[im]
                    tmp2_2 = tmp2_1 @ Sijim.T
                    r2_2one -= tmp2_2 * lFmi[m,j]

                    tmp5 = Sijim @ (t2_ij[im] - t2_ij[im].swapaxes(0,1))
                    r2_6 += tmp5 @ Wmbej_ijim[ijm].T

                    tmp6 = Sijim @ t2_ij[im]
                    tmp7 = Wmbej_ijim[ijm] + Wmbje_ijim[ijm]
                    r2_7 += tmp6 @ tmp7.T

                    tmp8 = Sijmj @ t2_ij[mj]
                    tmp9 = Wmbie_ijmj[ijm]
                    r2_8 += tmp8 @ tmp9.T

                    for n in range(self.no):
                        mn = m*self.no + n

                        Sijmn = QL[ij].T @ QL[mn]

                        tmp4_1 = Sijmn @ t2_ij[mn]
                        tmp4_2 = tmp4_1 @ Sijmn.T
                        r2_3 += 0.5 * tmp4_2 * lWmnij[m,n,i,j]

                nr2_ij.append(r2 + r2_1one + r2_2one + r2_3 + r2_4 + r2_6 + r2_7 + r2_8)
        else:
            for ij in range(self.no*self.no):
                i = ij //self.no
                j = ij % self.no
                ii = i*self.no + i
                jj = j*self.no + j
    
                r2 = np.zeros(dim[ij],dim[ij])
                r2_1one = np.zeros_like(r2)
                r2_4 = np.zeros_like(r2)
                r2_11 = np.zeros_like(r2)        
    
                r2 = 0.5 * ERIoovv_ij[ij][i,j]
    
                r2_1one = contract('ae,be->ab', t2_ij[ij], Fae_ij[ij])
    
                r2_4 = 0.5 * contract('ef,abef->ab', t2_ij[ij],ERIvvvv_ij[ij])
                r2_4tmp = contract('abcd,aA->Abcd',self.H.ERI[v,v,v,v], QL[ij])
                r2_4tmp1 = contract('Abcd,bB->ABcd',r2_4tmp, QL[ij])
                r2_4tmp2 = contract('ABcd,cC->ABCd',r2_4tmp1, QL[ii])
                r2_4tmp3 = contract('ABCd,dD->ABCD',r2_4tmp2, QL[jj])
                r2_4 = r2_4.copy() + 0.5 *contract('e,f,abef->ab',t1_ii[i],t1_ii[j],r2_4tmp3.copy())
    
                tmp15 = contract('abc,aA->Abc',self.H.ERI[v,v,v,j], QL[ij])
                tmp16 = contract('Abc,bB->ABc',tmp15, QL[ij])
                tmp17 = contract('ABc,cC->ABC',tmp16, QL[ii])
                r2_11 = contract('E,abE->ab', t1_ii[i], tmp17)
    
                r2_1two = np.zeros_like(r2)
                r2_2one = np.zeros_like(r2)
                r2_2two = np.zeros_like(r2)
                r2_3 = np.zeros_like(r2)
                r2_5 = np.zeros_like(r2)
                r2_6 = np.zeros_like(r2)
                r2_7 = np.zeros_like(r2)
                r2_8 = np.zeros_like(r2)
                r2_9 = np.zeros_like(r2)
                r2_10 = np.zeros_like(r2)
                r2_12 = np.zeros_like(r2)
                for m in range(self.no):
                    mm = m *self.no + m
                    im = i*self.no + m
                    ijm = ij*self.no + m
    
                    Sijmm = QL[ij].T @ QL[mm]
    
                    tmp = Sijmm @ t1_ii[m]
                    tmp1 = contract('b,e->be', tmp, Fme_ij[ij][m])
                    r2_1two -= 0.5 * t2_ij[ij] @ tmp1.T
    
                    im = i*self.no + m
                    Sijim = QL[ij].T @ QL[im]
    
                    tmp2_1 = Sijim @ t2_ij[im]
                    tmp2_2 = tmp2_1 @ Sijim.T
                    r2_2one -= tmp2_2 * lFmi[m,j]
    
                    tmp3 = contract('E,E->',t1_ii[j], Fme_ij[jj][m])
                    r2_2two -= 0.5 * tmp2_2 * tmp3
    
                    r2_5 -= contract('a,b->ab',tmp.copy(),Zmbij_ij[ij][m])
    
                    tmp5 = Sijim @ (t2_ij[im] - t2_ij[im].swapaxes(0,1))
                    r2_6 += tmp5 @ Wmbej_ijim[ijm].T
    
                    tmp6 = Sijim @ t2_ij[im]
                    tst  = Wmbje_ijim[ijm]
                    tmp7 = Wmbej_ijim[ijm] + tst
                    r2_7 += tmp6 @ tmp7.T
    
                    mj = m*self.no + j
                    Sijmj = QL[ij].T @ QL[mj]
    
                    tmp8 = Sijmj @ t2_ij[mj]
                    tmp9 = Wmbie_ijmj[ijm]
                    r2_8 += tmp8 @ tmp9.T
    
                    tmp10 = QL[ij].T @ self.H.ERI[m,v,v,j]
                    tmp11 = tmp10 @ QL[ii]
                    tmp12 = contract ('E,a->Ea', t1_ii[i],tmp)
                    r2_9 -= tmp12.T @ tmp11.T
    
                    tmp13 = QL[ij].T @ self.H.ERI[m,v,j,v]
                    tmp14 = tmp13 @ QL[ii]
                    r2_10 -= tmp14 @ tmp12
    
                    r2_12 -= contract('a,b->ab',tmp, ERIovoo_ij[ij][m,:,i,j])
    
                    for n in range(self.no):
                        mn = m*self.no + n
                        nn = n*self.no + n
    
                        Sijmn = QL[ij].T @ QL[mn]
                        Sijnn = QL[ij].T @ QL[nn]
    
                        tmp4_1 = Sijmn @ t2_ij[mn] #self.build_ltau(mn,t1_ii,t2_ij)
                        tmp4_2 = tmp4_1 @ Sijmn.T
                        tmp4_3 = Sijnn @ t1_ii[n]
                        r2_3 += 0.5 * tmp4_2 * lWmnij[m,n,i,j]
                        r2_3 = r2_3 + 0.5 * contract('a,b->ab',tmp, tmp4_3) * lWmnij[m,n,i,j]
                nr2_ij.append(r2 + r2_1one + r2_1two + r2_2one + r2_2two + r2_3 + r2_4 + r2_5 + r2_6 + r2_7 +
                r2_8 + r2_9 + r2_10 + r2_11 + r2_12)
    
        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j
                ji = j*self.no + i
                r2_ij.append(nr2_ij[ij].copy() + nr2_ij[ji].copy().transpose())
        #self.r2_t.stop()
        return r2_ij 

    def lcc_energy(self,Fov_ij,Loovv_ij,t1_ii,t2_ij):
        #self.energy_t.start()
        QL = self.QL
        v = self.v
        ecc_ii = 0
        ecc_ij = 0
        ecc = 0
        
        if self.model == 'CCD':
            for i in range(self.no):
                for j in range(self.no):
                    ij = i*self.no + j

                    ecc_ij = contract('ab,ab->',t2_ij[ij],Loovv_ij[ij][i,j])
                    ecc += ecc_ij
        else:        
            for i in range(self.no):
                ii = i*self.no + i

                ecc_ii = 2.0 *contract ('a,a->',Fov_ij[ii][i], t1_ii[i])
                ecc += ecc_ii

                for j in range(self.no):
                    ij = i*self.no + j
                    ii = i*self.no + i
                    jj = j*self.no + j

                    Sijii = QL[ij].T @ QL[ii]
                    Sijjj = QL[ij].T @ QL[jj]

                    tmp = Sijii @ t1_ii[i]
                    tmp1 = t1_ii[j] @ Sijjj.T

                    ecc_ij = contract('ab,ab->',t2_ij[ij],Loovv_ij[ij][i,j])
                    tmp2 = QL[ii].T @ self.H.L[i,j,v,v] @ QL[jj]
                    ecc_ij = ecc_ij + contract('a,b,ab->',t1_ii[i], t1_ii[j], tmp2)
                    ecc += ecc_ij
        #self.energy_t.stop()
        return ecc
