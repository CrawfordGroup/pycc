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
                for j in range(self.no):
                    ij = i*self.no + j

                    self.t2_ij[ij] -= r2_ij[ij]/(self.eps[ij].reshape(1,-1) + self.eps[ij].reshape(-1,1)
                    - self.H.F[i,i] - self.H.F[j,j])

                    rms_t2 += contract('ZY,ZY->',r2_ij[ij],r2_ij[ij])

            rms = np.sqrt(rms_t2)
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

        Wmbej_ijim = []
        Wmbje_ijim = []
        Wmbie_ijmj = []
        Zmbij_ij = []

        r1_ii = []
        r2_ij = []

        Fae_ij = self.build_lFae(Fae_ij, self.Local.Fvv_ij, self.Local.Fov_ij, 
        self.Local.Lovvv_ij, self.Local.Loovv_ij, t1_ii, t2_ij)
        lFmi = self.build_lFmi(o, F, self.Local.Fov_ij, self.Local.Looov_ij, self.Local.Loovv_ij, t1_ii, t2_ij)
        Fme_ij = self.build_lFme(Fme_ij, self.Local.Fov_ij, self.Local.Loovv_ij, t1_ii)
        lWmnij = self.build_lWmnij(o, ERI, self.Local.ERIooov_ij, self.Local.ERIoovo_ij, 
        self.Local.ERIoovv_ij, t1_ii, t2_ij)
        Zmbij = self.build_lZmbij(Zmbij_ij, self.Local.ERIovvv_ij, t1_ii, t2_ij)
        Wmbej_ijim = self.build_lWmbej(Wmbej_ijim, self.Local.ERIoovv_ij, self.Local.ERIovvo_ij, 
        self.Local.ERIovvv_ij,self.Local.ERIoovo_ij, self.Local.Loovv_ij, t1_ii, t2_ij)
        Wmbje_ijim, Wmbie_ijmj = self.build_lWmbje(Wmbje_ijim, Wmbie_ijmj, self.Local.ERIovov_ij, 
        self.Local.ERIovvv_ij,self.Local.ERIoovv_ij, self.Local.ERIooov_ij, t1_ii, t2_ij)

        r1_ii = self.lr_T1(r1_ii, self.Local.Fov_ij , self.Local.ERIovvv_ij, self.Local.Lovvo_ij, self.Local.Loovo_ij, 
        t1_ii, t2_ij,Fae_ij, Fme_ij, lFmi)
        r2_ij = self.lr_T2(r2_ij, self.Local.ERIoovv_ij, self.Local.ERIvvvv_ij, self.Local.ERIovvo_ij, self.Local.ERIovoo_ij, 
        self.Local.ERIvvvo_ij,self.Local.ERIovov_ij, t1_ii, t2_ij, Fae_ij ,lFmi,Fme_ij, lWmnij, Zmbij_ij, Wmbej_ijim, Wmbje_ijim, Wmbie_ijmj)

        return r1_ii, r2_ij    

    def build_lFae(self, Fae_ij, Fvv_ij,Fov_ij, Lovvv_ij, Loovv_ij, t1_ii, t2_ij):
        #self.fae_t.start()
        o = self.o
        v = self.v
        QL = self.QL

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
        #self.fae_t.stop()
        return Fae_ij

    def build_lFmi(self, o, F, Fov_ij, Looov_ij, Loovv_ij, t1_ii, t2_ij):
        #self.fmi_t.start()
        v = self.v
        QL = self.QL

        Fmi = F[o,o].copy()

        Fmi_3 = np.zeros_like(Fmi)
        for j in range(self.no):
           for n in range(self.no):
               jn = j*self.no + n

               Fmi_3[:,j] += contract('EF,mEF->m',t2_ij[jn],Loovv_ij[jn][:,n,:,:])

        Fmi_tot = Fmi + Fmi_3
        #self.fmi_t.stop()
        return Fmi_tot 

    def build_lFme(self, Fme_ij, Fov_ij, Loovv_ij, t1_ii):
        #self.fme_t.start()
        #self.fme_t.stop()
        return

    def build_lWmnij(self, o, ERI, ERIooov_ij, ERIoovo_ij, ERIoovv_ij, t1_ii, t2_ij):
        #self.wmnij_t.start()

        Wmnij = ERI[o,o,o,o].copy()

        Wmnij_3 = np.zeros_like(Wmnij)
        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j

                Wmnij_3[:,:,i,j] += contract('ef,mnef->mn',t2_ij[ij], ERIoovv_ij[ij])

        Wmnij_tot = Wmnij + Wmnij_3
        #self.wmnij_t.stop()
        return Wmnij_tot

    def build_lZmbij(self, Zmbij_ij, ERIovvv_ij, t1_ii, t2_ij):
        #self.zmbij_t.start()
        #self.zmbij_t.stop()
        return

    def build_lWmbej(self, Wmbej_ijim, ERIoovv_ij, ERIovvo_ij, ERIovvv_ij, ERIoovo_ij, Loovv_ij, t1_ii, t2_ij):
        #self.wmbej_t.start()
        v = self.v
        o = self.o
        QL = self.QL
        dim = self.dim

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
        #self.wmbej_t.stop()
        return Wmbej_ijim

    def build_lWmbje(self, Wmbje_ijim,Wmbie_ijmj,ERIovov_ij, ERIovvv_ij, ERIoovv_ij, ERIooov_ij, t1_ii, t2_ij):
        #self.wmbje_t.start()
        o = self.o
        v = self.v
        QL = self.QL
        dim = self.dim

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
        #self.wmbje_t.stop()
        return Wmbje_ijim, Wmbie_ijmj

    def lr_T1(self, r1_ii, Fov_ij , ERIovvv_ij, Lovvo_ij, Loovo_ij, t1_ii, t2_ij, Fae_ij , Fme_ij, lFmi):
        #self.r1_t.start()
        for i in range(self.no):
            r1_ii.append(np.zeros_like(t1_ii[i]))
        #self.r1_t.stop()
        return r1_ii

    def lr_T2(self,r2_ij,ERIoovv_ij, ERIvvvv_ij, ERIovvo_ij, ERIovoo_ij, ERIvvvo_ij, ERIovov_ij, t1_ii,
    t2_ij, Fae_ij,lFmi,Fme_ij, lWmnij, Zmbij_ij, Wmbej_ijim, Wmbje_ijim, Wmbie_ijmj):
        #self.r2_t.start()
        v = self.v
        QL = self.QL
        dim = self.dim

        nr2_ij = []
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

        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j
                ji = j*self.no + i
 
                r2_ij.append(nr2_ij[ij].copy() + nr2_ij[ji].copy().transpose())
        
        #self.r2_t.stop()
        return r2_ij 

    def lcc_energy(self,Fov_ij,Loovv_ij,t1_ii,t2_ij):
        #self.energy_t.start()
        ecc_ij = 0
        ecc = 0

        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j

                ecc_ij = contract('ab,ab->',t2_ij[ij],Loovv_ij[ij][i,j])
                ecc += ecc_ij

        #self.energy_t.stop()
        return ecc
