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
                for j in range(self.no):
                    ij = i*self.no + j

                    self.t2[ij] -= r2[ij]/(self.eps[ij].reshape(1,-1) + self.eps[ij].reshape(-1,1)
                    - self.H.F[i,i] - self.H.F[j,j])

                    rms_t2 += contract('ZY,ZY->',r2[ij],r2[ij])

            rms = np.sqrt(rms_t2)
            elcc = self.lcc_energy(self.Local.Fov,self.Local.Loovv,self.t1, self.t2)
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
        Wmbje, Wmbie = self.build_Wmbje(Wmbje, Wmbie, ERI, self.Local.ERIoovo, self.Local.Sijin, self.Local.Sijjn, t1, t2)

        r1 = self.r_T1(r1, self.Local.Fov , ERI, L, self.Local.Loovo, self.Local.Siimm, self.Local.Siiim, self.Local.Siimn,  
        t1, t2, Fae, Fmi, Fme)
        r2 = self.r_T2(r2, ERI, self.Local.ERIoovv, self.Local.ERIvvvv, self.Local.ERIovoo, self.Local.Sijmm, self.Local.Sijim, 
        self.Local.Sijmj, self.Local.Sijmn, t1, t2, Fae ,Fmi, Fme, Wmnij, Zmbij, Wmbej, Wmbje, Wmbie)

        return r1, r2    

    def build_Fae(self, Fae_ij, L, Fvv, Fov, Sijmm, Sijmn, t1, t2):
        #self.fae_t.start()
        o = self.o
        v = self.v
        QL = self.QL

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
        #self.fae_t.stop()
        return Fae_ij

    def build_Fmi(self, o, F, L, Fov, Looov, Loovv, t1, t2):
        #self.fmi_t.start()
        v = self.v
        QL = self.QL
        Fmi = F[o,o].copy()

        for j in range(self.no):
           for n in range(self.no):
               jn = j*self.no + n

               Fmi[:,j] += contract('EF,mEF->m',t2[jn],Loovv[jn][:,n,:,:])

        #self.fmi_t.stop()
        return Fmi 

    def build_Fme(self, Fme_ij, L, Fov, t1):
        #self.fme_t.start()
        #self.fme_t.stop()
        return 

    def build_Wmnij(self, o, ERI, ERIooov, ERIoovo, ERIoovv, t1, t2):
        #self.wmnij_t.start()
        Wmnij = ERI[o,o,o,o].copy()

        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j

                Wmnij[:,:,i,j] += contract('ef,mnef->mn',t2[ij], ERIoovv[ij])

        #self.wmnij_t.stop()
        return Wmnij

    def build_Zmbij(self, Zmbij_ij, ERI, ERIovvv, t1, t2):
        #self.zmbij_t.start()
        #self.zmbij_t.stop()
        return

    def build_Wmbej(self, Wmbej_ijim, ERI, L, ERIoovo, Sijnn, Sijnj, Sijjn, t1, t2):
        #self.wmbej_t.start()
        v = self.v
        o = self.o
        QL = self.QL
        dim = self.dim

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
        #self.wmbej_t.stop()
        return Wmbej_ijim

    def build_Wmbje(self, Wmbje_ijim, Wmbie_ijmj, ERI, ERIooov, Sijin, Sijjn, t1, t2):
        #self.wmbje_t.start()
        o = self.o
        v = self.v
        QL = self.QL
        dim = self.dim

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
        #self.wmbje_t.stop()
        return Wmbje_ijim, Wmbie_ijmj

    def r_T1(self, r1_ii, Fov , ERI, L, Loovo, Siimm, Siiim, Siimn, t1, t2, Fae, Fmi, Fme):
        #self.r1_t.start()
        for i in range(self.no):
            r1_ii.append(np.zeros_like(t1[i]))
        #self.r1_t.stop()
        return r1_ii

    def r_T2(self,r2_ij, ERI, ERIoovv, ERIvvvv, ERIovoo, Sijmm, Sijim, Sijmj, Sijmn, t1, t2, Fae ,Fmi, Fme, Wmnij, Zmbij, Wmbej, Wmbje, Wmbie):
        #self.r2_t.start()
        v = self.v
        QL = self.QL
        dim = self.dim

        nr2 = []
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

        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j
                ji = j*self.no + i
 
                r2_ij.append(nr2[ij].copy() + nr2[ji].copy().transpose())
        
        #self.r2_t.stop()
        return r2_ij 

    def lcc_energy(self, Fov, Loovv, t1, t2):
        #self.energy_t.start()
        ecc_ij = 0
        ecc = 0

        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j

                ecc_ij = contract('ab,ab->',t2[ij],Loovv[ij][i,j])
                ecc += ecc_ij

        #self.energy_t.stop()
        return ecc
