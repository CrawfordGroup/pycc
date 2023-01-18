"""
ccwfn.py: CC T-amplitude Solver
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import time
import numpy as np
import h5py
from opt_einsum import contract
from utils import helper_diis
from hamiltonian import Hamiltonian
from local import Local
from timer import Timer
from debug import Debug

class lccwfn_test(object):
    """
    An RHF-CC wave function and energy object.

    Attributes
    ----------
    ref : Psi4 SCF Wavefunction object
        the reference wave function built by Psi4 energy() method
    eref : float
        the energy of the reference wave function (including nuclear repulsion contribution)
    nfzc : int
        the number of frozen core orbitals
    no : int
        the number of active occupied orbitals
    nv : int
        the number of active virtual orbitals
    nmo : int
        the number of active orbitals
    H : Hamiltonian object
        the normal-ordered Hamiltonian, which includes the Fock matrix, the ERIs, and the spin-adapted ERIs (L)
    o : NumPy slice
        occupied orbital subspace
    v : NumPy slice
        virtual orbital subspace
    Dia : NumPy array
        one-electron energy denominator
    Dijab : NumPy array
        two-electron energy denominator
    t1 : NumPy array
        T1 amplitudes
    t2 : NumPy array
        T2 amplitudes
    ecc | float
        the final CC correlation energy

    Methods
    -------
    solve_cc()
        Solves the CC T amplitude equations
    residuals()
        Computes the T1 and T2 residuals for a given set of amplitudes and Fock operator
    """

    def __init__(self, scf_wfn, **kwargs):
        """
        Parameters
        ----------
        scf_wfn : Psi4 Wavefunction Object
            computed by Psi4 energy() method

        Returns
        -------
        None
        """

        time_init = time.time()

        valid_cc_models = ['CCD', 'CC2', 'CCSD', 'CCSD(T)', 'CC3']
        model = kwargs.pop('model','CCSD')
        if model not in valid_cc_models:
            raise Exception("%s is not an allowed CC model." % (model))
        self.model = model
        
        # models requiring singles
        self.need_singles = ['CCSD', 'CCSD(T)']

        # models requiring T1-transformed integrals
        self.need_t1_transform = ['CC2', 'CC3']

        valid_local_models = [None, 'PNO', 'PAO','PNO++']
        local = kwargs.pop('local', None)
        # TODO: case-protect this kwarg
        if local not in valid_local_models:
            raise Exception("%s is not an allowed local-CC model." % (local))
        self.local = local
        self.local_cutoff = kwargs.pop('local_cutoff', 1e-5)

        valid_local_MOs = ['PIPEK_MEZEY', 'BOYS']
        local_MOs = kwargs.pop('local_mos', 'PIPEK_MEZEY')
        if local_MOs not in valid_local_MOs:
            raise Exception("%s is not an allowed MO localization method." % (local_MOs))
        self.local_MOs = local_MOs

        valid_init_t2 = [None,'OPT']
        init_t2 = kwargs.pop('init_t2', None)
        # TODO: case-protect this kwarg
        if init_t2 not in valid_init_t2:
            raise Exception("%s is not an allowed initial t2 amplitudes." % (init_t2))
        self.init_t2 = init_t2

        self.ref = scf_wfn
        self.eref = self.ref.energy()
        self.nfzc = self.ref.frzcpi()[0]                # assumes symmetry c1
        self.no = self.ref.doccpi()[0] - self.nfzc      # active occ; assumes closed-shell
        self.nmo = self.ref.nmo()                       # all MOs/AOs
        self.nv = self.nmo - self.no - self.nfzc        # active virt
        self.nact = self.no + self.nv                   # all active MOs

        print("NMO = %d; NACT = %d; NO = %d; NV = %d" % (self.nmo, self.nact, self.no, self.nv))

        # orbital subspaces
        self.o = slice(0, self.no)
        self.v = slice(self.no, self.nmo)
        print("self.no")
        print(self.no)
        # For convenience
        o = self.o
        v = self.v

        # Get MOs
        C = self.ref.Ca_subset("AO", "ACTIVE")
        npC = np.asarray(C)  # as numpy array
        self.C = C

        # Localize occupied MOs if requested
        if (local is not None):
            C_occ = self.ref.Ca_subset("AO", "ACTIVE_OCC")
            LMOS = psi4.core.Localizer.build(self.local_MOs, self.ref.basisset(), C_occ)
            LMOS.localize()
            npL = np.asarray(LMOS.L)
            npC[:,:self.no] = npL
            C = psi4.core.Matrix.from_array(npC)
            self.C = C
          
        self.H = Hamiltonian(self.ref, self.C, self.C, self.C, self.C)
        
        if local is not None:
            self.Local = Local(local, self.C, self.nfzc, self.no, self.nv, self.H, self.local_cutoff, self.init_t2)        
            self.transform_integral(o,v)
             
        self.Debug = Debug(self.no,self.nv) 
        # denominators
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        self.Dia = eps_occ.reshape(-1,1) - eps_vir
        self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        print("mp2 energy without truncation")
        t2_test = self.H.ERI[o,o,v,v]/self.Dijab

        print(contract('ijab,ijab->', t2_test, self.H.L[o,o,v,v]))

        # first-order amplitudes
        self.t1 = np.zeros((self.no, self.nv))

        if local is not None:
            t1_ii = []
            t2_ij = []
            emp2 = 0
  
            for i in range(self.no):
                ii = i*self.no + i                

                X = self.Local.Q[ii].T @ self.t1[i]
                t1_ii.append(self.Local.L[ii].T @ X)
           
                for j in range(self.no):
                    ij = i*self.no + j
 
                    X = self.Local.L[ij].T @ self.Local.Q[ij].T @ self.H.ERI[i,j,v,v] @ self.Local.Q[ij] @ self.Local.L[ij] 
                    t2_ij.append( -1*X/ (self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1)
                    - self.H.F[i,i] - self.H.F[j,j])) 

                    L_ij = 2.0 * t2_ij[ij] - t2_ij[ij].T
                    mp2_ij = contract('ab,ab->',self.ERIoovv_ij[ij][i,j], L_ij)  
                    emp2 += mp2_ij

            print("mp2 energy in the local basis")    
            print(emp2)
            self.t1_ii = t1_ii
            self.t2_ij = t2_ij

            #for ij in range(self.no*self.no):
                #i = ij // self.no
                #j = ij % self.no
                #ji = j*self.no + i 
       
                #print("t2_ij", ij)
                #print(t2_ij[ij])
                #print("t2_ji", ji)
                #print(t2_ij[ij])
  
        print("CC object initialized in %.3f seconds." % (time.time() - time_init))

    def solve_localcc(self, e_conv=1e-7,r_conv=1e-7,maxiter=1,max_diis=8,start_diis=8):
        lccsd_tstart = time.time()

        o = self.o 
        v = self.v        
        F = self.H.F
        L = self.H.L
        Dia = self.Dia
        Dijab = self.Dijab

        emp2 = 0
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
 
            L_ij = 2.0 * self.t2_ij[ij] - self.t2_ij[ij].T
            mp2_ij = contract('ab,ab->',self.ERIoovv_ij[ij][i,j], L_ij)
            emp2 += mp2_ij

        print("lmp2 energy before CC iteration")    
        print(emp2)
        
        #initialize variables for timing each function
        self.fae_t = Timer("Fae")
        self.fme_t = Timer("Fme") 
        self.fmi_t = Timer("Fmi")
        self.wmnij_t = Timer("Wmnij")
        self.zmbij_t = Timer("Zmbij")
        self.wmbej_t = Timer("Wmbej")
        self.wmbje_t = Timer("Wmbje")
        self.tau_t = Timer("tau")
        self.r1_t = Timer("r1")
        self.r2_t = Timer("r2")
        self.energy_t = Timer("energy") 

        ecc = self.lcc_energy(self.Fov_ij,self.Loovv_ij,self.t1_ii,self.t2_ij)
        print("CC Iter %3d: CC Ecorr = %.15f dE = % .5E MP2" % (0,ecc,-ecc)) 

        for niter in range(1, maxiter+1):

            ecc_last = ecc  

            r1_ii, r2_ij = self.local_residuals(self.t1_ii, self.t2_ij)
       
            rms = 0  
            rms_t1 = 0
            rms_t2 = 0

            for i in range(self.no):
                ii = i*self.no + i 
                
                for a in range(self.Local.dim[ii]):
                    self.t1_ii[i][a] += r1_ii[i][a]/(self.H.F[i,i] - self.Local.eps[ii][a])               

                rms_t1 += contract('Z,Z->',r1_ii[i], r1_ii[i]) 
                
                for j in range(self.no):
                    ij = i*self.no + j

                    self.t2_ij[ij] -= r2_ij[ij]/(self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) 
                    - self.H.F[i,i] - self.H.F[j,j])

                    rms_t2 += contract('ZY,ZY->',r2_ij[ij],r2_ij[ij])
            print("t2_ij[1]", 1)
            print(self.t2_ij[1])
            rms = np.sqrt(rms_t1 + rms_t2)
            ecc = self.lcc_energy(self.Fov_ij,self.Loovv_ij,self.t1_ii,self.t2_ij)
            ediff = ecc - ecc_last
            print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, ecc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nCC has converged in %.3f seconds.\n" % (time.time() - lccsd_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                print("E(%s) = %20.15f" % (self.model, ecc))
                print("E(TOT)  = %20.15f" % (ecc + self.eref))
                self.ecc = ecc
                print(Timer.timers)
                return ecc        
   
    def transform_integral(self,o,v):
        """
        Transforming all the necessary integrals to the semi-canonical PNO basis, stored in a list with length occ*occ 
        Naming scheme will have the tensor name with _ij such as Fov_ij
        """   
        trans_intstart = time.time() 
        #Initializing the transformation matrices 
        Q = self.Local.Q
        L = self.Local.L
        
        #contraction notation i,j,a,b typically MO; A,B,C,D virtual semicanonical PNO;
        
        #Initializing transformation and integral list 
        QL = []
        QLT = []

        Fov_ij = []
        Fvv_ij = []

        ERIoovo_ij = []
        ERIooov_ij = []
        ERIovvv_ij = []
        ERIvvvv_ij = []
        ERIoovv_ij = []
        ERIovvo_ij = []
        ERIvvvo_ij = []
        ERIovov_ij = []        
        ERIovoo_ij = [] 

        Loovv_ij = []
        Lovvv_ij = []
        Looov_ij = [] 
        Loovo_ij = []
        Lovvo_ij = [] 

        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no

            QL.append(Q[ij] @ L[ij])

            Fov_ij.append(self.H.F[o,v] @ QL[ij])
                
            Fvv_ij.append(L[ij].T @ Q[ij].T @ self.H.F[v,v] @ QL[ij])
          
            ERIoovo_ij.append(contract('ijak,aA->ijAk', self.H.ERI[o,o,v,o],QL[ij]))

            ERIooov_ij.append(contract('ijka,aA->ijkA', self.H.ERI[o,o,o,v],QL[ij]))
            
            tmp = contract('ijab,aA->ijAb',self.H.ERI[o,o,v,v], QL[ij])
            ERIoovv_ij.append(contract('ijAb,bB->ijAB',tmp,QL[ij]))

            tmp1 = contract('iabc,aA->iAbc',self.H.ERI[o,v,v,v], QL[ij])
            tmp2 = contract('iAbc,bB->iABc',tmp1, QL[ij])
            ERIovvv_ij.append(contract('iABc,cC->iABC',tmp2, QL[ij]))            
            
            tmp3 = contract('abcd,aA->Abcd',self.H.ERI[v,v,v,v], QL[ij])
            tmp4 = contract('Abcd,bB->ABcd',tmp3, QL[ij])
            tmp5 = contract('ABcd,cC->ABCd',tmp4, QL[ij])
            ERIvvvv_ij.append(contract('ABCd,dD->ABCD',tmp5, QL[ij]))         
            
            tmp6 = contract('iabj,aA->iAbj',self.H.ERI[o,v,v,o], QL[ij]) 
            ERIovvo_ij.append(contract('iAbj,bB->iABj',tmp6,QL[ij]))
            
            tmp7 = contract('abci,aA->Abci',self.H.ERI[v,v,v,o], QL[ij]) 
            tmp8 = contract('Abci,bB->ABci',tmp7, QL[ij])
            ERIvvvo_ij.append(contract('ABci,cC->ABCi',tmp8, QL[ij]))
            
            tmp9 = contract('iajb,aA->iAjb',self.H.ERI[o,v,o,v], QL[ij])
            ERIovov_ij.append(contract('iAjb,bB->iAjB', tmp9, QL[ij]))
            
            ERIovoo_ij.append(contract('iajk,aA->iAjk', self.H.ERI[o,v,o,o], QL[ij]))

            Loovo_ij.append(contract('ijak,aA->ijAk', self.H.L[o,o,v,o],QL[ij]))
            
            tmp10 = contract('ijab,aA->ijAb',self.H.L[o,o,v,v], QL[ij])
            Loovv_ij.append(contract('ijAb,bB->ijAB',tmp10,QL[ij]))
            
            tmp11 = contract('iabc,aA->iAbc',self.H.L[o,v,v,v], QL[ij])
            tmp12 = contract('iAbc,bB->iABc',tmp11, QL[ij])
            Lovvv_ij.append(contract('iABc,cC->iABC',tmp12, QL[ij]))
            
            Looov_ij.append(contract('ijka,aA->ijkA',self.H.L[o,o,o,v], QL[ij]))
            
            tmp13 = contract('iabj,aA->iAbj',self.H.L[o,v,v,o], QL[ij])
            Lovvo_ij.append(contract('iAbj,bB->iABj',tmp13,QL[ij]))

        #Storing the list to this class  
        self.QL = QL
        self.QLT = QLT 
        self.Fov_ij = Fov_ij
        self.Fvv_ij = Fvv_ij

        self.ERIoovo_ij = ERIoovo_ij
        self.ERIooov_ij = ERIooov_ij
        self.ERIovvv_ij = ERIovvv_ij 
        self.ERIvvvv_ij = ERIvvvv_ij
        self.ERIoovv_ij = ERIoovv_ij
        self.ERIovvo_ij = ERIovvo_ij
        self.ERIvvvo_ij = ERIvvvo_ij
        self.ERIovov_ij = ERIovov_ij 
        self.ERIovoo_ij = ERIovoo_ij      

        self.Loovv_ij = Loovv_ij 
        self.Lovvv_ij = Lovvv_ij
        self.Looov_ij = Looov_ij 
        self.Loovo_ij = Loovo_ij
        self.Lovvo_ij = Lovvo_ij

        print("Integrals transformed in %.3f seconds." % (time.time() - trans_intstart))

    def local_residuals(self, t1_ii, t2_ij):
        """
        Constructing the two- and four-index intermediates 
        Then evaluating the singles and doubles residuals, storing them in a list of length occ (single) and length occ*occ (doubles) 
        Naming scheme same as integrals where _ij is attach as a suffix to the intermeidate name ... special naming scheme for those involving 
        two different pair space ij and im -> _ijm      

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
        
        Fae_ij = self.build_lFae(Fae_ij, self.Fvv_ij, self.Fov_ij, self.Lovvv_ij, self.Loovv_ij, t1_ii, t2_ij)
        lFmi = self.build_lFmi(o, F, self.Fov_ij, self.Looov_ij, self.Loovv_ij, t1_ii, t2_ij)
        Fme_ij = self.build_lFme(Fme_ij, self.Fov_ij, self.Loovv_ij, t1_ii) 
        lWmnij = self.build_lWmnij(o, ERI, self.ERIooov_ij, self.ERIoovo_ij, self.ERIoovv_ij, t1_ii, t2_ij)
        Zmbij = self.build_lZmbij(Zmbij_ij, self.ERIovvv_ij, t1_ii, t2_ij)
        Wmbej_ijim = self.build_lWmbej(Wmbej_ijim, self.ERIoovv_ij, self.ERIovvo_ij, self.ERIovvv_ij, 
        self.ERIoovo_ij, self.Loovv_ij, t1_ii, t2_ij)
        Wmbje_ijim, Wmbie_ijmj = self.build_lWmbje(Wmbje_ijim, Wmbie_ijmj, self.ERIovov_ij, self.ERIovvv_ij, 
        self.ERIoovv_ij, self.ERIooov_ij, t1_ii, t2_ij)        
        
        r1_ii = self.lr_T1(r1_ii, self.Fov_ij , self.ERIovvv_ij, self.Lovvo_ij, self.Loovo_ij, t1_ii, t2_ij, 
        Fae_ij, Fme_ij, lFmi)
        r2_ij = self.lr_T2(r2_ij, self.ERIoovv_ij, self.ERIvvvv_ij, self.ERIovvo_ij, self.ERIovoo_ij, self.ERIvvvo_ij, 
        self.ERIovov_ij, t1_ii, t2_ij, Fae_ij ,lFmi,Fme_ij, lWmnij, Zmbij_ij, Wmbej_ijim, Wmbje_ijim, Wmbie_ijmj)

        return r1_ii, r2_ij

    def build_lFae(self, Fae_ij, Fvv_ij,Fov_ij, Lovvv_ij, Loovv_ij, t1_ii, t2_ij):
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms
        """
        self.fae_t.start()
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
        self.fae_t.stop()    
        return Fae_ij  
       
    def build_lFmi(self, o, F, Fov_ij, Looov_ij, Loovv_ij, t1_ii, t2_ij):
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms
        
        Concern/To do
        -------
        Need Fmi and Fmj but would generating the Fmi matrix be enough? Or need a seperate matrix for Fmj to call for?
        """
        self.fmi_t.start()
        v = self.v
        QL = self.QL
        
        Fmi = F[o,o].copy()

        Fmi_3 = np.zeros_like(Fmi)
        for j in range(self.no):
           for n in range(self.no):
               jn = j*self.no + n     
               Fmi_3[:,j] += contract('EF,mEF->m',t2_ij[jn],Loovv_ij[jn][:,n,:,:])

        Fmi_tot = Fmi + Fmi_3
        self.fmi_t.stop() 
        return Fmi_tot #Fmj_ij

    def build_lFme(self, Fme_ij, Fov_ij, Loovv_ij, t1_ii):
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms
        """
        self.fme_t.start()
        self.fme_t.stop()
        return 

    def build_lWmnij(self, o, ERI, ERIooov_ij, ERIoovo_ij, ERIoovv_ij, t1_ii, t2_ij):
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms
        """
        self.wmnij_t.start()
        Wmnij = ERI[o,o,o,o].copy()
 
        Wmnij_3 = np.zeros_like(Wmnij)
        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no + j
                
                Wmnij_3[:,:,i,j] += contract('ef,mnef->mn',self.build_ltau(ij,t1_ii,t2_ij), ERIoovv_ij[ij])

        Wmnij_tot = Wmnij + Wmnij_3
        self.wmnij_t.stop()
        return Wmnij_tot

    def build_lZmbij(self, Zmbij_ij, ERIovvv_ij, t1_ii, t2_ij): 
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms
        """
        self.zmbij_t.start()
        self.zmbij_t.stop()
        return
        
    def build_lWmbej(self, Wmbej_ijim, ERIoovv_ij, ERIovvo_ij, ERIovvv_ij, ERIoovo_ij, Loovv_ij, t1_ii, t2_ij):
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms
        """
        self.wmbej_t.start()
        v = self.v
        o = self.o
        QL = self.QL 
        Q = self.Local.Q
        L = self.Local.L
            
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no 
            jj = j*self.no + j  
           
            for m in range(self.no):
                im = i*self.no + m 
                
                Wmbej = np.zeros((self.Local.dim[ij],self.Local.dim[im]))

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

                    tmp5 = self.build_ltau(jn,t1_ii,t2_ij, 0.5, 1.0) @ Sijjn.T 
                    tmp6_1 = QL[im].T @ self.H.ERI[m,n,v,v]
                    tmp6_2 = tmp6_1 @ QL[jn]
                    Wmbej_3 -= tmp5.T @ tmp6_2.T 

                    Sijnj = QL[ij].T @ QL[nj]                    
                    
                    tmp7 = t2_ij[nj] @ Sijnj.T
                    tmp8_1 = QL[im].T @ self.H.L[m,n,v,v]
                    tmp8_2 = tmp8_1 @ QL[nj] 
                    Wmbej_4 += 0.5 * tmp7.T @ tmp8_2.T 
                    
                Wmbej_ijim.append(Wmbej + Wmbej_3 + Wmbej_4)
        self.wmbej_t.stop()
        return Wmbej_ijim

    def build_lWmbje(self, Wmbje_ijim,Wmbie_ijmj,ERIovov_ij, ERIovvv_ij, ERIoovv_ij, ERIooov_ij, t1_ii, t2_ij):
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms
        """
        self.wmbje_t.start()
        o = self.o
        v = self.v
        QL = self.QL

        for ij in range(self.no*self.no):
            i = ij // self.no 
            j = ij % self.no 
            ii = i*self.no + i
            jj = j*self.no + j 
 
            for m in range(self.no):
                im = i*self.no + m
                mj = m*self.no + j
                
                Wmbje = np.zeros(self.Local.dim[ij],self.Local.dim[im])
                Wmbie = np.zeros(self.Local.dim[ij],self.Local.dim[mj])

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
                    
                    tmp5 = self.build_ltau(jn,t1_ii,t2_ij, 0.5, 1.0) @ Sijjn.T
                    tmp6_1 = QL[jn].T @ self.H.ERI[m,n,v,v]
                    tmp6_2 = tmp6_1 @ QL[im]
                    Wmbje_3 += tmp5.T @ tmp6_2
             
                    tmp5_mj = self.build_ltau(_in,t1_ii,t2_ij, 0.5, 1.0) @ Sijin.T
                    tmp6_1mj = QL[_in].T @ self.H.ERI[m,n,v,v]
                    tmp6_2mj = tmp6_1mj @ QL[mj] 
                    Wmbie_3 += tmp5_mj.T @ tmp6_2mj

                Wmbje_ijim.append(Wmbje + Wmbje_3)
                Wmbie_ijmj.append(Wmbie + Wmbie_3)
        self.wmbje_t.stop()
        return Wmbje_ijim, Wmbie_ijmj

    def build_ltau(self,ij,t1_ii,t2_ij,fact1=1.0, fact2=1.0):
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms
        """
        self.tau_t.start()
        self.tau_t.stop()       
        return fact1 * t2_ij[ij] #+ fact1 * contract('a,b->ab',tmp,tmp1)

    def lr_T1(self, r1_ii, Fov_ij , ERIovvv_ij, Lovvo_ij, Loovo_ij, t1_ii, t2_ij, Fae_ij , Fme_ij, lFmi):
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms 
        """
        self.r1_t.start()
        for i in range(self.no):
            r1_ii.append(np.zeros_like(t1_ii[i]))
        self.r1_t.stop()
        return r1_ii

    def lr_T2(self,r2_ij,ERIoovv_ij, ERIvvvv_ij, ERIovvo_ij, ERIovoo_ij, ERIvvvo_ij, ERIovov_ij, t1_ii, 
    t2_ij, Fae_ij,lFmi,Fme_ij, lWmnij, Zmbij_ij, Wmbej_ijim, Wmbje_ijim, Wmbie_ijmj):
        """
        Implemented up to CCSD but debugging at the CCD level to narrow down terms
        """

        nr2_ij = []
        nr2T_ij = []
        r2_ij1 = []
        r2_1one_ij = []
        r2_2one_ij = []
        r2_3_ij = [] 
        r2_4_ij = []
        r2_6_ij = []
        r2_7_ij = [] 
        r2_8_ij = []
        self.r2_t.start()
        v = self.v
        QL = self.QL
        Q = self.Local.Q
        L = self.Local.L
        
        for ij in range(self.no*self.no):
            i = ij //self.no
            j = ij % self.no
            ii = i*self.no + i 
            jj = j*self.no + j
       
            r2 = np.zeros(self.Local.dim[ij],self.Local.dim[ij])
            r2_1one = np.zeros_like(r2)
            r2_4 = np.zeros_like(r2)
 
            r2 = 0.5 * ERIoovv_ij[ij][i,j]
 
            r2_1one = t2_ij[ij] @ Fae_ij[ij].T 

            r2_4 = 0.5 * contract('ef,abef->ab',self.build_ltau(ij,t1_ii,t2_ij),ERIvvvv_ij[ij])

            r2_2one = np.zeros_like(r2)
            r2_3 = np.zeros_like(r2)
            r2_6 = np.zeros_like(r2) 
            r2_7 = np.zeros_like(r2)
            r2_8 = np.zeros_like(r2)
            for m in range(self.no):
                mm = m *self.no + m
                im = i*self.no + m
                ijm = ij*self.no + m

                Sijmm = QL[ij].T @ QL[mm]

                tmp = Sijmm @ t1_ii[m]   

                im = i*self.no + m 
                Sijim = QL[ij].T @ QL[im]

                tmp2_1 = Sijim @ t2_ij[im]            
                tmp2_2 = tmp2_1 @ Sijim.T 
                r2_2one -= tmp2_2 * lFmi[m,j]

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
        
                for n in range(self.no): 
                    mn = m*self.no + n
                    
                    Sijmn = QL[ij].T @ QL[mn]
                    
                    tmp4_1 = Sijmn @ self.build_ltau(mn,t1_ii,t2_ij) 
                    tmp4_2 = tmp4_1 @ Sijmn.T 
                    r2_3 += 0.5 * tmp4_2 * lWmnij[m,n,i,j]
            r2_ij1.append(r2)
            r2_1one_ij.append(r2_1one)
            r2_2one_ij.append(r2_2one)
            r2_3_ij.append(r2_3)
            r2_4_ij.append(r2_4) 
            r2_6_ij.append(r2_6)
            r2_7_ij.append(r2_7)
            r2_8_ij.append(r2_8)
            nr2_ij.append(r2 + r2_1one + r2_2one + r2_3 + r2_4 + r2_6 + r2_7 + r2_8)
        
        for i in range(self.no):
            for j in range(self.no): 
                ij = i*self.no + j 
                ji = j*self.no + i 
  
                nr2T_ij = np.zeros_like(nr2_ij[ij])
                for a in range(self.Local.dim[ij]):
                    for b in range(self.Local.dim[ij]):
                        if a == b: 
                            nr2T_ij[b][a] = nr2_ij[ij][a][b]
                        else:
                            nr2T_ij[b][a] = -1.0 * nr2_ij[ij][a][b]
      
                #r2_ij.append(nr2_ij[ij].copy() + nr2T_ij)
                r2_ij.append(nr2_ij[ij].copy() + nr2_ij[ji].copy().transpose())

                if ij == 8:

                    #self.Debug._store_t2(r2_ij1[ij], "r2")                    

                    #self.Debug._store_t2(r2_1one_ij[ij], "r2_1one")

                    #self.Debug._store_t2(r2_2one_ij[ij], "r2_2one")
   
                    #self.Debug._store_t2(r2_3_ij[ij], "r2_3")

                    #self.Debug._store_t2(r2_4_ij[ij], "r2_4")

                    #self.Debug._store_t2(r2_6_ij[ij], "r2_6")

                    #self.Debug._store_t2(r2_7_ij[ij], "r2_7")
   
                    #self.Debug._store_t2(r2_8_ij[ij], "r2_8")

                    self.Debug._store_t2(nr2_ij[ij], "nr2")
                    print("nr2_ij", nr2_ij[ij])

                    self.Debug._store_t2(nr2_ij[ji].transpose(), "nr2_ji")
                    print("nr2_ji.T", nr2_ij[ji].copy().transpose())        

                    #for a in range(self.Local.dim[ij]):
                        #for b in range(self.Local.dim[ij]):
                            #if a == b: 
                                 #continue
                            #else:
                                #nr2_ij[ji][b][a] = -1.0 * nr2_ij[ij][a][b]
                    #print("nr2_ji using manual", nr2_ij[ji])

                    self.Debug._store_t2(r2_ij[ij], "tot_r2")
                    #print("r2_ij[ij]", r2_ij[ij]) 
  
        self.r2_t.stop()
        return r2_ij

    def lcc_energy(self,Fov_ij,Loovv_ij,t1_ii,t2_ij): 
        self.energy_t.start()
        ecc_ij = 0
        ecc = 0
        
        for i in range(self.no): 
            for j in range(self.no):    
                ij = i*self.no + j
                #ltau = self.build_ltau(ij,t1_ii,t2_ij)
                ecc_ij = contract('ab,ab->',t2_ij[ij],Loovv_ij[ij][i,j]) 
                #print("ecc_ij", ij, ecc_ij)
                self.Debug._store_Eij(ij, ecc_ij)
                #if ij == 8:
                    #print("t2_ij", ij, t2_ij[ij])
                    #print("ltau", ltau)
                    #print("Loovv_ij", Loovv_ij[ij][i,j])             
                ecc += ecc_ij
        self.energy_t.stop()
        return ecc

