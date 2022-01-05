import psi4
import numpy as np
from opt_einsum import contract


class Local(object):
    """
    A Local object for simulating different (virtual-space) localization schemes.

    Attributes
    ----------
    C: Psi4.core.Matrix object
        localized active orbital coefficients
    nfzc: int
        number of frozen core orbitals
    no: int
        number of (active) occupied orbitals
    nv: int
        number of virtual orbitals
    H: PyCC Hamiltonian object
        Hamiltonian object with integrals, Fock matrices, and orbital coefficients
    cutoff: double
        main cutoff parameter for virtual space localizer
    core_cut: double (optional)
        used in PAO calculations for determining the norm-cutoff of the virtual space
    lindep_cut: double (optional)
        used in PAO calculations for determining linear dependence threshold

    Parameters
    ----------
    local: string
        type of local calculation ("PAO", "LPNO", etc.)

    Methods
    -------
    filter_amps(): Returns t1 and t2 amplitude numpy arrays
        rotates amps to local vir space, applies CC step, and back-transforms
    filter_res(): Returns t1 and t2 amplitude numpy arrays
        applies forward and reverse localized-virtual transforms (no CC step)
    _build(): runs requested local build
    _build_PAO(): build PAO orbital rotation tensors
    _build_LPNO(): build LPNO orbital rotation tensors
    """

    def __init__(self, local, C, nfzc, no, nv, H, cutoff,
            core_cut=5E-2,
            lindep_cut=1E-6):

        self.cutoff = cutoff
        self.nfzc = nfzc
        self.no = no
        self.nv = nv
        self.H = H
        self.C = C.to_array()
        self.core_cut = core_cut
        self.lindep_cut = lindep_cut

        self._build(local)
    
    def _build(self,local):
        if local.upper() in ["LPNO"]:
            self._build_LPNO()
        elif local.upper() in ["PAO"]:
            self._build_PAO()
        else:
            raise Exception("Not a valid local type!")

    def _build_PAO(self):
        """
        Attributes
        ----------
        Q: transform between canonical VMO and PAO spaces
        L: transform between PAO and semicanonical PAO spaces
        dim: dimension of PAO space
        eps: semicananonical PAO energies

        Notes
        -----
        - Equation numbers from Hampel & Werner 1996 [10.1063/1.471289]
        - D includes the sum over the inactive occ orbs (C_{pk}@C_{qk}^T) 
          because we still need to project frzn core out of the virtual space.
          everywhere else, use only active occ space C
          (this matches the Psi3 implementation, if norm-cutting is removed)

        TODO
        ----
        - choose initial domains by charge (instead of just the first atom)
        - print domain atoms (not just # of functions)
        - simplify a2ao map generation
        """
        # SETUP
        bs = self.H.basisset
        mints = psi4.core.MintsHelper(bs)
        no_all = self.no + self.nfzc
        D = self.H.C_all[:,:no_all] @ self.H.C_all[:,:no_all].T
        S = mints.ao_overlap().to_array() 
        nao = self.no + self.nv + self.nfzc

        # map the number of atoms, basis functions per atom, and their indices
        a2ao = {} # atom-to-nao dict
        for i in range(bs.nshell()): # initialize for every atom
            a2ao[str(bs.shell_to_center(i))] = []
        natom = len(list(a2ao.keys()))
        a2nao = [0]*natom
        for i in range(bs.nshell()): # find nbasis per atom
            a2nao[bs.shell_to_center(i)] += bs.shell(i).nfunction
        for i in range(natom): # store basis indices for each atom
            bsi = [b+sum(a2nao[:i]) for b in range(a2nao[i])]
            a2ao[str(i)] = bsi

        # now, a2ao is a dict (keys str(0-natom)) w/ vals as indices
        # of AOs on each atom

        # SINGLES DOMAINS
        atom_domains = [] # atom indices
        AO_domains = []   # AO basis function indices
        for i in range(0,self.no):
            # population matrix for orbital i
            charges = [0]*natom
            for j in range(natom):
                for k in a2ao[str(j)]:
                    SC = contract('l,l->',S[k,:],self.C[:,i])
                    charges[j] += (SC*self.C[k,i])

            print("Charge analysis for occupied orbital %3d:" % i)
            print(np.round(charges,2))

            atoms = [i for i in range(natom)]
            zipped = zip(charges,atoms)
            sort = sorted(zipped, reverse=True)
            tups = zip(*sort)
            charges,atoms = [list(t) for t in tups] # sorted!

            # choose which atoms belong to the domain based on charge
            atom_domains.append([])
#            charge = 0
#            for n in range(0,natom):
#                atom_domains[i].append(atoms.pop(0))
#                charge += charges.pop(0)
#                if charge>1.8:
#                    break
            atom_domains[i].append(atoms.pop(0))

            # AOs associated with domain atoms
            AOi = []
            for n in atom_domains[i]:
                AOi += a2ao[str(n)]
            AOi = sorted(AOi) # set low-high ordering so two spaces with 
                              # the same orbitals will have them in the same order

            chk = 1
            while chk > self.cutoff:
                # Eq 8, SRp = SR
                # let A == S, B == SR
                # form and solve ARp = B
                A = np.zeros((len(AOi),len(AOi)))
                SB = np.zeros((len(AOi),nao))
                for x,a in enumerate(AOi):
                    for y,b in enumerate(AOi):
                        A[x,y] = S[a,b]
                    for y,b in enumerate(range(nao)):
                        SB[x,y] = S[a,b]
                B = contract('mp,p->m',SB,self.C[:,i])
                Rp = np.linalg.solve(A,B)
    
                # Eq 9, completeness check
                chk = 1 - contract('m,mn,n->',Rp,SB,self.C[:,i])
                print("BP completeness check: %.3f" % chk)

                if chk > self.cutoff:
                    try:
                        n = atoms.pop(0)
                        atom_domains[i].append(n)
                        AOi += a2ao[str(n)]
                        AOi = sorted(AOi) 
                    except IndexError:
                        print("Ran out of atoms. How did that happen?")
                        raise IndexError
                else:
                    print("Completeness threshold fulfilled.")
                    print("PAO domain %3d contains %3d/%3d orbitals." 
                            % (i,len(AOi),self.no+self.nv))
            AO_domains.append(AOi)

        # at this point, atom_domains contains the indices for each atom in the 
        # PAO space for each occupied orbital
        # and AO_domains contains the (sorted low->high) indices of the
        # AOs which make up the PAO space for each occupied orbital

        # Eq 3, total virtual-space projector
        Rt_full = np.eye(S.shape[0]) - contract('ik,kj->ij',D,S)

        # remove PAOs with negligible norms
        for i in range(nao):
            norm = np.linalg.norm(Rt_full[:,i]) # column-norm
            if norm < self.core_cut:
                print("Norm of orbital %4d = %20.12f... deleting" % (i,norm))
                Rt_full[:,i] = 0

        # Eq 5, first two terms RHS
        # R^+.S for virtual space, we will use this to compute the LMO->PAO 
        # transformation matrix
        RS = self.C[:,self.no:].T @ S

        Q = []  # MOij->PAOij
        L = []  # PAOij->PAOij(semi-canonical)
        eps = [] # semi-canonical orbital energies
        dim = [] # dimension of final PAO pair spaces
        for ij in range(self.no**2):
            i = ij // self.no
            j = ij % self.no
            ij_domain = list(set(AO_domains[i]+AO_domains[j]))

            # Eq 5, last term RHS for a given pair
            Rt = np.zeros((nao,len(ij_domain)))
            for x,a in enumerate(ij_domain):
                Rt[:,x] = Rt_full[:,a] # virtual-space projector for pair ij

            # Eq 73, MO -> PAO (redundant)
            # called the "Local residual vector" in psi3
            # equivalent to Q in PNO code
            # used to transform the LMO-basis residual matrix into the
            # projected (redundant, non-canonical) PAO basis
            V = contract('ap,pq->aq',RS,Rt)
            Q.append(V)

            # Eq 5, PAO -> semicanonical PAO
            # check for linear dependencies 
            St = contract('pq,pr,rs->qs',Rt,S,Rt) 
            evals,evecs = np.linalg.eigh(St)
            toss = np.abs(evals) < self.lindep_cut 

            # Eq 53, normalized nonredundant transform 
            # (still not semi-canonical)
            Xt = np.delete(evecs,toss,axis=1)
            evals = np.delete(evals,toss)
            for c in range(Xt.shape[1]):
                Xt[:,c] = Xt[:,c] / evals[c]**0.5
            dim.append(Xt.shape[1])

            # just below Eq 51, redundant PAO Fock 
            Ft = contract('pq,pr,rs->qs',Rt,self.H.F_ao,Rt)

            # Eq 54, non-redundant PAO Fock
            # diagonalize to get semi-canonical space
            Fbar = contract('pq,pr,rs->qs',Xt,Ft,Xt)
            evals,evecs = np.linalg.eigh(Fbar)

            # Eq 51, W 
            # rotates the redundant PAO-basis amplitudes 
            # directly into the into the non-redundant, semi-canonical basis
            W = contract('pq,qs->ps',Xt,evecs)

            eps.append(evals)
            L.append(W)

            print("Pair domain (%1d,%1d) contains %3d/%3d orbitals." 
                                % (i,j,len(ij_domain),nao))

        print("Average PAO dimension: %d" % (np.average(dim)))
        print("Number of canonical VMOs: %d" % (self.nv))

        self.Q = Q  # transform between canonical VMO and PAO spaces
        self.L = L  # transform between PAO and semicanonical PAO spaces
        self.dim = dim  # dimension of PAO space
        self.eps = eps  # semicananonical PAO energies

    def _build_LPNO(self):
        """
        Attributes
        ----------
        Q: transform between canonical VMO and LPNO spaces
        L: transform between LPNO and semicanonical LPNO spaces
        dim: dimension of LPNO space
        eps: semicananonical LPNO energies
        """

        o = slice(0, self.no)
        v = slice(self.no, self.no+self.nv)

        # Compute MP2 amplitudes in non-canonical MO basis
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # initial guess amplitudes
        t2 = self.H.ERI[o,o,v,v]/Dijab

        emp2 = contract('ijab,ijab->', t2, self.H.L[o,o,v,v])
        print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E" % (0, emp2, -emp2))

        e_conv = 1e-7
        r_conv = 1e-7
        maxiter = 100
        ediff = emp2
        rmsd = 0.0
        niter = 0

        while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
            elast = emp2

            r2 = 0.5 * self.H.ERI[o,o,v,v].copy()
            r2 += contract('ijae,be->ijab', t2, self.H.F[v,v])
            r2 -= contract('imab,mj->ijab', t2, self.H.F[o,o])
            r2 = r2 + r2.swapaxes(0,1).swapaxes(2,3)

            t2 += r2/Dijab

            rmsd = np.sqrt(contract('ijab,ijab->', r2/Dijab, r2/Dijab))

            emp2 = contract('ijab,ijab->', t2, self.H.L[o,o,v,v])
            ediff = emp2 - elast

            print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E  rmsd = % .5E" % (niter, emp2, ediff, rmsd))

        # Build LPNOs and store transformation matrices
        print("Computing PNOs.  Canonical VMO dim: %d" % (self.nv))
        T_ij = t2.copy().reshape((self.no*self.no, self.nv, self.nv))
        Tt_ij = 2.0 * T_ij - T_ij.swapaxes(1,2)
        D = np.zeros_like(T_ij)
        Q_full = np.zeros_like(T_ij)
        Q = []  # truncated LPNO list
        occ = np.zeros((self.no*self.no, self.nv))
        dim = np.zeros((self.no*self.no), dtype=int)  # dimension of LPNO space for each pair
        L = []
        eps = []
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no

            # Compute pair density
            D[ij] = contract('ab,bc->ac', T_ij[ij], Tt_ij[ij].T) + contract('ab,bc->ac', T_ij[ij].T, Tt_ij[ij])
            D[ij] *= 2.0/(1 + int(i == j))

            # Compute PNOs and truncate
            occ[ij], Q_full[ij] = np.linalg.eigh(D[ij])
            if (occ[ij] < 0).any(): # Check for negative occupation numbers
                neg = occ[ij][(occ[ij]<0)].min()
                print("Warning! Negative occupation numbers up to {} detected. \
                        Using absolute values - please check if your input is correct.".format(neg))
            dim[ij] = (np.abs(occ[ij]) > self.cutoff).sum()
            Q.append(Q_full[ij, :, (self.nv-dim[ij]):])

            # Compute semicanonical virtual space
            F = Q[ij].T @ self.H.F[v,v] @ Q[ij]  # Fock matrix in PNO basis
            eval, evec = np.linalg.eigh(F)
            eps.append(eval)
            L.append(evec)

            print("PNO dimension of pair %d = %d" % (ij, dim[ij]))

        print("Average PNO dimension: %d" % (np.average(dim)))
        print("Number of canonical VMOs: %d" % (self.nv))

        self.Q = Q  # transform between canonical VMO and LPNO spaces
        self.dim = dim  # dimension of LPNO space
        self.eps = eps  # semicananonical LPNO energies
        self.L = L  # transform between LPNO and semicanonical LPNO spaces

    def filter_amps(self, r1, r2):
        no = self.no
        nv = self.nv
        dim = self.dim

        t1 = np.zeros((no,nv))
        for i in range(no):
            ii = i * no + i

            X = self.Q[ii].T @ r1[i]
            Y = self.L[ii].T @ X

            for a in range(dim[ii]):
                Y[a] = Y[a]/(self.H.F[i,i] - self.eps[ii][a])

            X = self.L[ii] @ Y
            t1[i] = self.Q[ii] @ X

        t2 = np.zeros((no,no,nv,nv))
        for ij in range(no*no):
            i = ij // no
            j = ij % no

            X = self.Q[ij].T @ r2[i,j] @ self.Q[ij]
            Y = self.L[ij].T @ X @ self.L[ij]

            for a in range(dim[ij]):
                for b in range(dim[ij]):
                    Y[a,b] = Y[a,b]/(self.H.F[i,i] + self.H.F[j,j] - self.eps[ij][a] - self.eps[ij][b])

            X = self.L[ij] @ Y @ self.L[ij].T
            t2[i,j] = self.Q[ij] @ X @ self.Q[ij].T

        return t1, t2

    def filter_res(self, r1, r2):
        no = self.no
        nv = self.nv

        t1 = np.zeros((no,nv)).astype('complex128')
        for i in range(no):
            ii = i * no + i

            X = self.Q[ii].T @ r1[i]
            t1[i] = self.Q[ii] @ X

        t2 = np.zeros((no,no,nv,nv)).astype('complex128')
        for ij in range(no*no):
            i = ij // no
            j = ij % no

            X = self.Q[ij].T @ r2[i,j] @ self.Q[ij]
            t2[i,j] = self.Q[ij] @ X @ self.Q[ij].T

        return t1, t2
