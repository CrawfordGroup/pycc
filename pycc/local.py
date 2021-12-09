import psi4
import numpy as np
from opt_einsum import contract


class Local(object):

    def __init__(self, local, C, no, nv, H, cutoff):

        self.cutoff = cutoff
        self.no = no
        self.nv = nv
        self.H = H
        self.C = C.to_array()

        self._build(local)
    
    def _build(self,local):
        if local.upper() in ["LPNO"]:
            self._build_LPNO()
        elif local.upper() in ["PAO"]:
            self._build_PAO()
        else:
            raise Exception("Not a valid local type!")

    def _build_PAO(self):
        bs = self.H.basisset
        mints = psi4.core.MintsHelper(bs)
        D = self.C[:,:self.no] @ self.C[:,:self.no].T # 1/2 OPDM
        S = mints.ao_overlap().to_array() # overlap matrix

        a2ao = {} # atom-to-ao dict
        # forgive me Father, for this is the only way to get natom w/o the mol object
        for i in range(bs.nshell()):
            a2ao[str(bs.shell_to_center(i))] = []
        natom = len(list(a2ao.keys()))
        for i in range(bs.nshell()):
            a2ao[str(bs.shell_to_center(i))].append(i)

        # build total basis
        R = np.eye(D.shape[0]) - D @ S

        # build up domain for each occ orb
        domains = []

        for i in range(0,self.no):
            # population matrix for orbital i
            Pi = np.einsum('p,r->pr',D[:,i],S[:,i])
            dP = np.diag(Pi)

            # get relative charge for each atom
            charges = []
            for n in range(0,natom):
                nb = len(a2ao[str(n)])
                charges.append(self.H.mol.Z(n) - sum(dP[:nb]))
                dP = np.delete(dP,range(nb))
            print("Charge analysis:")
            print("Unsorted charges: {}".format(charges))

            atoms = [i for i in range(natom)]
            zipped = zip(charges,atoms)
            sort = sorted(zipped, reverse=True)
            tups = zip(*sort)
            charges,atoms = [list(t) for t in tups] # sorted!
            print(charges)
            print(atoms)

            # choose which atoms belong to the domain based on charge
            domains.append([])
            charge = 0
            for n in range(0,natom):
                domains[i].append(atoms.pop(n))
                charge += charges.pop(n)
                if charge>1.8:
                    break
            print("charge: {}".format(charge))
            print("domain: {}".format(domains[i]))

            # AOs associated with domain atoms
            AOi = []
            for n in domains[i]:
                AOi += a2ao[str(n)]
            print("domain basis functions: {}".format(AOi))

#            # grab the PAOs associated with domain atoms
#            Ri = R[:,domains[i]]
#
#            # form approximate MO i by solving SRi'=SRi==Z
#            Z = np.einsum('mr,ri->mi',S,Ri)
#            print(S[AOi][AOi].shape)
#            print(Z.shape)
#            Rip = np.linalg.solve(S[AOi][AOi],Z)
#
#            # completeness check
#            chk = 1 - np.einsum('mi,mn,ni->',Rip,S[AOi][:],Ri)
#            print(chk)

        pass

    def _build_LPNO(self):

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
