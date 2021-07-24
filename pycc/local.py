import numpy as np
from opt_einsum import contract
import scipy.linalg


class Local(object):

    def __init__(self, no, nv, H, cutoff):

        o = slice(0, no)
        v = slice(no, no+nv)

        ## Compute MP2 amplitudes in non-canonical MO basis
        eps_occ = np.diag(H.F)[o]
        eps_vir = np.diag(H.F)[v]
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # initial guess amplitudes
        t2 = H.ERI[o,o,v,v]/Dijab

        emp2 = contract('ijab,ijab->', t2, H.L[o,o,v,v])
        print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E" % (0, emp2, -emp2))

        e_conv=1e-7
        r_conv=1e-7
        maxiter=100
        ediff = emp2
        rmsd = 0.0
        niter = 0

        while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
            elast = emp2

            r2 = 0.5 * H.ERI[o,o,v,v].copy()
            r2 += contract('ijae,be->ijab', t2, H.F[v,v])
            r2 -= contract('imab,mj->ijab', t2, H.F[o,o])
            r2 = r2 + r2.swapaxes(0,1).swapaxes(2,3)

            t2 += r2/Dijab

            rmsd = np.sqrt(contract('ijab,ijab->', r2/Dijab, r2/Dijab))

            emp2 = contract('ijab,ijab->', t2, H.L[o,o,v,v])
            ediff = emp2 - elast

            print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E  rmsd = % .5E" % (niter, emp2, ediff, rmsd))

        ## Build LPNOs and store transformation matrices
        print("Computing PNOs.  Canonical VMO dim: %d" % (nv))
        T_ij = t2.copy().reshape((no*no, nv, nv))
        Tt_ij = 2.0 * T_ij - T_ij.swapaxes(1,2)
        D = np.zeros_like(T_ij)
        Q_full = np.zeros_like(T_ij)
        Q = []  # truncated LPNO list
        occ = np.zeros((no*no, nv))
        dim = np.zeros((no*no), dtype=int)  # dimension of LPNO space for each pair
        L = []
        eps = []
        for ij in range(no*no):
            i = ij // no
            j = ij % no

            # Compute pair density
            D[ij] = contract('ab,bc->ac', T_ij[ij], Tt_ij[ij].T) + contract('ab,bc->ac', T_ij[ij].T, Tt_ij[ij])
            D[ij] *= 2.0/(1 + int(i==j))

            # Compute PNOs and truncate
            occ[ij], Q_full[ij] = np.linalg.eigh(D[ij])
            dim[ij] = (occ[ij] > cutoff).sum()
            Q.append(Q_full[ij, :, (nv-dim[ij]):])

            # Compute semicanonical virtual space
            F = Q[ij].T @ H.F[v,v] @ Q[ij]  # Fock matrix in PNO basis
            eval, evec = np.linalg.eigh(F)
            eps.append(eval)
            L.append(evec)

            print("PNO dimension of pair %d = %d" % (ij, dim[ij]))

        print("Average PNO dimension: %d" % (np.average(dim)))
        print("Number of canonical VMOs: %d" % (nv))

        self.cutoff = cutoff
        self.no = no
        self.nv = nv
        self.H = H
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
        dim = self.dim

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

