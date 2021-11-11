import numpy as np
from opt_einsum import contract


class LocalCC(object):

    def __init__(self, ccwfn, cclambda, cutoff):

        no = ccwfn.no
        nv = ccwfn.nv
        o = slice(0, no)
        v = slice(no, no+nv)

        # Build LPNOs and store transformation matrices
        print("Computing PNOs.  Canonical VMO dim: %d" % (nv))
        T_ij = ccwfn.t2.copy().reshape((no*no, nv, nv))
        L_ij = cclambda.l2.copy().reshape((no*no, nv, nv))
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
            pd = contract('mb,ma->ab',ccwfn.t1,cclambda.l1) + contract('be,ae->ab', T_ij[ij], L_ij[ij]) 
#            print(np.allclose(pd,pd.T))
            D[ij] = (pd + pd.T) / 2 # symmetrize
#            print(np.allclose(D[ij],D[ij].T))
#            print(np.linalg.cholesky(D[ij]+np.eye(D[ij].shape[0])*1E-16))

            # Compute PNOs and truncate
            occ[ij], Q_full[ij] = np.linalg.eigh(D[ij])
#            print(occ[ij])
            if (occ[ij] < 0).any(): # Check for negative occupation numbers
                neg = occ[ij][(occ[ij]<0)].min()
                print("Warning! Negative occupation numbers up to {} detected. \
                        Using absolute values - please check if your input is correct.".format(neg))
#            dim[ij] = (np.abs(occ[ij]) > cutoff).sum()
#            Q.append(Q_full[ij, :, (nv-dim[ij]):])


            checks = np.abs(occ[ij])>cutoff
            dim[ij] = checks.sum()
            cuts = [i for i, x in enumerate(checks) if not x]
            Q.append(np.delete(Q_full[ij],cuts,axis=1))
#            print(cuts)
#            print(Q_full[ij].shape)
#            print(Q[-1].shape)

            # Compute semicanonical virtual space
            F = Q[ij].T @ ccwfn.H.F[v,v] @ Q[ij]  # Fock matrix in PNO basis
            eval, evec = np.linalg.eigh(F)
            eps.append(eval)
            L.append(evec)

            print("PNO dimension of pair %d = %d" % (ij, dim[ij]))

        print("Average PNO dimension: %d" % (np.average(dim)))
        print("Number of canonical VMOs: %d" % (nv))

        self.cutoff = cutoff
        self.no = no
        self.nv = nv
        self.H = ccwfn.H
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
