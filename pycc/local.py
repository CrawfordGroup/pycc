import psi4
import numpy as np
from opt_einsum import contract


class Local(object):

    def __init__(self, no, nv, H, cutoff, extent=False):

        o = slice(0, no)
        v = slice(no, no+nv)

        # Compute MP2 amplitudes in non-canonical MO basis
        eps_occ = np.diag(H.F)[o]
        eps_vir = np.diag(H.F)[v]
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # initial guess amplitudes
        t2 = H.ERI[o,o,v,v]/Dijab

        emp2 = contract('ijab,ijab->', t2, H.L[o,o,v,v])
        print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E" % (0, emp2, -emp2))

        e_conv = 1e-7
        r_conv = 1e-7
        maxiter = 100
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

        # Build LPNOs and store transformation matrices
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

        if extent:
            # determine orbital extent
            mints = psi4.core.MintsHelper(H.ref.basisset())

            # AO-basis quadrupole integrals (XX + YY + ZZ)
            # stored as upper triangle[[0,1,2],[.,3,4],[.,.,5]]
            C = H.C.to_array()
            ao_r2 = mints.ao_quadrupole()[0].to_array()
            ao_r2 += mints.ao_quadrupole()[3].to_array()
            ao_r2 += mints.ao_quadrupole()[5].to_array()

            # to (localized-occ) MO basis
            mo_r2 = contract('ki,kl,lj->ij',C,ao_r2,C)

        for ij in range(no*no):
            i = ij // no
            j = ij % no

            # Compute pair density
            D[ij] = contract('ab,bc->ac', T_ij[ij], Tt_ij[ij].T) + contract('ab,bc->ac', T_ij[ij].T, Tt_ij[ij])
            D[ij] *= 2.0/(1 + int(i == j))

            # Compute PNOs and truncate
            occ[ij], Q_full[ij] = np.linalg.eigh(D[ij])
            print("Q:\n{}".format(Q_full[ij]))
            print("OCC N:\n{}".format(occ[ij]))

            if extent:
                # orb extents to PNO basis
                r2 = contract('Aa,ab,bB->AB',Q_full[ij].T,mo_r2[no:,no:],Q_full[ij])

                print("Orbital extents:")
                print(r2.diagonal())

                # now zip these against the orbitals, pull a few w/ the highest extent,
                # then truncate from the remaining set by occ number and combine the two
                # transpose of Q is needed to sort by columns instead of rows
                zipped = zip(np.abs(r2.diagonal()),occ[ij],list(Q_full[ij].T))
                by_r2 = sorted(zipped)
                tups = zip(*by_r2)
                sorted_r2, sorted_occ, sorted_q = [list(tup) for tup in tups]

                print("sorted_Q:\n{}".format(sorted_q))
                print("sorted_occ:\n{}".format(sorted_occ))
                print("sorted_r2:\n{}".format(sorted_r2))

                # take the _n orbitals with the highest spatial extent
                # transpose to get "rows" (inner index of 2d list) back to columns
                ext_space = np.asarray(sorted_q[-extent:]).T
                int_space = np.asarray(sorted_q[:-extent]).T
                int_occ = np.asarray(sorted_occ[:-extent]).T
                # below for if extent=0, but (if extent) keeps this case from happening now
                #else: # a[-_n:] gives the entire space, which would reserve ALL orbitals
                #    ext_space = np.asarray([])
                #    int_space = np.asarray(sorted_q).T
                #    int_occ = np.asarray(sorted_occ).T

                print("external space:\n{}".format(ext_space))
                print("internal space:\n{}".format(int_space))
                print("internal occ:\n{}".format(int_occ))
    
                # get PNOs as normal from the internal space
                checks = np.abs(int_occ)>cutoff
                dim_int = checks.sum()
                cuts = [i for i, x in enumerate(checks) if not x]
                print("PNO CUTS:\n{}".format(cuts))
                int_space = np.delete(int_space,cuts,axis=1)
                if _n:
                    Q.append(np.append(int_space,ext_space,axis=1))
                else:
                    Q.append(int_space)
                dim[ij] = dim_int + _n 

                print("Final space:\n{}".format(Q[-1]))
                test = Q[-1].T @ Q[-1]
                print("Should be 1's:\n{}".format(sum(test>1e-15)))

            else:
                if (occ[ij] < 0).any(): # Check for negative occupation numbers
                    neg = occ[ij][(occ[ij]<0)].min()
                    print("Warning! Negative occupation numbers up to {} detected. \
                            Using absolute values - please check if your input is correct.".format(neg))
                dim[ij] = (np.abs(occ[ij]) > cutoff).sum()
                Q.append(Q_full[ij, :, (nv-dim[ij]):])

                # check if the PNO-cutoff method above works in the regular case (it does)
#                checks = np.abs(occ[ij])>cutoff
#                dim[ij] = checks.sum()
#                cuts = [i for i, x in enumerate(checks) if not x]
#                Q.append(np.delete(Q_full[ij],cuts,axis=1))

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
