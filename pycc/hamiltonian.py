if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import numpy as np

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

class Hamiltonian(object):


    def __init__(self, ref, local=False):
        self.ref = ref

        # Get MOs
        C = self.ref.Ca_subset("AO", "ACTIVE")
        npC = np.asarray(C)  # as numpy array
        self.C = C

        # Localize occupied MOs if requested
        if (local == True):
            C_occ = self.ref.Ca_subset("AO", "ACTIVE_OCC")
            npC_occ = np.asarray(C_occ)
            no = self.ref.doccpi()[0] - self.ref.frzcpi()[0]  # assumes symmetry c1
            Local = psi4.core.Localizer.build("PIPEK_MEZEY", ref.basisset(), C_occ)
            Local.localize()
            npL = np.asarray(Local.L)
            npC[:,:no] = npL
            C = psi4.core.Matrix.from_array(npC)
            self.C = C

        # Generate MO Fock matrix
        self.F = np.asarray(self.ref.Fa())
        # self.F = np.einsum('uj,vi,uv', npC, npC, self.F)
        self.F = npC.T @ self.F @ npC

        # Get MO two-electron integrals in Dirac notation
        mints = psi4.core.MintsHelper(self.ref.basisset())
        self.ERI = np.asarray(mints.mo_eri(C, C, C, C))     # (pr|qs)
        self.ERI = self.ERI.swapaxes(1,2)                   # <pq|rs>
        self.L = 2.0 * self.ERI - self.ERI.swapaxes(2,3)    # 2 <pq|rs> - <pq|sr>

        print("AO Fock matrix from reference wave function:")
        print(np.asarray(self.ref.Fa()))
        print("MO Fock matrix:")
        print(self.F)
