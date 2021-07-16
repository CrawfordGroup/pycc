if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import numpy as np

class Hamiltonian(object):


    def __init__(self, ref, local=None):
        self.ref = ref

        if (local != None):
            C_occ = self.ref.Ca_subset("AO", "ACTIVE_OCC")
            Local = psi4.core.Localizer.build("PIPEK_MEZEY", ref.bassiset(), C_occ)

        # Get MOs
        C = self.ref.Ca_subset("AO", "ACTIVE")
        npC = np.asarray(C)  # as numpy array

        # Get MO Fock matrix
        self.F = np.asarray(self.ref.Fa())
        self.F = np.einsum('uj,vi,uv', npC, npC, self.F)

        # Get MO two-electron integrals in Dirac notation
        mints = psi4.core.MintsHelper(self.ref.basisset())
        self.ERI = np.asarray(mints.mo_eri(C, C, C, C))     # (pr|qs)
        self.ERI = self.ERI.swapaxes(1,2)                   # <pq|rs>
        self.L = 2.0 * self.ERI - self.ERI.swapaxes(2,3)    # 2 <pq|rs> - <pq|sr>
