if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import numpy as np


class Hamiltonian(object):

    def __init__(self, ref, C):

        npC = np.asarray(C)

        # Generate MO Fock matrix
        self.F = np.asarray(ref.Fa())
        self.F = npC.T @ self.F @ npC

        # Get MO two-electron integrals in Dirac notation
        mints = psi4.core.MintsHelper(ref.basisset())
        self.ERI = np.asarray(mints.mo_eri(C, C, C, C))     # (pr|qs)
        self.ERI = self.ERI.swapaxes(1,2)                   # <pq|rs>
        self.L = 2.0 * self.ERI - self.ERI.swapaxes(2,3)    # 2 <pq|rs> - <pq|sr>
