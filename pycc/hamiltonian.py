if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import numpy as np


class Hamiltonian(object):

    def __init__(self, ref, Cp, Cr, Cq, Cs):

        npCp = np.asarray(Cp)
        npCr = np.asarray(Cr)

        # Generate MO Fock matrix
        self.F = np.asarray(ref.Fa())
        self.F = npCp.T @ self.F @ npCr

        # Get MO two-electron integrals in Dirac notation
        mints = psi4.core.MintsHelper(ref.basisset())
        self.ERI = np.asarray(mints.mo_eri(Cp, Cr, Cq, Cs))  # (pr|qs)
        self.ERI = self.ERI.swapaxes(1,2)                    # <pq|rs>
        self.L = 2.0 * self.ERI - self.ERI.swapaxes(2,3)     # 2 <pq|rs> - <pq|sr>

        self.mol = ref.molecule()
        self.basisset = ref.basisset()
        self.C_all = ref.Ca().to_array() # includes frozen core
        self.F_ao = ref.Fa().to_array()
