if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import numpy as np


class Hamiltonian(object):
    """
    A molecular Hamiltonian object.

    Attributes
    ----------
    F : NumPy array
        MO-basis Fock matrix (can be non-diagonal)
    ERI : NumPy array
        MO-basis electron repulsion integrals in Dirac notation: <pq|rs>
    L : NumPy array
        MO-basis spin-adapted ERIs: L_pqrs = 2 <pq|rs> - <pq|sr>
    mu : NumPy array
        MO-basis electric dipole integrals (length)
    m : NumPy array
        MO-basis magnetic dipole integrals
    """
    def __init__(self, ref, Cp, Cr, Cq, Cs):

        npCp = np.asarray(Cp)
        npCr = np.asarray(Cr)
        npCq = np.asarray(Cq)
        npCs = np.asarray(Cs)

        # AO->SO transformation from Psi4's petite list.
        # Use to_array(dense=True) to get a single (nao, nso) matrix regardless
        # of the number of irreps — np.asarray() fails for nirrep > 1.
        aotoso_full = ref.aotoso().to_array(dense=True)   # (nao, nso)

        # Transform C matrices from SO basis to AO basis.
        npCp_ao = aotoso_full @ npCp
        npCr_ao = aotoso_full @ npCr
        npCq_ao = aotoso_full @ npCq
        npCs_ao = aotoso_full @ npCs

        # Rebuild Psi4 Matrix objects in the AO basis for mo_eri()
        Cp_ao = psi4.core.Matrix.from_array(npCp_ao)
        Cr_ao = psi4.core.Matrix.from_array(npCr_ao)
        Cq_ao = psi4.core.Matrix.from_array(npCq_ao)
        Cs_ao = psi4.core.Matrix.from_array(npCs_ao)

        # Generate MO Fock matrix.
        # ref.Fa() is SO-basis blocked; use to_array(dense=True) to get (nso, nso).
        F_so = ref.Fa().to_array(dense=True)             # (nso, nso)
        F_ao = aotoso_full @ F_so @ aotoso_full.T        # (nao, nao)
        self.F = npCp_ao.T @ F_ao @ npCr_ao

        # Get MO two-electron integrals in Dirac notation using AO-basis C matrices
        mints = psi4.core.MintsHelper(ref.basisset())
        self.ERI = np.asarray(mints.mo_eri(Cp_ao, Cr_ao, Cq_ao, Cs_ao))  # (pr|qs)
        self.ERI = self.ERI.swapaxes(1, 2)                                  # <pq|rs>
        self.L = 2.0 * self.ERI - self.ERI.swapaxes(2, 3)

        self.mol = ref.molecule()
        self.basisset = ref.basisset()
        self.C_all = ref.Ca().to_array()  # includes frozen core (SO basis)
        self.F_ao = F_ao                   # AO-basis Fock matrix

        ## One-electron property integrals
        # mints.ao_*() methods always return pure AO-basis matrices.

        # Electric dipole integrals (length): -e r
        dipole_ints = mints.ao_dipole()
        self.mu = []
        for axis in range(3):
            mu_so = dipole_ints[axis].to_array(dense=True)   # (nso, nso)
            mu_ao = aotoso_full @ mu_so @ aotoso_full.T       # (nao, nao)
            self.mu.append(npCp_ao.T @ mu_ao @ npCr_ao)

        # Magnetic dipole integrals: -(e/2 m_e) L
        m_ints = mints.ao_angular_momentum()
        self.m = []
        for axis in range(3):
            m_so = m_ints[axis].to_array(dense=True)
            m_ao = aotoso_full @ m_so @ aotoso_full.T
            m = npCp_ao.T @ (m_ao * -0.5) @ npCr_ao
            self.m.append(m * 1.0j)

        # Linear momentum integrals: (-e)(-i hbar) Del
        p_ints = mints.ao_nabla()
        self.p = []
        for axis in range(3):
            p_so = p_ints[axis].to_array(dense=True)
            p_ao = aotoso_full @ p_so @ aotoso_full.T
            p = npCp_ao.T @ p_ao @ npCr_ao
            self.p.append(p * 1.0j)

        # Traceless quadrupole
        Q_ints = mints.ao_traceless_quadrupole()
        self.Q = []
        ij = 0
        for axis1 in range(3):
            for axis2 in range(axis1, 3):
                Q_so = Q_ints[ij].to_array(dense=True)
                Q_ao = aotoso_full @ Q_so @ aotoso_full.T
                self.Q.append(npCp_ao.T @ Q_ao @ npCr_ao)
                ij += 1
