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

        # Build the AO->SO transformation matrix by stacking per-irrep blocks
        # horizontally.  ref.aotoso() is a blocked psi4.core.Matrix with shape
        # (nao, nso_h) per irrep; to_array(dense=True) stacks them vertically
        # (incorrectly for our purposes), so we use nph[] directly instead.
        aotoso_raw = ref.aotoso()
        aotoso_full = np.hstack([np.array(aotoso_raw.nph[h])
                                 for h in range(aotoso_raw.nirrep())
                                 if np.array(aotoso_raw.nph[h]).shape[1] > 0])  # (nao, nso)

        # Generate MO Fock matrix.
        # ref.Fa() is in the SO basis; back-transform to AO basis, then to MO.
        # npCp contains AO->MO coefficients (from Ca_subset("AO", "ACTIVE")),
        # so we need F in the AO basis for the transform to be consistent.
        F_so = ref.Fa().to_array(dense=True)             # (nso, nso)
        F_ao = aotoso_full @ F_so @ aotoso_full.T        # (nao, nao)
        self.F = npCp.T @ F_ao @ npCr

        # Get MO two-electron integrals in Dirac notation.
        # mo_eri() expects AO-basis C matrices, which Cp and Cr already are.
        mints = psi4.core.MintsHelper(ref.basisset())
        self.ERI = np.asarray(mints.mo_eri(Cp, Cr, Cq, Cs))  # (pr|qs)
        self.ERI = self.ERI.swapaxes(1, 2)                     # <pq|rs>
        self.L = 2.0 * self.ERI - self.ERI.swapaxes(2, 3)

        self.mol = ref.molecule()
        self.basisset = ref.basisset()
        self.C_all = ref.Ca().to_array(dense=True)  # includes frozen core (SO basis)
        self.F_ao = F_ao

        ## One-electron property integrals
        # mints.ao_*() returns SO-basis matrices despite the "ao" prefix;
        # back-transform to AO basis before the MO transform.

        # Electric dipole integrals (length): -e r
        dipole_ints = mints.ao_dipole()
        self.mu = []
        for axis in range(3):
            mu_so = dipole_ints[axis].to_array(dense=True)
            mu_ao = aotoso_full @ mu_so @ aotoso_full.T
            self.mu.append(npCp.T @ mu_ao @ npCr)

        # Magnetic dipole integrals: -(e/2 m_e) L
        m_ints = mints.ao_angular_momentum()
        self.m = []
        for axis in range(3):
            m_so = m_ints[axis].to_array(dense=True)
            m_ao = aotoso_full @ m_so @ aotoso_full.T
            self.m.append((npCp.T @ (m_ao * -0.5) @ npCr) * 1.0j)

        # Linear momentum integrals: (-e)(-i hbar) Del
        p_ints = mints.ao_nabla()
        self.p = []
        for axis in range(3):
            p_so = p_ints[axis].to_array(dense=True)
            p_ao = aotoso_full @ p_so @ aotoso_full.T
            self.p.append((npCp.T @ p_ao @ npCr) * 1.0j)

        # Traceless quadrupole
        Q_ints = mints.ao_traceless_quadrupole()
        self.Q = []
        ij = 0
        for axis1 in range(3):
            for axis2 in range(axis1, 3):
                Q_so = Q_ints[ij].to_array(dense=True)
                Q_ao = aotoso_full @ Q_so @ aotoso_full.T
                self.Q.append(npCp.T @ Q_ao @ npCr)
                ij += 1
