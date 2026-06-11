from __future__ import annotations

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


from typing import Any

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
    def __init__(self, ref: Any, Cp: Any, Cr: Any, Cq: Any, Cs: Any) -> None:

        npCp = np.asarray(Cp)
        npCr = np.asarray(Cr)

        # Generate MO Fock matrix.
        F_ao = np.asarray(ref.Fa_subset("AO"))
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

        # Electric dipole integrals (length): -e r
        dipole_ints = mints.ao_dipole()
        self.mu = []
        for axis in range(3):
            mu_ao = np.asarray(dipole_ints[axis])
            self.mu.append(npCp.T @ mu_ao @ npCr)

        # Magnetic dipole integrals: -(e/2 m_e) L
        m_ints = mints.ao_angular_momentum()
        self.m = []
        for axis in range(3):
            m_ao = np.asarray(m_ints[axis])
            self.m.append((npCp.T @ (m_ao * -0.5) @ npCr) * 1.0j)

        # Linear momentum integrals: (-e)(-i hbar) Del
        p_ints = mints.ao_nabla()
        self.p = []
        for axis in range(3):
            p_ao = np.asarray(p_ints[axis])
            self.p.append((npCp.T @ p_ao @ npCr) * 1.0j)

        # Traceless quadrupole
        Q_ints = mints.ao_traceless_quadrupole()
        self.Q = []
        ij = 0
        for axis1 in range(3):
            for axis2 in range(axis1, 3):
                Q_ao = np.asarray(Q_ints[ij])
                self.Q.append(npCp.T @ Q_ao @ npCr)
                ij += 1
