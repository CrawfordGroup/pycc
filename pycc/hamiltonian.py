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


class SpinOrbitalHamiltonian(object):
    """A molecular spin-orbital Hamiltonian object.

    The spin-orbital sibling of :class:`Hamiltonian`, used for open-shell
    (UHF/ROHF) references. It exposes the same attribute surface (``F``, ``ERI``,
    and the property integrals ``mu``/``m``/``p``/``Q``) so the correlation code
    can read either Hamiltonian uniformly -- with one deliberate difference: the
    spin-adapted ``L`` does not exist in spin orbitals, so there is no ``L``
    attribute. The natural object is the antisymmetrized ``ERI = <pq||rs>``.

    Spin orbitals are ordered ``[alpha-occ, beta-occ, alpha-vir, beta-vir]``; the
    caller supplies the per-spin-orbital ``spin`` (0=alpha, 1=beta) and ``spat``
    (index into that spin's spatial active space) maps. Everything is built by
    spin-masked assignment from the spatial MO integrals -- no Python loops over
    orbital quadruples.

    Attributes
    ----------
    F : NumPy array
        spin-orbital Fock matrix, block-diagonal in spin (can be non-diagonal
        within a spin block, e.g. ROHF/semicanonical -- not assumed diagonal)
    ERI : NumPy array
        antisymmetrized spin-orbital ERIs in Dirac notation: <pq||rs>
    mu, m, p, Q : list of NumPy array
        spin-orbital property integrals (electric dipole, magnetic dipole, linear
        momentum, traceless quadrupole), block-diagonal in spin. Built for parity
        with :class:`Hamiltonian`; used by later (deferred) response work, not by
        the energy.
    """
    def __init__(self, ref: Any, Ca: Any, Cb: Any, spin: Any, spat: Any,
                 nocc_a: int, nocc_b: int) -> None:
        npCa = np.asarray(Ca)
        npCb = np.asarray(Cb)
        nact = spin.shape[0]

        # Spin-orbital indices grouped by spin, and their spatial indices. ix_ on
        # these reproduces the spin-block structure without explicit loops.
        a = np.where(spin == 0)[0]   # spin orbitals that are alpha
        b = np.where(spin == 1)[0]   # spin orbitals that are beta
        sa = spat[a]                 # their spatial indices (alpha active space)
        sb = spat[b]                 # (beta active space)

        # Semicanonicalize each spin: form the MO Fock, diagonalize its occ-occ and
        # vir-vir blocks, and rotate the occupied/virtual MO column-blocks by the
        # eigenvectors. This makes each spin's Fock block-diagonal (occ and vir
        # separately), so the orbital energies -- and hence the non-iterative MP2/(T)/CC3
        # denominators -- are well-defined even for ROHF (whose occ-vir Fock block stays
        # nonzero, feeding the MP2 singles). The AO-basis Fock ("AO" subset) is
        # symmetry-collapsed to a single irrep, so it transforms cleanly under symmetry.
        # For a canonical reference (RHF/UHF) the blocks are already diagonal, so the
        # rotation is the identity -- a no-op. ROHF feeds identical Ca/Cb but distinct
        # Fa/Fb, so its semicanonical alpha/beta orbitals diverge (UHF-like), as intended.
        Fa_ao = np.asarray(ref.Fa_subset("AO"))
        Fb_ao = np.asarray(ref.Fb_subset("AO"))
        npCa, Fa = self._semicanonicalize(npCa, Fa_ao, nocc_a)
        npCb, Fb = self._semicanonicalize(npCb, Fb_ao, nocc_b)

        # Spin-orbital Fock: alpha/beta MO Fock placed on the matching spin blocks; the
        # alpha-beta blocks vanish.
        F = np.zeros((nact, nact))
        F[np.ix_(a, a)] = Fa[np.ix_(sa, sa)]
        F[np.ix_(b, b)] = Fb[np.ix_(sb, sb)]
        self.F = F

        # All subsequent AO->MO transforms use the semicanonical MOs.
        Ca = psi4.core.Matrix.from_array(npCa)
        Cb = psi4.core.Matrix.from_array(npCb)

        # Antisymmetrized two-electron integrals <pq||rs>. Build the chemist-notation
        # spin-orbital integral (pr|qs) first -- nonzero only when spin_p==spin_r and
        # spin_q==spin_s, i.e. on the four spin-conserving blocks -- then convert to
        # physicist <pq|rs> (swap the middle indices) and antisymmetrize.
        mints = psi4.core.MintsHelper(ref.basisset())
        ERI_AA = np.asarray(mints.mo_eri(Ca, Ca, Ca, Ca))  # (pr|qs), alpha electrons
        ERI_BB = np.asarray(mints.mo_eri(Cb, Cb, Cb, Cb))  # (pr|qs), beta electrons
        ERI_AB = np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb))  # (aa|bb)
        ERI_BA = ERI_AB.transpose(2, 3, 0, 1)              # (bb|aa)

        chem = np.zeros((nact, nact, nact, nact))
        chem[np.ix_(a, a, a, a)] = ERI_AA[np.ix_(sa, sa, sa, sa)]
        chem[np.ix_(b, b, b, b)] = ERI_BB[np.ix_(sb, sb, sb, sb)]
        chem[np.ix_(a, a, b, b)] = ERI_AB[np.ix_(sa, sa, sb, sb)]
        chem[np.ix_(b, b, a, a)] = ERI_BA[np.ix_(sb, sb, sa, sa)]

        phys = chem.swapaxes(1, 2)                  # (pr|qs) -> <pq|rs>
        self.ERI = phys - phys.swapaxes(2, 3)       # <pq|rs> - <pq|sr> = <pq||rs>

        self.mol = ref.molecule()
        self.basisset = ref.basisset()

        ## One-electron property integrals (block-diagonal in spin)

        def _spin_block(O_ao):
            Oa = npCa.T @ O_ao @ npCa
            Ob = npCb.T @ O_ao @ npCb
            O = np.zeros((nact, nact))
            O[np.ix_(a, a)] = Oa[np.ix_(sa, sa)]
            O[np.ix_(b, b)] = Ob[np.ix_(sb, sb)]
            return O

        # Electric dipole (length): -e r
        dipole_ints = mints.ao_dipole()
        self.mu = [_spin_block(np.asarray(dipole_ints[axis])) for axis in range(3)]

        # Magnetic dipole: -(e/2 m_e) L  (imaginary, matching Hamiltonian's convention)
        m_ints = mints.ao_angular_momentum()
        self.m = [_spin_block(np.asarray(m_ints[axis]) * -0.5) * 1.0j for axis in range(3)]

        # Linear momentum: (-e)(-i hbar) Del
        p_ints = mints.ao_nabla()
        self.p = [_spin_block(np.asarray(p_ints[axis])) * 1.0j for axis in range(3)]

        # Traceless quadrupole (6 unique Cartesian components)
        Q_ints = mints.ao_traceless_quadrupole()
        self.Q = [_spin_block(np.asarray(Q_ints[ij])) for ij in range(6)]

    @staticmethod
    def _semicanonicalize(npC, F_ao, nocc):
        """Rotate the occupied and virtual MO blocks of ``npC`` so the occ-occ and
        vir-vir blocks of the MO Fock become diagonal (semicanonical).

        ``npC`` columns are ordered [occupied (nocc), virtual]. Returns the rotated
        coefficients and the resulting MO Fock; its occ-occ and vir-vir blocks are
        diagonal (orbital energies on the diagonal), while the occ-vir block may be
        nonzero (as for ROHF -- this is what feeds the MP2 singles). For a canonical
        reference both blocks are already diagonal and the rotation is the identity.
        """
        Fmo = npC.T @ F_ao @ npC
        C = npC.copy()
        if nocc > 0:
            _, Uo = np.linalg.eigh(Fmo[:nocc, :nocc])
            C[:, :nocc] = npC[:, :nocc] @ Uo
        if nocc < npC.shape[1]:
            _, Uv = np.linalg.eigh(Fmo[nocc:, nocc:])
            C[:, nocc:] = npC[:, nocc:] @ Uv
        return C, C.T @ F_ao @ C
