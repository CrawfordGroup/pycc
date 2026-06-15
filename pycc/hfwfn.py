"""
hfwfn.py: Hartree-Fock wavefunction and MO-basis analytic derivative properties.
"""

from __future__ import annotations

from typing import Any

import psi4
import numpy as np

from .wavefunction import Wavefunction
from .derivatives import Derivatives
from .cphf import CPHF
from .utils import diag


class HFwfn(Wavefunction):
    """An RHF wavefunction on the shared :class:`Wavefunction` base, and the home
    for MO-basis HF analytic derivative properties (gradient now; Hessian, APTs,
    AATs, and a CPHF solver to follow).

    The SCF energy is the base's reference energy (``self.eref``); this class adds
    the response/derivative engine, including a :class:`Derivatives` provider.

    Attributes
    ----------
    derivatives : Derivatives
        MO-basis derivative-integral provider (promotable to the base later)
    cphf : CPHF
        coupled-perturbed HF orbital-response solver (promotable to the base later)
    grad : numpy.ndarray
        the most recently computed nuclear gradient, shape (natom, 3)
    """

    def __init__(self, scf_wfn: Any, **kwargs) -> None:
        # HF properties are all-electron: always use the full MO space, regardless
        # of any frozen core the reference was run with.
        kwargs.pop('frozen_core', None)
        super().__init__(scf_wfn, frozen_core=False, **kwargs)
        self.derivatives = Derivatives(self)
        self.cphf = CPHF(self)

    def gradient(self) -> np.ndarray:
        """RHF analytic energy gradient (a.u.), shape (natom, 3).

        Closed-shell, MO-basis, CPHF-free RHF gradient (i, j over occupied; the
        ``^x`` skeleton derivatives come from :class:`Derivatives`, ``eps_i`` are the
        occupied orbital energies, and the ``-2 eps_i S^x_ii`` term is the
        energy-weighted-density / orbital-response contribution that makes the HF
        gradient CPHF-free)::

            dE/dX = sum_i 2 h^x_ii + sum_ij (2 (ii|jj)^x - (ij|ij)^x)
                    - sum_i 2 eps_i S^x_ii + dV_NN/dX

        The derivative integrals are transformed with the base's symmetry-handled
        ``self.C`` (single irrep block, global energy order), so this works with
        molecular symmetry left on. HFwfn always uses the full (all-electron) MO
        space, so a frozen core on the reference does not affect the gradient.
        """
        o = self.o
        no = self.no
        # Occupied block of the symmetry-handled MO coefficients, and the matching
        # (energy-ordered) occupied orbital energies from the Fock diagonal.
        Cocc = psi4.core.Matrix.from_array(np.asarray(self.C)[:, :no])
        eps = np.asarray(diag(self.H.F))[o]

        d = self.derivatives
        grad = np.zeros((d.natom, 3))
        Vnn = d.nuclear_repulsion()
        for atom in range(d.natom):
            Sx = d.overlap(atom, Cocc, Cocc)
            hx = d.core(atom, Cocc, Cocc)
            erix = d.eri(atom, Cocc, Cocc, Cocc, Cocc)
            for c in range(3):
                grad[atom, c] = (2.0 * np.trace(hx[c])
                                 + 2.0 * np.einsum('iijj->', erix[c])
                                 - np.einsum('ijij->', erix[c])
                                 - 2.0 * np.einsum('i,ii->', eps, Sx[c])
                                 + Vnn[atom, c])
        self.grad = grad
        return grad

    def polarizability(self) -> np.ndarray:
        """Static electric-dipole polarizability tensor (a.u.), shape (3, 3).

        Solves the electric-field CPHF orbital response (:class:`CPHF`) and contracts
        with the MO dipole integrals. This is the first validation target for the
        CPHF machinery: the field perturbation does not move the basis functions, so
        the response has no overlap/Pulay contribution.
        """
        self.alpha = self.cphf.polarizability()
        return self.alpha
