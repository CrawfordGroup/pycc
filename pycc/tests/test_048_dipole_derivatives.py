"""
Test the RHF nuclear dipole derivatives / atomic polar tensors (HFwfn CPHF nuclear
response) against finite difference of the SCF dipole.

The dipole derivative d(mu_alpha)/d(X_A,beta) is sensitive to the occupied-virtual
orbital response (unlike the energy gradient, which is variationally insensitive to
it), so it validates the nuclear CPHF solution.
"""

import psi4
import pycc
import numpy as np
from ..data.molecules import *


def test_dipole_derivatives_h2o(rhf_wfn):
    """H2O STO-3G APTs vs finite difference of the SCF dipole (frame locked)."""
    wfn = rhf_wfn("H2O", "STO-3G", geom_extra="\nsymmetry c1\nnoreorient\nnocom",
                  e_convergence=1e-11, d_convergence=1e-11)
    analytic = pycc.HFwfn(wfn).dipole_derivatives()   # (natom, beta, alpha)

    # Reference: central finite difference of Psi4's SCF dipole under nuclear
    # displacement. Lock the molecular frame (fix com/orientation and stop Psi4 from
    # reinterpreting the cartesians from the internal coordinate entry) so a
    # displacement survives instead of being reoriented away.
    mol = wfn.molecule()
    mol.fix_com(True)
    mol.fix_orientation(True)
    mol.update_geometry()
    mol.reinterpret_coordentry(False)
    geom0 = np.asarray(mol.geometry())
    natom = mol.natom()
    h = 1e-4

    def scf_dipole(geom):
        mol.set_geometry(psi4.core.Matrix.from_array(geom))
        mol.update_geometry()
        psi4.energy('scf')
        return np.asarray(psi4.variable('SCF DIPOLE'))

    fd = np.zeros((natom, 3, 3))
    for A in range(natom):
        for beta in range(3):
            gp = geom0.copy(); gp[A, beta] += h
            gm = geom0.copy(); gm[A, beta] -= h
            fd[A, beta] = (scf_dipole(gp) - scf_dipole(gm)) / (2.0 * h)

    # restore the reference geometry
    mol.set_geometry(psi4.core.Matrix.from_array(geom0))
    mol.update_geometry()

    assert np.max(np.abs(analytic - fd)) < 1e-6
