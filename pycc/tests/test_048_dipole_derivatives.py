"""
Test the RHF nuclear dipole derivatives / atomic polar tensors (HFwfn CPHF nuclear
response) against a finite-difference-validated frozen reference.

The dipole derivative d(mu_alpha)/d(X_A,beta) is sensitive to the occupied-virtual
orbital response (unlike the energy gradient, which is variationally insensitive to
it), so it validates the nuclear CPHF solution.
"""

import psi4
import pycc
import numpy as np
from ..data.molecules import *


# H2O/STO-3G RHF APT [3*atom + beta, alpha], frozen reference (frame locked; closed-shell, so it is
# platform-reproducible).  Validated (once) against a 7-point O(h^6) central finite difference of the
# SCF dipole under nuclear displacement, agreeing to ~5e-12 (see _findiff_apt, the regeneration recipe):
#   apt[A, beta, alpha] = d mu_alpha / d X_Ab.
APT_REF = np.array([[-0.4715839734,  0.0,          0.0],
                    [ 0.0,           0.0448743816, 0.0],
                    [ 0.0,           0.0,          0.1006383942],
                    [ 0.2357919867,  0.0,          0.0],
                    [ 0.0,          -0.0224371908, 0.1439765481],
                    [ 0.0,           0.2017507447, -0.0503191971],
                    [ 0.2357919867,  0.0,          0.0],
                    [ 0.0,          -0.0224371908, -0.1439765481],
                    [ 0.0,          -0.2017507447, -0.0503191971]])


def _findiff_apt(wfn, h=0.005):
    """Regeneration recipe for APT_REF (not run in the tests): a 7-point O(h^6) central finite
    difference of Psi4's SCF dipole under nuclear displacement.  Lock the molecular frame (fix
    com/orientation and stop Psi4 reinterpreting the cartesians from the internal coordinate entry)
    so a displacement survives instead of being reoriented away.  Restores the reference geometry."""
    mol = wfn.molecule()
    mol.fix_com(True)
    mol.fix_orientation(True)
    mol.update_geometry()
    mol.reinterpret_coordentry(False)
    geom0 = np.asarray(mol.geometry())
    natom = mol.natom()

    def scf_dipole(geom):
        mol.set_geometry(psi4.core.Matrix.from_array(geom))
        mol.update_geometry()
        psi4.energy('scf')
        return np.asarray(psi4.variable('SCF DIPOLE'))

    fd = np.zeros((natom, 3, 3))
    for A in range(natom):
        for beta in range(3):
            acc = np.zeros(3)
            for k, c in [(-3, -1), (-2, 9), (-1, -45), (1, 45), (2, -9), (3, 1)]:
                g = geom0.copy(); g[A, beta] += k * h
                acc += c * scf_dipole(g)
            fd[A, beta] = acc / (60.0 * h)
    mol.set_geometry(psi4.core.Matrix.from_array(geom0))
    mol.update_geometry()
    return fd.reshape(-1, 3)


def test_dipole_derivatives_h2o(rhf_wfn):
    """H2O STO-3G RHF APTs vs the finite-difference-validated frozen reference (frame locked)."""
    wfn = rhf_wfn("H2O", "STO-3G", geom_extra="\nsymmetry c1\nnoreorient\nnocom",
                  e_convergence=1e-11, d_convergence=1e-11)
    analytic = np.asarray(pycc.HFwfn(wfn).dipole_derivatives()).reshape(-1, 3)
    assert np.max(np.abs(analytic - APT_REF)) < 1e-8
