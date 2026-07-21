"""
CCSD analytic nuclear gradient -- pycc.gradient(ccwfn) / CCderiv, spatial closed-shell RHF.

Anchored on tight finite differences of pycc's own CCSD correlation energy: each frozen reference is
the analytic correlation gradient itself, validated once against a 5-point O(h^4) central FD of the
CCSD correlation energy (h=0.005, CC converged to 1e-13) to ~6e-12 (see _findiff_gradient, the
regeneration recipe).  All-electron and frozen-core; STO-3G and cc-pVDZ (a real virtual space --
polarization functions, several virtuals per irrep, A2-symmetry MOs).  A further pycc-vs-pycc
cross-check stays live: the gradient-convention densities reconstructing the CCSD correlation energy.
"""

import contextlib
import os

import numpy as np
import psi4
import pycc

# H2O, fixed frame (bohr), from the original test geometry (0.118 / 0.758 / 0.472 Angstrom).
ANG2BOHR = 1.8897259886
ATOMS = ['O', 'H', 'H']
REF = ANG2BOHR * np.array([
    [0.0,  0.0,     0.118],
    [0.0,  0.758,  -0.472],
    [0.0, -0.758,  -0.472],
])

# Frozen references: the analytic CCSD *correlation* gradient (a.u.; frame locked; closed-shell, so
# platform-reproducible to ~1e-14).  Each validated once against a 5-point O(h^4) FD of the CCSD
# correlation energy (see _findiff_gradient) -- STO-3G AE to 5.7e-12, cc-pVDZ AE to 2.5e-12,
# STO-3G FC to 6.7e-12.
GRAD_REF_STO3G = np.array([
    [0.0,  0.0,                -4.970192865099e-02],
    [0.0, -2.344510629594e-02,  2.485096432549e-02],
    [0.0,  2.344510629594e-02,  2.485096432549e-02],
])
GRAD_REF_PVDZ = np.array([
    [0.0,  0.0,                -2.716560855565e-02],
    [0.0, -1.281727832151e-02,  1.358280427783e-02],
    [0.0,  1.281727832151e-02,  1.358280427783e-02],
])
GRAD_REF_FC_STO3G = np.array([
    [0.0,  0.0,                -4.974459331509e-02],
    [0.0, -2.346970548696e-02,  2.487229665755e-02],
    [0.0,  2.346970548696e-02,  2.487229665755e-02],
])


def _geom(coords):
    body = "\n".join(f"{s} {c[0]:.15f} {c[1]:.15f} {c[2]:.15f}" for s, c in zip(ATOMS, coords))
    return body + "\nsymmetry c1\nunits bohr\nno_com\nno_reorient\n"


def _scf_wfn(coords, basis, frozen_core=False):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.be_quiet()
    psi4.geometry(_geom(coords))
    psi4.set_options({'basis': basis, 'scf_type': 'pk',
                      'freeze_core': 'true' if frozen_core else 'false',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _ccwfn(coords, basis, frozen_core=False):
    cc = pycc.ccwfn(_scf_wfn(coords, basis, frozen_core))
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        cc.solve_cc(1e-12, 1e-12, 200)
    return cc


def _ecorr(coords, basis, frozen_core=False):
    """CCSD correlation energy at a geometry (for the FD regeneration recipe)."""
    return _ccwfn(coords, basis, frozen_core).ecc


def _findiff_gradient(basis, frozen_core=False, h=0.005):
    """Regeneration recipe for the GRAD_REF_* (not run in the tests): 5-point O(h^4) central finite
    difference of the CCSD correlation energy under nuclear displacement."""
    g = np.zeros((3, 3))
    for a in range(3):
        for x in range(3):
            e = {}
            for k in (-2, -1, 1, 2):
                c = REF.copy(); c[a, x] += k * h
                e[k] = _ecorr(c, basis, frozen_core)
            g[a, x] = (e[-2] - 8 * e[-1] + 8 * e[1] - e[2]) / (12 * h)
    return g


def test_ccsd_gradient_vs_findiff():
    """STO-3G CCSD correlation gradient reproduces the FD-validated frozen reference, and the
    pycc.gradient facade decomposes exactly (total = nuclear + reference + correlation)."""
    cc = _ccwfn(REF, "STO-3G")
    g = np.asarray(pycc.CCderiv(cc).gradient())
    assert np.max(np.abs(g - GRAD_REF_STO3G)) < 1e-11, g
    r = pycc.gradient(pycc.CCderiv(cc))
    assert np.max(np.abs(r.total - (r.nuclear + r.reference + r.correlation))) < 1e-12
    assert np.max(np.abs(np.asarray(r.correlation) - GRAD_REF_STO3G)) < 1e-11


def test_ccsd_gradient_density_energy():
    """The gradient-convention densities (CCderiv adapter) reproduce the CCSD correlation energy:
    E_corr = contract(D, F) + contract(Gamma, ERI) (no prefactor on the two-particle term)."""
    cc = _ccwfn(REF, "STO-3G")
    dens = pycc.CCderiv(cc).ccdensity
    ecc = dens.compute_energy()
    D, G = dens.gradient_densities()
    E = cc.contract('pq,pq->', D, np.asarray(cc.H.F)) + cc.contract('pqrs,pqrs->', G, np.asarray(cc.H.ERI))
    assert abs(E - ecc) < 1e-12, (E, ecc)


def test_ccsd_gradient_ccpvdz():
    """cc-pVDZ CCSD correlation gradient vs the FD-validated frozen reference -- a real virtual space
    (polarization functions, several virtuals per irrep, A2-symmetry MOs) that STO-3G lacks -- and
    the gradient-convention densities reconstruct E_corr."""
    cc = _ccwfn(REF, "cc-pVDZ")
    g = np.asarray(pycc.CCderiv(cc).gradient())
    assert np.max(np.abs(g - GRAD_REF_PVDZ)) < 1e-11, g
    dens = pycc.CCderiv(cc).ccdensity
    D, G = dens.gradient_densities()
    E = cc.contract('pq,pq->', D, np.asarray(cc.H.F)) + cc.contract('pqrs,pqrs->', G, np.asarray(cc.H.ERI))
    assert abs(E - dens.compute_energy()) < 1e-12


def test_ccsd_gradient_frozen_core():
    """Frozen-core CCSD gradient vs the FD-validated frozen reference (exercising the
    core<->active-occupied P_co response).  psi4 has no frozen-core CC gradient, so the FD of the
    frozen-core CCSD energy is the oracle."""
    cc = _ccwfn(REF, "STO-3G", frozen_core=True)
    assert cc.nfzc > 0
    deriv = pycc.CCderiv(cc)
    g_zvector = np.asarray(deriv.gradient())
    assert np.max(np.abs(g_zvector - GRAD_REF_FC_STO3G)) < 1e-11, g_zvector
