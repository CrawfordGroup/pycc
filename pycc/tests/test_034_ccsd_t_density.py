"""
Test CCSD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import numpy as np
from ..data.molecules import *
from ..cctriples import t_vikings, t_vikings_inverted, t_tjl

def test_ccsd_t_h2o(rhf_wfn):
    """H2O cc-pVDZ"""
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    geom = """
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
"""
    wfn = rhf_wfn(geom, "STO-3G", freeze_core="false")

    etot = psi4.energy('CCSD(T)')
    ecc_psi4 = psi4.variable('CCSD(T) CORRELATION ENERGY')

    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True)
    ecc = cc.solve_cc(e_conv, r_conv, maxiter, max_diis=0)
    assert (abs(ecc_psi4 - ecc) < 1e-11)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis=0)
    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc_density = ccdensity.compute_energy()
    eone = ccdensity.eone
    etwo = ccdensity.etwo

    lambda_psi4 = -0.069084521221746
    eone_psi4 = 0.104463374777302
    etwo_psi4 = -0.175243393781829
    assert (abs(lambda_psi4 - lcc) < 1e-11)
    assert (abs(eone_psi4 - eone) < 1e-11)
    assert (abs(etwo_psi4 - etwo) < 1e-11)

    psi4.core.clean()

    wfn = rhf_wfn(geom, "cc-pVDZ", freeze_core="false")

    etot = psi4.energy('CCSD(T)')
    ecc_psi4 = psi4.variable('CCSD(T) CORRELATION ENERGY')

    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True)
    ecc = cc.solve_cc(e_conv, r_conv, maxiter)
    assert (abs(ecc_psi4 - ecc) < 1e-11)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc_density = ccdensity.compute_energy()
    eone = ccdensity.eone
    etwo = ccdensity.etwo

    lambda_psi4 = -0.227199866607450
    eone_psi4 = 0.251210862963227
    etwo_psi4 = -0.479006477929931
    assert (abs(lambda_psi4 - lcc) < 1e-11)
    assert (abs(eone_psi4 - eone) < 1e-11)
    assert (abs(etwo_psi4 - etwo) < 1e-11)


# Frozen Fock-perturbation FD oracle for the (T) unrelaxed correlation dipole mu_z (H2O/6-31G,
# frame locked; closed-shell, platform-reproducible).  Validated (once) against a 5-point FD of
# E_corr[CCSD(T)] w.r.t. V = -F*mu_z (fixed orbitals), agreeing to 1.9e-13 (see _findiff_t_dipole_z,
# the regeneration recipe).
T_DIPOLE_Z_REF = 0.08826237289720

_T_DIPOLE_GEOM = """
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
no_com
no_reorient
"""


def _findiff_t_dipole_z(wfn, h=1e-4):
    """Regeneration recipe for T_DIPOLE_Z_REF (not run in the tests): 5-point central finite
    difference of the CCSD(T) correlation energy w.r.t. a static Fock perturbation V = -s*mu_z
    (fixed orbitals)."""
    def E(s):
        c = pycc.ccwfn(wfn, model='ccsd(t)')
        c.H.F = np.asarray(c.H.F) - s * np.asarray(c.H.mu[2])
        return c.solve_cc(1e-12, 1e-12, 100)
    return -(-E(2 * h) + 8 * E(h) - 8 * E(-h) + E(-2 * h)) / (12 * h)


def test_ccsd_t_dipole(rhf_wfn):
    """The analytic (T) unrelaxed correlation dipole mu_z matches the finite-difference-validated
    frozen reference (H2O/6-31G) -- the guard that the (T) 1-PDM carries only the physical blocks
    {Dov, diag(Doo), diag(Dvv)}.  The off-diagonal Doo/Dvv appears in neither Lee-Rendell nor Hald
    et al.; it would corrupt Tr(D.mu) yet stay invisible to the eone reconstruction above (the
    canonical Fock is diagonal, so only diag(D) enters).  6-31G, not STO-3G: STO-3G has too few
    same-symmetry virtuals for mu_z to probe an off-diagonal Dvv."""
    wfn = rhf_wfn(_T_DIPOLE_GEOM, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True)
    cc.solve_cc(1e-12, 1e-12, 100)
    cclambda = pycc.cclambda(cc, pycc.cchbar(cc))
    cclambda.solve_lambda(1e-12, 1e-12)
    ccdensity = pycc.ccdensity(cc, cclambda)
    ccdensity.compute_energy()
    opdm = ccdensity.compute_onepdm(cc.t1, cc.t2, cclambda.l1, cclambda.l2)
    mu_analytic = cc.contract('pq,pq->', cc.H.mu[2], opdm)
    # frozen ref is the analytic dipole itself (matched the Fock-perturbation FD to 1.9e-13); the
    # deterministic recomputation reproduces it to ~3e-15, so 1e-11 is a tight, robust guard.
    assert abs(mu_analytic - T_DIPOLE_Z_REF) < 1e-11, mu_analytic

    psi4.core.clean()


# Frozen 5-point O(h^4) finite-field oracle for the CCSD(T) RELAXED correlation dipole mu_z
# (H2O/6-31G, _T_DIPOLE_GEOM).  Unlike the unrelaxed T_DIPOLE_Z_REF above (a fixed-orbital Fock
# perturbation of the (T) density only), this is the full CCSD(T) relaxed correlation dipole
# CCderiv.relaxed_dipole = Tr(D_rel . mu), with D_rel the gradient's relaxed density (correlation D +
# frozen-core P_co + (T) kappa_oo/kappa_vv + ov Z-vector).  Validated (once) against a 5-point finite
# field of pycc's own CCSD(T) correlation energy (E_CCSD(T) - E_SCF) under a field-relaxed SCF (psi4
# perturb_h, F=0.0005): 1.4e-12 all-electron, 1.1e-12 frozen core (see _findiff_relaxed_dipole_z).
RELAXED_DIPOLE_Z_REF = 0.0888679557555765
RELAXED_DIPOLE_Z_REF_FC = 0.0889835346143946


def _findiff_relaxed_dipole_z(freeze_core='false', F=0.0005):
    """Regeneration recipe for RELAXED_DIPOLE_Z_REF[_FC] (not run in the tests): 5-point O(h^4)
    finite field of pycc's CCSD(T) correlation energy (E_CCSD(T) - E_SCF) under a field-relaxed SCF
    (the SCF orbitals relax in the field via psi4 perturb_h), on _T_DIPOLE_GEOM / 6-31G."""
    def ecorr(Fz):
        psi4.core.clean(); psi4.core.clean_options()
        psi4.geometry(_T_DIPOLE_GEOM)
        opt = {'basis': '6-31G', 'scf_type': 'pk', 'freeze_core': freeze_core,
               'e_convergence': 1e-13, 'd_convergence': 1e-13}
        if Fz:
            opt.update({'perturb_h': True, 'perturb_with': 'dipole', 'perturb_dipole': [0., 0., Fz]})
        psi4.set_options(opt)
        _, wfn = psi4.energy('scf', return_wfn=True)
        return pycc.ccwfn(wfn, model='ccsd(t)').solve_cc(1e-13, 1e-13, 200)
    e = {s: ecorr(s * F) for s in (-2, -1, 1, 2)}
    return (-e[2] + 8 * e[1] - 8 * e[-1] + e[-2]) / (12 * F)


def test_ccsdt_relaxed_dipole(rhf_wfn):
    """The analytic CCSD(T) RELAXED correlation dipole mu_z (CCderiv.relaxed_dipole = Tr(D_rel . mu))
    matches the finite-field-validated frozen reference (H2O/6-31G), and the pycc.dipole facade
    decomposes exactly.  A static field does not move the AO basis (S^F = <pq|rs>^F = 0), so the
    relaxed dipole reuses the gradient's relaxed density -- the (T) density and its oo/vv
    dependent-pair ride along inside D_rel with no separate handling."""
    wfn = rhf_wfn(_T_DIPOLE_GEOM, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True)
    cc.solve_cc(1e-12, 1e-12, 100)
    mu = np.asarray(pycc.CCderiv(cc).relaxed_dipole())
    assert abs(mu[2] - RELAXED_DIPOLE_Z_REF) < 1e-10, mu[2]
    # facade: total = nuclear + reference + correlation, and correlation == relaxed_dipole
    r = pycc.dipole(pycc.CCderiv(cc))
    assert np.max(np.abs(np.asarray(r.total) - (np.asarray(r.nuclear)
                  + np.asarray(r.reference) + np.asarray(r.correlation)))) < 1e-12
    assert np.max(np.abs(np.asarray(r.correlation) - mu)) < 1e-12
    psi4.core.clean()


def test_ccsdt_relaxed_dipole_spinorbital(rhf_wfn):
    """SO == spatial: a closed-shell RHF driven through the spin-orbital path reproduces the spatial
    CCSD(T) relaxed correlation dipole."""
    wfn = rhf_wfn(_T_DIPOLE_GEOM, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True, orbital_basis='spinorbital')
    cc.solve_cc(1e-12, 1e-12, 100)
    mu = np.asarray(pycc.CCderiv(cc).relaxed_dipole())
    assert abs(mu[2] - RELAXED_DIPOLE_Z_REF) < 1e-9, mu[2]
    psi4.core.clean()


def test_ccsdt_relaxed_dipole_frozen_core(rhf_wfn):
    """Frozen-core CCSD(T) relaxed correlation dipole (H2O/6-31G, O 1s frozen): the core<->active (T)
    response rides the P_co divide and the active oo/vv the dependent pair, inside D_rel."""
    wfn = rhf_wfn(_T_DIPOLE_GEOM, "6-31G", freeze_core="true")
    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True)
    cc.solve_cc(1e-12, 1e-12, 100)
    assert cc.nfzc > 0
    mu = np.asarray(pycc.CCderiv(cc).relaxed_dipole())
    assert abs(mu[2] - RELAXED_DIPOLE_Z_REF_FC) < 1e-10, mu[2]
    psi4.core.clean()


# Off-axis relaxed CCSD(T) dipole.  The mu_z checks above only probe the totally-symmetric (z)
# direction of yz-plane water, so the non-totally-symmetric (off-axis) blocks of the (T) relaxed
# density went untested (mu_x = mu_y = 0 there by symmetry).  Here the molecule is tilted into a
# general C1 orientation (a fixed rotation of _T_DIPOLE_GEOM) so mu has all three Cartesian
# components nonzero.  Each reference component was validated (once) against mu_a = -dE_corr/dF_a --
# a 5-point O(h^4) field-relaxed energy gradient of pycc's own CCSD(T) correlation energy -- to
# 3.6e-11, all-electron and frozen core.
_T_DIPOLE_TILTED_GEOM = """
O  0.074487494137924 -0.055774775873582  0.108878068289408
H -0.951260432480889 -1.066139182313485 -1.390453518773079
H -0.230910151065614  1.951325348189316 -0.337520431951801
symmetry c1
units bohr
no_com
no_reorient
"""
RELAXED_DIPOLE_OFFAXIS_REF = np.array([0.0462175718, -0.0346068120, 0.0675560373])      # 6-31G, AE
RELAXED_DIPOLE_OFFAXIS_REF_FC = np.array([0.0462776809, -0.0346518205, 0.0676438986])   # 6-31G, FC


def test_ccsdt_relaxed_dipole_offaxis(rhf_wfn):
    """The full CCSD(T) relaxed correlation dipole vector (all three Cartesian components) matches its
    field-gradient-validated reference for a general (tilted) orientation, in BOTH the spin-orbital
    and spatial bases (SO == spatial keystone).  Unlike the yz-plane mu_z checks above, this probes
    the off-axis (non-totally-symmetric) blocks of the (T) relaxed density."""
    wfn = rhf_wfn(_T_DIPOLE_TILTED_GEOM, "6-31G", freeze_core="false")
    mu = {}
    for basis in ('spinorbital', 'spatial'):
        cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True, orbital_basis=basis)
        cc.solve_cc(1e-12, 1e-12, 100)
        mu[basis] = np.asarray(pycc.CCderiv(cc).relaxed_dipole())
    assert np.max(np.abs(mu['spatial'] - RELAXED_DIPOLE_OFFAXIS_REF)) < 1e-8, mu['spatial']
    assert np.max(np.abs(mu['spinorbital'] - RELAXED_DIPOLE_OFFAXIS_REF)) < 1e-8, mu['spinorbital']
    assert np.max(np.abs(mu['spinorbital'] - mu['spatial'])) < 1e-9, (mu['spinorbital'], mu['spatial'])
    psi4.core.clean()


def test_ccsdt_relaxed_dipole_offaxis_frozen_core(rhf_wfn):
    """Frozen-core analog of :func:`test_ccsdt_relaxed_dipole_offaxis` (O 1s frozen): the off-axis (T)
    relaxed dipole with the core<->active P_co divide and the active oo/vv dependent pairs inside
    D_rel, again in both bases."""
    wfn = rhf_wfn(_T_DIPOLE_TILTED_GEOM, "6-31G", freeze_core="true")
    mu = {}
    for basis in ('spinorbital', 'spatial'):
        cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True, orbital_basis=basis)
        cc.solve_cc(1e-12, 1e-12, 100)
        assert cc.nfzc > 0
        mu[basis] = np.asarray(pycc.CCderiv(cc).relaxed_dipole())
    assert np.max(np.abs(mu['spatial'] - RELAXED_DIPOLE_OFFAXIS_REF_FC)) < 1e-8, mu['spatial']
    assert np.max(np.abs(mu['spinorbital'] - RELAXED_DIPOLE_OFFAXIS_REF_FC)) < 1e-8, mu['spinorbital']
    assert np.max(np.abs(mu['spinorbital'] - mu['spatial'])) < 1e-9, (mu['spinorbital'], mu['spatial'])
    psi4.core.clean()
