"""
Spin-orbital CCSD dipole polarizability via the symmetric linear-response function
(right-hand perturbed amplitudes only); docs/ENHANCEMENT_PLAN_2026-06.md.

Validation:
  * SO-RHF dynamic polarizability (omega != 0) vs Psi4's CCSD polarizability
    (isotropic) -- the symmetric kernel reproduces the spin-adapted result.
  * UHF/ROHF *static* polarizability vs an in-place finite-difference of the CCSD
    energy. No code computes open-shell CC *dynamic* polarizabilities, but the
    dynamic kernel is validated for RHF above and the static value is checked
    directly here; the finite difference perturbs the Fock F -> F - F_str*mu_z and
    differentiates the correlation energy (the fixed-orbital reference is linear in
    the field, so it does not contribute to the second derivative).
"""

import psi4
import pycc
import numpy as np

# Open-shell doublet (hydroxyl radical).
OH = """
0 2
O 0.0 0.0 0.0
H 0.0 0.0 1.83
units bohr
no_com
no_reorient
symmetry c1
"""


def _findiff_alpha_zz(cc, d=0.001):
    """Static alpha_zz from a 5-point finite difference of the CCSD correlation
    energy w.r.t. a z-dipole field. Restores cc.H.F on return."""
    F0 = np.asarray(cc.H.F).copy()
    muz = np.asarray(cc.H.mu[2])
    g1, g2 = cc.mp.t1.copy(), cc.mp.t2.copy()

    def ecc(Fstr):
        cc.H.F = F0 - Fstr * muz
        cc.t1 = g1.copy()
        cc.t2 = g2.copy()
        return cc.solve_cc(e_conv=1e-13, r_conv=1e-12, maxiter=300)

    e0, ep, e2p, em, e2m = ecc(0.0), ecc(d), ecc(2*d), ecc(-d), ecc(-2*d)
    cc.H.F = F0
    return -(-e2p + 16*ep - 30*e0 + 16*em - e2m) / (12 * d * d)


def test_so_rhf_polar_vs_psi4(rhf_wfn):
    """SO-RHF dynamic CCSD polarizability (isotropic, omega=0.1) vs Psi4."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12, r_convergence=1e-12)
    cc = pycc.CCwfn(wfn, frozen_core=False, orbital_basis="spinorbital")
    cc.solve_cc(e_conv=1e-11, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-11, r_conv=1e-11)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    resp = pycc.ccresponse(dens)
    iso = np.trace(resp.polarizability(0.1)) / 3.0

    psi4.set_options({'omega': [0.1, 'au'], 'gauge': 'length'})
    psi4.properties('ccsd', properties=['polarizability'])
    ref = next(psi4.variable(k) for k in psi4.core.variables()
               if 'DIPOLE POLARIZABILITY' in k.upper())
    assert abs(iso - ref) < 1e-9


def test_uhf_polar_findiff(uhf_wfn):
    """UHF static CCSD polarizability (alpha_zz) vs finite difference."""
    wfn = uhf_wfn(OH, "STO-3G", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13)
    cc = pycc.CCwfn(wfn, frozen_core=False)
    cc.solve_cc(e_conv=1e-12, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-12, r_conv=1e-11)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    resp = pycc.ccresponse(dens)
    azz = resp.polarizability(0.0)[2, 2]   # analytic, before perturbing cc
    azz_fd = _findiff_alpha_zz(cc)
    assert abs(azz - azz_fd) < 1e-6


def test_rohf_polar_findiff(rohf_wfn):
    """ROHF static CCSD polarizability (alpha_zz) vs finite difference."""
    wfn = rohf_wfn(OH, "STO-3G", freeze_core="false",
                   e_convergence=1e-13, d_convergence=1e-13)
    cc = pycc.CCwfn(wfn, frozen_core=False)
    cc.solve_cc(e_conv=1e-12, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-12, r_conv=1e-11)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    resp = pycc.ccresponse(dens)
    azz = resp.polarizability(0.0)[2, 2]
    azz_fd = _findiff_alpha_zz(cc)
    assert abs(azz - azz_fd) < 1e-6
