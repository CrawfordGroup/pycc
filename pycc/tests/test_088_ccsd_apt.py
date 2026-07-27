"""CCSD atomic polar tensors (nuclear dipole derivatives, the IR-intensity tensor) via the shared
2n+1 machinery -- P_corr[A,beta,alpha] = d(mu_alpha)/dX_{A,beta} = -d^2 E_corr/dF_alpha dX_{A,beta},
the CCSD analog of test_087 (CCSD(T)) and the mixed field/nuclear analog of the CCSD polarizability
(test_084).

The correlation APT is the base CorrelatedDerivs.dipole_derivatives 2n+1 assembly on the CCSD
relaxed and perturbed densities (no (T), so make_t3_density is not needed); CCderiv.dipole_derivatives
adds only the method guards and delegates to the base.

Oracle: CFOUR (xcfour, CALC=CCSD, VIB=EXACT, SCF_CONV=13, CC_CONV=12).  The DIPDER file holds the
TOTAL APT (nuclear + HF + correlation); the correlation part is DIPDER(CCSD) minus the SCF DIPDER
(the nuclear + HF part; frozen core is a no-op for HF, so the frozen-core correlation subtracts the
same all-electron SCF, matching pycc whose reference block is the all-electron SCF).  DIPDER indexes
[mu, atom, coord]; pycc indexes [A, beta, alpha] = d mu_alpha / d R_{A,beta}, so the oracle is
transpose(DIPDER, (1,2,0)).  CFOUR's exact GENBAS "6-31G" is transcribed here (same string as
test_084/test_087) so pycc's AO basis matches to all printed digits; pycc then reproduces the
10-digit DIPDER to ~1e-10, on both the spin-orbital and spatial routes.

Planar HOF (Cs, molecular plane xy) is the off-diagonal case; water/6-31G (C2v, plane yz) is the
cheaper second molecule whose spin-orbital case is fast enough (~5 s) to keep the SO route
CFOUR-anchored in the default suite; the SO HOF case is marked slow.
"""

import psi4
import pycc
import numpy as np
import pytest


# CFOUR's exact GENBAS 6-31G (cartesian; identical to test_084/test_087 so pycc's AO basis matches
# the CFOUR run that produced the DIPDER oracle to all printed digits).
CFOUR_631G = """cartesian
****
H     0
S   3   1.00
     18.7311370             0.0334946
      2.8253944             0.2347270
      0.6401217             0.8137573
S   1   1.00
      0.1612778             1.0000000
****
O     0
S   6   1.00
   5484.6716600             0.0018311
    825.2349460             0.0139502
    188.0469580             0.0684451
     52.9645000             0.2327143
     16.8975704             0.4701929
      5.7996353             0.3585209
S   3   1.00
     15.5396162            -0.1107775
      3.5999336            -0.1480263
      1.0137618             1.1307670
S   1   1.00
      0.2700058             1.0000000
P   3   1.00
     15.5396162             0.0708743
      3.5999336             0.3397528
      1.0137618             0.7271586
P   1   1.00
      0.2700058             1.0000000
****
F     0
S   6   1.00
   7001.7130900             0.0018196
   1051.3660900             0.0139161
    239.2856900             0.0684053
     67.3974453             0.2331858
     21.5199573             0.4712674
      7.4031013             0.3566185
S   3   1.00
     20.8479528            -0.1085070
      4.8083083            -0.1464517
      1.3440699             1.1286886
S   1   1.00
      0.3581514             1.0000000
P   3   1.00
     20.8479528             0.0716287
      4.8083083             0.3459121
      1.3440699             0.7224700
P   1   1.00
      0.3581514             1.0000000
****
"""

# Planar HOF (Cs, plane xy), same frame as the CFOUR run (units bohr, no_com/no_reorient).
HOF = """
O  0.00000  0.00000  0.00000
F  2.56070  0.93200  0.00000
H -0.83450  1.62360  0.00000
symmetry c1
units bohr
no_com
no_reorient
"""

# Equilibrium H2O (C2v), for the cheaper SO == spatial keystone.
WATER = """
O  0.00000  0.00000  0.00000
H  0.00000  1.43121 -1.10664
H  0.00000 -1.43121 -1.10664
symmetry c1
units bohr
no_com
no_reorient
"""

# CFOUR CCSD APT oracles (a.u.), HOF (O, F, H) / CFOUR GENBAS 6-31G, indexed [A, beta, alpha] =
# d mu_alpha / d R_{A,beta}.  Correlation = DIPDER(CCSD) - DIPDER(SCF); total = DIPDER(CCSD).
CFOUR_APT_HOF_631G = np.array(
    [[[-0.2283803199, -0.0899610722, 0.0],
      [-0.1933533783, 0.0530123216, 0.0],
      [0.0, 0.0, -0.013744654]],
     [[0.2499221636, 0.0335685423, 0.0],
      [0.108335889, 0.0643549674, 0.0],
      [0.0, 0.0, 0.0445811407]],
     [[-0.0215418436, 0.0563925299, 0.0],
      [0.0850174893, -0.117367289, 0.0],
      [0.0, 0.0, -0.0308364867]]])
CFOUR_APT_HOF_631G_FC = np.array(
    [[[-0.2283937397, -0.0900510356, 0.0],
      [-0.1934653497, 0.0529769499, 0.0],
      [0.0, 0.0, -0.0137019437]],
     [[0.2499973509, 0.0335809077, 0.0],
      [0.1083735179, 0.0644076981, 0.0],
      [0.0, 0.0, 0.0445741776]],
     [[-0.0216036112, 0.0564701279, 0.0],
      [0.0850918319, -0.117384648, 0.0],
      [0.0, 0.0, -0.030872234]]])
CFOUR_APT_HOF_631G_TOTAL = np.array(
    [[[-0.123795074, -0.0503029583, 0.0],
      [0.107823156, 0.095526844, 0.0],
      [0.0, 0.0, -0.3786006089]],
     [[-0.1417913991, -0.0005095643, 0.0],
      [-0.1371636569, -0.1792474625, 0.0],
      [0.0, 0.0, -0.0868714428]],
     [[0.2655864731, 0.0508125226, 0.0],
      [0.0293405009, 0.0837206185, 0.0],
      [0.0, 0.0, 0.4654720517]]])
CFOUR_APT_HOF_631G_TOTAL_FC = np.array(
    [[[-0.1238084938, -0.0503929217, 0.0],
      [0.1077111846, 0.0954914723, 0.0],
      [0.0, 0.0, -0.3785578986]],
     [[-0.1417162118, -0.0004971989, 0.0],
      [-0.137126028, -0.1791947318, 0.0],
      [0.0, 0.0, -0.0868784059]],
     [[0.2655247055, 0.0508901206, 0.0],
      [0.0294148435, 0.0837032595, 0.0],
      [0.0, 0.0, 0.4654363044]]])

# CFOUR CCSD APT oracles (a.u.), water (O, H, H) / CFOUR GENBAS 6-31G, indexed [A, beta, alpha].
# C2v, molecular plane yz (x perpendicular); the fast SO route anchor.
CFOUR_APT_WATER_631G = np.array(
    [[[0.0368350628, 0.0, 0.0],
      [0.0, 0.1049013758, 0.0],
      [0.0, 0.0, 0.111175279]],
     [[-0.0184175315, 0.0, 0.0],
      [0.0, -0.0524506879, 0.0332568917],
      [0.0, 0.0263151125, -0.0555876395]],
     [[-0.0184175315, 0.0, 0.0],
      [0.0, -0.0524506879, -0.0332568917],
      [0.0, -0.0263151125, -0.0555876395]]])
CFOUR_APT_WATER_631G_FC = np.array(
    [[[0.0369506702, 0.0, 0.0],
      [0.0, 0.1050980957, 0.0],
      [0.0, 0.0, 0.1112302379]],
     [[-0.0184753351, 0.0, 0.0],
      [0.0, -0.0525490478, 0.0332955133],
      [0.0, 0.0263464715, -0.055615119]],
     [[-0.0184753351, 0.0, 0.0],
      [0.0, -0.0525490478, -0.0332955133],
      [0.0, -0.0263464715, -0.055615119]]])
CFOUR_APT_WATER_631G_TOTAL = np.array(
    [[[-0.8978937423, 0.0, 0.0],
      [0.0, -0.392987091, 0.0],
      [0.0, 0.0, -0.2745420927]],
     [[0.4489468711, 0.0, 0.0],
      [0.0, 0.1964935455, 0.1316260017],
      [0.0, 0.1952019257, 0.1372710464]],
     [[0.4489468711, 0.0, 0.0],
      [0.0, 0.1964935455, -0.1316260017],
      [0.0, -0.1952019257, 0.1372710464]]])
CFOUR_APT_WATER_631G_TOTAL_FC = np.array(
    [[[-0.8977781349, 0.0, 0.0],
      [0.0, -0.3927903711, 0.0],
      [0.0, 0.0, -0.2744871338]],
     [[0.4488890675, 0.0, 0.0],
      [0.0, 0.1963951856, 0.1316646233],
      [0.0, 0.1952332847, 0.1372435669]],
     [[0.4488890675, 0.0, 0.0],
      [0.0, 0.1963951856, -0.1316646233],
      [0.0, -0.1952332847, 0.1372435669]]])


def _cfour_wfn(geom, freeze_core):
    """RHF reference in CFOUR's exact GENBAS 6-31G (via basis_helper), for the CFOUR-oracle tests."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    psi4.basis_helper(CFOUR_631G, name='cfour631g')
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _ccsd_apt(geom, freeze_core, orbital_basis):
    """Converged CCSD APT PropertyComponents for the CFOUR-basis reference."""
    wfn = _cfour_wfn(geom, freeze_core)
    cc = pycc.ccwfn(wfn, orbital_basis=orbital_basis)      # model defaults to CCSD
    cc.solve_cc(1e-12, 1e-12, 100)
    return pycc.apt(pycc.CCderiv(cc))


def _check(r, corr_ref, total_ref, Z, perp):
    """Shared assertions for a planar-molecule CCSD APT run vs the CFOUR oracle: correlation and total
    tensors, the facade decomposition (total == nuclear + reference + correlation, nuclear == Z_A
    delta), the out-of-plane block vanishing, and the translational (acoustic) sum rule.  ``Z`` is the
    nuclear-charge list; ``perp`` is the axis perpendicular to the molecular plane (2/z for HOF in the
    xy plane, 0/x for water in the yz plane)."""
    corr = np.asarray(r.correlation)
    total = np.asarray(r.total)
    assert np.max(np.abs(corr - corr_ref)) < 1e-9, corr
    assert np.max(np.abs(total - total_ref)) < 1e-9, total
    parts = np.asarray(r.nuclear) + np.asarray(r.reference) + np.asarray(r.correlation)
    assert np.max(np.abs(total - parts)) < 1e-12
    assert np.max(np.abs(np.asarray(r.nuclear) - np.asarray(Z)[:, None, None] * np.eye(3))) < 1e-12
    other = [a for a in range(3) if a != perp]
    assert np.max(np.abs(corr[:, other][:, :, perp])) < 1e-9
    assert np.max(np.abs(corr[:, perp][:, other])) < 1e-9
    assert np.max(np.abs(np.sum(total, axis=0))) < 1e-9
    assert np.max(np.abs(np.sum(corr, axis=0))) < 1e-9


def test_ccsd_apt_cfour_hof_spatial():
    """All-electron spatial (closed-shell RHF) CCSD APT for HOF vs the CFOUR DIPDER oracle -- the
    correlation and total tensors, the exact facade decomposition, the Cs out-of-plane zeros, and the
    translational sum rule."""
    _check(_ccsd_apt(HOF, 'false', 'spatial'),
           CFOUR_APT_HOF_631G, CFOUR_APT_HOF_631G_TOTAL, [8.0, 9.0, 1.0], perp=2)
    psi4.core.clean()


def test_fc_ccsd_apt_cfour_hof_spatial():
    """Frozen-core spatial CCSD APT for HOF vs the CFOUR frozen-core oracle -- exercises the
    frozen-core core<->active response in the perturbed relaxed density; the SCF reference is
    unchanged by freezing, so the oracle subtracts the same all-electron SCF DIPDER."""
    _check(_ccsd_apt(HOF, 'true', 'spatial'),
           CFOUR_APT_HOF_631G_FC, CFOUR_APT_HOF_631G_TOTAL_FC, [8.0, 9.0, 1.0], perp=2)
    psi4.core.clean()


def test_ccsd_apt_cfour_water_spatial():
    """All-electron and frozen-core spatial CCSD APT for water/6-31G (C2v, plane yz) vs the CFOUR
    oracle -- a second molecule/frame for the spatial route, and the spatial half of the fast SO
    anchor."""
    _check(_ccsd_apt(WATER, 'false', 'spatial'),
           CFOUR_APT_WATER_631G, CFOUR_APT_WATER_631G_TOTAL, [8.0, 1.0, 1.0], perp=0)
    psi4.core.clean()
    _check(_ccsd_apt(WATER, 'true', 'spatial'),
           CFOUR_APT_WATER_631G_FC, CFOUR_APT_WATER_631G_TOTAL_FC, [8.0, 1.0, 1.0], perp=0)
    psi4.core.clean()


def test_so_ccsd_apt_cfour_water():
    """Spin-orbital CCSD APT for water/6-31G vs the CFOUR oracle (all-electron and frozen core) -- the
    direct SO-vs-CFOUR anchor that runs fast enough for the default suite (C2v water, ~5 s, vs the
    slow HOF SO case).  Exercises the genuine SO path (SO perturbed amplitudes/Lambda, inline orbital
    Hessian)."""
    _check(_ccsd_apt(WATER, 'false', 'spinorbital'),
           CFOUR_APT_WATER_631G, CFOUR_APT_WATER_631G_TOTAL, [8.0, 1.0, 1.0], perp=0)
    psi4.core.clean()
    _check(_ccsd_apt(WATER, 'true', 'spinorbital'),
           CFOUR_APT_WATER_631G_FC, CFOUR_APT_WATER_631G_TOTAL_FC, [8.0, 1.0, 1.0], perp=0)
    psi4.core.clean()


@pytest.mark.slow
def test_so_ccsd_apt_cfour_hof():
    """Spin-orbital CCSD APT for HOF vs the CFOUR oracle (all-electron and frozen core) -- the
    off-diagonal (Cs) SO anchor.  Marked slow; the fast SO coverage is test_so_ccsd_apt_cfour_water."""
    _check(_ccsd_apt(HOF, 'false', 'spinorbital'),
           CFOUR_APT_HOF_631G, CFOUR_APT_HOF_631G_TOTAL, [8.0, 9.0, 1.0], perp=2)
    psi4.core.clean()
    _check(_ccsd_apt(HOF, 'true', 'spinorbital'),
           CFOUR_APT_HOF_631G_FC, CFOUR_APT_HOF_631G_TOTAL_FC, [8.0, 9.0, 1.0], perp=2)
    psi4.core.clean()


def test_ccsd_apt_routes_agree_water():
    """The two 2n+1 APT routes agree for CCSD (water/6-31G, spatial): '2n+1-field' (default, 3 field
    solves) and '2n+1-nuclear' (3N nuclear solves) build the same tensor."""
    wfn = _cfour_wfn(WATER, 'false')
    cc = pycc.ccwfn(wfn, orbital_basis='spatial')
    cc.solve_cc(1e-12, 1e-12, 100)
    Pf = np.asarray(pycc.CCderiv(cc).dipole_derivatives(route='2n+1-field'))
    Pn = np.asarray(pycc.CCderiv(cc).dipole_derivatives(route='2n+1-nuclear'))
    assert np.max(np.abs(Pf - Pn)) < 1e-10
    psi4.core.clean()


def test_ccsd_apt_route_guard():
    """An unknown route raises rather than silently returning a wrong tensor."""
    wfn = _cfour_wfn(WATER, 'false')
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    with pytest.raises(ValueError):
        pycc.CCderiv(cc).dipole_derivatives(route='bogus')
    psi4.core.clean()
