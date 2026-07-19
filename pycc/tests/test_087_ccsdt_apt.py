"""CCSD(T) atomic polar tensors (nuclear dipole derivatives, the IR-intensity tensor) via the
shared 2n+1 machinery -- P_corr[A,beta,alpha] = d(mu_alpha)/dX_{A,beta} = -d^2 E_corr/dF_alpha
dX_{A,beta}, the mixed field/nuclear analog of the CCSD(T) polarizability (test_084).

The (T) contribution needs no APT-specific code: the base CorrelatedDerivs.dipole_derivatives 2n+1
assembly consumes the same (T)-aware relaxed and perturbed densities that the CCSD(T) gradient and
polarizability already build (dt3 threaded through the perturbed amplitudes/Lambda/densities, the
canonical perturbed-MO oo/vv dependent pairs).  CCderiv.dipole_derivatives overrides the base only
to add the method guards (supported model, make_t3_density), exactly as CCderiv.polarizability does.

Oracle: CFOUR (xcfour, CALC=CCSD(T), VIB=EXACT, SCF_CONV=13, CC_CONV=12).  The DIPDER file holds the
TOTAL APT (nuclear + HF + correlation).  The correlation contribution is DIPDER(CCSD(T)) minus the
SCF DIPDER (which is the nuclear + HF part -- frozen core is a no-op for HF, so the frozen-core
correlation subtracts the same all-electron SCF, matching pycc whose reference block is the
all-electron SCF).  DIPDER indexes [mu, atom, coord]; pycc indexes [A, beta, alpha] = d mu_alpha /
d R_{A,beta}, so the oracle is transpose(DIPDER, (1,2,0)).  To match CFOUR to all printed digits the
AO basis must be identical -- Psi4's "6-31G" differs from CFOUR's GENBAS by ~4e-8 -- so CFOUR's exact
GENBAS "6-31G" is transcribed here (the same string as test_084); pycc then reproduces the 10-digit
DIPDER to ~1e-10 (the DIPDER truncation), on both the spin-orbital and spatial routes.

Planar HOF (Cs, molecular plane xy) is the off-diagonal case: the in-plane block is full while the
out-of-plane (z) couplings vanish.  Water/6-31G is the cheaper SO == spatial keystone.
"""

import psi4
import pycc
import numpy as np
import pytest


# CFOUR's exact GENBAS 6-31G (cartesian; transcribed in test_084 -- identical here so pycc's AO basis
# matches the CFOUR run that produced the DIPDER oracle to all printed digits).
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

# Planar HOF (Cs, molecular plane xy), tilted off the axes so the in-plane APT block is full; the
# same frame as the CFOUR run (units bohr, no_com/no_reorient) so the off-diagonal xy elements line up.
HOF = """
O  0.00000  0.00000  0.00000
F  2.56070  0.93200  0.00000
H -0.83450  1.62360  0.00000
symmetry c1
units bohr
no_com
no_reorient
"""

# Equilibrium H2O (C2v), for the cheaper SO == spatial keystone (no CFOUR oracle needed).
WATER = """
O  0.00000  0.00000  0.00000
H  0.00000  1.43121 -1.10664
H  0.00000 -1.43121 -1.10664
symmetry c1
units bohr
no_com
no_reorient
"""

# CFOUR CCSD(T) APT oracles (a.u.), HOF (O, F, H) / CFOUR GENBAS 6-31G, indexed [A, beta, alpha] =
# d mu_alpha / d R_{A,beta}.  Correlation = DIPDER(CCSD(T)) - DIPDER(SCF); total = DIPDER(CCSD(T)).
CFOUR_APT_T_HOF_631G = np.array(
    [[[-0.215755342, -0.0887708031, 0.0],
      [-0.190937797, 0.0625843162, 0.0],
      [0.0, 0.0, -0.0113236757]],
     [[0.2417792404, 0.0286946511, 0.0],
      [0.1044037051, 0.0624437659, 0.0],
      [0.0, 0.0, 0.044003958]],
     [[-0.0260238983, 0.060076152, 0.0],
      [0.0865340919, -0.1250280821, 0.0],
      [0.0, 0.0, -0.0326802823]]])
CFOUR_APT_T_HOF_631G_FC = np.array(
    [[[-0.2156168499, -0.0888064181, 0.0],
      [-0.1909555635, 0.0625488937, 0.0],
      [0.0, 0.0, -0.0112698848]],
     [[0.2416916297, 0.0286764607, 0.0],
      [0.1043805326, 0.0624624679, 0.0],
      [0.0, 0.0, 0.0439799958]],
     [[-0.0260747797, 0.0601299574, 0.0],
      [0.0865750309, -0.1250113616, 0.0],
      [0.0, 0.0, -0.032710111]]])
CFOUR_APT_T_HOF_631G_TOTAL = np.array(
    [[[-0.1111700961, -0.0491126892, 0.0],
      [0.1102387373, 0.1050988386, 0.0],
      [0.0, 0.0, -0.3761796306]],
     [[-0.1499343223, -0.0053834555, 0.0],
      [-0.1410958408, -0.181158664, 0.0],
      [0.0, 0.0, -0.0874486255]],
     [[0.2611044184, 0.0544961447, 0.0],
      [0.0308571035, 0.0760598254, 0.0],
      [0.0, 0.0, 0.4636282561]]])
CFOUR_APT_T_HOF_631G_TOTAL_FC = np.array(
    [[[-0.111031604, -0.0491483042, 0.0],
      [0.1102209708, 0.1050634161, 0.0],
      [0.0, 0.0, -0.3761258397]],
     [[-0.150021933, -0.0054016459, 0.0],
      [-0.1411190133, -0.181139962, 0.0],
      [0.0, 0.0, -0.0874725877]],
     [[0.261053537, 0.0545499501, 0.0],
      [0.0308980425, 0.0760765459, 0.0],
      [0.0, 0.0, 0.4635984274]]])


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


def _ccsdt_apt(geom, freeze_core, orbital_basis):
    """Converged CCSD(T) APT PropertyComponents for the CFOUR-basis reference."""
    wfn = _cfour_wfn(geom, freeze_core)
    cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis=orbital_basis, make_t3_density=True)
    cc.solve_cc(1e-12, 1e-12, 100)
    return pycc.apt(cc)


def _check_hof(r, corr_ref, total_ref):
    """Shared assertions for an HOF CCSD(T) APT run: correlation and total vs the CFOUR oracle, the
    facade decomposition (total == nuclear + reference + correlation, nuclear == Z_A delta), the Cs
    out-of-plane (z) block vanishing, and the translational (acoustic) sum rule."""
    corr = np.asarray(r.correlation)
    total = np.asarray(r.total)
    assert np.max(np.abs(corr - corr_ref)) < 1e-9, corr
    assert np.max(np.abs(total - total_ref)) < 1e-9, total
    # facade decomposition is exact
    parts = np.asarray(r.nuclear) + np.asarray(r.reference) + np.asarray(r.correlation)
    assert np.max(np.abs(total - parts)) < 1e-12
    # nuclear block is Z_A delta_{alpha,beta}
    Z = np.array([8.0, 9.0, 1.0])
    assert np.max(np.abs(np.asarray(r.nuclear) - Z[:, None, None] * np.eye(3))) < 1e-12
    # Cs (molecular plane xy): out-of-plane couplings vanish -- d mu_{x,y}/dz = d mu_z/d{x,y} = 0
    assert np.max(np.abs(corr[:, 2, :2])) < 1e-9 and np.max(np.abs(corr[:, :2, 2])) < 1e-9
    # translational (acoustic) sum rule: sum over atoms vanishes for a neutral molecule
    assert np.max(np.abs(np.sum(total, axis=0))) < 1e-9
    assert np.max(np.abs(np.sum(corr, axis=0))) < 1e-9


def test_ccsdt_apt_cfour_hof_spatial():
    """All-electron spatial (closed-shell RHF) CCSD(T) APT for HOF vs the CFOUR DIPDER oracle -- the
    correlation and total tensors, the exact facade decomposition, the Cs out-of-plane zeros, and the
    translational sum rule."""
    _check_hof(_ccsdt_apt(HOF, 'false', 'spatial'),
               CFOUR_APT_T_HOF_631G, CFOUR_APT_T_HOF_631G_TOTAL)
    psi4.core.clean()


def test_fc_ccsdt_apt_cfour_hof_spatial():
    """Frozen-core spatial CCSD(T) APT for HOF vs the CFOUR frozen-core oracle -- exercises the (T)
    core<->active and active oo/vv dependent-pair response in the perturbed relaxed density; the SCF
    reference (and thus nuclear + HF part) is unchanged by freezing, so the oracle subtracts the same
    all-electron SCF DIPDER."""
    r = _ccsdt_apt(HOF, 'true', 'spatial')
    _check_hof(r, CFOUR_APT_T_HOF_631G_FC, CFOUR_APT_T_HOF_631G_TOTAL_FC)
    psi4.core.clean()


@pytest.mark.slow
def test_so_ccsdt_apt_cfour_hof():
    """Spin-orbital CCSD(T) APT for HOF vs the CFOUR oracle (all-electron and frozen core).  The
    genuinely spin-orbital path (SO perturbed amplitudes/Lambda/(T) intermediates and the inline
    orbital Hessian); marked slow (the SO HOF (T) solve is the expensive case)."""
    _check_hof(_ccsdt_apt(HOF, 'false', 'spinorbital'),
               CFOUR_APT_T_HOF_631G, CFOUR_APT_T_HOF_631G_TOTAL)
    psi4.core.clean()
    _check_hof(_ccsdt_apt(HOF, 'true', 'spinorbital'),
               CFOUR_APT_T_HOF_631G_FC, CFOUR_APT_T_HOF_631G_TOTAL_FC)
    psi4.core.clean()


def test_ccsdt_apt_so_eq_spatial_keystone_water():
    """SO == spatial keystone (water/6-31G, all-electron and frozen core): the spin-orbital CCSD(T)
    correlation APT of an RHF reference forced into the spin-orbital basis reproduces the spatial
    (closed-shell) value.  Cheaper than HOF, and independent of the CFOUR oracle -- it carries the
    spatial validation onto the SO machinery (SO perturbed densities, inline orbital Hessian)."""
    for fc in ('false', 'true'):
        wfn = _cfour_wfn(WATER, fc)
        cc_sa = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spatial', make_t3_density=True)
        cc_sa.solve_cc(1e-12, 1e-12, 100)
        P_sa = np.asarray(pycc.CCderiv(cc_sa).dipole_derivatives())
        cc_so = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spinorbital', make_t3_density=True)
        cc_so.solve_cc(1e-12, 1e-12, 100)
        P_so = np.asarray(pycc.CCderiv(cc_so).dipole_derivatives())
        assert np.max(np.abs(P_so - P_sa)) < 1e-10, (fc, np.max(np.abs(P_so - P_sa)))
    psi4.core.clean()


def test_ccsdt_apt_routes_agree_water():
    """The two 2n+1 APT routes agree for CCSD(T) (water/6-31G, spatial): '2n+1-field' (default, 3
    field solves) and '2n+1-nuclear' (3N nuclear solves) build the same tensor from the same (T)-aware
    responses."""
    wfn = _cfour_wfn(WATER, 'false')
    cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spatial', make_t3_density=True)
    cc.solve_cc(1e-12, 1e-12, 100)
    Pf = np.asarray(pycc.CCderiv(cc).dipole_derivatives(route='2n+1-field'))
    Pn = np.asarray(pycc.CCderiv(cc).dipole_derivatives(route='2n+1-nuclear'))
    assert np.max(np.abs(Pf - Pn)) < 1e-10
    psi4.core.clean()


def test_ccsdt_apt_guards():
    """The misused paths raise rather than silently returning a wrong tensor: CCSD(T) built without
    the (T) density intermediates (make_t3_density), and an unknown route."""
    wfn = _cfour_wfn(WATER, 'false')
    cc_t = pycc.ccwfn(wfn, model='ccsd(t)')                     # CCSD(T), no make_t3_density
    cc_t.solve_cc(1e-12, 1e-12, 100)
    with pytest.raises(ValueError):                             # (T) needs make_t3_density
        pycc.CCderiv(cc_t).dipole_derivatives()
    cc = pycc.ccwfn(wfn)                                        # plain CCSD: reaches the route check
    cc.solve_cc(1e-12, 1e-12, 100)
    with pytest.raises(ValueError):                             # unknown route
        pycc.CCderiv(cc).dipole_derivatives(route='bogus')
    psi4.core.clean()
