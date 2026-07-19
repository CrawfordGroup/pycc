"""CCSD(T) molecular (nuclear) Hessian (correlation contribution) via the 2n+1 route --
H_corr[Aa,Bb] = d^2 E_corr / dX_{Aa} dX_{Bb}, the nuclear-nuclear analog of the CCSD(T)
polarizability (test_084) and APT (test_087).

As for the APT, the (T) contribution needs no Hessian-specific code: the base
CorrelatedDerivs.hessian 2n+1 assembly (3N nuclear perturbed solves plus the full nuclear-nuclear
second skeletons) consumes the same (T)-aware relaxed and perturbed densities the CCSD(T) gradient,
polarizability, and APT already build (dt3 threaded through the perturbed amplitudes/Lambda/
densities, canonical perturbed-MO oo/vv dependent pairs).  CCderiv.hessian adds only the method
guards (supported model; CCSD(T) requires make_t3_density) and delegates to the base.

Oracle: CFOUR (xcfour, CALC=CCSD(T), VIB=EXACT, SCF_CONV=13, CC_CONV=12).  FCMFINAL holds the TOTAL
force-constant matrix; the correlation contribution is FCMFINAL(CCSD(T)) - FCMFINAL(SCF) (frozen core
is a no-op for HF, so the frozen-core correlation subtracts the same all-electron SCF FCMFINAL,
matching pycc whose reference block is the all-electron SCF).  FCMFINAL is row-major (A*3+a, B*3+b),
the same index order as pycc.  CFOUR's exact GENBAS 6-31G is transcribed here (same string as
test_084/087) so the AO basis matches to all printed digits; pycc reproduces the 10-digit FCMFINAL to
~1e-10.

The Hessian's 3N nuclear solves make it much costlier than the APT/polarizability, so the tiers are
tighter: water/6-31G (C2v) carries the default suite -- spatial (AE + FC) and the spin-orbital
all-electron case (the direct SO-(T) CFOUR anchor, ~55 s).  The spin-orbital frozen-core water case
and the HOF (Cs) spatial cross-check are marked slow.  The spin-orbital HOF CCSD(T) Hessian (~8.5 min
per build) is not included; SO-(T) is anchored on water and HOF-(T) spatially.
"""

import psi4
import pycc
import numpy as np
import pytest


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

WATER = """
O  0.00000  0.00000  0.00000
H  0.00000  1.43121 -1.10664
H  0.00000 -1.43121 -1.10664
symmetry c1
units bohr
no_com
no_reorient
"""

HOF = """
O  0.00000  0.00000  0.00000
F  2.56070  0.93200  0.00000
H -0.83450  1.62360  0.00000
symmetry c1
units bohr
no_com
no_reorient
"""

# CFOUR CCSD(T) correlation Hessians (a.u.), FCMFINAL(CCSD(T)) - FCMFINAL(SCF), row-major (A*3+a, B*3+b).
CFOUR_HESS_T_WATER = np.array(
    [[-0.0318157571,  0.          ,  0.          ,  0.0159078786,  0.          ,  0.          ,
       0.0159078786, -0.          ,  0.          ],
     [ 0.          , -0.0108003877,  0.          ,  0.          ,  0.0054001939,  0.0081247505,
      -0.          ,  0.0054001939, -0.0081247505],
     [ 0.          ,  0.          , -0.039088771 ,  0.          ,  0.0047811046,  0.0195443856,
       0.          , -0.0047811046,  0.0195443856],
     [ 0.0159078786,  0.          ,  0.          , -0.0146150076, -0.          ,  0.          ,
      -0.0012928709,  0.          ,  0.          ],
     [ 0.          ,  0.0054001939,  0.0047811046, -0.          , -0.0060683374, -0.0064529276,
       0.          ,  0.0006681436,  0.001671823 ],
     [ 0.          ,  0.0081247507,  0.0195443856,  0.          , -0.0064529277, -0.0132921518,
       0.          , -0.0016718231, -0.0062522337],
     [ 0.0159078786, -0.          ,  0.          , -0.0012928709,  0.          ,  0.          ,
      -0.0146150076,  0.          ,  0.          ],
     [-0.          ,  0.0054001939, -0.0047811046,  0.          ,  0.0006681436, -0.001671823 ,
       0.          , -0.0060683374,  0.0064529276],
     [ 0.          , -0.0081247507,  0.0195443856,  0.          ,  0.0016718231, -0.0062522337,
       0.          ,  0.0064529277, -0.0132921518]])
CFOUR_HESS_T_WATER_FC = np.array(
    [[-0.0318820814,  0.          ,  0.          ,  0.0159410408,  0.          ,  0.          ,
       0.0159410408, -0.          ,  0.          ],
     [ 0.          , -0.0106817843,  0.          ,  0.          ,  0.0053408921,  0.0081962453,
      -0.          ,  0.0053408921, -0.0081962453],
     [ 0.          ,  0.          , -0.038974217 ,  0.          ,  0.0048511178,  0.0194871086,
       0.          , -0.0048511178,  0.0194871086],
     [ 0.0159410408,  0.          ,  0.          , -0.0146415199, -0.          ,  0.          ,
      -0.0012995208,  0.          ,  0.          ],
     [ 0.          ,  0.0053408921,  0.0048511178, -0.          , -0.0060145929, -0.0065236817,
       0.          ,  0.0006737007,  0.0016725637],
     [ 0.          ,  0.0081962453,  0.0194871086,  0.          , -0.0065236817, -0.013245804 ,
       0.          , -0.0016725637, -0.0062413046],
     [ 0.0159410408, -0.          ,  0.          , -0.0012995208,  0.          ,  0.          ,
      -0.0146415199,  0.          ,  0.          ],
     [-0.          ,  0.0053408921, -0.0048511178,  0.          ,  0.0006737007, -0.0016725637,
       0.          , -0.0060145929,  0.0065236817],
     [ 0.          , -0.0081962453,  0.0194871086,  0.          ,  0.0016725637, -0.0062413046,
       0.          ,  0.0065236817, -0.013245804 ]])
CFOUR_HESS_T_HOF = np.array(
    [[-0.0267392581, -0.0009548925,  0.          ,  0.0107396156, -0.0029320561,  0.          ,
       0.0159996425,  0.0038869486, -0.          ],
     [-0.0009548924, -0.0360010422,  0.          , -0.0085422723,  0.0234676607,  0.          ,
       0.0094971648,  0.0125333815,  0.          ],
     [ 0.          , -0.          , -0.038063371 ,  0.          ,  0.          ,  0.0217778677,
      -0.          ,  0.          ,  0.0162855032],
     [ 0.0107396156, -0.0085422722,  0.          , -0.0150803169,  0.0060529887, -0.          ,
       0.0043407013,  0.0024892836,  0.          ],
     [-0.0029320561,  0.0234676606,  0.          ,  0.0060529887, -0.021831902 ,  0.          ,
      -0.0031209325, -0.0016357587,  0.          ],
     [ 0.          ,  0.          ,  0.0217778676, -0.          ,  0.          , -0.0215857036,
       0.          , -0.          , -0.0001921641],
     [ 0.0159996425,  0.0094971648, -0.          ,  0.0043407013, -0.0031209325,  0.          ,
      -0.0203403439, -0.0063762322,  0.          ],
     [ 0.0038869485,  0.0125333816,  0.          ,  0.0024892837, -0.0016357588, -0.          ,
      -0.0063762322, -0.0108976228, -0.          ],
     [-0.          ,  0.          ,  0.0162855033,  0.          ,  0.          , -0.0001921642,
       0.          ,  0.          , -0.0160933392]])
CFOUR_HESS_T_HOF_FC = np.array(
    [[-0.0267132577, -0.0009874858,  0.          ,  0.010721736 , -0.0029446097,  0.          ,
       0.0159915217,  0.0039320954, -0.          ],
     [-0.0009874858, -0.0359045783,  0.          , -0.0085618486,  0.0234649825,  0.          ,
       0.0095493343,  0.0124395958,  0.          ],
     [ 0.          , -0.          , -0.0380915628,  0.          ,  0.          ,  0.0217862544,
      -0.          ,  0.          ,  0.0163053083],
     [ 0.0107217359, -0.0085618486,  0.          , -0.0150694409,  0.0060666971, -0.          ,
       0.004347705 ,  0.0024951514,  0.          ],
     [-0.0029446097,  0.0234649826,  0.          ,  0.0060666971, -0.0218330826,  0.          ,
      -0.0031220874, -0.0016319001,  0.          ],
     [ 0.          ,  0.          ,  0.0217862544, -0.          ,  0.          , -0.0215928145,
       0.          , -0.          , -0.00019344  ],
     [ 0.0159915218,  0.0095493343, -0.          ,  0.0043477049, -0.0031220874,  0.          ,
      -0.0203392267, -0.0064272468,  0.          ],
     [ 0.0039320954,  0.0124395957,  0.          ,  0.0024951514, -0.0016319   , -0.          ,
      -0.0064272468, -0.0108076957, -0.          ],
     [-0.          ,  0.          ,  0.0163053083,  0.          ,  0.          , -0.00019344  ,
       0.          ,  0.          , -0.0161118684]])


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


def _ccsdt_hess(geom, freeze_core, orbital_basis):
    """Converged CCSD(T) Hessian PropertyComponents for the CFOUR-basis reference."""
    wfn = _cfour_wfn(geom, freeze_core)
    cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis=orbital_basis, make_t3_density=True)
    cc.solve_cc(1e-12, 1e-12, 100)
    return pycc.hessian(cc)


def _check(r, corr_ref):
    """Shared assertions for a CCSD(T) Hessian run vs the CFOUR oracle: the full 9x9 correlation
    matrix, the facade decomposition (total == nuclear + reference + correlation), and the FD-free
    physics checks -- symmetry and the translational (acoustic) sum rule (E_corr is invariant under
    uniform translation, so summing over the second atom vanishes)."""
    corr = np.asarray(r.correlation)
    total = np.asarray(r.total)
    assert corr.shape == (9, 9)
    assert np.max(np.abs(corr - corr_ref)) < 1e-9, np.max(np.abs(corr - corr_ref))
    parts = np.asarray(r.nuclear) + np.asarray(r.reference) + np.asarray(r.correlation)
    assert np.max(np.abs(total - parts)) < 1e-12
    assert np.max(np.abs(corr - corr.T)) < 1e-8
    assert np.max(np.abs(corr.reshape(3, 3, 3, 3).sum(axis=2))) < 1e-8


def test_ccsdt_hessian_cfour_water_spatial():
    """All-electron and frozen-core spatial CCSD(T) Hessian for water/6-31G vs the CFOUR FCMFINAL
    oracle -- the full correlation matrix, the facade decomposition, symmetry, and the translational
    sum rule.  Exercises the (T) core<->active and active oo/vv dependent-pair response in the
    perturbed relaxed density (frozen-core case)."""
    _check(_ccsdt_hess(WATER, 'false', 'spatial'), CFOUR_HESS_T_WATER)
    psi4.core.clean()
    _check(_ccsdt_hess(WATER, 'true', 'spatial'), CFOUR_HESS_T_WATER_FC)
    psi4.core.clean()


def test_so_ccsdt_hessian_cfour_water():
    """All-electron spin-orbital CCSD(T) Hessian for water/6-31G vs the CFOUR oracle -- the direct
    SO-(T) CFOUR anchor kept in the default suite (~55 s).  The genuine SO path (SO perturbed
    amplitudes/Lambda/(T) intermediates, inline orbital Hessian, nuclear-nuclear second skeletons)."""
    _check(_ccsdt_hess(WATER, 'false', 'spinorbital'), CFOUR_HESS_T_WATER)
    psi4.core.clean()


@pytest.mark.slow
def test_fc_so_ccsdt_hessian_cfour_water():
    """Frozen-core spin-orbital CCSD(T) Hessian for water/6-31G vs the CFOUR oracle -- the SO (T)
    frozen-core case (core<->active response through the SO perturbed densities).  Marked slow."""
    _check(_ccsdt_hess(WATER, 'true', 'spinorbital'), CFOUR_HESS_T_WATER_FC)
    psi4.core.clean()


@pytest.mark.slow
def test_ccsdt_hessian_cfour_hof_spatial():
    """Second-geometry cross-check: spatial CCSD(T) Hessian for HOF/6-31G (Cs) vs the CFOUR oracle
    (all-electron and frozen core).  Marked slow."""
    _check(_ccsdt_hess(HOF, 'false', 'spatial'), CFOUR_HESS_T_HOF)
    psi4.core.clean()
    _check(_ccsdt_hess(HOF, 'true', 'spatial'), CFOUR_HESS_T_HOF_FC)
    psi4.core.clean()


def test_ccsdt_hessian_guards():
    """The misused paths raise rather than silently returning a wrong matrix: CCSD(T) built without
    the (T) density intermediates (make_t3_density), and an unknown route."""
    wfn = _cfour_wfn(WATER, 'false')
    cc_t = pycc.ccwfn(wfn, model='ccsd(t)')                     # CCSD(T), no make_t3_density
    cc_t.solve_cc(1e-12, 1e-12, 100)
    with pytest.raises(ValueError):                             # (T) needs make_t3_density
        pycc.CCderiv(cc_t).hessian()
    cc = pycc.ccwfn(wfn)                                        # plain CCSD: reaches the route check
    cc.solve_cc(1e-12, 1e-12, 100)
    with pytest.raises(ValueError):                             # unknown route
        pycc.CCderiv(cc).hessian(route='bogus')
    psi4.core.clean()
