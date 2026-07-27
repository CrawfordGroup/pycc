"""CCSD molecular (nuclear) Hessian (correlation contribution) via the 2n+1 route --
H_corr[Aa,Bb] = d^2 E_corr / dX_{Aa} dX_{Bb}, the nuclear-nuclear analog of the CCSD polarizability
(test_084) and APT (test_088).

The correlation Hessian is the base CorrelatedDerivs.hessian 2n+1 assembly (3N nuclear perturbed
solves of the relaxed / energy-weighted density plus the full nuclear-nuclear second skeletons) on
the CCSD relaxed and perturbed densities; CCderiv.hessian adds only the method guards and delegates
to the base. The reference (SCF) Hessian carries the nuclear-repulsion second derivative and is kept
separate; pycc.hessian sums nuclear + reference + this correlation part.

Oracle: CFOUR (xcfour, CALC=CCSD, VIB=EXACT, SCF_CONV=13, CC_CONV=12).  The FCMFINAL file holds the
TOTAL force-constant matrix (nuclear + HF + correlation); the correlation contribution is
FCMFINAL(CCSD) - FCMFINAL(SCF) (frozen core is a no-op for HF, so the frozen-core correlation
subtracts the same all-electron SCF FCMFINAL, matching pycc whose reference block is the all-electron
SCF).  FCMFINAL is row-major (A*3+a, B*3+b), the same index order as pycc.  CFOUR's exact GENBAS
6-31G is transcribed here (same string as test_084/087) so the AO basis matches to all printed
digits; pycc reproduces the 10-digit FCMFINAL to ~1e-10 (the FCMFINAL truncation).

Water/6-31G (C2v) is the primary anchor and, being cheap, carries the default (non-slow) suite for
both the spatial and spin-orbital routes; HOF/6-31G (Cs) is a slower second-geometry cross-check.
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

# CFOUR CCSD correlation Hessians (a.u.), FCMFINAL(CCSD) - FCMFINAL(SCF), row-major (A*3+a, B*3+b).
CFOUR_HESS_WATER = np.array(
    [[-0.0308188741,  0.          ,  0.          ,  0.0154094371,  0.          ,  0.          ,
       0.0154094371, -0.          ,  0.          ],
     [ 0.          , -0.0092226159,  0.          ,  0.          ,  0.0046113079,  0.008349328 ,
      -0.          ,  0.0046113079, -0.008349328 ],
     [ 0.          ,  0.          , -0.0363638331,  0.          ,  0.0054010052,  0.0181819166,
       0.          , -0.0054010052,  0.0181819166],
     [ 0.0154094371,  0.          ,  0.          , -0.0141424341, -0.          ,  0.          ,
      -0.001267003 ,  0.          ,  0.          ],
     [ 0.          ,  0.0046113079,  0.0054010052, -0.          , -0.0050377578, -0.0068751667,
       0.          ,  0.0004264498,  0.0014741614],
     [ 0.          ,  0.008349328 ,  0.0181819166,  0.          , -0.0068751667, -0.0123007408,
       0.          , -0.0014741614, -0.0058811758],
     [ 0.0154094371, -0.          ,  0.          , -0.001267003 ,  0.          ,  0.          ,
      -0.0141424341,  0.          ,  0.          ],
     [-0.          ,  0.0046113079, -0.0054010052,  0.          ,  0.0004264498, -0.0014741614,
       0.          , -0.0050377578,  0.0068751667],
     [ 0.          , -0.008349328 ,  0.0181819166,  0.          ,  0.0014741614, -0.0058811758,
       0.          ,  0.0068751667, -0.0123007408]])
CFOUR_HESS_WATER_FC = np.array(
    [[-0.0308883645,  0.          ,  0.          ,  0.0154441824,  0.          ,  0.          ,
       0.0154441824, -0.          ,  0.          ],
     [ 0.          , -0.0091156512,  0.          ,  0.          ,  0.0045578256,  0.0084175471,
      -0.          ,  0.0045578256, -0.0084175471],
     [ 0.          ,  0.          , -0.0362627154,  0.          ,  0.0054677955,  0.0181313578,
       0.          , -0.0054677955,  0.0181313578],
     [ 0.0154441824,  0.          ,  0.          , -0.0141704769, -0.          ,  0.          ,
      -0.0012737054,  0.          ,  0.          ],
     [ 0.          ,  0.0045578256,  0.0054677956, -0.          , -0.0049891396, -0.0069426715,
       0.          ,  0.000431314 ,  0.0014748758],
     [ 0.          ,  0.0084175472,  0.0181313578,  0.          , -0.0069426715, -0.0122597573,
       0.          , -0.0014748759, -0.0058716004],
     [ 0.0154441824, -0.          ,  0.          , -0.0012737054,  0.          ,  0.          ,
      -0.0141704769,  0.          ,  0.          ],
     [-0.          ,  0.0045578256, -0.0054677956,  0.          ,  0.000431314 , -0.0014748758,
       0.          , -0.0049891396,  0.0069426715],
     [ 0.          , -0.0084175472,  0.0181313578,  0.          ,  0.0014748759, -0.0058716004,
       0.          ,  0.0069426715, -0.0122597573]])
CFOUR_HESS_HOF = np.array(
    [[-0.0187302056,  0.0019866565,  0.          ,  0.0042677948, -0.0050917598,  0.          ,
       0.0144624108,  0.0031051032, -0.          ],
     [ 0.0019866566, -0.0297889719,  0.          , -0.0111083661,  0.0194693731,  0.          ,
       0.0091217096,  0.0103195988,  0.          ],
     [ 0.          , -0.          , -0.0347973108,  0.          ,  0.          ,  0.0193876249,
      -0.          ,  0.          ,  0.0154096858],
     [ 0.0042677948, -0.011108366 ,  0.          , -0.0081442929,  0.0076985441, -0.          ,
       0.0038764981,  0.0034098219,  0.          ],
     [-0.0050917598,  0.0194693731,  0.          ,  0.0076985441, -0.0184029217,  0.          ,
      -0.0026067843, -0.0010664515,  0.          ],
     [ 0.          ,  0.          ,  0.0193876249, -0.          ,  0.          , -0.0192495431,
       0.          , -0.          , -0.0001380818],
     [ 0.0144624108,  0.0091217096, -0.          ,  0.0038764981, -0.0026067843,  0.          ,
      -0.0183389089, -0.0065149251,  0.          ],
     [ 0.0031051031,  0.0103195988,  0.          ,  0.003409822 , -0.0010664515, -0.          ,
      -0.0065149251, -0.0092531473, -0.          ],
     [-0.          ,  0.          ,  0.0154096858,  0.          ,  0.          , -0.0001380818,
       0.          ,  0.          , -0.0152716041]])
CFOUR_HESS_HOF_FC = np.array(
    [[-0.018737059 ,  0.0019535633,  0.          ,  0.0042803695, -0.00509794  ,  0.          ,
       0.0144566895,  0.0031443766, -0.          ],
     [ 0.0019535632, -0.0297108973,  0.          , -0.0111255029,  0.0194780839,  0.          ,
       0.0091719397,  0.0102328134,  0.          ],
     [ 0.          , -0.          , -0.0348365012,  0.          ,  0.          ,  0.0194058422,
      -0.          ,  0.          ,  0.015430659 ],
     [ 0.0042803695, -0.0111255029,  0.          , -0.0081665372,  0.0077065306, -0.          ,
       0.0038861677,  0.0034189723,  0.          ],
     [-0.0050979399,  0.0194780839,  0.          ,  0.0077065306, -0.0184162175,  0.          ,
      -0.0026085906, -0.0010618666,  0.          ],
     [ 0.          ,  0.          ,  0.0194058422, -0.          ,  0.          , -0.0192665042,
       0.          , -0.          , -0.0001393381],
     [ 0.0144566895,  0.0091719396, -0.          ,  0.0038861677, -0.0026085905,  0.          ,
      -0.0183428572, -0.006563349 ,  0.          ],
     [ 0.0031443766,  0.0102328134,  0.          ,  0.0034189723, -0.0010618666, -0.          ,
      -0.006563349 , -0.0091709468, -0.          ],
     [-0.          ,  0.          ,  0.015430659 ,  0.          ,  0.          , -0.0001393381,
       0.          ,  0.          , -0.015291321 ]])


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


def _ccsd_hess(geom, freeze_core, orbital_basis):
    """Converged CCSD Hessian PropertyComponents for the CFOUR-basis reference."""
    wfn = _cfour_wfn(geom, freeze_core)
    cc = pycc.ccwfn(wfn, orbital_basis=orbital_basis)      # model defaults to CCSD
    cc.solve_cc(1e-12, 1e-12, 100)
    return pycc.hessian(pycc.CCderiv(cc))


def _check(r, corr_ref):
    """Shared assertions for a CCSD Hessian run vs the CFOUR oracle: the full 9x9 correlation matrix,
    the facade decomposition (total == nuclear + reference + correlation), and the FD-free physics
    checks -- symmetry and the translational (acoustic) sum rule (E_corr is invariant under uniform
    translation, so summing over the second atom vanishes)."""
    corr = np.asarray(r.correlation)
    total = np.asarray(r.total)
    assert corr.shape == (9, 9)
    assert np.max(np.abs(corr - corr_ref)) < 1e-9, np.max(np.abs(corr - corr_ref))
    parts = np.asarray(r.nuclear) + np.asarray(r.reference) + np.asarray(r.correlation)
    assert np.max(np.abs(total - parts)) < 1e-12
    assert np.max(np.abs(corr - corr.T)) < 1e-8
    assert np.max(np.abs(corr.reshape(3, 3, 3, 3).sum(axis=2))) < 1e-8


def test_ccsd_hessian_cfour_water_spatial():
    """All-electron and frozen-core spatial (closed-shell RHF) CCSD Hessian for water/6-31G vs the
    CFOUR FCMFINAL oracle -- the full correlation matrix, the facade decomposition, symmetry, and the
    translational sum rule."""
    _check(_ccsd_hess(WATER, 'false', 'spatial'), CFOUR_HESS_WATER)
    psi4.core.clean()
    _check(_ccsd_hess(WATER, 'true', 'spatial'), CFOUR_HESS_WATER_FC)
    psi4.core.clean()


def test_so_ccsd_hessian_cfour_water():
    """Spin-orbital CCSD Hessian for water/6-31G vs the CFOUR oracle (all-electron and frozen core) --
    the direct SO-vs-CFOUR anchor, fast enough for the default suite (SO perturbed amplitudes/Lambda,
    inline orbital Hessian, and the nuclear-nuclear second skeletons)."""
    _check(_ccsd_hess(WATER, 'false', 'spinorbital'), CFOUR_HESS_WATER)
    psi4.core.clean()
    _check(_ccsd_hess(WATER, 'true', 'spinorbital'), CFOUR_HESS_WATER_FC)
    psi4.core.clean()


@pytest.mark.slow
def test_ccsd_hessian_cfour_hof_spatial():
    """Second-geometry cross-check: spatial CCSD Hessian for HOF/6-31G (Cs) vs the CFOUR oracle
    (all-electron and frozen core).  Marked slow."""
    _check(_ccsd_hess(HOF, 'false', 'spatial'), CFOUR_HESS_HOF)
    psi4.core.clean()
    _check(_ccsd_hess(HOF, 'true', 'spatial'), CFOUR_HESS_HOF_FC)
    psi4.core.clean()


@pytest.mark.slow
def test_so_ccsd_hessian_cfour_hof():
    """Spin-orbital CCSD Hessian for HOF/6-31G vs the CFOUR oracle (all-electron and frozen core) --
    the off-diagonal (Cs) SO cross-check.  Marked slow."""
    _check(_ccsd_hess(HOF, 'false', 'spinorbital'), CFOUR_HESS_HOF)
    psi4.core.clean()
    _check(_ccsd_hess(HOF, 'true', 'spinorbital'), CFOUR_HESS_HOF_FC)
    psi4.core.clean()
