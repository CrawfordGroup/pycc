"""
CISD diagonal Born-Oppenheimer correction (CIderiv.dboc) against the
analytic values of Gauss, Tajti, Kallay, Stanton, and Szalay,
J. Chem. Phys. 125, 144111 (2006), Table I(a), cc-pVDZ.

Geometries are the exact all-electron CCSD(T)/cc-pVQZ structures of the
paper's Ref. 14 (HEAT: Tajti et al., JCP 121, 11599 (2004), footnote);
with them the CISD values match the paper to its printed precision.
Nuclear (bare) masses are used inside CIderiv.dboc,
matching the paper's convention.
"""

import psi4
import pycc
from ..cideriv import CIderiv

HARTREE_CM = 219474.6313632

# exact ae-CCSD(T)/cc-pVQZ structures of the paper's Ref. 14 (HEAT:
# Tajti et al., J. Chem. Phys. 121, 11599 (2004), footnote)
GEOMS = {
    'h2': """
0 1
H
H 1 0.74186
symmetry c1
""",
    'hf': """
0 1
F
H 1 0.91516
symmetry c1
""",
    'h2o': """
0 1
O
H 1 0.95623
H 1 0.95623 2 104.25
symmetry c1
""",
}


def _cisd_dboc_cm(geom):
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    psi4.geometry(geom)
    e, wfn = psi4.energy('scf', return_wfn=True)
    ci = pycc.CIwfn(wfn, model='CISD')
    ci.solve_ci(e_conv=1e-11, r_conv=1e-11, maxiter=200)
    return CIderiv(ci).dboc() * HARTREE_CM


def test_cisd_dboc_h2():
    assert abs(_cisd_dboc_cm(GEOMS['h2']) - 111.91) < 0.01


def test_cisd_dboc_hf():
    assert abs(_cisd_dboc_cm(GEOMS['hf']) - 616.38) < 0.01


def test_cisd_dboc_h2o():
    assert abs(_cisd_dboc_cm(GEOMS['h2o']) - 615.69) < 0.01


def _cisd_wfn(geom, basis='6-31G'):
    """Converged CISD wavefunction only (the reference checks above go through _cisd_dboc_cm,
    which returns the DBOC directly and so cannot be reused for a two-driver comparison).

    The basis differs from this module's cc-pVDZ reference checks, and may: the test below is an
    INVARIANCE test, comparing two computations of the same quantity against each other, so it
    needs no external oracle and the basis is free to pick on cost and conditioning.  6-31G is
    both (2.4 s here against 5.0 s at cc-pVDZ) and is the best conditioned of the options
    measured: worst DIIS cond(B) 2.0e26, against 1.1e27 at cc-pVDZ, 4.8e29 at STO-3G, and 5.4e31
    for H2/STO-3G, which is what made this test fail on the Linux CI runners while passing on
    macOS -- Linux LAPACK rejects that system as exactly singular where macOS returns a value.
    See the helper_diis note in docs/DERIVATIVES_PLAN_2026-06.md: every one of those numbers is
    past double precision, so this choice reduces exposure rather than removing it."""
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': basis, 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    psi4.geometry(geom)
    _e, wfn = psi4.energy('scf', return_wfn=True)
    ci = pycc.CIwfn(wfn, model='CISD')
    ci.solve_ci(e_conv=1e-11, r_conv=1e-11, maxiter=200)
    return ci


def test_cisd_dboc_perturbed_mo_gauge_invariance():
    """The CISD DBOC is invariant to the perturbed-MO gauge, the canonical vs non-canonical choice
    governing the *nuclear* perturbations it is built from.

    That choice is real, but it is NOT what the former ``dboc(gauge=...)`` argument selected: that
    one was the *imaginary*-perturbation gauge, which the DBOC's nuclear-only perturbations never
    consume, so it was bitwise inert and the cross-check it invited could not fail.  The genuine
    choice reaches the CPCI solve from the driver, and this guards it.

    One molecule is enough for an invariance claim; H2 is deliberately not used, being the
    worst-conditioned case available (see :func:`_cisd_wfn`)."""
    ci = _cisd_wfn(GEOMS['h2o'])
    nc = CIderiv(ci).dboc()
    cd = CIderiv(ci)
    cd._gauge_override = 'canonical'
    ca = cd.dboc()
    assert abs(nc - ca) < 1e-9 * max(1.0, abs(nc)), (nc, ca)
