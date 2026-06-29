import numpy as np
import pytest
import psi4
import pycc
from scipy.integrate import complex_ode as ode

from pycc.rt.lasers import sine_square_laser
from pycc.rt.integrators import rk4
from pycc.uccwfn import make_ucc_fns, UCCWfn
from pycc.rt.rtcc_ucc import rtcc_ucc
from pycc.data.molecules import *

def _run_scf(mol_str, basis, **opts):
    psi4.core.clean()
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('test_rtuccsd.dat', False)
    defaults = dict(
        scf_type='pk',
        freeze_core='false',
        e_convergence=1e-13,
        d_convergence=1e-13,
        r_convergence=1e-13,
        diis=1,
    )
    defaults.update(opts)
    psi4.set_options({'basis': basis, **defaults})
    mol = psi4.geometry(mol_str)
    _, wfn = psi4.energy('SCF', return_wfn=True)
    return wfn


def _run_ccsd(wfn, e_conv=1e-12, r_conv=1e-12):
    cc = pycc.ccwfn(wfn, model='CCSD')
    cc.solve_cc(e_conv, r_conv, maxiter=75)
    return cc


def _run_uccsd(cc, e_conv=1e-11, r_conv=1e-11):
    ucc, energy_fn, residuals_fn = make_ucc_fns(cc, e_conv=e_conv, r_conv=r_conv)
    # Transfer converged UCC amplitudes back onto cc so rtcc_ucc uses them
    cc.t1 = ucc.t1
    cc.t2 = ucc.t2
    return ucc, energy_fn, residuals_fn

def test_uccsd_rdm_trace():
    wfn = _run_scf(
        """
        O  0.000000000000  -0.143225816552   0.000000000000
        H  1.638036840407   1.136548822547  -0.000000000000
        H -1.638036840407   1.136548822547  -0.000000000000
        units bohr
        """,
        basis='6-31g',
    )
    cc  = _run_ccsd(wfn)
    ucc, energy_fn, residuals_fn = _run_uccsd(cc)

    # Build density at ground-state (real) amplitudes
    rt = rtcc_ucc(cc, lambda t: 0.0, energy_fn, residuals_fn, uccwfn=ucc)
    t1, t2 = cc.t1, cc.t2

    t_vo,    t_vvoo    = rt._to_ajay(t1, t2)
    tdag_vo, tdag_vvoo = rt._prep_tdag(t_vo, t_vvoo)

    opdm = ucc.compute_onepdm(t_vo, t_vvoo, tdag_vo, tdag_vvoo)

    no  = cc.no
    nv  = cc.nv
    nmo = no + nv
    o   = slice(0, no)
    v   = slice(no, nmo)

    trace_oo  = np.trace(opdm[o, o])
    trace_vv  = np.trace(opdm[v, v])
    trace_tot = trace_oo + trace_vv

    print()
    print("=" * 55)
    print("  UCC 1-RDM trace checks  (H2O / 6-31g)")
    print("=" * 55)
    print(f"  tr(D_oo)            : {trace_oo.real:.8f}  (im: {trace_oo.imag:.2e})")
    print(f"  tr(D_vv)            : {trace_vv.real:.8f}  (im: {trace_vv.imag:.2e})")
    print(f"  tr(D_oo) + tr(D_vv) : {trace_tot.real:.8f}  (correlation; should be 0 at HF limit)")
    print(f"  tr(D_oo) + tr(D_vv) = -tr(D_vv) + tr(D_vv) conservation: "
          f"{abs(trace_oo.real + trace_vv.real):.2e}  (occ depletion = virt occupation)")
    print(f"  |Im(trace)|         : {abs(trace_tot.imag):.2e}  (should be 0)")
    print(f"  gamma Hermitian?    : max|gamma-gamma.T| = "
          f"{np.max(np.abs(opdm - opdm.conj().T)):.2e}  (should be 0)")
    print("=" * 55)
    exit()

def test_rtuccsd_he_dipole():
    """
    He / cc-pVDZ, sine-squared laser, tf = 1.0 a.u.
    """
    wfn = _run_scf(
        "He 0 0 0",
        basis='cc-pVDZ',
    )
    cc  = _run_ccsd(wfn)
    ucc, energy_fn, residuals_fn = _run_uccsd(cc)

    # Laser pulse
    F_str  = 1.0
    omega  = 2.87
    tprime = 5.0
    V = sine_square_laser(F_str, omega, tprime)

    # RT-UCCSD propagation with scipy vode integrator
    t0 = 0.0
    tf = 1.0
    h  = 0.01

    rt = rtcc_ucc(cc, V, energy_fn, residuals_fn, kick='z', uccwfn=ucc)
    y0 = rt.collect_amps(cc.t1, cc.t2, 0.0 + 0.0j)

    ODE = ode(rt.f).set_integrator('vode', atol=1e-13, rtol=1e-13)
    ODE.set_initial_value(y0, t0)

    # Initial dipole
    t1_0, t2_0, _ = rt.extract_amps(y0)
    mu0_x, mu0_y, mu0_z = rt.dipole(t1_0, t2_0)

    print()
    print("=" * 55)
    print("  RT-UCCSD dipole test  (He / cc-pVDZ)")
    print("=" * 55)
    print(f"  mu_z(t=0) : {mu0_z.real:.10f}")

    mu_z = None
    while ODE.successful() and ODE.t < tf:
        y = ODE.integrate(ODE.t + h)
        t = ODE.t
        t1, t2, phase = rt.extract_amps(y)
        mu_x, mu_y, mu_z = rt.dipole(t1, t2)
        ecc = rt.energy(t, t1, t2)

    print(f"  mu_z(t={tf}) : {mu_z.real:.10f}")
    print(f"  |Im(mu_z)|  : {abs(mu_z.imag):.2e}")
    print("=" * 55)

if __name__ == "__main__":
    test_uccsd_rdm_trace()
    test_rtuccsd_he_dipole()
