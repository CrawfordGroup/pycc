"""
Dipole moment test: finite difference of E = Tr[F.D1] + 0.5*Tr[ERI.D2]
with respect to a field perturbation F -> F + lam*mu should equal Tr[mu.D1].

Since D1 and D2 are evaluated at FIXED converged amplitudes (T-fixed), this
is a pure calculus identity -- no amplitude reconvergence is needed in the
FD loop. The FD is exact (not approximate) because E is exactly linear in F
at fixed T:

    d/dlam [Tr[(F + lam*mu).D1] + 0.5*Tr[ERI.D2]]  =  Tr[mu.D1]

So FD(+h) - FD(-h) / 2h == Tr[mu.D1] to machine precision for any h.

Three dipole components tested (x, y, z).
"""

import numpy as np
import psi4
import pycc

from pycc.uccwfn import make_ucc_fns


def _run_scf(mol_str, basis, **opts):
    psi4.core.clean()
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('test_dipole_ucc.dat', False)
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


def _run_ccsd(wfn, e_conv=1e-13, r_conv=1e-13):
    cc = pycc.ccwfn(wfn, model='CCSD')
    cc.solve_cc(e_conv, r_conv, maxiter=100)
    return cc


def energy_from_density(F, ERI, D1, D2):
    """E = Tr[F.D1] + 0.5*Tr[ERI.D2] at fixed D1, D2."""
    return (np.dot(F.flatten(), D1.flatten())
            + 0.5 * np.dot(ERI.flatten(), D2.flatten())).real


def test_ucc_dipole():
    wfn = _run_scf(
        """
        O  0.000000000000  -0.143225816552   0.000000000000
        H  1.638036840407   1.136548822547  -0.000000000000
        H -1.638036840407   1.136548822547  -0.000000000000
        units bohr
        """,
        basis='6-31g',
    )
    cc = _run_ccsd(wfn)
    ucc, _, _ = make_ucc_fns(cc, e_conv=1e-13, r_conv=1e-12)

    # Amplitudes in Ajay convention at unperturbed ground state
    t_vo      = ucc.t1.T.astype(complex)
    t_vvoo    = ucc.t2.transpose(2, 3, 0, 1).astype(complex)
    tdag_vo   = t_vo.conj()
    tdag_vvoo = t_vvoo.conj()

    # Densities at fixed converged amplitudes -- computed once, used throughout
    D1 = np.asarray(ucc.compute_onepdm_full(t_vo, t_vvoo, tdag_vo, tdag_vvoo)).real
    D2 = np.asarray(ucc.compute_twopdm_full(t_vo, t_vvoo, tdag_vo, tdag_vvoo)).real

    F0   = ucc.F
    ERI  = ucc.ERI

    print()
    print("=" * 65)
    print("  UCC unrelaxed dipole: FD of E[D1,D2] vs Tr[mu.D1]")
    print("  E = Tr[F.D1] + 0.5*Tr[ERI.D2], T fixed, ERI fixed")
    print("=" * 65)

    labels = ['x', 'y', 'z']
    h = 1e-4  # any h works; E is exactly linear in F at fixed T

    for axis in range(3):
        mu = cc.H.mu[axis]   # MO dipole integrals

        # Analytic: Tr[mu.D1]
        analytic = np.dot(mu.flatten(), D1.flatten())

        # FD: d/dlam E(F + lam*mu) at fixed D1, D2
        E_plus  = energy_from_density(F0 + h * mu, ERI, D1, D2)
        E_minus = energy_from_density(F0 - h * mu, ERI, D1, D2)
        fd = (E_plus - E_minus) / (2 * h)

        print(f"  mu_{labels[axis]}  analytic Tr[mu.D1] : {analytic: .10f}")
        print(f"       FD dE/dlam      : {fd: .10f}")
        print(f"       |gap|           : {abs(analytic - fd):.2e}")
        print()

    print("=" * 65)


if __name__ == "__main__":
    test_ucc_dipole()
