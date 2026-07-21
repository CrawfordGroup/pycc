"""
Test: E_bch2 = 2*Tr[F.D1] + Tr[L.D2],  L = 2*ERI - ERI.swapaxes(2,3)

Two modes:
  A) Using compute_twopdm_full() from uccwfn.py (once that transcription is
     debugged/verified). Run via test_ucc_energy_from_density_internal().

  B) Using Ajay's externally-saved D2_pppp.npy (from running his raw script).
     Run via test_ucc_energy_from_density_external(path_to_npy_dir).
     This is the immediate usable test until compute_twopdm_full() is verified.

D1 in both cases comes from compute_onepdm() (blocked, already validated for
shape/finiteness). The energy reference is energy_bch2() (independently
transcribed from Ajay's BCH2 energy script, validated for shape/runtime).
"""

import numpy as np
import psi4
import pycc

from pycc.uccwfn import make_ucc_fns


def _run_scf(mol_str, basis, **opts):
    psi4.core.clean()
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('test_ucc_energy_from_density.dat', False)
    defaults = dict(
        scf_type='pk', freeze_core='false',
        e_convergence=1e-13, d_convergence=1e-13,
        r_convergence=1e-13, diis=1,
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


def energy_bch2(ucc, F, ERI, t_vo, t_vvoo, tdag_vo, tdag_vvoo):
    """BCH2 energy -- literal transcription of Ajay's script. F/ERI sliced
    directly from arguments (does not touch self.g_* / self.f_*)."""
    no, nv = ucc.no, ucc.nv
    o, v = slice(0, no), slice(no, no + nv)
    f_oo, f_vv, f_ov, f_vo = F[o, o], F[v, v], F[o, v], F[v, o]
    g_oooo = ERI[o, o, o, o]; g_oovv = ERI[o, o, v, v]; g_vvoo = ERI[v, v, o, o]
    g_ovov = ERI[o, v, o, v]; g_ovvo = ERI[o, v, v, o]; g_ovoo = ERI[o, v, o, o]
    g_ooov = ERI[o, o, o, v]; g_vvvv = ERI[v, v, v, v]
    g_vvov = ERI[v, v, o, v]; g_ovvv = ERI[o, v, v, v]
    c = ucc._contract
    t_dag_ov_fn   = lambda: c('ai->ia', tdag_vo)
    t_dag_oovv_fn = lambda: c('abic->icab', tdag_vvoo)
    E = 0.0 + 0.0j
    t_dag_oovv = t_dag_oovv_fn()
    E += -4 * c('ia,ia->', c('iabc,cbad->di', t_dag_oovv, t_vvoo), f_oo)
    t_dag_oovv = t_dag_oovv_fn()
    E += -1 * c('ia,ai->', c('iabc,ci->ab', t_dag_oovv, f_vo), t_vo)
    I = c('iabc,bdei->cead', g_ovvo, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E += -4 * c('iabc,iabc->', I, t_dag_oovv)
    I = c('iabc,bdei->cead', g_ovvo, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  2 * c('iabc,iacb->', I, t_dag_oovv)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  2 * c('ia,ai->', c('iabc,dbia->dc', t_dag_oovv, g_ovoo), t_vo)
    I = c('iabc,cdei->bead', g_ovov, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  2 * c('iabc,iabc->', I, t_dag_oovv)
    I = c('iabc,cdei->bead', g_ovov, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E += -4 * c('iabc,iacb->', I, t_dag_oovv)
    t_dag_oovv = t_dag_oovv_fn()
    E += -1 * c('iabc,cbia->', t_dag_oovv, g_vvoo)
    I = c('iabc,bdie->cead', g_ovvo, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E += -4 * c('iabc,iacb->', I, t_dag_oovv)
    I = c('iabc,bdie->cead', g_ovvo, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  8 * c('iabc,iabc->', I, t_dag_oovv)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  2 * c('ia,ai->', c('iabc,bi->ac', t_dag_oovv, f_vo), t_vo)
    I = c('iabc,cdie->bead', g_ovov, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  2 * c('iabc,iacb->', I, t_dag_oovv)
    I = c('iabc,cdie->bead', g_ovov, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E += -4 * c('iabc,iabc->', I, t_dag_oovv)
    I = c('iabc,deia->bcde', g_oooo, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E += -1 * c('iabc,iacb->', I, t_dag_oovv)
    I = c('iabc,deia->bcde', g_oooo, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  2 * c('iabc,iabc->', I, t_dag_oovv)
    t_dag_oovv = t_dag_oovv_fn()
    E += -4 * c('ia,ai->', c('iabc,dcia->db', t_dag_oovv, g_ovoo), t_vo)
    I = c('abcd,cdie->eiab', g_vvvv, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E += -1 * c('iabc,iabc->', I, t_dag_oovv)
    I = c('abcd,cdie->ieab', g_vvvv, t_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  2 * c('iabc,iabc->', I, t_dag_oovv)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  4 * c('ia,ai->', c('iabc,bcid->ad', t_dag_oovv, g_vvov), t_vo)
    t_dag_oovv = t_dag_oovv_fn()
    E += -2 * c('ab,ab->', c('iabc,dbia->cd', t_dag_oovv, t_vvoo), f_vv)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  4 * c('ab,ab->', c('iabc,dbai->cd', t_dag_oovv, t_vvoo), f_vv)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  2 * c('ia,ia->', c('iabc,bcad->di', t_dag_oovv, t_vvoo), f_oo)
    t_dag_oovv = t_dag_oovv_fn()
    E +=  2 * c('iabc,bcia->', t_dag_oovv, g_vvoo)
    t_dag_oovv = t_dag_oovv_fn()
    E += -2 * c('ia,ai->', c('iabc,cbid->ad', t_dag_oovv, g_vvov), t_vo)
    t_dag_ov = t_dag_ov_fn()
    E += -2 * c('ia,ia->', c('ia,bi->ab', f_oo, t_vo), t_dag_ov)
    t_dag_ov = t_dag_ov_fn()
    E += -1 * c('ia,ia->', c('ia,baic->cb', t_dag_ov, g_vvoo), t_dag_ov_fn())
    t_dag_ov = t_dag_ov_fn()
    E +=  2 * c('ia,ia->', c('ab,bi->ia', f_vv, t_vo), t_dag_ov)
    t_dag_ov = t_dag_ov_fn()
    E += -2 * c('ia,ia->', c('iabc,bcdi->da', g_ovvv, t_vvoo), t_dag_ov)
    t_dag_ov = t_dag_ov_fn()
    E +=  4 * c('ia,ia->', c('iabc,bi->ca', g_ovvo, t_vo), t_dag_ov)
    t_dag_ov = t_dag_ov_fn()
    E +=  4 * c('ia,ia->', c('iabc,bcid->da', g_ovvv, t_vvoo), t_dag_ov)
    t_dag_ov = t_dag_ov_fn()
    E +=  2 * c('ia,ia->', c('iabc,cdia->bd', g_ooov, t_vvoo), t_dag_ov)
    t_dag_ov = t_dag_ov_fn()
    E += -4 * c('ia,ia->', c('iabc,cdai->bd', g_ooov, t_vvoo), t_dag_ov)
    E +=  2 * c('ia,ai->', t_dag_ov_fn(), f_vo)
    t_dag_ov = t_dag_ov_fn()
    E +=  2 * c('ia,ia->', c('ia,abic->cb', f_ov, t_vvoo), t_dag_ov)
    t_dag_ov = t_dag_ov_fn()
    E += -2 * c('ia,ia->', c('iabc,ci->ba', g_ovov, t_vo), t_dag_ov)
    t_dag_ov = t_dag_ov_fn()
    E += -1 * c('ia,ia->', c('ia,abci->cb', f_ov, t_vvoo), t_dag_ov)
    t_dag_ov = t_dag_ov_fn()
    E +=  2 * c('ia,ia->', c('ia,abic->cb', t_dag_ov, g_vvoo), t_dag_ov_fn())
    E += -1 * c('ia,ai->', c('iabc,ba->ic', g_oovv, t_vo), t_vo)
    E +=  2 * c('iabc,bcia->', g_oovv, t_vvoo)
    E += -1 * c('iabc,bcai->', g_oovv, t_vvoo)
    E +=  2 * c('ia,ai->', f_ov, t_vo)
    E +=  2 * c('ia,ai->', c('iabc,bi->ac', g_oovv, t_vo), t_vo)
    return E


def print_comparison(label, E_bch2, E_density, fD1, LD2):
    print()
    print("=" * 65)
    print(f"  {label}")
    print("=" * 65)
    print(f"  E_bch2 (direct, from amplitudes)   : {E_bch2: .10f}")
    print(f"  2 * Tr[F . D1]                      : {2.0*fD1: .10f}")
    print(f"  Tr[L . D2]                           : {LD2: .10f}")
    print(f"  2*Tr[F.D1] + Tr[L.D2]               : {E_density: .10f}")
    print(f"  |E_bch2 - (2*fD1 + LD2)|            : {abs(E_bch2 - E_density):.2e}")
    print("=" * 65)


def run_test(cc, ucc, label):
    no, nv = ucc.no, ucc.nv
    t_vo = ucc.t1.T.astype(complex)
    t_vvoo = ucc.t2.transpose(2, 3, 0, 1).astype(complex)
    tdag_vo, tdag_vvoo = t_vo.conj(), t_vvoo.conj()

    F = ucc.F
    ERI = ucc.ERI
    L = 2.0 * ERI - ERI.swapaxes(2, 3)

    E_bch2 = energy_bch2(ucc, F, ERI, t_vo, t_vvoo, tdag_vo, tdag_vvoo).real

    # D1 from blocked compute_onepdm (already validated)
    D1 = np.asarray(ucc.compute_onepdm(t_vo, t_vvoo, tdag_vo, tdag_vvoo))
    fD1 = np.dot(F.flatten(), D1.flatten()).real

    # D2 from compute_twopdm_full
    D2 = ucc.compute_twopdm(t_vo, t_vvoo, tdag_vo, tdag_vvoo)
    LD2 = np.dot(ERI.flatten(), D2.flatten()).real

    E_density = fD1 + 0.5*LD2
    print(D1)
    print(D2)
    print_comparison(label, E_bch2, E_density, fD1, LD2)


def test_ucc_energy_from_density():
    wfn = _run_scf(
        """
        O   -0.702196054  -0.056060256   0.009942262
H   -1.022193224   0.846775782  -0.011488714
H    0.257521062   0.042121496   0.005218999
        """,
        basis='6-31g',
    )
    cc = _run_ccsd(wfn)
    ucc, _, _ = make_ucc_fns(cc, e_conv=1e-13, r_conv=1e-12)
    run_test(cc, ucc, "E_bch2 vs 2*Tr[F.D1]+Tr[L.D2]  (H2O/6-31G)")


def test_ucc_energy_from_external_d2(cc=None, ucc=None):
    """
    Mode B: load Ajay's externally-computed D2_pppp.npy and use that
    instead of compute_twopdm_full(). Useful for validating the energy
    formula independent of any uccwfn.py transcription issues.

    d2_npy_path: path to D2_pppp.npy produced by Ajay's raw script.
    cc, ucc: if None, re-runs SCF/CCSD/UCC internally.
    """
    if cc is None or ucc is None:
        wfn = _run_scf(
            """
            O   -0.702196054  -0.056060256   0.009942262
H   -1.022193224   0.846775782  -0.011488714
H    0.257521062   0.042121496   0.005218999
            """,
            basis='6-31g',
        )
        cc = _run_ccsd(wfn)
        ucc, _, _ = make_ucc_fns(cc, e_conv=1e-13, r_conv=1e-12)

    no, nv = ucc.no, ucc.nv
    t_vo = ucc.t1.T.astype(complex)
    t_vvoo = ucc.t2.transpose(2, 3, 0, 1).astype(complex)
    tdag_vo, tdag_vvoo = t_vo.conj(), t_vvoo.conj()

    F = ucc.F
    ERI = ucc.ERI
    L = 2.0 * ERI - ERI.swapaxes(2, 3)

    E_bch2 = energy_bch2(ucc, F, ERI, t_vo, t_vvoo, tdag_vo, tdag_vvoo).real
    D1 = np.asarray(ucc.compute_onepdm(t_vo, t_vvoo, tdag_vo, tdag_vvoo))
    fD1 = np.dot(F.flatten(), D1.flatten()).real

    D2 = np.load(d2_npy_path)
    print(f"  Loaded D2_pppp from {d2_npy_path}, shape {D2.shape}")
    LD2 = np.dot(L.flatten(), D2.flatten()).real

    E_density = 2.0 * fD1 + LD2
    print_comparison("E_bch2 vs 2*Tr[F.D1]+Tr[L.D2] (external D2)", E_bch2, E_density, fD1, LD2)


if __name__ == "__main__":
    import sys
    if False:
        # Mode B: python test_ucc_energy_from_density.py /path/to/D2_pppp.npy
        test_ucc_energy_from_external_d2()
    else:
        # Mode A: uses compute_twopdm_full()
        test_ucc_energy_from_density()
