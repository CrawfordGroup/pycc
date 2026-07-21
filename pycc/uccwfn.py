"""
UCCSD wavefunction object for PyCC.  Mirrors the structure of ccwfn.py.

Energy   : BCH2  (uccsd_bch2_cs.py  — 2-commutator truncation)
Residuals: BCH2  (uccsd_bch2_cs.py  — 2-commutator truncation)
1-/2-RDM : BCH2  (ucc_1rdm_full_cs.py / ucc_2rdm_full_cs.py — unblocked,
           full-space D_pp / D2_pppp; E = f·d1 + 0.5·g·d2)

All integral blocks are loaded ONCE at __init__ from ccwfn.H and stored
as attributes.  energy() and residuals() do zero disk I/O.

The t_dag convention - Ajay's SeQuant generator emits t_dag as a plain TRANSPOSE:
    t_dag_oovv = einsum('abic->icab', t_vvoo)   # correct only for real T
For complex RT amplitudes, rtcc_ucc._prep_tdag() pre-conjugates the
inputs BEFORE passing them here, so every transpose inside this file
produces the correct conjugate transpose T†.
So, do NOT add .conj() calls anywhere in this file.

Index conventions (SeQuant)
  t_vo    (nv, no)            [ai]
  t_vvoo  (nv, nv, no, no)   [abij]
  tdag_vo / tdag_vvoo : pre-conjugated inputs passed by rtcc_ucc

Interface (called by rtcc_ucc), just like normal stuff
  ucc.energy(F, ERI, t_vo, t_vvoo, tdag_vo, tdag_vvoo)    -> scalar
  ucc.residuals(F, ERI, t_vo, t_vvoo, tdag_vo, tdag_vvoo) -> (R1_vo, R2_vvoo)

Usage
  from uccwfn import make_ucc_fns
  from rtcc_ucc import rtcc_ucc

  energy_fn, residuals_fn = make_ucc_fns(ccwfn)
  rt = rtcc_ucc(ccwfn, V, energy_fn, residuals_fn, kick='z')
"""

import numpy as np
from opt_einsum import contract as ct


class UCCWfn:
    """
    UCCSD wavefunction.

    Parameters
    ----------
    ccwfn : PyCC ccwfn object
        Provides .no, .nv, .o, .v, .H.F, .H.ERI, .t1, .t2
    """

    def __init__(self, ccwfn):
        self.no = ccwfn.no
        self.nv = ccwfn.nv
        self.t1 = ccwfn.t1.copy()   # initial guess (PyCC convention: no,nv)
        self.t2 = ccwfn.t2.copy()

        o   = ccwfn.o
        v   = ccwfn.v
        F   = ccwfn.H.F
        ERI = ccwfn.H.ERI

        # ---- Store F and ERI for solve_ucc ----
        self.F   = F.copy()
        self.ERI = ERI

        # ---- Fock denominators for amplitude updates ----
        eps      = np.diag(F)         # full MO energies
        eps_o    = eps[:self.no]      # occupied
        eps_v    = eps[self.no:]      # virtual

        # Singles: shape (nv, no) to match Ajay's t_vo convention
        self.D1  = (eps_v[:, None] - eps_o[None, :])          # (nv, no)

        # Doubles: shape (nv, nv, no, no) to match t_vvoo convention
        self.D2  = (eps_v[:, None, None, None]
                  + eps_v[None, :, None, None]
                  - eps_o[None, None, :, None]
                  - eps_o[None, None, None, :])                # (nv, nv, no, no)

        # ---- Two-electron integral blocks (antisymmetrized, loaded once) ----
        self.g_oooo = ERI[o, o, o, o].copy()
        self.g_oovv = ERI[o, o, v, v].copy()
        self.g_vvoo = ERI[v, v, o, o].copy()
        self.g_ovov = ERI[o, v, o, v].copy()
        self.g_ovvo = ERI[o, v, v, o].copy()
        self.g_ovoo = ERI[o, v, o, o].copy()
        self.g_ooov = ERI[o, o, o, v].copy()
        self.g_vvvv = ERI[v, v, v, v].copy()
        self.g_vvov = ERI[v, v, o, v].copy()
        self.g_ovvv = ERI[o, v, v, v].copy()

        self._expr_cache = {}  # lazy path cache
        print(f"UCCWfn: integrals loaded  no={self.no}  nv={self.nv}")

    # UCC Ground State Solver  (BCH2 residuals -> BCH2 energy)
    def solve_ucc(self, e_conv=1e-12, r_conv=1e-8, maxiter=200):
        """
        Converge UCC amplitudes using t-CCSD residuals (mirrors Ajay's MPQC).
        Once converged, evaluates BCH2 energy on the converged amplitudes.

        Updates self.t1, self.t2 in place (PyCC convention).
        Returns final BCH2 correlation energy (float).
        todo - add diis
        """
        t_vo   = np.zeros((self.nv, self.no), dtype=complex)
        t_vvoo = np.zeros((self.nv, self.nv, self.no, self.no), dtype=complex)

        Eold = 0.0 + 0.0j

        print(f"\n{'UCC Iter':>8}  {'E(BCH2)':>18}  {'dE':>12}  {'rms(R)':>12}")
        print("-" * 60)

        for i in range(maxiter):
            tdag_vo   = t_vo.conj()
            tdag_vvoo = t_vvoo.conj()

            # BCH2 residuals
            R1_vo, R2_vvoo = self.residuals(self.F, self.ERI, t_vo, t_vvoo, tdag_vo, tdag_vvoo)

            # Amplitude update with Fock denominator (steepest descent)
            t_vo   -= R1_vo   / self.D1
            t_vvoo -= R2_vvoo / self.D2
            t_vvoo = 0.5 * (t_vvoo + t_vvoo.transpose(1, 0, 3, 2))  #very imp, we need to symmetrize T2!!

            # BCH2 energy on updated amplitudes
            tdag_vo   = t_vo.conj()
            tdag_vvoo = t_vvoo.conj()
            # Cheap CCSD-like energy for convergence monitoring, or we can use bch2 energy (more lines of code!!)
            f_ov  = self.F[:self.no, self.no:]          # (no, nv)
            t_ov  = t_vo.T                              # (no, nv)  
            g_oovv = self.g_oovv                        # (no, no, nv, nv)
            # Singles contribution
            E_s = np.einsum('ia,ia->', f_ov, t_ov)
            # Doubles contribution  
            E_d = 0.25 * np.einsum('abij,ijab->', t_vvoo, g_oovv)
            # Singles^2 contribution
            E_ss = 0.5 * np.einsum('ai,bj,ijab->', t_vo, t_vo, g_oovv)

            E = E_s + E_d + E_ss

            dE  = abs(E - Eold)
            rms = np.sqrt((np.mean(np.abs(R1_vo)**2) + np.mean(np.abs(R2_vvoo)**2)) / 2)

            print(f"{i+1:>8d}  {E.real:>18.10f}  {dE:>12.2e}  {rms:>12.2e}")

            if dE < e_conv:
                print(f"UCC converged in {i+1} iterations.")
                break
            Eold = E
        else:
            self.last_energy = E.real
            print("Warning: UCC did not converge!")

        # Store back in PyCC convention
        self.t1 = t_vo.T.real
        self.t2 = t_vvoo.transpose(2, 3, 0, 1).real

        # Final BCH2 energy on converged amplitudes (called once) - this is smarter and faster. we can use bch2 energy too (same timings)
        tdag_vo_f   = t_vo.conj()
        tdag_vvoo_f = t_vvoo.conj()
        E_bch4 = self.energy(self.F, self.ERI, t_vo, t_vvoo, tdag_vo_f, tdag_vvoo_f)
        self.last_energy = E_bch4.real
        print(f"UCC BCH2 energy (final): {E_bch4.real:.10f}")

        return E_bch4.real

    # Field-dressed Fock blocks  (ERI blocks are field-independent)
    @staticmethod
    def _fock(F, no):
        return F[:no, :no], F[no:, no:], F[:no, no:], F[no:, :no]

    # BCH2 Energy
    def _contract(self, spec, *operands):
        """Lazy-cached opt_einsum contract_expression."""
        key = (spec,) + tuple(op.shape for op in operands)
        expr = self._expr_cache.get(key)
        if expr is None:
            from opt_einsum import contract_expression
            expr = contract_expression(spec, *(op.shape for op in operands))
            self._expr_cache[key] = expr
        return expr(*operands)

    def energy(self, F, ERI, t_vo, t_vvoo, tdag_vo, tdag_vvoo):
        """UCCSD BCH2 energy  (uccsd_bch2_cs.py, SeQuant CS, hbar_comm_rank=2).

        Consistent with the BCH2 residuals below.  t_dag is a transpose of the
        PRE-CONJUGATED tdag_* inputs (rtcc_ucc._prep_tdag), so no .conj() here.
        """
        nocc = self.no
        nvirt = self.nv
        nmo = nocc + nvirt
        o = slice(0, nocc)
        v = slice(nocc, nmo)
        E = 0.0
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,cbad->di', t_dag_oovv, t_vvoo)
        f_oo = F[o, o]
        E += -4 * self._contract('ia,ia->', I_oo, f_oo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        f_vo = F[v, o]
        I_ov += self._contract('iabc,ci->ab', t_dag_oovv, f_vo)
        E += -1 * self._contract('ia,ai->', I_ov, t_vo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovvo = ERI[o, v, v, o]
        I_oovv += self._contract('iabc,bdei->cead', g_ovvo, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += -4 * self._contract('iabc,iabc->', I_oovv, t_dag_oovv)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovvo = ERI[o, v, v, o]
        I_oovv += self._contract('iabc,bdei->cead', g_ovvo, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += 2 * self._contract('iabc,iacb->', I_oovv, t_dag_oovv)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ov += self._contract('iabc,dbia->dc', t_dag_oovv, g_ovoo)
        E += 2 * self._contract('ia,ai->', I_ov, t_vo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovov = ERI[o, v, o, v]
        I_oovv += self._contract('iabc,cdei->bead', g_ovov, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += 2 * self._contract('iabc,iabc->', I_oovv, t_dag_oovv)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovov = ERI[o, v, o, v]
        I_oovv += self._contract('iabc,cdei->bead', g_ovov, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += -4 * self._contract('iabc,iacb->', I_oovv, t_dag_oovv)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        E += -1 * self._contract('iabc,cbia->', t_dag_oovv, g_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovvo = ERI[o, v, v, o]
        I_oovv += self._contract('iabc,bdie->cead', g_ovvo, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += -4 * self._contract('iabc,iacb->', I_oovv, t_dag_oovv)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovvo = ERI[o, v, v, o]
        I_oovv += self._contract('iabc,bdie->cead', g_ovvo, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += 8 * self._contract('iabc,iabc->', I_oovv, t_dag_oovv)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        f_vo = F[v, o]
        I_ov += self._contract('iabc,bi->ac', t_dag_oovv, f_vo)
        E += 2 * self._contract('ia,ai->', I_ov, t_vo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovov = ERI[o, v, o, v]
        I_oovv += self._contract('iabc,cdie->bead', g_ovov, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += 2 * self._contract('iabc,iacb->', I_oovv, t_dag_oovv)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovov = ERI[o, v, o, v]
        I_oovv += self._contract('iabc,cdie->bead', g_ovov, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += -4 * self._contract('iabc,iabc->', I_oovv, t_dag_oovv)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_oooo = ERI[o, o, o, o]
        I_oovv += self._contract('iabc,deia->bcde', g_oooo, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += -1 * self._contract('iabc,iacb->', I_oovv, t_dag_oovv)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_oooo = ERI[o, o, o, o]
        I_oovv += self._contract('iabc,deia->bcde', g_oooo, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += 2 * self._contract('iabc,iabc->', I_oovv, t_dag_oovv)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ov += self._contract('iabc,dcia->db', t_dag_oovv, g_ovoo)
        E += -4 * self._contract('ia,ai->', I_ov, t_vo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_vvvv = ERI[v, v, v, v]
        I_oovv += self._contract('abcd,cdie->eiab', g_vvvv, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += -1 * self._contract('iabc,iabc->', I_oovv, t_dag_oovv)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_vvvv = ERI[v, v, v, v]
        I_oovv += self._contract('abcd,cdie->ieab', g_vvvv, t_vvoo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        E += 2 * self._contract('iabc,iabc->', I_oovv, t_dag_oovv)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ov += self._contract('iabc,bcid->ad', t_dag_oovv, g_vvov)
        E += 4 * self._contract('ia,ai->', I_ov, t_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dbia->cd', t_dag_oovv, t_vvoo)
        f_vv = F[v, v]
        E += -2 * self._contract('ab,ab->', I_vv, f_vv)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dbai->cd', t_dag_oovv, t_vvoo)
        f_vv = F[v, v]
        E += 4 * self._contract('ab,ab->', I_vv, f_vv)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,bcad->di', t_dag_oovv, t_vvoo)
        f_oo = F[o, o]
        E += 2 * self._contract('ia,ia->', I_oo, f_oo)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        E += 2 * self._contract('iabc,bcia->', t_dag_oovv, g_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ov += self._contract('iabc,cbid->ad', t_dag_oovv, g_vvov)
        E += -2 * self._contract('ia,ai->', I_ov, t_vo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        f_oo = F[o, o]
        I_ov += self._contract('ia,bi->ab', f_oo, t_vo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += -2 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvoo = ERI[v, v, o, o]
        I_ov += self._contract('ia,baic->cb', t_dag_ov, g_vvoo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += -1 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        f_vv = F[v, v]
        I_ov += self._contract('ab,bi->ia', f_vv, t_vo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += 2 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_ov += self._contract('iabc,bcdi->da', g_ovvv, t_vvoo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += -2 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_ovvo = ERI[o, v, v, o]
        I_ov += self._contract('iabc,bi->ca', g_ovvo, t_vo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += 4 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_ov += self._contract('iabc,bcid->da', g_ovvv, t_vvoo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += 4 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_ov += self._contract('iabc,cdia->bd', g_ooov, t_vvoo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += 2 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_ov += self._contract('iabc,cdai->bd', g_ooov, t_vvoo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += -4 * self._contract('ia,ia->', I_ov, t_dag_ov)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        f_vo = F[v, o]
        E += 2 * self._contract('ia,ai->', t_dag_ov, f_vo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        f_ov = F[o, v]
        I_ov += self._contract('ia,abic->cb', f_ov, t_vvoo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += 2 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_ovov = ERI[o, v, o, v]
        I_ov += self._contract('iabc,ci->ba', g_ovov, t_vo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += -2 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        f_ov = F[o, v]
        I_ov += self._contract('ia,abci->cb', f_ov, t_vvoo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += -1 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvoo = ERI[v, v, o, o]
        I_ov += self._contract('ia,abic->cb', t_dag_ov, g_vvoo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        E += 2 * self._contract('ia,ia->', I_ov, t_dag_ov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_ov += self._contract('iabc,ba->ic', g_oovv, t_vo)
        E += -1 * self._contract('ia,ai->', I_ov, t_vo)
        g_oovv = ERI[o, o, v, v]
        E += 2 * self._contract('iabc,bcia->', g_oovv, t_vvoo)
        g_oovv = ERI[o, o, v, v]
        E += -1 * self._contract('iabc,bcai->', g_oovv, t_vvoo)
        f_ov = F[o, v]
        E += 2 * self._contract('ia,ai->', f_ov, t_vo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_ov += self._contract('iabc,bi->ac', g_oovv, t_vo)
        E += 2 * self._contract('ia,ai->', I_ov, t_vo)
        return E

    def residuals(self, F, ERI, t_vo, t_vvoo, tdag_vo, tdag_vvoo):
        """UCCSD BCH2 amplitude residuals  (uccsd_bch2_cs.py, SeQuant CS)."""
        nocc = self.no
        nvirt = self.nv
        nmo = nocc + nvirt
        o = slice(0, nocc)
        v = slice(nocc, nmo)
        R1_vo = np.zeros((nvirt, nocc), order='F', dtype=complex)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,bcad->di', t_dag_oovv, t_vvoo)
        f_vo = F[v, o]
        R1_vo += 1/2 * self._contract('ia,ba->bi', I_oo, f_vo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ov += self._contract('iabc,cbid->ad', t_dag_oovv, g_vvov)
        R1_vo += self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,cbda->di', t_dag_oovv, t_vvoo)
        g_ovoo = ERI[o, v, o, o]
        R1_vo += 2 * self._contract('ia,ibac->bc', I_oo, g_ovoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,cbda->di', t_dag_oovv, t_vvoo)
        g_ovoo = ERI[o, v, o, o]
        R1_vo += -1 * self._contract('ia,ibca->bc', I_oo, g_ovoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ooov += self._contract('iabc,dbie->edac', t_dag_oovv, g_ovoo)
        R1_vo += -1 * self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ooov += self._contract('iabc,dbie->edac', t_dag_oovv, g_ovoo)
        R1_vo += 2 * self._contract('iabc,dcab->di', I_ooov, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_vv += self._contract('iabc,dcai->db', t_dag_oovv, g_vvoo)
        R1_vo += 1/2 * self._contract('ab,bi->ai', I_vv, t_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dbia->cd', t_dag_oovv, t_vvoo)
        g_vvov = ERI[v, v, o, v]
        R1_vo += -2 * self._contract('ab,caib->ci', I_vv, g_vvov)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dcai->bd', t_dag_oovv, t_vvoo)
        g_vvov = ERI[v, v, o, v]
        R1_vo += self._contract('ab,acib->ci', I_vv, g_vvov)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dbia->dc', t_dag_oovv, t_vvoo)
        f_vo = F[v, o]
        R1_vo += 1/2 * self._contract('ab,bi->ai', I_vv, f_vo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ov += self._contract('iabc,bcid->ad', t_dag_oovv, g_vvov)
        R1_vo += -2 * self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ov += self._contract('iabc,bcid->ad', t_dag_oovv, g_vvov)
        R1_vo += 4 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ovvv += self._contract('iabc,deia->debc', t_dag_oovv, g_ovoo)
        R1_vo += -1 * self._contract('iabc,bcdi->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ovvv += self._contract('iabc,deia->debc', t_dag_oovv, g_ovoo)
        R1_vo += 2 * self._contract('iabc,bcid->ad', I_ovvv, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ooov += self._contract('iabc,dcei->edab', t_dag_oovv, g_ovoo)
        R1_vo += 2 * self._contract('iabc,dcab->di', I_ooov, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ooov += self._contract('iabc,dcei->edab', t_dag_oovv, g_ovoo)
        R1_vo += -1 * self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ov += self._contract('iabc,dbia->dc', t_dag_oovv, g_ovoo)
        R1_vo += -1 * self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ov += self._contract('iabc,dbia->dc', t_dag_oovv, g_ovoo)
        R1_vo += 2 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ovvv += self._contract('iabc,dbie->adec', t_dag_oovv, g_vvov)
        R1_vo += -2 * self._contract('iabc,bcdi->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ovvv += self._contract('iabc,dbie->adec', t_dag_oovv, g_vvov)
        R1_vo += self._contract('iabc,bcid->ad', I_ovvv, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ooov += self._contract('iabc,dcie->edab', t_dag_oovv, g_ovoo)
        R1_vo += 2 * self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ooov += self._contract('iabc,dcie->edab', t_dag_oovv, g_ovoo)
        R1_vo += -1 * self._contract('iabc,dcab->di', I_ooov, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ooov += self._contract('iabc,cbde->diae', t_dag_oovv, g_vvov)
        R1_vo += -2 * self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ooov += self._contract('iabc,bcde->diae', t_dag_oovv, g_vvov)
        R1_vo += self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dcia->bd', t_dag_oovv, t_vvoo)
        g_vvov = ERI[v, v, o, v]
        R1_vo += 4 * self._contract('ab,caib->ci', I_vv, g_vvov)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dbai->dc', t_dag_oovv, t_vvoo)
        f_vo = F[v, o]
        R1_vo += -1 * self._contract('ab,bi->ai', I_vv, f_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dbai->cd', t_dag_oovv, t_vvoo)
        g_vvov = ERI[v, v, o, v]
        R1_vo += -2 * self._contract('ab,acib->ci', I_vv, g_vvov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        f_vo = F[v, o]
        I_ov += self._contract('iabc,bi->ac', t_dag_oovv, f_vo)
        R1_vo += -1 * self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        f_vo = F[v, o]
        I_ov += self._contract('iabc,bi->ac', t_dag_oovv, f_vo)
        R1_vo += 2 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        f_vo = F[v, o]
        I_ov += self._contract('iabc,ci->ab', t_dag_oovv, f_vo)
        R1_vo += -1 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        f_vo = F[v, o]
        I_ov += self._contract('iabc,ci->ab', t_dag_oovv, f_vo)
        R1_vo += 1/2 * self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_ov += self._contract('iabc,ca->ib', t_dag_oovv, t_vo)
        g_vvoo = ERI[v, v, o, o]
        R1_vo += -1 * self._contract('ia,baic->bc', I_ov, g_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_ov += self._contract('iabc,ca->ib', t_dag_oovv, t_vo)
        g_vvoo = ERI[v, v, o, o]
        R1_vo += 2 * self._contract('ia,baci->bc', I_ov, g_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ovvv += self._contract('iabc,cdie->adeb', t_dag_oovv, g_vvov)
        R1_vo += -2 * self._contract('iabc,bcdi->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ovvv += self._contract('iabc,cdie->adeb', t_dag_oovv, g_vvov)
        R1_vo += self._contract('iabc,bcid->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ovvv += self._contract('iabc,dcie->adeb', t_dag_oovv, g_vvov)
        R1_vo += -2 * self._contract('iabc,bcid->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ovvv += self._contract('iabc,dcie->adeb', t_dag_oovv, g_vvov)
        R1_vo += self._contract('iabc,bcdi->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ovvv += self._contract('iabc,bdie->adec', t_dag_oovv, g_vvov)
        R1_vo += -2 * self._contract('iabc,bcid->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ovvv += self._contract('iabc,bdie->adec', t_dag_oovv, g_vvov)
        R1_vo += 4 * self._contract('iabc,bcdi->ad', I_ovvv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,cbad->di', t_dag_oovv, t_vvoo)
        f_vo = F[v, o]
        R1_vo += -1 * self._contract('ia,ba->bi', I_oo, f_vo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,cbad->di', t_dag_oovv, t_vvoo)
        g_ovoo = ERI[o, v, o, o]
        R1_vo += -4 * self._contract('ia,ibac->bc', I_oo, g_ovoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,cbad->di', t_dag_oovv, t_vvoo)
        g_ovoo = ERI[o, v, o, o]
        R1_vo += 2 * self._contract('ia,ibca->bc', I_oo, g_ovoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ov += self._contract('iabc,dcia->db', t_dag_oovv, g_ovoo)
        R1_vo += -4 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ov += self._contract('iabc,dcia->db', t_dag_oovv, g_ovoo)
        R1_vo += 2 * self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oo += self._contract('iabc,bcdi->da', t_dag_oovv, g_vvoo)
        R1_vo += 1/2 * self._contract('ia,ba->bi', I_oo, t_vo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_ov += self._contract('iabc,ba->ic', t_dag_oovv, t_vo)
        g_vvoo = ERI[v, v, o, o]
        R1_vo += 1/2 * self._contract('ia,baic->bc', I_ov, g_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_ov += self._contract('iabc,ba->ic', t_dag_oovv, t_vo)
        g_vvoo = ERI[v, v, o, o]
        R1_vo += -1 * self._contract('ia,baci->bc', I_ov, g_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ooov += self._contract('iabc,dbei->edac', t_dag_oovv, g_ovoo)
        R1_vo += -4 * self._contract('iabc,dcab->di', I_ooov, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_ovoo = ERI[o, v, o, o]
        I_ooov += self._contract('iabc,dbei->edac', t_dag_oovv, g_ovoo)
        R1_vo += 2 * self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_vv += self._contract('iabc,dbai->dc', t_dag_oovv, g_vvoo)
        R1_vo += -1 * self._contract('ab,bi->ai', I_vv, t_vo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oo += self._contract('iabc,bcid->da', t_dag_oovv, g_vvoo)
        R1_vo += -1 * self._contract('ia,ba->bi', I_oo, t_vo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvov = ERI[v, v, o, v]
        I_ov += self._contract('iabc,cbid->ad', t_dag_oovv, g_vvov)
        R1_vo += -2 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_oo += self._contract('ia,ab->bi', t_dag_ov, t_vo)
        g_ovoo = ERI[o, v, o, o]
        R1_vo += self._contract('ia,ibca->bc', I_oo, g_ovoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_oo += self._contract('ia,ab->bi', t_dag_ov, t_vo)
        g_ovoo = ERI[o, v, o, o]
        R1_vo += -2 * self._contract('ia,ibac->bc', I_oo, g_ovoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baci->cb', t_dag_ov, t_vvoo)
        f_vv = F[v, v]
        R1_vo += self._contract('ia,ba->bi', I_ov, f_vv)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baci->cb', t_dag_ov, t_vvoo)
        g_ovvo = ERI[o, v, v, o]
        R1_vo += 2 * self._contract('ia,ibac->bc', I_ov, g_ovvo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baci->cb', t_dag_ov, t_vvoo)
        g_ovov = ERI[o, v, o, v]
        R1_vo += -1 * self._contract('ia,ibca->bc', I_ov, g_ovov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baci->cb', t_dag_ov, t_vvoo)
        f_oo = F[o, o]
        R1_vo += -1 * self._contract('ia,ib->ab', I_ov, f_oo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovov = ERI[o, v, o, v]
        I_ovvv += self._contract('ia,bcid->bcda', t_dag_ov, g_ovov)
        R1_vo += self._contract('iabc,bcid->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovov = ERI[o, v, o, v]
        I_ovvv += self._contract('ia,bcid->bcda', t_dag_ov, g_ovov)
        R1_vo += -2 * self._contract('iabc,bcdi->ad', I_ovvv, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_oooo = ERI[o, o, o, o]
        I_ooov += self._contract('ia,bcdi->dbca', t_dag_ov, g_oooo)
        R1_vo += -1 * self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_oooo = ERI[o, o, o, o]
        I_ooov += self._contract('ia,bcid->dbca', t_dag_ov, g_oooo)
        R1_vo += 2 * self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvvv = ERI[v, v, v, v]
        I_ovvv += self._contract('ia,bacd->ibdc', t_dag_ov, g_vvvv)
        R1_vo += -1 * self._contract('iabc,bcdi->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvvv = ERI[v, v, v, v]
        I_ovvv += self._contract('ia,bacd->ibdc', t_dag_ov, g_vvvv)
        R1_vo += 2 * self._contract('iabc,bcid->ad', I_ovvv, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        f_oo = F[o, o]
        I_ov += self._contract('ia,bi->ba', t_dag_ov, f_oo)
        R1_vo += -2 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        f_oo = F[o, o]
        I_ov += self._contract('ia,bi->ba', t_dag_ov, f_oo)
        R1_vo += self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvov = ERI[v, v, o, v]
        I_vv += self._contract('ia,abic->bc', t_dag_ov, g_vvov)
        R1_vo += 2 * self._contract('ab,bi->ai', I_vv, t_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        f_vo = F[v, o]
        I_vv += self._contract('ia,bi->ba', t_dag_ov, f_vo)
        R1_vo += -1/2 * self._contract('ab,bi->ai', I_vv, t_vo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovov = ERI[o, v, o, v]
        I_ooov += self._contract('ia,bacd->cbid', t_dag_ov, g_ovov)
        R1_vo += self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovov = ERI[o, v, o, v]
        I_ooov += self._contract('ia,bacd->cbid', t_dag_ov, g_ovov)
        R1_vo += -2 * self._contract('iabc,dcab->di', I_ooov, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovoo = ERI[o, v, o, o]
        I_oo += self._contract('ia,baic->cb', t_dag_ov, g_ovoo)
        R1_vo += self._contract('ia,ba->bi', I_oo, t_vo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvoo = ERI[v, v, o, o]
        R1_vo += 2 * self._contract('ia,baci->bc', t_dag_ov, g_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baic->cb', t_dag_ov, t_vvoo)
        g_ovov = ERI[o, v, o, v]
        R1_vo += 1/2 * self._contract('ia,ibca->bc', I_ov, g_ovov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baic->cb', t_dag_ov, t_vvoo)
        g_ovvo = ERI[o, v, v, o]
        R1_vo += -1 * self._contract('ia,ibac->bc', I_ov, g_ovvo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baic->cb', t_dag_ov, t_vvoo)
        f_oo = F[o, o]
        R1_vo += 1/2 * self._contract('ia,ib->ab', I_ov, f_oo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baic->cb', t_dag_ov, t_vvoo)
        f_vv = F[v, v]
        R1_vo += -1/2 * self._contract('ia,ba->bi', I_ov, f_vv)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovvo = ERI[o, v, v, o]
        I_ovvv += self._contract('ia,bcdi->bcda', t_dag_ov, g_ovvo)
        R1_vo += -2 * self._contract('iabc,bcid->ad', I_ovvv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovvo = ERI[o, v, v, o]
        I_ovvv += self._contract('ia,bcdi->bcda', t_dag_ov, g_ovvo)
        R1_vo += self._contract('iabc,bcdi->ad', I_ovvv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovoo = ERI[o, v, o, o]
        I_oo += self._contract('ia,baci->cb', t_dag_ov, g_ovoo)
        R1_vo += -2 * self._contract('ia,ba->bi', I_oo, t_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_vv += self._contract('ia,bi->ab', t_dag_ov, t_vo)
        g_vvov = ERI[v, v, o, v]
        R1_vo += -1 * self._contract('ab,acib->ci', I_vv, g_vvov)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_vv += self._contract('ia,bi->ab', t_dag_ov, t_vo)
        g_vvov = ERI[v, v, o, v]
        R1_vo += 2 * self._contract('ab,caib->ci', I_vv, g_vvov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovvo = ERI[o, v, v, o]
        I_ov += self._contract('ia,baci->bc', t_dag_ov, g_ovvo)
        R1_vo += 4 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovvo = ERI[o, v, v, o]
        I_ov += self._contract('ia,baci->bc', t_dag_ov, g_ovvo)
        R1_vo += -2 * self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        f_vv = F[v, v]
        I_ov += self._contract('ia,ab->ib', t_dag_ov, f_vv)
        R1_vo += 2 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        f_vv = F[v, v]
        I_ov += self._contract('ia,ab->ib', t_dag_ov, f_vv)
        R1_vo += -1 * self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        f_vo = F[v, o]
        I_oo += self._contract('ia,ab->bi', t_dag_ov, f_vo)
        R1_vo += -1/2 * self._contract('ia,ba->bi', I_oo, t_vo)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvoo = ERI[v, v, o, o]
        R1_vo += -1 * self._contract('ia,baic->bc', t_dag_ov, g_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovvo = ERI[o, v, v, o]
        I_ooov += self._contract('ia,bacd->dbic', t_dag_ov, g_ovvo)
        R1_vo += self._contract('iabc,dcab->di', I_ooov, t_vvoo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovvo = ERI[o, v, v, o]
        I_ooov += self._contract('ia,bacd->dbic', t_dag_ov, g_ovvo)
        R1_vo += -2 * self._contract('iabc,dcba->di', I_ooov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovov = ERI[o, v, o, v]
        I_ov += self._contract('ia,baic->bc', t_dag_ov, g_ovov)
        R1_vo += -2 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovov = ERI[o, v, o, v]
        I_ov += self._contract('ia,baic->bc', t_dag_ov, g_ovov)
        R1_vo += self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvov = ERI[v, v, o, v]
        I_vv += self._contract('ia,baic->bc', t_dag_ov, g_vvov)
        R1_vo += -1 * self._contract('ab,bi->ai', I_vv, t_vo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oo += self._contract('iabc,bcdi->da', g_oovv, t_vvoo)
        R1_vo += self._contract('ia,ba->bi', I_oo, t_vo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_ov += self._contract('iabc,ca->ib', g_oovv, t_vo)
        R1_vo += 4 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_ov += self._contract('iabc,ca->ib', g_oovv, t_vo)
        R1_vo += -2 * self._contract('ia,baic->bc', I_ov, t_vvoo)
        g_ooov = ERI[o, o, o, v]
        R1_vo += -2 * self._contract('iabc,dcia->db', g_ooov, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_vv += self._contract('iabc,dcai->db', g_oovv, t_vvoo)
        R1_vo += self._contract('ab,bi->ai', I_vv, t_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_vv += self._contract('iabc,dbai->dc', g_oovv, t_vvoo)
        R1_vo += -2 * self._contract('ab,bi->ai', I_vv, t_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        f_ov = F[o, v]
        I_vv += self._contract('ia,bi->ba', f_ov, t_vo)
        R1_vo += -1 * self._contract('ab,bi->ai', I_vv, t_vo)
        f_vv = F[v, v]
        R1_vo += self._contract('ab,bi->ai', f_vv, t_vo)
        g_ovvo = ERI[o, v, v, o]
        R1_vo += 2 * self._contract('iabc,bi->ac', g_ovvo, t_vo)
        g_ovov = ERI[o, v, o, v]
        R1_vo += -1 * self._contract('iabc,ci->ab', g_ovov, t_vo)
        f_ov = F[o, v]
        R1_vo += -1 * self._contract('ia,baic->bc', f_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_ov += self._contract('iabc,ba->ic', g_oovv, t_vo)
        R1_vo += self._contract('ia,baic->bc', I_ov, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_ov += self._contract('iabc,ba->ic', g_oovv, t_vo)
        R1_vo += -2 * self._contract('ia,baci->bc', I_ov, t_vvoo)
        g_ooov = ERI[o, o, o, v]
        R1_vo += self._contract('iabc,dcai->db', g_ooov, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_vv += self._contract('iabc,ci->ab', g_ovvv, t_vo)
        R1_vo += -1 * self._contract('ab,bi->ai', I_vv, t_vo)
        f_oo = F[o, o]
        R1_vo += -1 * self._contract('ia,bi->ba', f_oo, t_vo)
        f_vo = F[v, o]
        R1_vo += self._contract('ai->ai', f_vo)
        f_ov = F[o, v]
        R1_vo += 2 * self._contract('ia,baci->bc', f_ov, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_oo += self._contract('iabc,ci->ba', g_ooov, t_vo)
        R1_vo += self._contract('ia,ba->bi', I_oo, t_vo)
        g_ovvv = ERI[o, v, v, v]
        R1_vo += 2 * self._contract('iabc,bcid->ad', g_ovvv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oo += self._contract('iabc,bcid->da', g_oovv, t_vvoo)
        R1_vo += -2 * self._contract('ia,ba->bi', I_oo, t_vo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_oo += self._contract('iabc,ca->bi', g_ooov, t_vo)
        R1_vo += -2 * self._contract('ia,ba->bi', I_oo, t_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_vv += self._contract('iabc,bi->ac', g_ovvv, t_vo)
        R1_vo += 2 * self._contract('ab,bi->ai', I_vv, t_vo)
        g_ovvv = ERI[o, v, v, v]
        R1_vo += -1 * self._contract('iabc,bcdi->ad', g_ovvv, t_vvoo)

        R2_vvoo = np.zeros((nvirt, nvirt, nocc, nocc), order='F', dtype=complex)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oovv += self._contract('iabc,dcie->eadb', t_dag_oovv, g_vvoo)
        R2_vvoo += self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        g_ovov = ERI[o, v, o, v]
        R2_vvoo += -2 * self._contract('iabc,dcei->daeb', g_ovov, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oovv += self._contract('iabc,dcie->eadb', t_dag_oovv, g_vvoo)
        R2_vvoo += self._contract('iabc,dcae->dbie', I_oovv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,cbad->di', t_dag_oovv, t_vvoo)
        g_vvoo = ERI[v, v, o, o]
        R2_vvoo += -2 * self._contract('ia,bcda->bcdi', I_oo, g_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oo += self._contract('iabc,cbdi->da', t_dag_oovv, g_vvoo)
        R2_vvoo += -2 * self._contract('ia,bcda->bcdi', I_oo, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dcia->db', t_dag_oovv, t_vvoo)
        g_vvoo = ERI[v, v, o, o]
        R2_vvoo += -2 * self._contract('ab,cbid->caid', I_vv, g_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oovv += self._contract('iabc,dcei->eadb', t_dag_oovv, g_vvoo)
        R2_vvoo += self._contract('iabc,dcae->bdie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oovv += self._contract('iabc,dcei->eadb', t_dag_oovv, g_vvoo)
        R2_vvoo += -2 * self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oovv += self._contract('iabc,dbei->eadc', t_dag_oovv, g_vvoo)
        R2_vvoo += 4 * self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oovv += self._contract('iabc,dbei->eadc', t_dag_oovv, g_vvoo)
        R2_vvoo += -2 * self._contract('iabc,dcae->bdie', I_oovv, t_vvoo)
        I_vvvv = np.zeros((nvirt, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_vvvv += self._contract('iabc,deia->debc', t_dag_oovv, g_vvoo)
        R2_vvoo += 1/2 * self._contract('abcd,cdie->abie', I_vvvv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oovv += self._contract('iabc,dbie->eadc', t_dag_oovv, g_vvoo)
        R2_vvoo += self._contract('iabc,dcae->bdie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oovv += self._contract('iabc,dbie->eadc', t_dag_oovv, g_vvoo)
        R2_vvoo += -2 * self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        I_oooo = np.zeros((nocc, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oooo += self._contract('iabc,bcde->deia', t_dag_oovv, g_vvoo)
        R2_vvoo += 1/2 * self._contract('iabc,debc->deia', I_oooo, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vv += self._contract('iabc,dbia->dc', t_dag_oovv, t_vvoo)
        g_vvoo = ERI[v, v, o, o]
        R2_vvoo += self._contract('ab,cbid->caid', I_vv, g_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_oo += self._contract('iabc,bcdi->da', t_dag_oovv, g_vvoo)
        R2_vvoo += self._contract('ia,bcda->bcdi', I_oo, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_vv += self._contract('iabc,dbai->dc', t_dag_oovv, g_vvoo)
        R2_vvoo += -2 * self._contract('ab,cbid->caid', I_vv, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        g_vvoo = ERI[v, v, o, o]
        I_vv += self._contract('iabc,dbia->dc', t_dag_oovv, g_vvoo)
        R2_vvoo += self._contract('ab,cbid->caid', I_vv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,cbda->di', t_dag_oovv, t_vvoo)
        g_vvoo = ERI[v, v, o, o]
        R2_vvoo += self._contract('ia,bcda->bcdi', I_oo, g_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_oovv += self._contract('iabc,bd->diac', g_ovvv, t_vo)
        R2_vvoo += -2 * self._contract('iabc,dcea->dbei', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_oovv += self._contract('iabc,bd->diac', g_ovvv, t_vo)
        R2_vvoo += -2 * self._contract('iabc,dcae->bdei', I_oovv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovoo = ERI[o, v, o, o]
        I_oo += self._contract('ia,baci->cb', t_dag_ov, g_ovoo)
        R2_vvoo += -4 * self._contract('ia,bcda->bcdi', I_oo, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovoo = ERI[o, v, o, o]
        I_oovv += self._contract('ia,bcid->dbca', t_dag_ov, g_ovoo)
        R2_vvoo += -4 * self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovoo = ERI[o, v, o, o]
        I_oovv += self._contract('ia,bcid->dbca', t_dag_ov, g_ovoo)
        R2_vvoo += 2 * self._contract('iabc,dcae->bdie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvov = ERI[v, v, o, v]
        I_oovv += self._contract('ia,bacd->cibd', t_dag_ov, g_vvov)
        R2_vvoo += 4 * self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvov = ERI[v, v, o, v]
        I_oovv += self._contract('ia,bacd->cibd', t_dag_ov, g_vvov)
        R2_vvoo += -2 * self._contract('iabc,dcae->bdie', I_oovv, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baci->cb', t_dag_ov, t_vvoo)
        g_vvov = ERI[v, v, o, v]
        R2_vvoo += 2 * self._contract('ia,bcda->bcdi', I_ov, g_vvov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baci->cb', t_dag_ov, t_vvoo)
        g_ovoo = ERI[o, v, o, o]
        R2_vvoo += -2 * self._contract('ia,ibcd->badc', I_ov, g_ovoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovoo = ERI[o, v, o, o]
        I_oovv += self._contract('ia,bcdi->dbca', t_dag_ov, g_ovoo)
        R2_vvoo += 2 * self._contract('iabc,dcae->dbie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovoo = ERI[o, v, o, o]
        I_oovv += self._contract('ia,bcdi->dbca', t_dag_ov, g_ovoo)
        R2_vvoo += 2 * self._contract('iabc,dcea->dbei', I_oovv, t_vvoo)
        I_vvvv = np.zeros((nvirt, nvirt, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvov = ERI[v, v, o, v]
        I_vvvv += self._contract('ia,bcid->cbda', t_dag_ov, g_vvov)
        R2_vvoo += -2 * self._contract('abcd,cdie->abie', I_vvvv, t_vvoo)
        I_oooo = np.zeros((nocc, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovoo = ERI[o, v, o, o]
        I_oooo += self._contract('ia,bacd->cdbi', t_dag_ov, g_ovoo)
        R2_vvoo += 2 * self._contract('iabc,debc->deia', I_oooo, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvov = ERI[v, v, o, v]
        I_oovv += self._contract('ia,abcd->cibd', t_dag_ov, g_vvov)
        R2_vvoo += -2 * self._contract('iabc,dcea->dbei', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvov = ERI[v, v, o, v]
        I_oovv += self._contract('ia,abcd->cibd', t_dag_ov, g_vvov)
        R2_vvoo += -2 * self._contract('iabc,dcae->bdei', I_oovv, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvov = ERI[v, v, o, v]
        I_vv += self._contract('ia,abic->bc', t_dag_ov, g_vvov)
        R2_vvoo += 4 * self._contract('ab,cbid->caid', I_vv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_oo += self._contract('ia,ab->bi', t_dag_ov, t_vo)
        g_vvoo = ERI[v, v, o, o]
        R2_vvoo += -1 * self._contract('ia,bcda->bcdi', I_oo, g_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        f_vo = F[v, o]
        I_vv += self._contract('ia,bi->ba', t_dag_ov, f_vo)
        R2_vvoo += -1 * self._contract('ab,cbid->caid', I_vv, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_vv += self._contract('ia,bi->ba', t_dag_ov, t_vo)
        g_vvoo = ERI[v, v, o, o]
        R2_vvoo += -1 * self._contract('ab,cbid->caid', I_vv, g_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        f_vo = F[v, o]
        I_oo += self._contract('ia,ab->bi', t_dag_ov, f_vo)
        R2_vvoo += -1 * self._contract('ia,bcda->bcdi', I_oo, t_vvoo)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baic->cb', t_dag_ov, t_vvoo)
        g_vvov = ERI[v, v, o, v]
        R2_vvoo += -1 * self._contract('ia,bcda->bcdi', I_ov, g_vvov)
        I_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_ov += self._contract('ia,baic->cb', t_dag_ov, t_vvoo)
        g_ovoo = ERI[o, v, o, o]
        R2_vvoo += self._contract('ia,ibcd->badc', I_ov, g_ovoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_ovoo = ERI[o, v, o, o]
        I_oo += self._contract('ia,baic->cb', t_dag_ov, g_ovoo)
        R2_vvoo += 2 * self._contract('ia,bcda->bcdi', I_oo, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        g_vvov = ERI[v, v, o, v]
        I_vv += self._contract('ia,baic->bc', t_dag_ov, g_vvov)
        R2_vvoo += -2 * self._contract('ab,cbid->caid', I_vv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oovv += self._contract('iabc,dcei->eadb', g_oovv, t_vvoo)
        R2_vvoo += -2 * self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_vv += self._contract('iabc,dbai->dc', g_oovv, t_vvoo)
        R2_vvoo += -4 * self._contract('ab,cbid->acdi', I_vv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oo += self._contract('iabc,bcid->da', g_oovv, t_vvoo)
        R2_vvoo += -4 * self._contract('ia,bcad->bcid', I_oo, t_vvoo)
        I_vvvv = np.zeros((nvirt, nvirt, nvirt, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_vvvv += self._contract('iabc,di->adbc', g_ovvv, t_vo)
        R2_vvoo += -2 * self._contract('abcd,cdie->abei', I_vvvv, t_vvoo)
        g_vvoo = ERI[v, v, o, o]
        R2_vvoo += self._contract('abic->abic', g_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        g_ovvo = ERI[o, v, v, o]
        I_ovvv += self._contract('iabc,di->cdab', g_ovvo, t_vo)
        R2_vvoo += -2 * self._contract('iabc,cd->abdi', I_ovvv, t_vo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_oovv += self._contract('iabc,da->bidc', g_ooov, t_vo)
        R2_vvoo += 2 * self._contract('iabc,dcae->dbie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_oovv += self._contract('iabc,da->bidc', g_ooov, t_vo)
        R2_vvoo += 2 * self._contract('iabc,dcea->dbei', I_oovv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oo += self._contract('iabc,bcad->di', g_oovv, t_vvoo)
        R2_vvoo += 2 * self._contract('ia,bcad->bcid', I_oo, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oovv += self._contract('iabc,dbei->eadc', g_oovv, t_vvoo)
        R2_vvoo += 4 * self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_vv += self._contract('iabc,dbia->dc', g_oovv, t_vvoo)
        R2_vvoo += 2 * self._contract('ab,cbid->acdi', I_vv, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_oo += self._contract('iabc,ci->ba', g_ooov, t_vo)
        R2_vvoo += 2 * self._contract('ia,bcda->bcdi', I_oo, t_vvoo)
        g_ovoo = ERI[o, v, o, o]
        R2_vvoo += -2 * self._contract('iabc,di->adcb', g_ovoo, t_vo)
        I_vvvv = np.zeros((nvirt, nvirt, nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_vvvv += self._contract('iabc,deia->debc', g_oovv, t_vvoo)
        R2_vvoo += self._contract('abcd,cdie->abie', I_vvvv, t_vvoo)
        g_ovvo = ERI[o, v, v, o]
        R2_vvoo += -2 * self._contract('iabc,dbie->daec', g_ovvo, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_oovv += self._contract('iabc,di->badc', g_ooov, t_vo)
        R2_vvoo += -4 * self._contract('iabc,dcea->dbei', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_oovv += self._contract('iabc,di->badc', g_ooov, t_vo)
        R2_vvoo += 2 * self._contract('iabc,dcae->dbei', I_oovv, t_vvoo)
        f_vv = F[v, v]
        R2_vvoo += 2 * self._contract('ab,cbid->caid', f_vv, t_vvoo)
        g_oooo = ERI[o, o, o, o]
        R2_vvoo += self._contract('iabc,deia->debc', g_oooo, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oovv += self._contract('iabc,dbie->eadc', g_oovv, t_vvoo)
        R2_vvoo += self._contract('iabc,dcae->bdie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oovv += self._contract('iabc,dbie->eadc', g_oovv, t_vvoo)
        R2_vvoo += -4 * self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        g_ovov = ERI[o, v, o, v]
        I_ovvv += self._contract('iabc,di->badc', g_ovov, t_vo)
        R2_vvoo += -2 * self._contract('iabc,cd->abdi', I_ovvv, t_vo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        f_ov = F[o, v]
        I_oo += self._contract('ia,ab->bi', f_ov, t_vo)
        R2_vvoo += -2 * self._contract('ia,bcda->bcdi', I_oo, t_vvoo)
        g_ovov = ERI[o, v, o, v]
        R2_vvoo += -2 * self._contract('iabc,dcie->adeb', g_ovov, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oovv += self._contract('iabc,dcie->eadb', g_oovv, t_vvoo)
        R2_vvoo += 2 * self._contract('iabc,dcea->bdie', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_oovv = ERI[o, o, v, v]
        I_oovv += self._contract('iabc,dbae->eidc', g_oovv, t_vvoo)
        R2_vvoo += self._contract('iabc,dcae->bdei', I_oovv, t_vvoo)
        f_oo = F[o, o]
        R2_vvoo += -2 * self._contract('ia,bcdi->bcda', f_oo, t_vvoo)
        I_ovvv = np.zeros((nocc, nvirt, nvirt, nvirt), order='F', dtype=complex)
        g_vvvv = ERI[v, v, v, v]
        I_ovvv += self._contract('abcd,ci->iabd', g_vvvv, t_vo)
        R2_vvoo += self._contract('iabc,cd->abid', I_ovvv, t_vo)
        I_ooov = np.zeros((nocc, nocc, nocc, nvirt), order='F', dtype=complex)
        g_oooo = ERI[o, o, o, o]
        I_ooov += self._contract('iabc,di->bcad', g_oooo, t_vo)
        R2_vvoo += self._contract('iabc,db->cdia', I_ooov, t_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        f_ov = F[o, v]
        I_vv += self._contract('ia,bi->ba', f_ov, t_vo)
        R2_vvoo += -2 * self._contract('ab,cbid->caid', I_vv, t_vvoo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_vv += self._contract('iabc,bi->ac', g_ovvv, t_vo)
        R2_vvoo += 4 * self._contract('ab,cbid->caid', I_vv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_oovv += self._contract('iabc,cd->diab', g_ovvv, t_vo)
        R2_vvoo += -2 * self._contract('iabc,dcae->dbei', I_oovv, t_vvoo)
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_oovv += self._contract('iabc,cd->diab', g_ovvv, t_vo)
        R2_vvoo += 4 * self._contract('iabc,dcea->dbei', I_oovv, t_vvoo)
        g_vvvv = ERI[v, v, v, v]
        R2_vvoo += self._contract('abcd,cdie->abie', g_vvvv, t_vvoo)
        I_oooo = np.zeros((nocc, nocc, nocc, nocc), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_oooo += self._contract('iabc,cd->bdia', g_ooov, t_vo)
        R2_vvoo += 2 * self._contract('iabc,debc->deia', I_oooo, t_vvoo)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        g_ooov = ERI[o, o, o, v]
        I_oo += self._contract('iabc,ca->bi', g_ooov, t_vo)
        R2_vvoo += -4 * self._contract('ia,bcda->bcdi', I_oo, t_vvoo)
        g_vvov = ERI[v, v, o, v]
        R2_vvoo += 2 * self._contract('abic,cd->abid', g_vvov, t_vo)
        I_vv = np.zeros((nvirt, nvirt), order='F', dtype=complex)
        g_ovvv = ERI[o, v, v, v]
        I_vv += self._contract('iabc,ci->ab', g_ovvv, t_vo)
        R2_vvoo += -2 * self._contract('ab,cbid->caid', I_vv, t_vvoo)
        g_ovvo = ERI[o, v, v, o]
        R2_vvoo += 4 * self._contract('iabc,dbei->daec', g_ovvo, t_vvoo)
        return R1_vo, R2_vvoo

    def compute_onepdm(self, t_vo, t_vvoo, tdag_vo, tdag_vvoo):
        """UCC BCH2 one-particle density matrix (ucc_1rdm_full_cs.py, SeQuant CS,
        unblocked).  Returns the full correlation 1-RDM D_pp (nmo, nmo),
        Hermitian by construction.  Energy: E1 = einsum('pq,pq->', f, D_pp).
        t_dag is a transpose of the pre-conjugated tdag_* inputs (no .conj()).
        """
        nocc = self.no
        nvirt = self.nv
        nmo = nocc + nvirt
        o = slice(0, nocc)
        v = slice(nocc, nmo)
        dim_p = nmo
        p = slice(0, nmo)
        EYE = np.eye(nmo, dtype=complex)
        D_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        I_vo = np.zeros((nvirt, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vo += self._contract('iabc,ci->ba', t_dag_oovv, t_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ai,ap->pi', I_vo, δ_vp)
        δ_po = EYE[p, o]
        D_pp += -1 * self._contract('pi,ai->ap', I_po, δ_po)
        I_vp = np.zeros((nvirt, dim_p), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        I_vp += self._contract('apib,acib->cp', I_vpoo, t_vvoo)
        δ_pv = EYE[p, v]
        D_pp += -2 * self._contract('ap,ba->bp', I_vp, δ_pv)
        I_vp = np.zeros((nvirt, dim_p), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        I_vp += self._contract('apib,acib->cp', I_vpoo, t_vvoo)
        δ_pv = EYE[p, v]
        D_pp += 4 * self._contract('ap,ba->bp', I_vp, δ_pv)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        I_vo = np.zeros((nvirt, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vo += self._contract('iabc,bi->ca', t_dag_oovv, t_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ai,ap->pi', I_vo, δ_vp)
        δ_po = EYE[p, o]
        D_pp += 2 * self._contract('pi,ai->ap', I_po, δ_po)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,bcid->da', t_dag_oovv, t_vvoo)
        δ_op = EYE[o, p]
        I_po += self._contract('ia,ip->pa', I_oo, δ_op)
        δ_po = EYE[p, o]
        D_pp += -4 * self._contract('pi,ai->ap', I_po, δ_po)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oo += self._contract('iabc,bcdi->da', t_dag_oovv, t_vvoo)
        δ_op = EYE[o, p]
        I_po += self._contract('ia,ip->pa', I_oo, δ_op)
        δ_po = EYE[p, o]
        D_pp += 2 * self._contract('pi,ai->ap', I_po, δ_po)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        I_oo = np.zeros((nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_oo += self._contract('ia,ab->bi', t_dag_ov, t_vo)
        δ_op = EYE[o, p]
        I_po += self._contract('ia,ip->pa', I_oo, δ_op)
        δ_po = EYE[p, o]
        D_pp += -2 * self._contract('pi,ai->ap', I_po, δ_po)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        I_vo = np.zeros((nvirt, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_vo += self._contract('ia,abci->bc', t_dag_ov, t_vvoo)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', I_vo, δ_pv)
        δ_op = EYE[o, p]
        D_pp += -1 * self._contract('pi,ia->pa', I_po, δ_op)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        I2_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        D_pp += 2 * self._contract('pi,ai->ap', I_po, I2_po)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        δ_po = EYE[p, o]
        D_pp += 2 * self._contract('pi,ai->ap', I_po, δ_po)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        δ_op = EYE[o, p]
        D_pp += 2 * self._contract('pi,ia->pa', I_po, δ_op)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        I_vo = np.zeros((nvirt, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_vo += self._contract('ia,abic->bc', t_dag_ov, t_vvoo)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', I_vo, δ_pv)
        δ_op = EYE[o, p]
        D_pp += 2 * self._contract('pi,ia->pa', I_po, δ_op)
        return D_pp

    def compute_twopdm(self, t_vo, t_vvoo, tdag_vo, tdag_vvoo):
        """UCC BCH2 two-particle density matrix (ucc_2rdm_full_cs.py, SeQuant CS,
        unblocked).  Returns the full correlation 2-RDM D2_pppp (nmo^4).
        Energy: E2 = 0.5 * einsum('pqrs,pqrs->', <pq|rs>, D2_pppp), so the
        total correlation energy is E1 + E2 with compute_onepdm.
        t_dag is a transpose of the pre-conjugated tdag_* inputs (no .conj()).
        """
        nocc = self.no
        nvirt = self.nv
        nmo = nocc + nvirt
        o = slice(0, nocc)
        v = slice(nocc, nmo)
        dim_p = nmo
        p = slice(0, nmo)
        EYE = np.eye(nmo, dtype=complex)
        D2_pppp = np.zeros((dim_p, dim_p, dim_p, dim_p), order='F', dtype=complex)
        I_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        δ_po = EYE[p, o]
        I_pp += self._contract('pi,ai->pa', I_po, δ_po)
        I2_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        δ_op = EYE[o, p]
        I2_pp += self._contract('pi,ia->ap', I_po, δ_op)
        D2_pppp += 4 * self._contract('pa,bc->cabp', I_pp, I2_pp)
        I_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        δ_po = EYE[p, o]
        I_pp += self._contract('pi,ai->pa', I_po, δ_po)
        I2_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        δ_op = EYE[o, p]
        I2_pp += self._contract('pi,ia->ap', I_po, δ_op)
        D2_pppp += -2 * self._contract('pa,bc->acbp', I_pp, I2_pp)
        I_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        δ_po = EYE[p, o]
        I_pp += self._contract('pi,ai->pa', I_po, δ_po)
        I2_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        δ_op = EYE[o, p]
        I2_pp += self._contract('pi,ia->ap', I_po, δ_op)
        D2_pppp += -2 * self._contract('pa,bc->capb', I_pp, I2_pp)
        I_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        δ_po = EYE[p, o]
        I_pp += self._contract('pi,ai->pa', I_po, δ_po)
        I2_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        δ_op = EYE[o, p]
        I2_pp += self._contract('pi,ia->ap', I_po, δ_op)
        D2_pppp += 4 * self._contract('pa,bc->acpb', I_pp, I2_pp)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        δ_pv = EYE[p, v]
        I_ppoo += self._contract('apib,ca->pcib', I_vpoo, δ_pv)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        D2_pppp += -2 * self._contract('pabi,ci->abcp', I_pppo, I_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        δ_pv = EYE[p, v]
        I_ppoo += self._contract('apib,ca->pcib', I_vpoo, δ_pv)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,bc->cpai', I_ppoo, δ_op)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        D2_pppp += 4 * self._contract('pabi,ci->abcp', I_pppo, I_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        δ_pv = EYE[p, v]
        I_ppoo += self._contract('apib,ca->pcib', I_vpoo, δ_pv)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        D2_pppp += 4 * self._contract('pabi,ci->abpc', I_pppo, I_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        δ_pv = EYE[p, v]
        I_ppoo += self._contract('apib,ca->pcib', I_vpoo, δ_pv)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,bc->cpai', I_ppoo, δ_op)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        D2_pppp += -2 * self._contract('pabi,ci->abpc', I_pppo, I_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        δ_pv = EYE[p, v]
        I_ppoo += self._contract('apib,ca->pcib', I_vpoo, δ_pv)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_op = EYE[o, p]
        D2_pppp += 4 * self._contract('pabi,ic->abpc', I_pppo, δ_op)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        δ_pv = EYE[p, v]
        I_ppoo += self._contract('apib,ca->pcib', I_vpoo, δ_pv)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_op = EYE[o, p]
        D2_pppp += -2 * self._contract('pabi,ic->abcp', I_pppo, δ_op)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        δ_vp = EYE[v, p]
        I_ppoo += self._contract('apib,ac->pcib', I_vpoo, δ_vp)
        I2_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        δ_pv = EYE[p, v]
        I2_ppoo += self._contract('apib,ca->pcib', I_vpoo, δ_pv)
        D2_pppp += 4 * self._contract('paib,cdib->cdpa', I_ppoo, I2_ppoo)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        δ_vp = EYE[v, p]
        I_ppoo += self._contract('apib,ac->cpib', I_vpoo, δ_vp)
        I2_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        δ_pv = EYE[p, v]
        I2_ppoo += self._contract('apib,ca->pcib', I_vpoo, δ_pv)
        D2_pppp += -2 * self._contract('paib,cdib->cdpa', I_ppoo, I2_ppoo)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acid->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acid->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acid->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 8 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acid->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 8 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acdi->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acid->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acid->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acdi->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acid->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acdi->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acdi->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acdi->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acid->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pb->apic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acid->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acdi->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        I2_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I2_vpoo += self._contract('abic,pa->bpic', t_vvoo, δ_pv)
        I_ppoo += self._contract('apib,acdi->pcdb', I_vpoo, I2_vpoo)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        δ_po = EYE[p, o]
        I_pp += self._contract('pi,ai->pa', I_po, δ_po)
        I2_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        δ_po = EYE[p, o]
        I2_pp += self._contract('pi,ai->pa', I_po, δ_po)
        D2_pppp += 4 * self._contract('pa,bc->acpb', I_pp, I2_pp)
        I_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        δ_po = EYE[p, o]
        I_pp += self._contract('pi,ai->pa', I_po, δ_po)
        I2_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        δ_vp = EYE[v, p]
        I_po += self._contract('ia,ap->pi', t_dag_ov, δ_vp)
        δ_po = EYE[p, o]
        I2_pp += self._contract('pi,ai->pa', I_po, δ_po)
        D2_pppp += -2 * self._contract('pa,bc->acbp', I_pp, I2_pp)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_vooo = np.zeros((nvirt, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_vooo += self._contract('ia,abcd->bcdi', t_dag_ov, t_vvoo)
        δ_pv = EYE[p, v]
        I_pooo += self._contract('aibc,pa->pibc', I_vooo, δ_pv)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->cpab', I_pooo, δ_op)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_vooo = np.zeros((nvirt, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_vooo += self._contract('ia,abcd->bcdi', t_dag_ov, t_vvoo)
        δ_pv = EYE[p, v]
        I_pooo += self._contract('aibc,pa->pibc', I_vooo, δ_pv)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->cpab', I_pooo, δ_op)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->cpab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_vooo = np.zeros((nvirt, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_vooo += self._contract('ia,abcd->bcdi', t_dag_ov, t_vvoo)
        δ_pv = EYE[p, v]
        I_pooo += self._contract('aibc,pa->pibc', I_vooo, δ_pv)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->cpab', I_pooo, δ_op)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->cbpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_vooo = np.zeros((nvirt, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_ov = np.zeros((nocc, nvirt), order='F', dtype=complex)
        t_dag_ov += self._contract('ai->ia', tdag_vo)
        I_vooo += self._contract('ia,bacd->bcdi', t_dag_ov, t_vvoo)
        δ_pv = EYE[p, v]
        I_pooo += self._contract('aibc,pa->pibc', I_vooo, δ_pv)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->cpab', I_pooo, δ_op)
        δ_op = EYE[o, p]
        I_pppo += self._contract('paib,ic->pcab', I_ppoo, δ_op)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        δ_op = EYE[o, p]
        I_pp += self._contract('pi,ia->ap', I_po, δ_op)
        I2_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        δ_op = EYE[o, p]
        I2_pp += self._contract('pi,ia->ap', I_po, δ_op)
        D2_pppp += 4 * self._contract('pa,bc->acpb', I_pp, I2_pp)
        I_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        δ_op = EYE[o, p]
        I_pp += self._contract('pi,ia->ap', I_po, δ_op)
        I2_pp = np.zeros((dim_p, dim_p), order='F', dtype=complex)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        δ_op = EYE[o, p]
        I2_pp += self._contract('pi,ia->ap', I_po, δ_op)
        D2_pppp += -2 * self._contract('pa,bc->acbp', I_pp, I2_pp)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,cp->bpia', t_dag_oovv, δ_vp)
        δ_vp = EYE[v, p]
        I_ppoo += self._contract('apib,ac->pcib', I_vpoo, δ_vp)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,cb->paci', I_ppoo, δ_po)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        D2_pppp += 4 * self._contract('pabi,ci->bcpa', I_pppo, I_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        δ_vp = EYE[v, p]
        I_ppoo += self._contract('apib,ac->pcib', I_vpoo, δ_vp)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,cb->paci', I_ppoo, δ_po)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        D2_pppp += -2 * self._contract('pabi,ci->bcpa', I_pppo, I_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        δ_vp = EYE[v, p]
        I_ppoo += self._contract('apib,ac->pcib', I_vpoo, δ_vp)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, I_po)
        δ_po = EYE[p, o]
        D2_pppp += 4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        δ_vp = EYE[v, p]
        I_ppoo += self._contract('apib,ac->cpib', I_vpoo, δ_vp)
        I_po = np.zeros((dim_p, nocc), order='F', dtype=complex)
        δ_pv = EYE[p, v]
        I_po += self._contract('ai,pa->pi', t_vo, δ_pv)
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, I_po)
        δ_po = EYE[p, o]
        D2_pppp += -2 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        δ_vp = EYE[v, p]
        I_ppoo += self._contract('apib,ac->cpib', I_vpoo, δ_vp)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, δ_po)
        δ_po = EYE[p, o]
        D2_pppp += -2 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_vpoo = np.zeros((nvirt, dim_p, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        δ_vp = EYE[v, p]
        I_vpoo += self._contract('iabc,bp->cpia', t_dag_oovv, δ_vp)
        δ_vp = EYE[v, p]
        I_ppoo += self._contract('apib,ac->pcib', I_vpoo, δ_vp)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, δ_po)
        δ_po = EYE[p, o]
        D2_pppp += 4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_oooo = np.zeros((nocc, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oooo += self._contract('iabc,bcde->deia', t_dag_oovv, t_vvoo)
        δ_op = EYE[o, p]
        I_pooo += self._contract('iabc,ip->pabc', I_oooo, δ_op)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->cpab', I_pooo, δ_op)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, δ_po)
        δ_po = EYE[p, o]
        D2_pppp += -2 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_oooo = np.zeros((nocc, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_oooo += self._contract('iabc,bcde->deia', t_dag_oovv, t_vvoo)
        δ_op = EYE[o, p]
        I_pooo += self._contract('iabc,ip->pabc', I_oooo, δ_op)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->pcab', I_pooo, δ_op)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, δ_po)
        δ_po = EYE[p, o]
        D2_pppp += 4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_vooo = np.zeros((nvirt, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vooo += self._contract('iabc,cd->bdia', t_dag_oovv, t_vo)
        δ_vp = EYE[v, p]
        I_pooo += self._contract('aibc,ap->pibc', I_vooo, δ_vp)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->cpab', I_pooo, δ_op)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, δ_po)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_vooo = np.zeros((nvirt, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vooo += self._contract('iabc,bd->cdia', t_dag_oovv, t_vo)
        δ_vp = EYE[v, p]
        I_pooo += self._contract('aibc,ap->pibc', I_vooo, δ_vp)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->pcab', I_pooo, δ_op)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, δ_po)
        δ_po = EYE[p, o]
        D2_pppp += 2 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_vooo = np.zeros((nvirt, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vooo += self._contract('iabc,bd->cdia', t_dag_oovv, t_vo)
        δ_vp = EYE[v, p]
        I_pooo += self._contract('aibc,ap->pibc', I_vooo, δ_vp)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->cpab', I_pooo, δ_op)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, δ_po)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        I_pppo = np.zeros((dim_p, dim_p, dim_p, nocc), order='F', dtype=complex)
        I_ppoo = np.zeros((dim_p, dim_p, nocc, nocc), order='F', dtype=complex)
        I_pooo = np.zeros((dim_p, nocc, nocc, nocc), order='F', dtype=complex)
        I_vooo = np.zeros((nvirt, nocc, nocc, nocc), order='F', dtype=complex)
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F', dtype=complex)
        t_dag_oovv += self._contract('abic->icab', tdag_vvoo)
        I_vooo += self._contract('iabc,cd->bdia', t_dag_oovv, t_vo)
        δ_vp = EYE[v, p]
        I_pooo += self._contract('aibc,ap->pibc', I_vooo, δ_vp)
        δ_op = EYE[o, p]
        I_ppoo += self._contract('piab,ic->pcab', I_pooo, δ_op)
        δ_po = EYE[p, o]
        I_pppo += self._contract('paib,ci->pacb', I_ppoo, δ_po)
        δ_po = EYE[p, o]
        D2_pppp += -4 * self._contract('pabi,ci->bcpa', I_pppo, δ_po)
        return D2_pppp


def make_ucc_fns(ccwfn, solve=True, e_conv=1e-12, r_conv=1e-8):
    ucc = UCCWfn(ccwfn)
    if solve:
        ucc.solve_ucc(e_conv=e_conv, r_conv=r_conv)
    return ucc, ucc.energy, ucc.residuals


