"""
Real-time UCCSD object for ODE propagation. Lives in PyCC alongside rtcc.py.
Uses Ajay's SeQuant-generated equations.
"""

import numpy as np
import opt_einsum


class rtcc_ucc:
    """
    Real-time UCC object. Mirrors pycc.rtcc interface.

    Parameters
    ----------
    ccwfn        : PyCC ccwfn object  (.no, .nv, .H)
    V            : callable           laser field V(t) -> float
    energy_fn    : callable           
    residuals_fn : callable           
    kick         : 'x','y','z' or None  (None = isotropic)
    """

    def __init__(self, ccwfn, V, energy_fn, residuals_fn, kick=None):
        self.ccwfn         = ccwfn
        self.V             = V
        self.no            = ccwfn.no
        self.nv            = ccwfn.nv
        self.contract      = opt_einsum.contract
        self._energy_fn    = energy_fn
        self._residuals_fn = residuals_fn

        self.mu = ccwfn.H.mu
        if kick:
            self.mu_tot = self.mu[{"x": 0, "y": 1, "z": 2}[kick.lower()]]
        else:
            self.mu_tot = sum(self.mu) / np.sqrt(3.0)

    # ------------------------------------------------------------------
    # Index-convention adapters
    # ------------------------------------------------------------------

    def _to_ajay(self, t1, t2):
        """PyCC (no,nv),(no,no,nv,nv)  ->  Ajay (nv,no),(nv,nv,no,no)."""
        return t1.T, t2.transpose(2, 3, 0, 1)

    def _from_ajay(self, R1_vo, R2_vvoo):
        """Ajay residuals -> PyCC convention + symmetrize R2."""
        r1 = R1_vo.T
        r2 = R2_vvoo.transpose(2, 3, 0, 1)
        r2 = 0.5 * (r2 + r2.transpose(1, 0, 3, 2))
        return r1, r2

    def _prep_tdag(self, t_vo, t_vvoo):
        """
        Pre-conjugate so Ajay's transpose-only t_dag gives correct T†
        for complex RT amplitudes. 
        """
        return t_vo.conj(), t_vvoo.conj()

    def _build_F(self, t):
        return self.ccwfn.H.F.copy() + self.mu_tot * self.V(t)

    # ------------------------------------------------------------------
    # Amplitude packing / unpacking
    # ------------------------------------------------------------------

    def collect_amps(self, t1, t2, phase):
        """Pack t1, t2, phase -> flat complex128 vector."""
        return np.concatenate(
            (t1.ravel(), t2.ravel(), np.array([phase]))
        ).astype("complex128")

    def extract_amps(self, y):
        """Unpack flat vector -> t1 (no,nv), t2 (no,no,nv,nv), phase."""
        no, nv = self.no, self.nv
        n1 = no * nv
        n2 = no * no * nv * nv
        return (y[:n1].reshape(no, nv),
                y[n1:n1 + n2].reshape(no, no, nv, nv),
                y[-1])

    # ------------------------------------------------------------------
    # Energy  (replaces lagrangian)
    # ------------------------------------------------------------------

    def energy(self, t, t1, t2):
        """UCC BCH energy <Phi_0| exp(-A) H(t) exp(A) |Phi_0>."""
        F = self._build_F(t)
        t_vo,    t_vvoo    = self._to_ajay(t1, t2)
        tdag_vo, tdag_vvoo = self._prep_tdag(t_vo, t_vvoo)
        return self._energy_fn(F, self.ccwfn.H.ERI,
                               t_vo, t_vvoo, tdag_vo, tdag_vvoo)

    # ------------------------------------------------------------------
    # ODE RHS
    # ------------------------------------------------------------------

    def f(self, t, y):
        """dy/dt = f(t,y). Returns flat [rt1 | rt2 | dphase]."""
        t1, t2, phase = self.extract_amps(y)
        F = self._build_F(t)
        t_vo,    t_vvoo    = self._to_ajay(t1, t2)
        tdag_vo, tdag_vvoo = self._prep_tdag(t_vo, t_vvoo)

        R1_vo, R2_vvoo = self._residuals_fn(F, self.ccwfn.H.ERI,
                                             t_vo, t_vvoo, tdag_vo, tdag_vvoo)
        rt1, rt2 = self._from_ajay(R1_vo, R2_vvoo)
        rt1 *= -1.0j
        rt2 *= -1.0j
        dphase = -1.0j * self.energy(t, t1, t2)
        return self.collect_amps(rt1, rt2, dphase)

    # ------------------------------------------------------------------
    # Autocorrelation  C(t) = <Psi(0)|Psi(t)>
    # ------------------------------------------------------------------

    def autocorrelation(self, y0, yt):
        """
        UCC ACF expanded to doubles level (five terms).
        |C(t)|^2 <= 1 by unitarity — no symmetrization needed.

        Terms:
          1                                              constant
          + t1*(0).t1(t)                                S-S
          + 0.25 * t2*(0).t2(t)                         D-D
          + 0.5 * [t1*(0).t1(t)]^2                      S^2-S^2
          + 0.5 * t2*(0)_ijab t1(t)_ia t1(t)_jb        D(0) x S^2(t)
          + 0.5 * t1*(0)_ia t1*(0)_jb t2(t)_ijab       S^2(0) x D(t)

        get SeQuant equations from Ajay and update
        """
        """
        UCC ACF C(t) = <Psi(0)|Psi(t)> at BCH2 level.
        SeQuant-generated equations from Ajay (uccsd_autocorr_k2_cs.py),
        converted from file-based to in-memory.

        y0 : packed amplitudes at t=0
        yt : packed amplitudes at time t
        """
        
        no, nv = self.no, self.nv
        nocc, nvirt = no, nv

        # Unpack amplitudes
        t1_0, t2_0, ph_0 = self.extract_amps(y0)
        t1_t, t2_t, ph_t = self.extract_amps(yt)

        # Convert to Ajay convention and conjugate t=0 amplitudes
        tp_vo_0   = t1_0.T.conj()                      # (nv, no) conj
        tp_vvoo_0 = t2_0.transpose(2,3,0,1).conj()     # (nv, nv, no, no) conj
        t_vo_t    = t1_t.T                              # (nv, no)
        t_vvoo_t  = t2_t.transpose(2,3,0,1)            # (nv, nv, no, no)



        # ======================================================================
        # Autocorrelation
        # ======================================================================

        C = 0.0
        I_ov = np.zeros((nocc, nvirt), order='F')
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        I_oovv += np.einsum('ai,bc->icab', tp_vo, tp_vo, optimize=True)
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_ov += np.einsum('iabc,ic->ab', I_oovv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_oovv
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        C += -1 * np.einsum('ia,ia->', I_ov, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ov
        I_ov = np.zeros((nocc, nvirt), order='F')
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        tp_vvoo = tp_vvoo_0.copy()
        I_ov += np.einsum('ia,abci->cb', tp_dag_ov, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_dag_ov
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        C += np.einsum('ia,ia->', I_ov, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ov
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iceg->abdf', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -4 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        t_vo = t_vo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', t_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vo
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        I_oovv += np.einsum('iabcde,bd->iace', I_ooovvv, tp_dag_ov, optimize=True)
        del tp_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 1/2 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iced->abfg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 4 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        t_vvoo = t_vvoo_t.copy()
        C += -1 * np.einsum('iabc,bcia->', t_dag_oovv, t_vvoo, optimize=True)
        del t_vvoo
        del t_dag_oovv
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        t_vo = t_vo_t.copy()
        C += -1 * np.einsum('ia,ai->', t_dag_ov, t_vo, optimize=True)
        del t_vo
        del t_dag_ov
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,ad->ibce', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -4 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vo = t_vo_t.copy()
        tp_vo = tp_vo_0.copy()
        I_oovv += np.einsum('ai,bc->icab', t_vo, tp_vo, optimize=True)
        del tp_vo
        del t_vo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -2 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,ibdf->aceg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 4 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', t_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vvoo
        tp_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        tp_dag_oovv += np.einsum('abic->icab', tp_vvoo, optimize=True)
        del tp_vvoo
        I_oovv += np.einsum('iabcdefg,bcfg->iade', I_oooovvvv, tp_dag_oovv, optimize=True)
        del tp_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1/2 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        I_oovv += np.einsum('ai,bc->icab', tp_vo, tp_vo, optimize=True)
        del tp_vo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 2 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        C += 1
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,id->abce', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 2 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,bd->iace', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -4 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_ov = np.zeros((nocc, nvirt), order='F')
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        tp_vvoo = tp_vvoo_0.copy()
        I_ov += np.einsum('ia,abic->cb', t_dag_ov, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_dag_ov
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        C += 2 * np.einsum('ia,ia->', I_ov, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ov
        I_ov = np.zeros((nocc, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        t_vo = t_vo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', t_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vo
        tp_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        tp_dag_oovv += np.einsum('abic->icab', tp_vvoo, optimize=True)
        del tp_vvoo
        I_ov += np.einsum('iabcde,abed->ic', I_ooovvv, tp_dag_oovv, optimize=True)
        del tp_dag_oovv
        del I_ooovvv
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        C += -1/2 * np.einsum('ia,ia->', I_ov, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ov
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        tp_vo = tp_vo_0.copy()
        C += -1 * np.einsum('ia,ai->', tp_dag_ov, tp_vo, optimize=True)
        del tp_vo
        del tp_dag_ov
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iade->bcfg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 2 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,ibeg->acdf', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_ov = np.zeros((nocc, nvirt), order='F')
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        I_oovv += np.einsum('ai,bc->icab', tp_vo, tp_vo, optimize=True)
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_ov += np.einsum('iabc,ib->ac', I_oovv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_oovv
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        C += 2 * np.einsum('ia,ia->', I_ov, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ov
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        t_vo = t_vo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', t_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vo
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        I_oovv += np.einsum('iabcde,ad->ibce', I_ooovvv, tp_dag_ov, optimize=True)
        del tp_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vo = t_vo_t.copy()
        tp_vo = tp_vo_0.copy()
        I_oovv += np.einsum('ai,bc->icab', t_vo, tp_vo, optimize=True)
        del tp_vo
        del t_vo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        t_vvoo = t_vvoo_t.copy()
        C += 1/2 * np.einsum('iabc,cbia->', t_dag_oovv, t_vvoo, optimize=True)
        del t_vvoo
        del t_dag_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        tp_vo = tp_vo_0.copy()
        I_ooovvv += np.einsum('abic,de->iceabd', t_vvoo, tp_vo, optimize=True)
        del tp_vo
        del t_vvoo
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        I_oovv += np.einsum('iabcde,be->iacd', I_ooovvv, tp_dag_ov, optimize=True)
        del tp_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1/2 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iaef->bcdg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 4 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_ov = np.zeros((nocc, nvirt), order='F')
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        tp_vvoo = tp_vvoo_0.copy()
        I_ov += np.einsum('ia,baci->cb', tp_dag_ov, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_dag_ov
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        C += -2 * np.einsum('ia,ia->', I_ov, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ov
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iagf->bcde', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1/2 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        tp_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        tp_dag_oovv += np.einsum('abic->icab', tp_vvoo, optimize=True)
        del tp_vvoo
        tp_vvoo = tp_vvoo_0.copy()
        C += 1/2 * np.einsum('iabc,cbia->', tp_dag_oovv, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_dag_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iade->bcfg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -2 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        t_vo = t_vo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', t_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vo
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        I_oovv += np.einsum('iabcde,be->iacd', I_ooovvv, tp_dag_ov, optimize=True)
        del tp_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 2 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,ibfd->aceg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -4 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        I_oovv += np.einsum('ai,bc->icab', tp_vo, tp_vo, optimize=True)
        del tp_vo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,ibgf->acde', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -4 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,ibgd->acef', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', t_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vvoo
        tp_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        tp_dag_oovv += np.einsum('abic->icab', tp_vvoo, optimize=True)
        del tp_vvoo
        I_oovv += np.einsum('iabcdefg,bcgf->iade', I_oooovvvv, tp_dag_oovv, optimize=True)
        del tp_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 1/4 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        t_vo = t_vo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', t_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vo
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        I_oovv += np.einsum('iabcde,bd->iace', I_ooovvv, tp_dag_ov, optimize=True)
        del tp_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,ae->ibcd', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 2 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,ic->abde', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 4 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        tp_vvoo = tp_vvoo_0.copy()
        C += -1 * np.einsum('iabc,cbia->', t_dag_oovv, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_dag_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iceg->abdf', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 4 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iafg->bcde', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,be->iacd', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 8 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,ie->abcd', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -4 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iadf->bceg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -4 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', t_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vvoo
        tp_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        tp_dag_oovv += np.einsum('abic->icab', tp_vvoo, optimize=True)
        del tp_vvoo
        I_oovv += np.einsum('iabcdefg,bcfg->iade', I_oooovvvv, tp_dag_oovv, optimize=True)
        del tp_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_ov = np.zeros((nocc, nvirt), order='F')
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        tp_vvoo = tp_vvoo_0.copy()
        I_ov += np.einsum('ia,baic->cb', t_dag_ov, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_dag_ov
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        C += -1 * np.einsum('ia,ia->', I_ov, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ov
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', t_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vvoo
        tp_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        tp_dag_oovv += np.einsum('abic->icab', tp_vvoo, optimize=True)
        del tp_vvoo
        I_oovv += np.einsum('iabcdefg,bcgf->iade', I_oooovvvv, tp_dag_oovv, optimize=True)
        del tp_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1/2 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_ov = np.zeros((nocc, nvirt), order='F')
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vo = t_vo_t.copy()
        tp_vo = tp_vo_0.copy()
        I_oovv += np.einsum('ai,bc->icab', t_vo, tp_vo, optimize=True)
        del tp_vo
        del t_vo
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        I_ov += np.einsum('iabc,ac->ib', I_oovv, tp_dag_ov, optimize=True)
        del tp_dag_ov
        del I_oovv
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        C += np.einsum('ia,ia->', I_ov, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ov
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,ibfd->aceg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        tp_vo = tp_vo_0.copy()
        I_ooovvv += np.einsum('abic,de->iceabd', t_vvoo, tp_vo, optimize=True)
        del tp_vo
        del t_vvoo
        tp_dag_ov = np.zeros((nocc, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_dag_ov += np.einsum('ai->ia', tp_vo, optimize=True)
        del tp_vo
        I_oovv += np.einsum('iabcde,be->iacd', I_ooovvv, tp_dag_ov, optimize=True)
        del tp_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        tp_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        tp_dag_oovv += np.einsum('abic->icab', tp_vvoo, optimize=True)
        del tp_vvoo
        tp_vvoo = tp_vvoo_0.copy()
        C += -1 * np.einsum('iabc,bcia->', tp_dag_oovv, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_dag_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,bc->iade', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -4 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_ov = np.zeros((nocc, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        t_vo = t_vo_t.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', t_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_vo
        tp_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        tp_dag_oovv += np.einsum('abic->icab', tp_vvoo, optimize=True)
        del tp_vvoo
        I_ov += np.einsum('iabcde,abde->ic', I_ooovvv, tp_dag_oovv, optimize=True)
        del tp_dag_oovv
        del I_ooovvv
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        C += np.einsum('ia,ia->', I_ov, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ov
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,ibeg->acdf', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iaeg->bcdf', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iced->abfg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -1 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,ic->abde', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += -2 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,icfe->abdg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_oooovvvv = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt), order='F')
        tp_vvoo = tp_vvoo_0.copy()
        I_oooovvvv += np.einsum('abic,defg->icfgabde', tp_vvoo, tp_vvoo, optimize=True)
        del tp_vvoo
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        I_oovv += np.einsum('iabcdefg,iaed->bcfg', I_oooovvvv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oooovvvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 1/2 * np.einsum('iabc,iacb->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        I_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        I_ooovvv = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt), order='F')
        tp_vo = tp_vo_0.copy()
        tp_vvoo = tp_vvoo_0.copy()
        I_ooovvv += np.einsum('ai,bcde->ideabc', tp_vo, tp_vvoo, optimize=True)
        del tp_vvoo
        del tp_vo
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        I_oovv += np.einsum('iabcde,bc->iade', I_ooovvv, t_dag_ov, optimize=True)
        del t_dag_ov
        del I_ooovvv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        C += 2 * np.einsum('iabc,iabc->', I_oovv, t_dag_oovv, optimize=True)
        del t_dag_oovv
        del I_oovv
        t_dag_oovv = np.zeros((nocc, nocc, nvirt, nvirt), order='F')
        t_vvoo = t_vvoo_t.copy()
        t_dag_oovv += np.einsum('abic->icab', t_vvoo, optimize=True)
        del t_vvoo
        tp_vvoo = tp_vvoo_0.copy()
        C += 2 * np.einsum('iabc,bcia->', t_dag_oovv, tp_vvoo, optimize=True)
        del tp_vvoo
        del t_dag_oovv
        t_dag_ov = np.zeros((nocc, nvirt), order='F')
        t_vo = t_vo_t.copy()
        t_dag_ov += np.einsum('ai->ia', t_vo, optimize=True)
        del t_vo
        tp_vo = tp_vo_0.copy()
        C += 2 * np.einsum('ia,ai->', t_dag_ov, tp_vo, optimize=True)
        del tp_vo
        del t_dag_ov
        return C * np.exp(-ph_0) * np.exp(ph_t)


    def dipole(self, t1, t2):
        """Placeholder. Needs Ajay's ucc_onepdm(). Same contraction as rtcc."""
        raise NotImplementedError(
            "Implement after Ajay provides ucc_onepdm(): "
            "opdm = ucc_onepdm(t_vo, t_vvoo, tdag_vo, tdag_vvoo); "
            "return mu[i].flatten() @ opdm.flatten() for i in 0,1,2."
        )
