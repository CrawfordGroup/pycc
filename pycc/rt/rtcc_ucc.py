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


    def dipole(self, t1, t2):
        """Placeholder. Needs Ajay's ucc_onepdm(). Same contraction as rtcc."""
        raise NotImplementedError(
            "Implement after Ajay provides ucc_onepdm(): "
            "opdm = ucc_onepdm(t_vo, t_vvoo, tdag_vo, tdag_vvoo); "
            "return mu[i].flatten() @ opdm.flatten() for i in 0,1,2."
        )
