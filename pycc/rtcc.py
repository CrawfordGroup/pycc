# We will assume that the ccwfn and cclambda objects already
# contain the t=0 amplitudes we need for the initial step

import psi4
import numpy as np
from .cc_eqs import build_tau
from .density_eqs import build_Dov, build_Dvo, build_Dvv, build_Doo
from .density_eqs import build_Doooo, build_Dvvvv, build_Dooov, build_Dvvvo
from .density_eqs import build_Dovov, build_Doovv
from opt_einsum import contract

class rtcc(object):
    def __init__(self, ccwfn, cclambda, ccdensity, V, axis):
        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.ccdensity = ccdensity
        self.V = V
        self.axis = axis

        # Prep the dipole integrals in MO basis
        mints = psi4.core.MintsHelper(ccwfn.ref.basisset())
        dipole_ints = mints.ao_dipole()
        C = np.asarray(ccwfn.ref.Ca_subset("AO", "ACTIVE"))
        self.mu = C.T @ np.asarray(dipole_ints[axis]) @ C

    def f(self, t, y):
        # Extract amplitude tensors
        t1, t2, l1, l2 = self.extract_amps(y)

        # Add the field to the Hamiltonian
        F = self.ccwfn.F.copy() + (self.mu * self.V(t))/np.sqrt(3.0)

        # Compute the current residuals
        rt1, rt2 = self.ccwfn.residuals(F, t1, t2)
        rt1 = rt1 * (-1.0j)
        rt2 = rt2 * (-1.0j)

        rl1, rl2 = self.cclambda.residuals(F, t1, t2, l1, l2)
        rl1 = rl1 * (+1.0j)
        rl2 = rl2 * (+1.0j)

        # Pack up the residuals
        y = self.collect_amps(rt1, rt2, rl1, rl2)

        return y

    def collect_amps(self, t1, t2, l1, l2):
        return np.concatenate((t1, t2, l1, l2), axis=None)

    def extract_amps(self, y):
        no = self.ccwfn.no
        nv = self.ccwfn.nv

        # Extract the amplitudes
        len1 = no*nv
        len2 = no*no*nv*nv
        t1 = np.reshape(y[:len1], (no, nv))
        t2 = np.reshape(y[len1:(len1+len2)], (no, no, nv, nv))
        l1 = np.reshape(y[(len1+len2):(len1+len2+len1)], (no, nv))
        l2 = np.reshape(y[(len1+len2+len1):], (no, no, nv, nv))

        return t1, t2, l1, l2

    def dipole(self, t1, t2, l1, l2):
        opdm = self.ccdensity.compute_onepdm(t1, t2, l1, l2)
        return self.mu.flatten().dot(opdm.flatten())

    def energy(self, t, t1, t2, l1, l2):
        o = self.ccwfn.o
        v = self.ccwfn.v
        F = self.ccwfn.F.copy() + self.mu * self.V(t)
        ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
        L = self.ccwfn.L
        ecc = ecc + contract('ijab,ijab->', build_tau(t1, t2), L[o,o,v,v])
        return ecc

    def lagrangian(self, t, t1, t2, l1, l2):
        o = self.ccwfn.o
        v = self.ccwfn.v
        ERI = self.ccwfn.ERI
        opdm = self.ccdensity.compute_onepdm(t1, t2, l1, l2)
        Doooo = build_Doooo(t1, t2, l2)
        Dvvvv = build_Dvvvv(t1, t2, l2)
        Dooov = build_Dooov(t1, t2, l1, l2)
        Dvvvo = build_Dvvvo(t1, t2, l1, l2)
        Dovov = build_Dovov(t1, t2, l1, l2)
        Doovv = build_Doovv(t1, t2, l1, l2)

        F = self.ccwfn.F.copy() + self.mu * self.V(t)
        eone = F.flatten().dot(opdm.flatten())
        oooo_energy = 0.5 * contract('ijkl,ijkl->', ERI[o,o,o,o], Doooo)
        vvvv_energy = 0.5 * contract('abcd,abcd->', ERI[v,v,v,v], Dvvvv)
        ooov_energy = contract('ijka,ijka->', ERI[o,o,o,v], Dooov)
        vvvo_energy = contract('abci,abci->', ERI[v,v,v,o], Dvvvo)
        ovov_energy = contract('iajb,iajb->', ERI[o,v,o,v], Dovov)
        oovv_energy = 0.5 * contract('ijab,ijab->', ERI[o,o,v,v], Doovv)
        etwo = oooo_energy + vvvv_energy + ooov_energy + vvvo_energy + ovov_energy + oovv_energy

        return eone + etwo
