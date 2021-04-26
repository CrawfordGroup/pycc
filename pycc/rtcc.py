# We will assume that the ccwfn and cclambda objects already
# contain the t=0 amplitudes we need for the initial step

import psi4
import numpy as np
from .ode import *

class rtcc(object):
    def __init__(self, ccwfn, cclambda, V, axis, t0, tf, h, method):
        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.V = V
        self.axis = axis
        self.method = method

        # Set up propagator for T and Lambda amps
        y0 = np.concatenate((ccwfn.t1, ccwfn.t2, cclambda.l1, cclambda.l2), axis=None)
        self.RK = RK(self.f, y0, t0, tf, h, method)

        # Prep the dipole integrals in MO basis
        mints = psi4.core.MintsHelper(ccwfn.ref.basisset())
        dipole_ints = mints.ao_dipole()
        C = np.asarray(ccwfn.ref.Ca_subset("AO", "ACTIVE"))
        self.mu = C.T @ np.asarray(dipole_ints[axis]) @ C

    def f(self, t, y):
        print("Starting f(): dtype of y is", y.dtype)

        # Extract amplitude tensors
        t1, t2, l1, l2 = self.extract_amps(y)

        # Add the field to the Hamiltonian
        F = self.ccwfn.F.copy() + self.mu * self.V(t)

        # Compute the current residuals
        rt1, rt2 = self.ccwfn.residuals(F, t1, t2)
        rt1 = rt1 * (-1.0j)
        rt2 = rt2 * (-1.0j)

        rl1, rl2 = self.cclambda.residuals(F, t1, t2, l1, l2)
        rl1 = rl1 * (+1.0j)
        rl2 = rl2 * (+1.0j)

        # Pack up the residuals
        y = np.concatenate((rt1, rt2, rl1, rl2), axis=None)

        print("Leaving f(): dtype of y is", y.dtype)
        return y

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
