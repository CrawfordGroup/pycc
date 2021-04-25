# We will assume that the ccwfn and cclambda objects already
# contain the t=0 amplitudes we need for the initial step

import numpy as np
from .ODE.helper_method import *

class rtcc(object):
    def __init__(self, ccwfn, cclambda, field, t0, tf, h, method):
        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.field = field
        self.method = method

        self.RK = RK(self.f, y0, t0, tf, h, method)
        self.y0 = concatenate((ccwfn.t1, ccwfn.t2, cclambda.l1, cclambda.l2), axis=None)

    def f(self, t, y):
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccefn.nv

        # extract the amplitudes
        len1 = no*nv
        len2 = no*no*nv*nv
        t1 = np.reshape(y[:len1], (no, nv))
        t2 = np.reshape(y[len1:(len1+len2], (no, no, nv, nv))
        l1 = np.reshape(y[(len1+len2):(len1+len2+len1)], (no, nv))
        l2 = np.reshape(y[(len1+len2+len1):], (no, no, nv, nv))

