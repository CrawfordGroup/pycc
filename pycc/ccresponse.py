"""
ccresponse.py: CC Response Functions
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np


class ccresponse(object):
    """
    An RHF-CC Response Property Object.

    Methods
    -------
    linresp():
        Compute a CC linear response function.
    """

    def __init__(self, ccdensity, 
            omega1 = 0,
            omega2 = 0):
        """
        Parameters
        ----------
        ccdensity : PyCC ccdensity object
            Contains all components of the CC one- and two-electron densities, as well as references to the underlying ccwfn, cchbar, and cclambda objects
        omega1 : scalar
            The first external field frequency (for linear and quadratic response functions)
        omega2 : scalar
            The second external field frequency (for quadratic response functions)

        Returns
        -------
        None
        """

        self.ccwfn = ccdensity.ccwfn
        self.cclambda = ccdensity.cclambda
        self.H = self.ccwfn.H
        self.hbar = self.cclambda.hbar
        self.contract = self.ccwfn.contract

        # Generate similarity-transformed property integrals
        self.mubar = []
        for axis in range(3):
            self.mubar.append(pertbar(self.H.mu[axis]))

        self.mbar = []
        for axis in range(3):
            self.mbar.append(pertbar(self.H.m[axis]))


class pertbar(object):
    def __init__(self, pert):
        o = self.ccwfn.o
        v = self.ccwfn.v
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2
        contract = self.ccwfn.contract

        self.Aov = pert[o,v].copy()

        self.Aoo = pert[o,o].copy()
        self.Aoo += contract('ie,me->mi', t1, pert[o,v])

        self.Avv = pert[v,v].copy()
        self.Avv -= contract('ma,me->ma', t1, pert[o,v])

        self.Avo = pert[v,o].copy()
        self.Avo += contract('ie,ae->ai', t1, pert[v,v])
        self.Avo -= contract('ma,mi->ai', t1, pert[o,o])
        self.Avo += contract('mieea,me->ai', (2.0*t2 - t2.swapaxis(2,3)), pert[o,v])
        self.Avo -= contract('ie,ma,me->ai', t1, t1, pert[o,v])

        self.Aovoo = contract('ijeb,me->mbij', t2, pert[o,v])

        self.Avvvo = -1.0*contract('miab,me->abei', t2, pert[o,v])
  
        self.Avvoo = contract('ijeb,ae->abij', t2, Avv)
        self.Avvoo -= contract('mjab,mi->abij', t2, Aoo)
