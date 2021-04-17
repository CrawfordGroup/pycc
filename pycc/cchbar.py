"""
cchbar.py: Builds the similarity-transformed Hamiltonian (one- and two-body terms only).
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import time
from .hbar_eqs import build_Hov, build_Hvv, build_Hoo
from .hbar_eqs import build_Hoooo, build_Hvvvv, build_Hvovv, build_Hooov
from .hbar_eqs import build_Hovvo, build_Hovov, build_Hvvvo, build_Hovoo


class cchbar(object):
    """
    An RHF-CCSD Similarity-Transformed Hamiltonian object.

    Attributes
    ----------
    Hov : NumPy array
        The occupied-virtual block of the one-body component HBAR.
    Hvv : NumPy array
        The virtual-virtual block of the one-body component HBAR.
    Hoo : NumPy array
        The occupied-occupied block of the one-body component HBAR.
    Hoooo : NumPy array
        The occ,occ,occ,occ block of the two-body component HBAR.
    Hvvvv : NumPy array
        The vir,vir,vir,vir block of the two-body component HBAR.
    Hvovv : NumPy array
        The vir,occ,vir,vir block of the two-body component HBAR.
    Hooov : NumPy array
        The occ,occ,occ,vir block of the two-body component HBAR.
    Hovvo : NumPy array
        The occ,vir,vir,occ block of the two-body component HBAR.
    Hovov : NumPy array
        The occ,vir,occ,vir block of the two-body component HBAR.
    Hvvvo : NumPy array
        The vir,vir,vir,occ block of the two-body component HBAR.
    Hovoo : NumPy array
        The occ,vir,occ,occ block of the two-body component HBAR.

    """
    def __init__(self, ccwfn):
        """
        Parameters
        ----------
        ccwfn : PyCC ccenergy object
            amplitudes instantiated to defaults or converged

        Returns
        -------
        None
        """

        time_init = time.time()

        o = ccwfn.o
        v = ccwfn.v
        F = ccwfn.F
        ERI = ccwfn.ERI
        L = ccwfn.L
        t1 = ccwfn.t1
        t2 = ccwfn.t2

        self.Hov = build_Hov(o, v, F, L, t1)
        self.Hvv = build_Hvv(o, v, F, L, t1, t2)
        self.Hoo = build_Hoo(o, v, F, L, t1, t2)
        self.Hoooo = build_Hoooo(o, v, ERI, t1, t2)
        self.Hvvvv = build_Hvvvv(o, v, ERI, t1, t2)
        self.Hvovv = build_Hvovv(o, v, ERI, t1)
        self.Hooov = build_Hooov(o, v, ERI, t1)
        self.Hovvo = build_Hovvo(o, v, ERI, L, t1, t2)
        self.Hovov = build_Hovov(o, v, ERI, t1, t2)
        self.Hvvvo = build_Hvvvo(o, v, ERI, L, self.Hov, self.Hvvvv, t1, t2)
        self.Hovoo = build_Hovoo(o, v, ERI, L, self.Hov, self.Hoooo, t1, t2)

        print("\nHBAR constructed in %.3f seconds.\n" % (time.time() - time_init))
