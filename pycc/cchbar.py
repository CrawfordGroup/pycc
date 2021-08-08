"""
cchbar.py: Builds the similarity-transformed Hamiltonian (one- and two-body terms only).
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import time
import numpy as np
from opt_einsum import contract


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
        ccwfn : PyCC ccwfn object
            amplitudes instantiated to defaults or converged

        Returns
        -------
        None
        """

        time_init = time.time()

        self.ccwfn = ccwfn
        o = ccwfn.o
        v = ccwfn.v
        F = ccwfn.H.F
        ERI = ccwfn.H.ERI
        L = ccwfn.H.L
        t1 = ccwfn.t1
        t2 = ccwfn.t2

        self.Hov = self.build_Hov(o, v, F, L, t1)
        self.Hvv = self.build_Hvv(o, v, F, L, t1, t2)
        self.Hoo = self.build_Hoo(o, v, F, L, t1, t2)
        self.Hoooo = self.build_Hoooo(o, v, ERI, t1, t2)
        self.Hvvvv = self.build_Hvvvv(o, v, ERI, t1, t2)
        self.Hvovv = self.build_Hvovv(o, v, ERI, t1)
        self.Hooov = self.build_Hooov(o, v, ERI, t1)
        self.Hovvo = self.build_Hovvo(o, v, ERI, L, t1, t2)
        self.Hovov = self.build_Hovov(o, v, ERI, t1, t2)
        self.Hvvvo = self.build_Hvvvo(o, v, ERI, L, self.Hov, self.Hvvvv, t1, t2)
        self.Hovoo = self.build_Hovoo(o, v, ERI, L, self.Hov, self.Hoooo, t1, t2)

        print("Hov norm = %20.15f" % np.linalg.norm(self.Hov))
        print("Hvv norm = %20.15f" % np.linalg.norm(self.Hvv))
        print("Hoo norm = %20.15f" % np.linalg.norm(self.Hoo))
        print("Hoooo norm = %20.15f" % np.linalg.norm(self.Hoooo))
        print("Hvvvv norm = %20.15f" % np.linalg.norm(self.Hvvvv))

        print("\nHBAR constructed in %.3f seconds.\n" % (time.time() - time_init))

    def build_Hov(self, o, v, F, L, t1):
        Hov = F[o,v].copy()
        Hov = Hov + contract('nf,mnef->me', t1, L[o,o,v,v])
        return Hov


    def build_Hvv(self, o, v, F, L, t1, t2):
        Hvv = F[v,v].copy()
        Hvv = Hvv - contract('me,ma->ae', F[o,v], t1)
        Hvv = Hvv + contract('mf,amef->ae', t1, L[v,o,v,v])
        Hvv = Hvv - contract('mnfa,mnfe->ae', self.ccwfn.build_tau(t1, t2), L[o,o,v,v])
        return Hvv


    def build_Hoo(self, o, v, F, L, t1, t2):
        Hoo = F[o,o].copy()
        Hoo = Hoo + contract('ie,me->mi', t1, F[o,v])
        Hoo = Hoo + contract('ne,mnie->mi', t1, L[o,o,o,v])
        Hoo = Hoo + contract('inef,mnef->mi', self.ccwfn.build_tau(t1, t2), L[o,o,v,v])
        return Hoo


    def build_Hoooo(self, o, v, ERI, t1, t2):
        Hoooo = ERI[o,o,o,o].copy()
        tmp = contract('je,mnie->mnij', t1, ERI[o,o,o,v])
        Hoooo = Hoooo + (tmp + tmp.swapaxes(0,1).swapaxes(2,3))
        Hoooo = Hoooo + contract('ijef,mnef->mnij', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
        return Hoooo


    def build_Hvvvv(self, o, v, ERI, t1, t2):
        Hvvvv = ERI[v,v,v,v].copy()
        tmp = contract('mb,amef->abef', t1, ERI[v,o,v,v])
        Hvvvv = Hvvvv - (tmp + tmp.swapaxes(0,1).swapaxes(2,3))
        Hvvvv = Hvvvv + contract('mnab,mnef->abef', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
        return Hvvvv


    def build_Hvovv(self, o, v, ERI, t1):
        Hvovv = ERI[v,o,v,v].copy()
        Hvovv = Hvovv - contract('na,nmef->amef', t1, ERI[o,o,v,v])
        return Hvovv


    def build_Hooov(self, o, v, ERI, t1):
        Hooov = ERI[o,o,o,v].copy()
        Hooov = Hooov + contract('if,nmef->mnie', t1, ERI[o,o,v,v])
        return Hooov


    def build_Hovvo(self, o, v, ERI, L, t1, t2):
        Hovvo = ERI[o,v,v,o].copy()
        Hovvo = Hovvo + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
        Hovvo = Hovvo - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
        Hovvo = Hovvo - contract('jnfb,mnef->mbej', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
        Hovvo = Hovvo + contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        return Hovvo


    def build_Hovov(self, o, v, ERI, t1, t2):
        Hovov = ERI[o,v,o,v].copy()
        Hovov = Hovov + contract('jf,bmef->mbje', t1, ERI[v,o,v,v])
        Hovov = Hovov - contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
        Hovov = Hovov - contract('jnfb,nmef->mbje', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
        return Hovov


    def build_Hvvvo(self, o, v, ERI, L, Hov, Hvvvv, t1, t2):
        Hvvvo = ERI[v,v,v,o].copy()
        Hvvvo = Hvvvo - contract('me,miab->abei', Hov, t2)
        Hvvvo = Hvvvo + contract('if,abef->abei', t1, Hvvvv)
        Hvvvo = Hvvvo + contract('mnab,mnei->abei', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,o])
        Hvvvo = Hvvvo - contract('imfa,bmfe->abei', t2, ERI[v,o,v,v])
        Hvvvo = Hvvvo - contract('imfb,amef->abei', t2, ERI[v,o,v,v])
        Hvvvo = Hvvvo + contract('mifb,amef->abei', t2, L[v,o,v,v])
        tmp = ERI[v,o,v,o].copy()
        tmp = tmp - contract('infa,mnfe->amei', t2, ERI[o,o,v,v])
        Hvvvo = Hvvvo - contract('mb,amei->abei', t1, tmp)
        tmp = ERI[v,o,o,v].copy()
        tmp = tmp - contract('infb,mnef->bmie', t2, ERI[o,o,v,v])
        tmp = tmp + contract('nifb,mnef->bmie', t2, L[o,o,v,v])
        Hvvvo = Hvvvo - contract('ma,bmie->abei', t1, tmp)
        return Hvvvo


    def build_Hovoo(self, o, v, ERI, L, Hov, Hoooo, t1, t2):
        Hovoo = ERI[o,v,o,o].copy()
        Hovoo = Hovoo + contract('me,ijeb->mbij', Hov, t2)
        Hovoo = Hovoo - contract('nb,mnij->mbij', t1, Hoooo)
        Hovoo = Hovoo + contract('ijef,mbef->mbij', self.ccwfn.build_tau(t1, t2), ERI[o,v,v,v])
        Hovoo = Hovoo - contract('ineb,nmje->mbij', t2, ERI[o,o,o,v])
        Hovoo = Hovoo - contract('jneb,mnie->mbij', t2, ERI[o,o,o,v])
        Hovoo = Hovoo + contract('njeb,mnie->mbij', t2, L[o,o,o,v])
        tmp = ERI[o,v,o,v].copy()
        tmp = tmp - contract('infb,mnfe->mbie', t2, ERI[o,o,v,v])
        Hovoo = Hovoo + contract('je,mbie->mbij', t1, tmp)
        tmp = ERI[v,o,o,v].copy()
        tmp = tmp - contract('jnfb,mnef->bmje', t2, ERI[o,o,v,v])
        tmp = tmp + contract('njfb,mnef->bmje', t2, L[o,o,v,v])
        Hovoo = Hovoo + contract('ie,bmje->mbij', t1, tmp)
        return Hovoo
