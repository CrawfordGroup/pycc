from opt_einsum import contract
from .cc_eqs import build_tau


def build_Doo(t1, t2, l1, l2):
    Doo = -1.0 * contract('ie,je->ij', t1, l1)
    Doo -= contract('imef,jmef->ij', t2, l2)
    return Doo


def build_Dvv(t1, t2, l1, l2):
    Dvv = contract('mb,ma->ab', t1, l1)
    Dvv += contract('mnab,mnae->ab', t2, l2)
    return Dvv


def build_Dvo(l1):
    return l1.copy();


def build_Dov(t1, t2, l1, l2):
    Dov = 2.0 * t1.copy()
    Dov += 2.0 * contract('me,imae->ia', l1, t2)
    Dov -= contract('me,miae->ia', l1, build_tau(t1, t2))
    tmp = contract('mnef,inef->mi', l2, t2)
    Dov -= contract('mi,ma->ia', tmp, t1)
    tmp = contract('mnef,mnaf->ea', l2, t2)
    Dov -= contract('ea,ie->ia', tmp, t1)
    return Dov


def build_Doooo(t1, t2, l2):
    return contract('ijef,klef->ijkl', build_tau(t1, t2), l2)


def build_Dvvvv(t1, t2, l2):
    return contract('mnab,mncd->abcd', build_tau(t1, t2), l2)


def build_Dooov(t1, t2, l1, l2):
    tmp = 2.0 * build_tau(t1, t2) - build_tau(t1, t2).swapaxes(2, 3)
    Gooov = -1.0 * contract('ke,ijea->ijka', l1, tmp)
    Gooov -= contract('ie,jkae->ijka', t1, l2)
    tmp = build_Goo(t2, l2)

    return Dooov


def build_Dvvvo(t1, t2, l1, l2):

    return Dvvvo


def build_Dovov(t1, t2, l1, l2):
    Dovov = -1.0 * contract('ia,jb->iajb', t1, l1)
    Dovov -= contract('mibe,jmea->iajb', build_tau(t1, t2), l2)
    Dovov -= contract('imbe,mjea->iajb', t2, l2)
    return Dovov


def build_Doovv(t1, t2, l1, l2):

    return Doovv
