from opt_einsum import contract
from .cc_eqs import build_tau


def build_Hov(o, v, F, L, t1):
    Hov = F[o,v].copy()
    Hov += contract('nf,mnef->me', t1, L[o,o,v,v])
    return Hov


def build_Hvv(o, v, F, L, t1, t2):
    Hvv = F[v,v].copy()
    Hvv -= contract('me,ma->ae', F[o,v], t1)
    Hvv += contract('mf,amef->ae', t1, L[v,o,v,v])
    Hvv -= contract('mnfa,mnfe->ae', build_tau(t1, t2), L[o,o,v,v])
    return Hvv


def build_Hoo(o, v, F, L, t1, t2):
    Hoo = F[o,o].copy()
    Hoo += contract('ie,me->mi', F[o,v], t1)
    Hoo += contract('ne,mnie->mi', t1, L[o,o,o,v])
    Hoo += contract('inef,mnef->mi', build_tau(t1, t2), L[o,o,v,v])
    return Hoo


def build_Hoooo(o, v, ERI, t1, t2):
    Hoooo = ERI[o,o,o,o].copy()
    Hoooo += 2.0 * contract('je,mnie->mnij', t1, ERI[o,o,o,v])
    Hoooo += contract('ijef,mnef->mnij', build_tau(t1, t2), ERI[o,o,v,v])
    return Hoooo


def build_Hvvvv(o, v, ERI, t1, t2):
    Hvvvv = ERI[v,v,v,v].copy()
    Hvvvv -= 2.0 * contract('mb,amef->abef', t1, ERI[v,o,v,v])
    Hvvvv += contract('mnab,mnef->abef', build_tau(t1, t2), ERI[o,o,v,v])
    return Hvvvv


def build_Hvovv(o, v, ERI, t1):
    Hvovv = ERI[v,o,v,v].copy()
    Hvovv -= contract('na,nmef->amef', t1, ERI[o,o,v,v])
    return Hvovv


def build_Hooov(o, v, ERI, t1):
    Hooov = ERI[o,o,o,v].copy()
    Hooov += contract('if,nmef->mnie', t1, ERI[o,o,v,v])
    return Hooov


def build_Hovvo(o, v, ERI, L, t1, t2):
    Hovvo = ERI[o,v,v,o].copy()
    Hovvo += contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
    Hovvo -= contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
    Hovvo -= contract('jnfb,mnef->mbej', build_tau(t1, t2), ERI[o,o,v,v])
    Hovvo += contract('njfb,mnef->mbej', t2, L[o,o,v,v])
    return Hovvo


def build_Hovov(o, v, ERI, t1, t2):
    Hovov = ERI[o,v,o,v].copy()
    Hovov += contract('jf,bmef->mbje', t1, ERI[v,o,v,v])
    Hovov -= contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
    Hovov -= contract('jnfb,nmef->mbje', build_tau(t1, t2), ERI[o,o,v,v])
    return Hovov


def build_Hvvvo(o, v, F, ERI, L, t1, t2):
    Hvvvo = ERI[v,v,v,o].copy()
    Hvvvo += contract('if,abef->abei', t1, ERI[v,v,v,v])
    Hvvvo -= contract('mb,amei->abei', t1, ERI[v,o,v,o])
    Hvvvo -= contract('ma,bmie->abei', t1, ERI[v,o,o,v])
    Hvvvo -= contract('imfa,mbef->abei', build_tau(t1, t2), ERI[o,v,v,v])
    Hvvvo -= contract('imfb,amef->abei', build_tau(t1, t2), ERI[v,o,v,v])
    Hvvvo += contract('mnab,mnei->abei', build_tau(t1, t2), ERI[o,o,v,o])
    Hvvvo -= contract('me,miab->abei', F[o,v], t2)
    Hvvvo += contract('mifb,amef->abei', t2, L[v,o,v,v])
    Hvvvo += contract('if,mnab,mnef->abei', t1, t2, ERI[o,o,v,v])
    Hvvvo += contract('ma,infb,mnef->abei', t1, t2, ERI[o,o,v,v])
    Hvvvo += contract('nb,miaf,mnef->abei', t1, t2, ERI[o,o,v,v])
    Hvvvo -= contract('mf,niab,mnfe->abei', t1, t2, L[o,o,v,v])
    Hvvvo -= contract('na,mifb,mnfe->abei', t1, t2, L[o,o,v,v])
    Hvvvo += contract('if,ma,nb,mnef->abei', t1, t1, t1, ERI[o,o,v,v])
    return Hvvvo


def build_Hovoo(o, v, F, ERI, L, t1, t2):
    Hovoo = ERI[o,v,o,o].copy()
    Hovoo += contract('je,mbie->mbij', t1, ERI[o,v,o,v])
    Hovoo += contract('ie,bmje->mbij', t1, ERI[v,o,o,v])
    Hovoo -= contract('nb,mnij->mbij', t1, ERI[o,o,o,o])
    Hovoo -= contract('ineb,nmje->mbij', build_tau(t1, t2), ERI[o,o,o,v])
    Hovoo -= contract('jneb,mnie->mbij', build_tau(t1, t2), ERI[o,o,o,v])
    Hovoo += contract('ijef,mbef->mbij', build_tau(t1, t2), ERI[o,v,v,v])
    Hovoo += contract('ijeb,me->mbij', t2, F[o,v])
    Hovoo += contract('njeb,mnie->mbij', t2, L[o,o,o,v])
    Hovoo -= contract('jf,ineb,mnef->mbij', t1, t2, ERI[o,o,v,v])
    Hovoo -= contract('ie,jnfb,mnef->mbij', t1, t2, ERI[o,o,v,v])
    Hovoo -= contract('nb,ijef,mnef->mbij', t1, t2, ERI[o,o,v,v])
    Hovoo += contract('ie,njfb,mnef->mbij', t1, t2, L[o,o,v,v])
    Hovoo += contract('nf,ijeb,mnef->mbij', t1, t2, L[o,o,v,v])
    Hovoo -= contract('jf,ie,nb,mnef->mbij', t1, t1, t1, ERI[o,o,v,v])
    return Hovoo
