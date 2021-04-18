from opt_einsum import contract
from .cc_eqs import build_tau


def build_Doo(t1, t2, l1, l2):  # complete
    Doo = -1.0 * contract('ie,je->ij', t1, l1)
    Doo -= contract('imef,jmef->ij', t2, l2)
    return Doo


def build_Dvv(t1, t2, l1, l2):  # complete
    Dvv = contract('mb,ma->ab', t1, l1)
    Dvv += contract('mnab,mnae->ab', t2, l2)
    return Dvv


def build_Dvo(l1):  # complete
    return l1.copy();


def build_Dov(t1, t2, l1, l2):  # complete
    Dov = 2.0 * t1.copy()
    Dov += 2.0 * contract('me,imae->ia', l1, t2)
    Dov -= contract('me,miae->ia', l1, build_tau(t1, t2))
    tmp = contract('mnef,inef->mi', l2, t2)
    Dov -= contract('mi,ma->ia', tmp, t1)
    tmp = contract('mnef,mnaf->ea', l2, t2)
    Dov -= contract('ea,ie->ia', tmp, t1)
    return Dov


def build_Doooo(t1, t2, l2):  # complete
    return contract('ijef,klef->ijkl', build_tau(t1, t2), l2)


def build_Dvvvv(t1, t2, l2):  # complete
    return contract('mnab,mncd->abcd', build_tau(t1, t2), l2)


def build_Dooov(t1, t2, l1, l2):  # complete
    tmp = 2.0 * build_tau(t1, t2) - build_tau(t1, t2).swapaxes(2, 3)
    Dooov = -1.0 * contract('ke,ijea->ijka', l1, tmp)
    Dooov -= contract('ie,jkae->ijka', t1, l2)
    Goo = build_Goo(t2, l2)
    Dooov -= 2.0 * contract('ik,ja->ijka', Goo, t1)
    Dooov += contract('jk,ia->ijka', Goo, t1)
    tmp = contract('jmaf,kmef->jake', t2, l2)
    Dooov -= 2.0 * contract('jake,ie->ijka', tmp, t1)
    Dooov += contract('iake,je->ijka', tmp, t1)
    tmp = contract('ijef,kmef->ijkm', t2, l2)
    Dooov -= contract('ijkm,ma->ijka', tmp, t1)
    tmp = contract('mjaf,kmef->jake', t2, l2)
    Dooov += contract('jake,ie->ijka', tmp, t1)
    Dooov += contract('iake,je->ijka', tmp, t1)
    tmp = contract('kmef,jf->kmej', l2, t1)
    tmp = contract('kmej,ie->kmij', tmp, t1)
    Dooov += contract('kmij,ma->ijka', tmp, t1)
    return Dooov


def build_Dvvvo(t1, t2, l1, l2):  # complete
    tmp = 2.0 * build_tau(t1, t2) - build_tau(t1, t2).swapaxes(2, 3)
    Dvvvo += contract('mc,miab->abci', l1, tmp)
    Dvvvo += contract('ma,imbc->abci', t1, l2)
    Gvv = build_Gvv(t2, l2)
    Dvvvo -= 2.0 * contract('ca,ib->abci', Gvv, t1)
    Dvvvo += contract('bc,ia->abci', Gvv, t1)
    tmp = contract('imbe,nmce->ibnc', t2, l2)
    Dvvvo += 2.0 * contract('ibnc,na->abci', tmp, t1)
    Dvvvo -= contract('ianc,nb->abci', tmp, t1)
    tmp = contract('nmab,nmce->abce', t2, l2)
    Dvvvo -= contract('abce,ie->abci', tmp, t1)
    tmp = contract('niae,nmce->iamc', t2, l2)
    Dvvvo -= contract('iamc,mb->abci', tmp, t1)
    Dvvvo -= contract('ibmc,na->abci', tmp, t1)
    tmp = contract('nmce,ie->nmci', l2, t1)
    tmp = contract('nmci,na->amci', tmp, t1)
    Dvvvo -= contract('amci,mb->abci', tmp, t1)
    return Dvvvo


def build_Dovov(t1, t2, l1, l2):  # complete
    Dovov = -1.0 * contract('ia,jb->iajb', t1, l1)
    Dovov -= contract('mibe,jmea->iajb', build_tau(t1, t2), l2)
    Dovov -= contract('imbe,mjea->iajb', t2, l2)
    return Dovov


def build_Doovv(t1, t2, l1, l2):
    Doovv = 4.0 * build_tau(t1, t2) - 2.0 * build_tau(t1, t2).swapaxes(2,3)

    Doovv += 4.0 * contract('ia,jb->ijab', t1, l1)

    Doovv += l2.copy()

    tmp1 = 2.0 * t2 - t2.swapaxes(2,3)
    tmp2 = 2.0 * contract('me,jmbe->jb', l1, tmp1)
    Doovv += 2.0 * contract('jb,ia->ijab', tmp2, t1)
    Doovv -= contract('ja,ib->ijab', tmp2, t1)
    tmp2 = 2.0 * contract('ijeb,me->ijmb', tmp1, l1)
    Doovv -= contract('ijmb,ma->ijab', tmp2, t1)
    tmp1 = 2.0 * build_tau(t1, t2) - build_tau(t1, t2).swapaxes(2,3)
    tmp2 = 2.0 * contract('jmba,me->jeba', tmp1, l1)
    Doovv -= contract('jeba,ma->ijab', tmp2, t1)

    Doovv += 4.0 * contract('imae,mjeb->ijab', t2, l2)
    Doovv -= 2.0 * contract('mjbe,imae->ijab', t2, l2)

    tmp1 = 0.5 * contract('ijef,mnef->ijmn', t2, l2)
    Doovv += 2.0 

    tmp1 = 0.5 * contract('njbf,mnef->jbme', t2, l2)

    return Doovv
