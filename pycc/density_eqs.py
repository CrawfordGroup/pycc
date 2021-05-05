from opt_einsum import contract
from .cc_eqs import build_tau
from .lambda_eqs import build_Goo, build_Gvv


def build_Doo(t1, t2, l1, l2):  # complete
    Doo = -1.0 * contract('ie,je->ij', t1, l1)
    Doo -= contract('imef,jmef->ij', t2, l2)
    return Doo


def build_Dvv(t1, t2, l1, l2):  # complete
    Dvv = contract('mb,ma->ab', t1, l1)
    Dvv += contract('mnbe,mnae->ab', t2, l2)
    return Dvv


def build_Dvo(l1):  # complete
    return l1.T.copy()


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
    Dooov += contract('ijkm,ma->ijka', tmp, t1)
    tmp = contract('mjaf,kmef->jake', t2, l2)
    Dooov += contract('jake,ie->ijka', tmp, t1)
    tmp = contract('imea,kmef->iakf', t2, l2)
    Dooov += contract('iakf,jf->ijka', tmp, t1)

    tmp = contract('kmef,jf->kmej', l2, t1)
    tmp = contract('kmej,ie->kmij', tmp, t1)
    Dooov += contract('kmij,ma->ijka', tmp, t1)
    return Dooov


def build_Dvvvo(t1, t2, l1, l2):  # complete
    tmp = 2.0 * build_tau(t1, t2) - build_tau(t1, t2).swapaxes(2, 3)
    Dvvvo = contract('mc,miab->abci', l1, tmp)
    Dvvvo += contract('ma,imbc->abci', t1, l2)

    Gvv = build_Gvv(t2, l2)
    Dvvvo -= 2.0 * contract('ca,ib->abci', Gvv, t1)
    Dvvvo += contract('cb,ia->abci', Gvv, t1)
    tmp = contract('imbe,nmce->ibnc', t2, l2)
    Dvvvo += 2.0 * contract('ibnc,na->abci', tmp, t1)
    Dvvvo -= contract('ianc,nb->abci', tmp, t1)

    tmp = contract('nmab,nmce->abce', t2, l2)
    Dvvvo -= contract('abce,ie->abci', tmp, t1)
    tmp = contract('niae,nmce->iamc', t2, l2)
    Dvvvo -= contract('iamc,mb->abci', tmp, t1)
    tmp = contract('mibe,nmce->ibnc', t2, l2)
    Dvvvo -= contract('ibnc,na->abci', tmp, t1)

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
    tau = build_tau(t1, t2)
    tau_spinad = 2.0 * tau - tau.swapaxes(2,3)

    Doovv = 4.0 * contract('ia,jb->ijab', t1, l1)
    Doovv += 2.0 * tau_spinad
    Doovv += l2

    tmp1 = 2.0 * t2 - t2.swapaxes(2,3)
    tmp2 = 2.0 * contract('me,jmbe->jb', l1, tmp1)
    Doovv += 2.0 * contract('jb,ia->ijab', tmp2, t1)
    Doovv -= contract('ja,ib->ijab', tmp2, t1)
    tmp2 = 2.0 * contract('ijeb,me->ijmb', tmp1, l1)
    Doovv -= contract('ijmb,ma->ijab', tmp2, t1)
    tmp2 = 2.0 * contract('jmba,me->jeba', tau_spinad, l1)
    Doovv -= contract('jeba,ie->ijab', tmp2, t1)

    Doovv += 4.0 * contract('imae,mjeb->ijab', t2, l2)
    Doovv -= 2.0 * contract('mjbe,imae->ijab', tau, l2)

    tmp_oooo = contract('ijef,mnef->ijmn', t2, l2)
    Doovv += contract('ijmn,mnab->ijab', tmp_oooo, t2)
    tmp1 = contract('njbf,mnef->jbme', t2, l2)
    Doovv += contract('jbme,miae->ijab', tmp1, t2)
    tmp1 = contract('imfb,mnef->ibne', t2, l2)
    Doovv += contract('ibne,njae->ijab', tmp1, t2)
    Gvv = build_Gvv(t2, l2)
    Doovv += 4.0 * contract('eb,ijae->ijab', Gvv, tau)
    Doovv -= 2.0 * contract('ea,ijbe->ijab', Gvv, tau)
    Goo = build_Goo(t2, l2)
    Doovv -= 4.0 * contract('jm,imab->ijab', Goo, tau)  # use tau_spinad?
    Doovv += 2.0 * contract('jm,imba->ijab', Goo, tau)
    tmp1 = contract('inaf,mnef->iame', t2, l2)
    Doovv -= 4.0 * contract('iame,mjbe->ijab', tmp1, tau)
    Doovv += 2.0 * contract('ibme,mjae->ijab', tmp1, tau)
    Doovv += 4.0 * contract('jbme,imae->ijab', tmp1, t2)
    Doovv -= 2.0 * contract('jame,imbe->ijab', tmp1, t2)

    # this can definitely be optimized better
    tmp = contract('nb,ijmn->ijmb', t1, tmp_oooo)
    Doovv += contract('ma,ijmb->ijab', t1, tmp)
    tmp = contract('ie,mnef->mnif', t1, l2)
    tmp = contract('jf,mnif->mnij', t1, tmp)
    Doovv += contract('mnij,mnab->ijab', tmp, t2)
    tmp = contract('ie,mnef->mnif', t1, l2)
    tmp = contract('mnif,njbf->mijb', tmp, t2)
    Doovv += contract('ma,mijb->ijab', t1, tmp)
    tmp = contract('jf,mnef->mnej', t1, l2)
    tmp = contract('mnej,miae->njia', tmp, t2)
    Doovv += contract('nb,njia->ijab', t1, tmp)
    tmp = contract('je,mnef->mnjf', t1, l2)
    tmp = contract('mnjf,imfb->njib', tmp, t2)
    Doovv += contract('na,njib->ijab', t1, tmp)
    tmp = contract('if,mnef->mnei', t1, l2)
    tmp = contract('mnei,njae->mija', tmp, t2)
    Doovv += contract('mb,mija->ijab', t1, tmp)

    tmp = contract('jf,mnef->mnej', t1, l2)
    tmp = contract('ie,mnej->mnij', t1, tmp)
    tmp = contract('nb,mnij->mbij', t1, tmp)
    Doovv += contract('ma,mbij->ijab', t1, tmp)

    return Doovv
