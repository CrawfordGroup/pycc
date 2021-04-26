from opt_einsum import contract


def build_tau(t1, t2, fact1=1.0, fact2=1.0):
    return fact1 * t2 + fact2 * contract('ia,jb->ijab', t1, t1)


def build_Fae(o, v, F, L, t1, t2):
    Fae = F[v,v].copy()
    Fae = Fae - 0.5 * contract('me,ma->ae', F[o,v], t1)
    Fae = Fae + contract('mf,mafe->ae', t1, L[o,v,v,v])
    Fae = Fae - contract('mnaf,mnef->ae', build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
    return Fae


def build_Fmi(o, v, F, L, t1, t2):
    Fmi = F[o,o].copy()
    Fmi = Fmi + 0.5 * contract('ie,me->mi', t1, F[o,v])
    Fmi = Fmi + contract('ne,mnie->mi', t1, L[o,o,o,v])
    Fmi = Fmi + contract('inef,mnef->mi', build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
    return Fmi


def build_Fme(o, v, F, L, t1):
    Fme = F[o,v].copy()
    Fme = Fme + contract('nf,mnef->me', t1, L[o,o,v,v])
    return Fme


def build_Wmnij(o, v, ERI, t1, t2):
    Wmnij = ERI[o,o,o,o].copy()
    Wmnij = Wmnij + contract('je,mnie->mnij', t1, ERI[o,o,o,v])
    Wmnij = Wmnij + contract('ie,mnej->mnij', t1, ERI[o,o,v,o])
    Wmnij = Wmnij + contract('ijef,mnef->mnij', build_tau(t1, t2), ERI[o,o,v,v])
    return Wmnij


def build_Wmbej(o, v, ERI, L, t1, t2):
    Wmbej = ERI[o,v,v,o].copy()
    Wmbej = Wmbej + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
    Wmbej = Wmbej - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
    Wmbej = Wmbej - contract('jnfb,mnef->mbej', build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
    Wmbej = Wmbej + 0.5 * contract('njfb,mnef->mbej', t2, L[o,o,v,v])
    return Wmbej


def build_Wmbje(o, v, ERI, t1, t2):
    Wmbje = -1.0 * ERI[o,v,o,v].copy()
    Wmbje = Wmbje - contract('jf,mbfe->mbje', t1, ERI[o,v,v,v])
    Wmbje = Wmbje + contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
    Wmbje = Wmbje + contract('jnfb,mnfe->mbje', build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
    return Wmbje


def build_Zmbij(o, v, ERI, t1, t2):
    return contract('mbef,ijef->mbij', ERI[o,v,v,v], build_tau(t1, t2))


def r_T1(o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi):
    r_T1 = F[o,v].copy()
    r_T1 = r_T1 + contract('ie,ae->ia', t1, Fae)
    r_T1 = r_T1 - contract('ma,mi->ia', t1, Fmi)
    r_T1 = r_T1 + contract('imae,me->ia', (2.0*t2 - t2.swapaxes(2,3)), Fme)
    r_T1 = r_T1 + contract('nf,nafi->ia', t1, L[o,v,v,o])
    r_T1 = r_T1 + contract('mief,maef->ia', (2.0*t2 - t2.swapaxes(2,3)), ERI[o,v,v,v])
    r_T1 = r_T1 - contract('mnae,nmei->ia', t2, L[o,o,v,o])
    return r_T1


def r_T2(o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij):
    r_T2 = 0.5 * ERI[o,o,v,v].copy()
    r_T2 = r_T2 + contract('ijae,be->ijab', t2, Fae)
    tmp = contract('mb,me->be', t1, Fme)
    r_T2 = r_T2 - 0.5 * contract('ijae,be->ijab', t2, tmp)
    r_T2 = r_T2 - contract('imab,mj->ijab', t2, Fmi)
    tmp = contract('je,me->jm', t1, Fme)
    r_T2 = r_T2 - 0.5 * contract('imab,jm->ijab', t2, tmp)
    r_T2 = r_T2 + 0.5 * contract('mnab,mnij->ijab', build_tau(t1, t2), Wmnij)
    r_T2 = r_T2 + 0.5 * contract('ijef,abef->ijab', build_tau(t1, t2), ERI[v,v,v,v])
    r_T2 = r_T2 - contract('ma,mbij->ijab', t1, Zmbij)
    r_T2 = r_T2 + contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), Wmbej)
    r_T2 = r_T2 + contract('imae,mbej->ijab', t2, (Wmbej + Wmbje.swapaxes(2,3)))
    r_T2 = r_T2 + contract('mjae,mbie->ijab', t2, Wmbje)
    tmp = contract('ie,ma->imea', t1, t1)
    r_T2 = r_T2 - contract('imea,mbej->ijab', tmp, ERI[o,v,v,o])
    r_T2 = r_T2 - contract('imeb,maje->ijab', tmp, ERI[o,v,o,v])
    r_T2 = r_T2 + contract('ie,abej->ijab', t1, ERI[v,v,v,o])
    r_T2 = r_T2 - contract('ma,mbij->ijab', t1, ERI[o,v,o,o])
    r_T2 = r_T2 + r_T2.swapaxes(0,1).swapaxes(2,3)
    return r_T2


def ccsd_energy(o, v, F, L, t1, t2):
    ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
    ecc = ecc + contract('ijab,ijab->', build_tau(t1, t2), L[o,o,v,v])
    return ecc
