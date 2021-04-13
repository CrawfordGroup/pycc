from opt_einsum import contract


def build_Goo(t2, l2):
    return contract('mjab,ijab->mi', t2, l2)


def build_Gvv(t2, l2):
    return -1.0 * contract('ijeb,ijab->ae', t2, l2)


def r_L1(o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
    r_l1 = 2.0 * Hov.copy()
    r_l1 += contract('ie,ea->ia', l1, Hvv)
    r_l1 -= contract('ma,im->ia', l1, Hoo)
    r_l1 += contract('me,ieam->ia', l1, (2.0 * Hovvo - Hovov.swapaxes(2,3)))
    r_l1 += contract('imef,efam->ia', l2, Hvvvo)
    r_l1 -= contract('mnae,iemn->ia', l2, Hovoo)
    r_l1 += 2.0 * contract('ef,eifa->ia', Gvv, Hvovv)
    r_l1 -= contract('ef,eiaf->ia', Gvv, Hvovv)
    r_l1 -= 2.0 * contract('mn,mina->ia', Goo, Hooov)
    r_l1 += contract('mn,imna->ia', Goo, Hooov)
    return r_l1


def r_L2(o, v, l1, l2, L, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hovov, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
    r_l2 = L[o,o,v,v].copy()
    r_l2 += 2.0 * contract('ia,jb->ijab', l1, Hov)
    r_l2 -= contract('ja,ib->ijab', l1, Hov)
    r_l2 += contract('ijeb,ea->ijab', l2, Hvv)
    r_l2 -= contract('mjab,im->ijab', l2, Hoo)
    r_l2 += 0.5 * contract('mnab,ijmn->ijab', l2, Hoooo)
    r_l2 += 0.5 * contract('ijef,efab->ijab', l2, Hvvvv)
    r_l2 += 2.0 * contract('ie,ejab->ijab', l1, Hvovv)
    r_l2 -= contract('ie,ejba->ijab', l1, Hvovv)
    r_l2 += 2.0 * contract('mb,jima->ijab', l1, Hooov)
    r_l2 -= contract('mb,ijma->ijab', l1, Hooov)
    r_l2 += contract('mjeb,ieam->ijab', l2, (2.0 * Hovvo - Hovov.swapaxes(2,3)))
    r_l2 -= contract('mibe,jema->ijab', l2, Hovov)
    r_l2 -= contract('mieb,jeam->ijab', l2, Hovvo)
    r_l2 += contract('ae,ijeb->ijab', Gvv, L[o,o,v,v])
    r_l2 -= contract('mi,mjab->ijab', Goo, L[o,o,v,v])
    r_l2 += r_l2.swapaxes(2,3)
    return r_l2


def pseudoenergy(o, v, ERI, l2):
    return 0.5 * contract('ijab,ijab->',ERI[o,o,v,v], l2)
