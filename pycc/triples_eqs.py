# Various triples formulations; useful for (T) corrections and CC3

import numpy as np
from opt_einsum import contract

def t3c_ijk(o, v, i, j, k, t2, ERI, F, WithDenom=True):

    t3 = contract('eab,ce->abc', ERI[i,v,v,v], t2[k,j])
    t3 += contract('eac,be->abc', ERI[i,v,v,v], t2[j,k])
    t3 += contract('eca,be->abc', ERI[k,v,v,v], t2[j,i])
    t3 += contract('ecb,ae->abc', ERI[k,v,v,v], t2[i,j])
    t3 += contract('ebc,ae->abc', ERI[j,v,v,v], t2[i,k])
    t3 += contract('eba,ce->abc', ERI[j,v,v,v], t2[k,i])

    t3 -= contract('mc,mab->abc', ERI[j,k,o,v], t2[i])
    t3 -= contract('mb,mac->abc', ERI[k,j,o,v], t2[i])
    t3 -= contract('mb,mca->abc', ERI[i,j,o,v], t2[k])
    t3 -= contract('ma,mcb->abc', ERI[j,i,o,v], t2[k])
    t3 -= contract('ma,mbc->abc', ERI[k,i,o,v], t2[j])
    t3 -= contract('mc,mba->abc', ERI[i,k,o,v], t2[j])

    if WithDenom is True:
        Fv = np.diag(F)[v]
        denom = np.zeros_like(t3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += F[i,i] + F[j,j] + F[k,k]
        return t3/denom
    else:
        return t3

def t3c_abc(o, v, a, b, c, t2, ERI, F, WithDenom=True):
    no = o.stop

    t3 = contract('ie,kje->ijk', ERI[o,v,a+no,b+no], t2[o,o,c])
    t3 += contract('ie,jke->ijk', ERI[o,v,a+no,c+no], t2[o,o,b])
    t3 += contract('ke,jie->ijk', ERI[o,v,c+no,a+no], t2[o,o,b])
    t3 += contract('ke,ije->ijk', ERI[o,v,c+no,b+no], t2[o,o,a])
    t3 += contract('je,ike->ijk', ERI[o,v,b+no,c+no], t2[o,o,a])
    t3 += contract('je,kie->ijk', ERI[o,v,b+no,a+no], t2[o,o,c])

    t3 -= contract('jkm,im->ijk', ERI[o,o,o,c+no], t2[o,o,a,b])
    t3 -= contract('kjm,im->ijk', ERI[o,o,o,b+no], t2[o,o,a,c])
    t3 -= contract('ijm,km->ijk', ERI[o,o,o,b+no], t2[o,o,c,a])
    t3 -= contract('jim,km->ijk', ERI[o,o,o,a+no], t2[o,o,c,b])
    t3 -= contract('kim,jm->ijk', ERI[o,o,o,a+no], t2[o,o,b,c])
    t3 -= contract('ikm,jm->ijk', ERI[o,o,o,c+no], t2[o,o,b,a])

    if WithDenom is True:
        Fo = np.diag(F)[o]
        denom = np.zeros_like(t3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= F[a+no,a+no] + F[b+no,b+no] + F[c+no,c+no]
        return t3/denom
    else:
        return t3

def t3d_ijk(o, v, i, j, k, t1, t2, ERI, F, WithDenom=True):
    t3 = contract('ab,c->abc', ERI[i,j,v,v], t1[k])
    t3 += contract('ac,b->abc', ERI[i,k,v,v], t1[j])
    t3 += contract('bc,a->abc', ERI[j,k,v,v], t1[i])
    t3 += contract('ab,c->abc', t2[i,j], F[k,v])
    t3 += contract('ac,b->abc', t2[i,k], F[j,v])
    t3 += contract('bc,a->abc', t2[j,k], F[i,v])

    if WithDenom is True:
        Fv = np.diag(F)[v]
        denom = np.zeros_like(t3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += F[i,i] + F[j,j] + F[k,k]
        return t3/denom
    else:
        return t3
