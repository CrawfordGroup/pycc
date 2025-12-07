#Triples class for (T) corrections, CC3, etc.

import numpy as np
import torch
import time

# Various triples drivers; useful for (T) corrections and CC3

def t3c_ijk(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract, WithDenom=True):

    t3 = contract('bae,ce->abc', Wvvvo[:,:,:,i], t2[k,j])
    t3 += contract('cae,be->abc', Wvvvo[:,:,:,i], t2[j,k])
    t3 += contract('ace,be->abc', Wvvvo[:,:,:,k], t2[j,i])
    t3 += contract('bce,ae->abc', Wvvvo[:,:,:,k], t2[i,j])
    t3 += contract('cbe,ae->abc', Wvvvo[:,:,:,j], t2[i,k])
    t3 += contract('abe,ce->abc', Wvvvo[:,:,:,j], t2[k,i])

    t3 -= contract('mc,mab->abc', Wovoo[:,:,j,k], t2[i])
    t3 -= contract('mb,mac->abc', Wovoo[:,:,k,j], t2[i])
    t3 -= contract('mb,mca->abc', Wovoo[:,:,i,j], t2[k])
    t3 -= contract('ma,mcb->abc', Wovoo[:,:,j,i], t2[k])
    t3 -= contract('ma,mbc->abc', Wovoo[:,:,k,i], t2[j])
    t3 -= contract('mc,mba->abc', Wovoo[:,:,i,k], t2[j])

    if WithDenom is True:
        if isinstance(t2, torch.Tensor):
            Fv = torch.diag(F)[v]
            denom = torch.zeros_like(t3)
        else:
            Fv = np.diag(F)[v]
            denom = np.zeros_like(t3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += F[i,i] + F[j,j] + F[k,k]
        return t3/denom
    else:
        return t3


def t3c_abc(o, v, a, b, c, t2, Wvvvo, Wovoo, F, contract, WithDenom=True):
    t3 = contract('ei,kje->ijk', Wvvvo[b,a], t2[:,:,c])
    t3 += contract('ei,jke->ijk', Wvvvo[c,a], t2[:,:,b])
    t3 += contract('ek,jie->ijk', Wvvvo[a,c], t2[:,:,b])
    t3 += contract('ek,ije->ijk', Wvvvo[b,c], t2[:,:,a])
    t3 += contract('ej,ike->ijk', Wvvvo[c,b], t2[:,:,a])
    t3 += contract('ej,kie->ijk', Wvvvo[a,b], t2[:,:,c])

    t3 -= contract('mjk,im->ijk', Wovoo[:,c,:,:], t2[:,:,a,b])
    t3 -= contract('mkj,im->ijk', Wovoo[:,b,:,:], t2[:,:,a,c])
    t3 -= contract('mij,km->ijk', Wovoo[:,b,:,:], t2[:,:,c,a])
    t3 -= contract('mji,km->ijk', Wovoo[:,a,:,:], t2[:,:,c,b])
    t3 -= contract('mki,jm->ijk', Wovoo[:,a,:,:], t2[:,:,b,c])
    t3 -= contract('mik,jm->ijk', Wovoo[:,c,:,:], t2[:,:,b,a])

    if WithDenom is True:
        no = o.stop
        if isinstance(t2, torch.Tensor):
            Fo = torch.diag(F)[o]
            denom = torch.zeros_like(t3)
        else:
            Fo = np.diag(F)[o]
            denom = np.zeros_like(t3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= F[a+no,a+no] + F[b+no,b+no] + F[c+no,c+no]
        return t3/denom
    else:
        return t3


def t3d_ijk(o, v, i, j, k, t1, t2, Woovv, F, contract, WithDenom=True):
    t3 = contract('ab,c->abc', Woovv[i,j], t1[k])
    t3 += contract('ac,b->abc', Woovv[i,k], t1[j])
    t3 += contract('bc,a->abc', Woovv[j,k], t1[i])
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

def t3d_abc(o, v, a, b, c, t1, t2, Woovv, F, contract, WithDenom=True):
    t3 = contract('ij,k->ijk', Woovv[:,:,a,b], t1[:,c])
    t3 += contract('ik,j->ijk', Woovv[:,:,a,c], t1[:,b])
    t3 += contract('jk,i->ijk', Woovv[:,:,b,c], t1[:,a])
    Fov = F[o,v]
    t3 += contract('ij,k->ijk', t2[:,:,a,b], Fov[:,c])
    t3 += contract('ik,j->ijk', t2[:,:,a,c], Fov[:,b])
    t3 += contract('jk,i->ijk', t2[:,:,b,c], Fov[:,a])

    if WithDenom is True:
        no = o.stop
        Fo = np.diag(F)[o]
        denom = np.zeros_like(t3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= F[a+no,a+no] + F[b+no,b+no] + F[c+no,c+no]
        return t3/denom
    else:
        return t3


# Lee and Rendell's formulation
def t_tjl(ccwfn):
    o = ccwfn.o
    v = ccwfn.v
    no = ccwfn.no
    nv = ccwfn.nv
    F = ccwfn.H.F
    ERI = ccwfn.H.ERI
    t1 = ccwfn.t1
    t2 = ccwfn.t2
    contract = ccwfn.contract

    ET = 0.0
    for i in range(no):
        for j in range(i+1):
            for k in range(j+1):
                W3 = t3c_ijk(o, v, i, j, k, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, contract, False)
                V3 = t3d_ijk(o, v, i, j, k, t1, t2, ERI[o,o,v,v], F, contract, False) + W3

                for a in range(nv):
                    for b in range(nv):
                        for c in range(nv):
                            V3[a,b,c] /= (1.0 + int(a == b) + int(a == c) + int(b == c))

                X3 = W3 * V3  # abc
                X3 += W3.swapaxes(1,2) * V3.swapaxes(1,2)  # acb
                X3 += W3.swapaxes(0,1) * V3.swapaxes(0,1)  # bac
                X3 += W3.swapaxes(0,1).swapaxes(1,2) * V3.swapaxes(0,1).swapaxes(1,2)  # bca
                X3 += W3.swapaxes(0,1).swapaxes(0,2) * V3.swapaxes(0,1).swapaxes(0,2)  # cab
                X3 += W3.swapaxes(0,2) * V3.swapaxes(0,2)  # cba

                Y3 = V3 + V3.swapaxes(0,1).swapaxes(1,2) + V3.swapaxes(0,1).swapaxes(0,2)
                Z3 = V3.swapaxes(1,2) + V3.swapaxes(0,1) + V3.swapaxes(0,2)

                Fv = np.diag(F)[v]
                denom = np.zeros_like(W3)
                denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
                denom += F[i,i] + F[j,j] + F[k,k]

                for a in range(nv):
                    for b in range(a+1):
                        for c in range(b+1):
                            ET += (
                                (Y3[a,b,c] - 2.0 * Z3[a,b,c]) * (W3[a,b,c] + W3[b,c,a] + W3[c,a,b])
                                + (Z3[a,b,c] - 2.0 * Y3[a,b,c]) * (W3[a,c,b] + W3[b,a,c] + W3[c,b,a])
                                + 3.0 * X3[a,b,c]) * (2.0 - (int(i == j) + int(i == k) + int(j == k)))/denom[a,b,c]

    return ET


# Vikings' formulation
def t_vikings(ccwfn):
    contract = ccwfn.contract
    o = ccwfn.o
    v = ccwfn.v
    no = ccwfn.no
    F = ccwfn.H.F
    ERI = ccwfn.H.ERI
    L = ccwfn.H.L
    t1 = ccwfn.t1
    t2 = ccwfn.t2
    if isinstance(t1, torch.Tensor):
        X1 = torch.zeros_like(ccwfn.t1)
        X2 = torch.zeros_like(ccwfn.t2)
    else:
        X1 = np.zeros_like(ccwfn.t1)
        X2 = np.zeros_like(ccwfn.t2)

    for i in range(no):
        for j in range(no):
            for k in range(no):
                t3 = t3c_ijk(o, v, i, j, k, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, contract)

                X1[i] += contract('abc,bc->a',(t3 - t3.swapaxes(0,2)), L[j,k,v,v])
                X2[i,j] += contract('abc,c->ab',(t3 - t3.swapaxes(0,2)), F[k,v])
                X2[i,j] += contract('abc,dbc->ad', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[v,k,v,v])
                X2[i] -= contract('abc,lc->lab', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[j,k,o,v])

    ET = 2.0 * contract('ia,ia->', t1, X1)
    ET += contract('ijab,ijab->', (4.0*t2 - 2.0*t2.swapaxes(2,3)), X2)

    return ET


# Vikings' formulation – inverted algorithm
def t_vikings_inverted(ccwfn):
    contract = ccwfn.contract
    o = ccwfn.o
    v = ccwfn.v
    no = ccwfn.no
    nv = ccwfn.nv
    F = ccwfn.H.F
    ERI = ccwfn.H.ERI
    L = ccwfn.H.L
    t1 = ccwfn.t1
    t2 = ccwfn.t2
    X1 = np.zeros_like(t1.T)
    X2 = np.zeros_like(t2.T)

    for a in range(nv):
        for b in range(nv):
            for c in range(nv):
                t3 = t3c_abc(o, v, a, b, c, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, contract, True)

                X1[a] += contract('ijk,jk->i',(t3 - t3.swapaxes(0,2)), L[o,o,b+no,c+no])
                X2[a,b] += contract('ijk,k->ij',(t3 - t3.swapaxes(0,2)), F[o,c+no])
                X2[a] += contract('ijk,dk->dij', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[v,o,b+no,c+no])
                X2[a,b] -= contract('ijk,jkl->il', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[o,o,o,c+no])

    ET = 2.0 * contract('ia,ia->', t1, X1.T)
    ET += contract('ijab,ijab->', (4.0*t2 - 2.0*t2.swapaxes(2,3)), X2.T)

    return ET

def l3_ijk(i, j, k, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    l3 = contract('ab,c->abc', L[i,j,v,v], l1[k]) - contract('ac,b->abc', L[i,j,v,v], l1[k])
    l3 += contract('ac,b->abc', L[i,k,v,v], l1[j]) - contract('ab,c->abc', L[i,k,v,v], l1[j])
    l3 += contract('ba,c->abc', L[j,i,v,v], l1[k]) - contract('bc,a->abc', L[j,i,v,v], l1[k])
    l3 += contract('ca,b->abc', L[k,i,v,v], l1[j]) - contract('cb,a->abc', L[k,i,v,v], l1[j])
    l3 += contract('bc,a->abc', L[j,k,v,v], l1[i]) - contract('ba,c->abc', L[j,k,v,v], l1[i])
    l3 += contract('cb,a->abc', L[k,j,v,v], l1[i]) - contract('ca,b->abc', L[k,j,v,v], l1[i])

    l3 += contract('a,bc->abc', Fov[i], l2[j,k]) - contract('b,ac->abc', Fov[i], l2[j,k])
    l3 += contract('a,cb->abc', Fov[i], l2[k,j]) - contract('c,ab->abc', Fov[i], l2[k,j])
    l3 += contract('b,ac->abc', Fov[j], l2[i,k]) - contract('a,bc->abc', Fov[j], l2[i,k])
    l3 += contract('c,ab->abc', Fov[k], l2[i,j]) - contract('a,cb->abc', Fov[k], l2[i,j])
    l3 += contract('b,ca->abc', Fov[j], l2[k,i]) - contract('c,ba->abc', Fov[j], l2[k,i])
    l3 += contract('c,ba->abc', Fov[k], l2[j,i]) - contract('b,ca->abc', Fov[k], l2[j,i])

    tmp_W = 2 * Wvovv - Wvovv.swapaxes(2,3)
    W = contract('eab,ce->abc',  tmp_W[:,j,:,:], l2[k,i])
    W += contract('eac,be->abc', tmp_W[:,k,:,:], l2[j,i])
    W += contract('eba,ce->abc', tmp_W[:,i,:,:], l2[k,j])
    W += contract('eca,be->abc', tmp_W[:,i,:,:], l2[j,k])
    W += contract('ebc,ae->abc', tmp_W[:,k,:,:], l2[i,j])
    W += contract('ecb,ae->abc', tmp_W[:,j,:,:], l2[i,k])

    W -= contract('ebc,ea->abc', Wvovv[:,i,:,:], l2[j,k,:,:])
    W -= contract('ecb,ea->abc', Wvovv[:,i,:,:], l2[k,j,:,:])
    W -= contract('eba,ec->abc', Wvovv[:,k,:,:], l2[j,i,:,:])
    W -= contract('eac,eb->abc', Wvovv[:,j,:,:], l2[i,k,:,:])
    W -= contract('eca,eb->abc', Wvovv[:,j,:,:], l2[k,i,:,:])
    W -= contract('eab,ec->abc', Wvovv[:,k,:,:], l2[i,j,:,:])

    tmp_W = 2 * Wooov - Wooov.swapaxes(0,1)
    W -= contract('ma,mcb->abc', tmp_W[j,i,:,:], l2[k])
    W -= contract('ma,mbc->abc', tmp_W[k,i,:,:], l2[j])
    W -= contract('mb,mca->abc', tmp_W[i,j,:,:], l2[k])
    W -= contract('mc,mba->abc', tmp_W[i,k,:,:], l2[j])
    W -= contract('mb,mac->abc', tmp_W[k,j,:,:], l2[i])
    W -= contract('mc,mab->abc', tmp_W[j,k,:,:], l2[i])

    W += contract('mc,mba->abc', Wooov[i,j,:,:], l2[k])
    W += contract('mb,mca->abc', Wooov[i,k,:,:], l2[j])
    W += contract('ma,mbc->abc', Wooov[k,j,:,:], l2[i])
    W += contract('mc,mab->abc', Wooov[j,i,:,:], l2[k])
    W += contract('ma,mcb->abc', Wooov[j,k,:,:], l2[i])
    W += contract('mb,mac->abc', Wooov[k,i,:,:], l2[j])

    l3 += W

    if WithDenom is True:
        if isinstance(l2, torch.Tensor):
            Fv = torch.diag(F)[v]
            denom = torch.zeros_like(l3)
        else:
            Fv = np.diag(F)[v]
            denom = np.zeros_like(l3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += F[i,i] + F[j,j] + F[k,k]
        return l3/denom
    else:
        return l3

def l3_abc(a, b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    if isinstance(l1, torch.Tensor):
        Loovv = L[o,o,v,v].clone()
    else:
        Loovv = L[o,o,v,v].copy()
    l3 = contract('ij,k->ijk', Loovv[:,:,a,b], l1[:,c]) - contract('ij,k->ijk', Loovv[:,:,a,c], l1[:,b])
    l3 += contract('ik,j->ijk', Loovv[:,:,a,c], l1[:,b]) - contract('ik,j->ijk', Loovv[:,:,a,b], l1[:,c])
    l3 += contract('ji,k->ijk', Loovv[:,:,b,a], l1[:,c]) - contract('ji,k->ijk', Loovv[:,:,b,c], l1[:,a])
    l3 += contract('ki,j->ijk', Loovv[:,:,c,a], l1[:,b]) - contract('ki,j->ijk', Loovv[:,:,c,b], l1[:,a])
    l3 += contract('jk,i->ijk', Loovv[:,:,b,c], l1[:,a]) - contract('jk,i->ijk', Loovv[:,:,b,a], l1[:,c])
    l3 += contract('kj,i->ijk', Loovv[:,:,c,b], l1[:,a]) - contract('kj,i->ijk', Loovv[:,:,c,a], l1[:,b])

    l3 += contract('i,jk->ijk', Fov[:,a], l2[:,:,b,c]) - contract('i,jk->ijk', Fov[:,b], l2[:,:,a,c])
    l3 += contract('i,kj->ijk', Fov[:,a], l2[:,:,c,b]) - contract('i,kj->ijk', Fov[:,c], l2[:,:,a,b])
    l3 += contract('j,ik->ijk', Fov[:,b], l2[:,:,a,c]) - contract('j,ik->ijk', Fov[:,a], l2[:,:,b,c])
    l3 += contract('k,ij->ijk', Fov[:,c], l2[:,:,a,b]) - contract('k,ij->ijk', Fov[:,a], l2[:,:,c,b])
    l3 += contract('j,ki->ijk', Fov[:,b], l2[:,:,c,a]) - contract('j,ki->ijk', Fov[:,c], l2[:,:,b,a])
    l3 += contract('k,ji->ijk', Fov[:,c], l2[:,:,b,a]) - contract('k,ji->ijk', Fov[:,b], l2[:,:,c,a])


    tmp_W = 2 * Wvovv - Wvovv.swapaxes(2,3)
    W = contract('ej,kie->ijk',  tmp_W[:,:,a,b], l2[:,:,c,:])
    W += contract('ek,jie->ijk', tmp_W[:,:,a,c], l2[:,:,b,:])
    W += contract('ei,kje->ijk', tmp_W[:,:,b,a], l2[:,:,c,:])
    W += contract('ei,jke->ijk', tmp_W[:,:,c,a], l2[:,:,b,:])
    W += contract('ek,ije->ijk', tmp_W[:,:,b,c], l2[:,:,a,:])
    W += contract('ej,ike->ijk', tmp_W[:,:,c,b], l2[:,:,a,:])

    W -= contract('ei,jke->ijk', Wvovv[:,:,b,c], l2[:,:,:,a])
    W -= contract('ei,kje->ijk', Wvovv[:,:,c,b], l2[:,:,:,a])
    W -= contract('ek,jie->ijk', Wvovv[:,:,b,a], l2[:,:,:,c])
    W -= contract('ej,ike->ijk', Wvovv[:,:,a,c], l2[:,:,:,b])
    W -= contract('ej,kie->ijk', Wvovv[:,:,c,a], l2[:,:,:,b])
    W -= contract('ek,ije->ijk', Wvovv[:,:,a,b], l2[:,:,:,c])

    tmp_W = 2 * Wooov - Wooov.swapaxes(0,1)
    W -= contract('jim,km->ijk', tmp_W[:,:,:,a], l2[:,:,c,b])
    W -= contract('kim,jm->ijk', tmp_W[:,:,:,a], l2[:,:,b,c])
    W -= contract('ijm,km->ijk', tmp_W[:,:,:,b], l2[:,:,c,a])
    W -= contract('ikm,jm->ijk', tmp_W[:,:,:,c], l2[:,:,b,a])
    W -= contract('kjm,im->ijk', tmp_W[:,:,:,b], l2[:,:,a,c])
    W -= contract('jkm,im->ijk', tmp_W[:,:,:,c], l2[:,:,a,b])

    W += contract('ijm,km->ijk', Wooov[:,:,:,c], l2[:,:,b,a])
    W += contract('ikm,jm->ijk', Wooov[:,:,:,b], l2[:,:,c,a])
    W += contract('kjm,im->ijk', Wooov[:,:,:,a], l2[:,:,b,c])
    W += contract('jim,km->ijk', Wooov[:,:,:,c], l2[:,:,a,b])
    W += contract('jkm,im->ijk', Wooov[:,:,:,a], l2[:,:,c,b])
    W += contract('kim,jm->ijk', Wooov[:,:,:,b], l2[:,:,a,c])

    l3 += W

    if WithDenom is True:
        no = o.stop
        if isinstance(l2, torch.Tensor):
            Fo = torch.diag(F)[o]
            denom = torch.zeros_like(l3)
        else:
            Fo = np.diag(F)[o]
            denom = np.zeros_like(l3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= F[a+no,a+no] + F[b+no,b+no] + F[c+no,c+no]
        return l3/denom
    else:
        return l3


# Efficient algorithm for l3
# Need further debugging
def l3_ijk_alt(i, j, k, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    l3 = contract('ab,c->abc', L[i,j,v,v], l1[k]) - contract('ac,b->abc', L[i,j,v,v], l1[k])
    l3 += contract('ac,b->abc', L[i,k,v,v], l1[j]) - contract('ab,c->abc', L[i,k,v,v], l1[j])
    l3 += contract('ba,c->abc', L[j,i,v,v], l1[k]) - contract('bc,a->abc', L[j,i,v,v], l1[k])
    l3 += contract('ca,b->abc', L[k,i,v,v], l1[j]) - contract('cb,a->abc', L[k,i,v,v], l1[j])
    l3 += contract('bc,a->abc', L[j,k,v,v], l1[i]) - contract('ba,c->abc', L[j,k,v,v], l1[i])
    l3 += contract('cb,a->abc', L[k,j,v,v], l1[i]) - contract('ca,b->abc', L[k,j,v,v], l1[i])

    l3 += contract('a,bc->abc', Fov[i], l2[j,k]) - contract('b,ac->abc', Fov[i], l2[j,k])
    l3 += contract('a,cb->abc', Fov[i], l2[k,j]) - contract('c,ab->abc', Fov[i], l2[k,j])
    l3 += contract('b,ac->abc', Fov[j], l2[i,k]) - contract('a,bc->abc', Fov[j], l2[i,k])
    l3 += contract('c,ab->abc', Fov[k], l2[i,j]) - contract('a,cb->abc', Fov[k], l2[i,j])
    l3 += contract('b,ca->abc', Fov[j], l2[k,i]) - contract('c,ba->abc', Fov[j], l2[k,i])
    l3 += contract('c,ba->abc', Fov[k], l2[j,i]) - contract('b,ca->abc', Fov[k], l2[j,i])

    W = contract('eab,ce->abc', Wvovv[:,j,:,:], l2[k,i])
    W += contract('eac,be->abc', Wvovv[:,k,:,:], l2[j,i])
    W += contract('eba,ce->abc', Wvovv[:,i,:,:], l2[k,j])
    W += contract('eca,be->abc', Wvovv[:,i,:,:], l2[j,k])
    W += contract('ebc,ae->abc', Wvovv[:,k,:,:], l2[i,j])
    W += contract('ecb,ae->abc', Wvovv[:,j,:,:], l2[i,k])

    W -= contract('ma,mcb->abc', Wooov[j,i,:,:], l2[k])
    W -= contract('ma,mbc->abc', Wooov[k,i,:,:], l2[j])
    W -= contract('mb,mca->abc', Wooov[i,j,:,:], l2[k])
    W -= contract('mc,mba->abc', Wooov[i,k,:,:], l2[j])
    W -= contract('mb,mac->abc', Wooov[k,j,:,:], l2[i])
    W -= contract('mc,mab->abc', Wooov[j,k,:,:], l2[i])

    l3 += 2 * W - W.swapaxes(0,1) - W.swapaxes(0,2)

    if WithDenom is True:
        if isinstance(l2, torch.Tensor):
            Fv = torch.diag(F)[v]
            denom = torch.zeros_like(l3)
        else:
            Fv = np.diag(F)[v]
            denom = np.zeros_like(l3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += F[i,i] + F[j,j] + F[k,k]
        return l3/denom
    else:
        return l3

def l3_abc_alt(a, b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    if isinstance(l1, torch.Tensor):
        Loovv = L[o,o,v,v].clone()
    else:
        Loovv = L[o,o,v,v].copy()
    l3 = contract('ij,k->ijk', Loovv[:,:,a,b], l1[:,c]) - contract('ij,k->ijk', Loovv[:,:,a,c], l1[:,b])
    l3 += contract('ik,j->ijk', Loovv[:,:,a,c], l1[:,b]) - contract('ik,j->ijk', Loovv[:,:,a,b], l1[:,c])
    l3 += contract('ji,k->ijk', Loovv[:,:,b,a], l1[:,c]) - contract('ji,k->ijk', Loovv[:,:,b,c], l1[:,a])
    l3 += contract('ki,j->ijk', Loovv[:,:,c,a], l1[:,b]) - contract('ki,j->ijk', Loovv[:,:,c,b], l1[:,a])
    l3 += contract('jk,i->ijk', Loovv[:,:,b,c], l1[:,a]) - contract('jk,i->ijk', Loovv[:,:,b,a], l1[:,c])
    l3 += contract('kj,i->ijk', Loovv[:,:,c,b], l1[:,a]) - contract('kj,i->ijk', Loovv[:,:,c,a], l1[:,b])

    l3 += contract('i,jk->ijk', Fov[:,a], l2[:,:,b,c]) - contract('i,jk->ijk', Fov[:,b], l2[:,:,a,c])
    l3 += contract('i,kj->ijk', Fov[:,a], l2[:,:,c,b]) - contract('i,kj->ijk', Fov[:,c], l2[:,:,a,b])
    l3 += contract('j,ik->ijk', Fov[:,b], l2[:,:,a,c]) - contract('j,ik->ijk', Fov[:,a], l2[:,:,b,c])
    l3 += contract('k,ij->ijk', Fov[:,c], l2[:,:,a,b]) - contract('k,ij->ijk', Fov[:,a], l2[:,:,c,b])
    l3 += contract('j,ki->ijk', Fov[:,b], l2[:,:,c,a]) - contract('j,ki->ijk', Fov[:,c], l2[:,:,b,a])
    l3 += contract('k,ji->ijk', Fov[:,c], l2[:,:,b,a]) - contract('k,ji->ijk', Fov[:,b], l2[:,:,c,a])

    W = contract('ej,kie->ijk', Wvovv[:,:,a,b], l2[:,:,c,:])
    W += contract('ek,jie->ijk', Wvovv[:,:,a,c], l2[:,:,b,:])
    W += contract('ei,kje->ijk', Wvovv[:,:,b,a], l2[:,:,c,:])
    W += contract('ei,jke->ijk', Wvovv[:,:,c,a], l2[:,:,b,:])
    W += contract('ek,ije->ijk', Wvovv[:,:,b,c], l2[:,:,a,:])
    W += contract('ej,ike->ijk', Wvovv[:,:,c,b], l2[:,:,a,:])

    W -= contract('jim,km->ijk', Wooov[:,:,:,a], l2[:,:,c,b])
    W -= contract('kim,jm->ijk', Wooov[:,:,:,a], l2[:,:,b,c])
    W -= contract('ijm,km->ijk', Wooov[:,:,:,b], l2[:,:,c,a])
    W -= contract('ikm,jm->ijk', Wooov[:,:,:,c], l2[:,:,b,a])
    W -= contract('kjm,im->ijk', Wooov[:,:,:,b], l2[:,:,a,c])
    W -= contract('jkm,im->ijk', Wooov[:,:,:,c], l2[:,:,a,b])

    l3 += 2 * W - W.swapaxes(0,1) - W.swapaxes(0,2)

    if WithDenom is True:
        no = o.stop
        if isinstance(l2, torch.Tensor):
            Fo = torch.diag(F)[o]
            denom = torch.zeros_like(l3)
        else:
            Fo = np.diag(F)[o]
            denom = np.zeros_like(l3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= F[a+no,a+no] + F[b+no,b+no] + F[c+no,c+no]
        return l3/denom
    else:
        return l3

# Triples drivers that are useful for density matrix calculation
# W_bc(ijka)
def t3c_bc(o, v, b, c, t2, Wvvvo, Wovoo, F, contract, WithDenom=True):

    t3 = contract('aei,kje->ijka', Wvvvo[b], t2[:,:,c])
    t3 += contract('aei,jke->ijka', Wvvvo[c], t2[:,:,b])
    t3 += contract('aek,jie->ijka', Wvvvo[:,c], t2[:,:,b])
    t3 += contract('ek,ijae->ijka', Wvvvo[b,c], t2)
    t3 += contract('ej,ikae->ijka', Wvvvo[c,b], t2)
    t3 += contract('aej,kie->ijka', Wvvvo[:,b], t2[:,:,c])

    t3 -= contract('mjk,ima->ijka', Wovoo[:,c,:,:], t2[:,:,:,b])
    t3 -= contract('mkj,ima->ijka', Wovoo[:,b,:,:], t2[:,:,:,c])
    t3 -= contract('mij,kma->ijka', Wovoo[:,b,:,:], t2[:,:,c])
    t3 -= contract('maji,km->ijka', Wovoo, t2[:,:,c,b])
    t3 -= contract('maki,jm->ijka', Wovoo, t2[:,:,b,c])
    t3 -= contract('mik,jma->ijka', Wovoo[:,c,:,:], t2[:,:,b])

    if WithDenom is True:
        no = o.stop
        if isinstance(t2, torch.Tensor):
            Fo = torch.diag(F)[o]
            Fv = torch.diag(F)[v]
            denom = torch.zeros_like(t3)
        else:
            Fo = np.diag(F)[o]
            Fv = np.diag(F)[v]
            denom = np.zeros_like(t3)
        denom += Fo.reshape(-1,1,1,1) + Fo.reshape(-1,1,1) + Fo.reshape(-1,1)
        denom -= Fv.reshape(1,1,1,-1)
        denom -= F[b+no,b+no] + F[c+no,c+no]
        return t3/denom
    else:
        return t3

def l3_bc(b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    if isinstance(l1, torch.Tensor):
        Loovv = L[o,o,v,v].clone()
    else:
        Loovv = L[o,o,v,v].copy()
    l3 = contract('ija,k->ijka', Loovv[:,:,:,b], l1[:,c]) - contract('ija,k->ijka', Loovv[:,:,:,c], l1[:,b])
    l3 += contract('ika,j->ijka', Loovv[:,:,:,c], l1[:,b]) - contract('ika,j->ijka', Loovv[:,:,:,b], l1[:,c])
    l3 += contract('jia,k->ijka', Loovv[:,:,b], l1[:,c]) - contract('ji,ka->ijka', Loovv[:,:,b,c], l1)
    l3 += contract('kia,j->ijka', Loovv[:,:,c], l1[:,b]) - contract('ki,ja->ijka', Loovv[:,:,c,b], l1)
    l3 += contract('jk,ia->ijka', Loovv[:,:,b,c], l1) - contract('jka,i->ijka', Loovv[:,:,b], l1[:,c])
    l3 += contract('kj,ia->ijka', Loovv[:,:,c,b], l1) - contract('kja,i->ijka', Loovv[:,:,c], l1[:,b])

    l3 += contract('ia,jk->ijka', Fov, l2[:,:,b,c]) - contract('i,jka->ijka', Fov[:,b], l2[:,:,:,c])
    l3 += contract('ia,kj->ijka', Fov, l2[:,:,c,b]) - contract('i,kja->ijka', Fov[:,c], l2[:,:,:,b])
    l3 += contract('j,ika->ijka', Fov[:,b], l2[:,:,:,c]) - contract('ja,ik->ijka', Fov, l2[:,:,b,c])
    l3 += contract('k,ija->ijka', Fov[:,c], l2[:,:,:,b]) - contract('ka,ij->ijka', Fov, l2[:,:,c,b])
    l3 += contract('j,kia->ijka', Fov[:,b], l2[:,:,c]) - contract('j,kia->ijka', Fov[:,c], l2[:,:,b])
    l3 += contract('k,jia->ijka', Fov[:,c], l2[:,:,b]) - contract('k,jia->ijka', Fov[:,b], l2[:,:,c])


    tmp_W = 2 * Wvovv - Wvovv.swapaxes(2,3)
    W = contract('eja,kie->ijka',  tmp_W[:,:,:,b], l2[:,:,c,:])
    W += contract('eka,jie->ijka', tmp_W[:,:,:,c], l2[:,:,b,:])
    W += contract('eia,kje->ijka', tmp_W[:,:,b], l2[:,:,c,:])
    W += contract('eia,jke->ijka', tmp_W[:,:,c], l2[:,:,b,:])
    W += contract('ek,ijae->ijka', tmp_W[:,:,b,c], l2)
    W += contract('ej,ikae->ijka', tmp_W[:,:,c,b], l2)

    W -= contract('ei,jkea->ijka', Wvovv[:,:,b,c], l2)
    W -= contract('ei,kjea->ijka', Wvovv[:,:,c,b], l2)
    W -= contract('eka,jie->ijka', Wvovv[:,:,b], l2[:,:,:,c])
    W -= contract('eja,ike->ijka', Wvovv[:,:,:,c], l2[:,:,:,b])
    W -= contract('eja,kie->ijka', Wvovv[:,:,c], l2[:,:,:,b])
    W -= contract('eka,ije->ijka', Wvovv[:,:,:,b], l2[:,:,:,c])

    tmp_W = 2 * Wooov - Wooov.swapaxes(0,1)
    W -= contract('jima,km->ijka', tmp_W, l2[:,:,c,b])
    W -= contract('kima,jm->ijka', tmp_W, l2[:,:,b,c])
    W -= contract('ijm,kma->ijka', tmp_W[:,:,:,b], l2[:,:,c])
    W -= contract('ikm,jma->ijka', tmp_W[:,:,:,c], l2[:,:,b])
    W -= contract('kjm,ima->ijka', tmp_W[:,:,:,b], l2[:,:,:,c])
    W -= contract('jkm,ima->ijka', tmp_W[:,:,:,c], l2[:,:,:,b])

    W += contract('ijm,kma->ijka', Wooov[:,:,:,c], l2[:,:,b])
    W += contract('ikm,jma->ijka', Wooov[:,:,:,b], l2[:,:,c])
    W += contract('kjma,im->ijka', Wooov, l2[:,:,b,c])
    W += contract('jim,kma->ijka', Wooov[:,:,:,c], l2[:,:,:,b])
    W += contract('jkma,im->ijka', Wooov, l2[:,:,c,b])
    W += contract('kim,jma->ijka', Wooov[:,:,:,b], l2[:,:,:,c])

    l3 += W

    if WithDenom is True:
        no = o.stop
        if isinstance(l2, torch.Tensor):
            Fo = torch.diag(F)[o]
            Fv = torch.diag(F)[v]
            denom = torch.zeros_like(l3)
        else:
            Fo = np.diag(F)[o]
            Fv = np.diag(F)[v]
            denom = np.zeros_like(l3)
        denom += Fo.reshape(-1,1,1,1) + Fo.reshape(-1,1,1) + Fo.reshape(-1,1)
        denom -= Fv.reshape(1,1,1,-1)
        denom -= F[b+no,b+no] + F[c+no,c+no]
        return l3/denom
    else:
        return l3

# Useful for RT-CC3
# Additional term in T3 equation when an external perturbation is present
def t3_pert_ijk(o, v, i, j, k, t2, t3_full, V, F, contract, WithDenom=True):
   
    #time1 = time.time() 
    t3 = -0.5 * contract('ad,dbc->abc', V[v,v], t3_full[i,j,k])
    t3 -= -0.5 * contract('l,labc->abc', V[o,i], t3_full[:,j,k])
    #print("Time(term1): ", time.time() - time1)

    #time1 = time.time()
    tmp = contract('ld,ad->al', V[o,v], t2[i,j])
    t3 -= contract('al,lcb->abc', tmp, t2[k])
    #print("Time(term2): ", time.time() - time1)

    if WithDenom is True:
        if isinstance(t2, torch.Tensor):
            Fv = torch.diag(F)[v]
            denom = torch.zeros_like(t3)
        else:
            Fv = np.diag(F)[v]
            denom = np.zeros_like(t3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += F[i,i] + F[j,j] + F[k,k]
        return t3/denom
    else:
        return t3

def t3_pert_abc(o, v, a, b, c, t2, t3_full, V, F, contract, WithDenom=True):
    
    t3 = -0.5 * contract('d,ijkd->ijk', V[a,v], t3_full[:,:,:,:,b,c])
    t3 -= -0.5 * contract('li,ljk->ijk', V[o,o], t3_full[:,:,:,a,b,c])
    
    tmp = contract('ld,ijd->ijl', V[o,v], t2[:,:,a])
    t3 -= contract('ijl,kl->ijk', tmp, t2[:,:,c,b])

    if WithDenom is True:
        no = o.stop
        if isinstance(t2, torch.Tensor):
            Fo = torch.diag(F)[o]
            denom = torch.zeros_like(t3)
        else:
            Fo = np.diag(F)[o]
            denom = np.zeros_like(t3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= F[a+no,a+no] + F[b+no,b+no] + F[c+no,c+no]
        return t3/denom
    else:
        return t3

def t3_pert_bc(o, v, b, c, t2, t3_full, V, F, contract, WithDenom=True):

    t3 = -0.5 * contract('ad,ijkd->ijka', V[v,v], t3_full[:,:,:,:,b,c])
    t3 -= -0.5 * contract('li,ljka->ijka', V[o,o], t3_full[:,:,:,:,b,c])

    tmp = contract('ld,ijad->ijal', V[o,v], t2)
    t3 -= contract('ijal,kl->ijka', tmp, t2[:,:,c,b])

    if WithDenom is True:
        no = o.stop
        if isinstance(t2, torch.Tensor):
            Fo = torch.diag(F)[o]
            Fv = torch.diag(F)[v]
            denom = torch.zeros_like(t3)
        else:
            Fo = np.diag(F)[o]
            Fv = np.diag(F)[v]
            denom = np.zeros_like(t3)
        denom += Fo.reshape(-1,1,1,1) + Fo.reshape(-1,1,1) + Fo.reshape(-1,1)
        denom -= Fv.reshape(1,1,1,-1)
        denom -= F[b+no,b+no] + F[c+no,c+no]
        return t3/denom
    else:
        return t3

def t3_ijkabc(o, v, i, j, k, a, b, c, t2, Wvvvo, Wovoo, F, contract, WithDenom=True):

    time0 = time.time()

    t3 = contract('e,e->', Wvvvo[b,a,:,i], t2[k,j,c])
    t3 += contract('e,e->', Wvvvo[c,a,:,i], t2[j,k,b])
    t3 += contract('e,e->', Wvvvo[a,c,:,k], t2[j,i,b])
    t3 += contract('e,e->', Wvvvo[b,c,:,k], t2[i,j,a])
    t3 += contract('e,e->', Wvvvo[c,b,:,j], t2[i,k,a])
    t3 += contract('e,e->', Wvvvo[a,b,:,j], t2[k,i,c])

    t3 -= contract('m,m->', Wovoo[:,c,j,k], t2[i,:,a,b])
    t3 -= contract('m,m->', Wovoo[:,b,k,j], t2[i,:,a,c])
    t3 -= contract('m,m->', Wovoo[:,b,i,j], t2[k,:,c,a])
    t3 -= contract('m,m->', Wovoo[:,a,j,i], t2[k,:,c,b])
    t3 -= contract('m,m->', Wovoo[:,a,k,i], t2[j,:,b,c])
    t3 -= contract('m,m->', Wovoo[:,c,i,k], t2[j,:,b,a])

    if WithDenom is True:
        no = o.stop
        denom = F[i,i] + F[j,j] + F[k,k]
        denom -= F[a+no,a+no] + F[b+no,b+no] + F[c+no,c+no]
        return t3/denom
    else:
        return t3

def t3_ijkabc_alt(o, v, t2, Wvvvo, Wovoo, F, contract, WithDenom=True):
    time0 = time.time()

    t3 = contract('baei,kjce->ijkabc', Wvvvo, t2)
    t3 += contract('caei,jkbe->ijkabc', Wvvvo, t2)
    t3 += contract('acek,jibe->ijkabc', Wvvvo, t2)
    t3 += contract('bcek,ijae->ijkabc', Wvvvo, t2)
    t3 += contract('cbej,ikae->ijkabc', Wvvvo, t2)
    t3 += contract('abej,kice->ijkabc', Wvvvo, t2)

    t3 -= contract('mcjk,imab->ijkabc', Wovoo, t2)
    t3 -= contract('mbkj,imac->ijkabc', Wovoo, t2)
    t3 -= contract('mbij,kmca->ijkabc', Wovoo, t2)
    t3 -= contract('maji,kmcb->ijkabc', Wovoo, t2)
    t3 -= contract('maki,jmbc->ijkabc', Wovoo, t2)
    t3 -= contract('mcik,jmba->ijkabc', Wovoo, t2)

    if WithDenom is True:
        if isinstance(t2, torch.Tensor):
            Fv = torch.diag(F)[v]
            Fo = torch.diag(F)[o]
            denom = torch.zeros_like(t3)
        else:
            Fv = np.diag(F)[v]
            Fo = np.diag(F)[o]
            denom = np.zeros_like(t3)
        denom += Fo.reshape(-1,1,1,1,1,1) + Fo.reshape(1,-1,1,1,1,1) + Fo.reshape(1,1,-1,1,1,1)
        denom -= Fv.reshape(1,1,1,-1,1,1) + Fv.reshape(1,1,1,1,-1,1) + Fv.reshape(1,1,1,1,1,-1)

        print("t3(full): ", time.time() - time0)
        return t3/denom
    else:
        return t3

