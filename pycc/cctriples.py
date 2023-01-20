# Triples class for (T) corrections, CC3, etc.

# Unfinished (Zhe, 0407)

import numpy as np
import torch

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
        Fv = np.diag(F)[v]
        denom = np.zeros_like(l3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += F[i,i] + F[j,j] + F[k,k]
        return l3/denom
    else:
        return l3

def l3_abc(a, b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    l3 = contract('ij,k->ijk', L[o,o,a,b], l1[:,c]) - contract('ij,k->ijk', L[o,o,a,c], l1[:,b])
    l3 += contract('ik,j->ijk', L[o,o,a,c], l1[:,b]) - contract('ik,j->ijk', L[o,o,a,b], l1[:,c])
    l3 += contract('ji,k->ijk', L[o,o,b,a], l1[:,c]) - contract('ji,k->ijk', L[o,o,b,c], l1[:,a])
    l3 += contract('ki,j->ijk', L[o,o,c,a], l1[:,b]) - contract('ki,j->ijk', L[o,o,c,b], l1[:,a])
    l3 += contract('jk,i->ijk', L[o,o,b,c], l1[:,a]) - contract('jk,i->ijk', L[o,o,b,a], l1[:,c])
    l3 += contract('kj,i->ijk', L[o,o,c,b], l1[:,a]) - contract('kj,i->ijk', L[o,o,c,a], l1[:,b])

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
        Fv = np.diag(F)[v]
        denom = np.zeros_like(l3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += F[i,i] + F[j,j] + F[k,k]
        return l3/denom
    else:
        return l3

def l3_abc_alt(a, b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    l3 = contract('ij,k->ijk', L[o,o,a,b], l1[:,c]) - contract('ij,k->ijk', L[o,o,a,c], l1[:,b])
    l3 += contract('ik,j->ijk', L[o,o,a,c], l1[:,b]) - contract('ik,j->ijk', L[o,o,a,b], l1[:,c])
    l3 += contract('ji,k->ijk', L[o,o,b,a], l1[:,c]) - contract('ji,k->ijk', L[o,o,b,c], l1[:,a])
    l3 += contract('ki,j->ijk', L[o,o,c,a], l1[:,b]) - contract('ki,j->ijk', L[o,o,c,b], l1[:,a])
    l3 += contract('jk,i->ijk', L[o,o,b,c], l1[:,a]) - contract('jk,i->ijk', L[o,o,b,a], l1[:,c])
    l3 += contract('kj,i->ijk', L[o,o,c,b], l1[:,a]) - contract('kj,i->ijk', L[o,o,c,a], l1[:,b])

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
        Fo = np.diag(F)[o]
        denom = np.zeros_like(l3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= F[a+no,a+no] + F[b+no,b+no] + F[c+no,c+no]
        return l3/denom
    else:
        return l3

