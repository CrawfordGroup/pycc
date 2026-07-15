#Triples class for (T) corrections, CC3, etc.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pycc.ccwfn import HAS_TORCH
if HAS_TORCH:
    import torch
from pycc.utils import zeros_like, diag, clone, permute_triples

if TYPE_CHECKING:
    from pycc.ccwfn import CCwfn

"""Various triples drivers; useful for (T) corrections and CC3.  Each driver
returns batches of triples amplitudes corresponding to the general expression:

    t3_ijkabc = -P(ijk/abc) [ t2_ijae W_bcek - t2_imab W_mcjk ]/D_ijkabc

Here the W quantity is either a two-electron integral <pq|rs> or a dressed
intermediate (e.g., T1-similarity-transformed integrals) depending on the
desired target.
"""

def t3c_ijk(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract, WithDenom=True):
    """Build the T3 amplitudes in batches for fixed i,j,k indices.

    Returns
    -------
    ndarray or torch.Tensor, shape (nv, nv, nv)

    Notes
    -----
    General expression:

    t3_ijkabc = -P(ijk/abc) [ t2_ijae W_bcek - t2_imab W_mcjk ]/D_ijkabc

    Here the W quantity is either a two-electron integral <pq|rs> or a dressed
    intermediate (e.g., T1-similarity-transformed integrals) depending on the
    desired target.

    """
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
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(t3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += Fo[i] + Fo[j] + Fo[k]
        return t3/denom
    else:
        return t3


def t3c_abc(o, v, a, b, c, t2, Wvvvo, Wovoo, F, contract, WithDenom=True):
    """Build the T3 amplitudes in batches for fixed a,b,c indices.

    Returns
    -------
    ndarray or torch.Tensor, shape (no, no, no)

    Notes
    -----
    General expression:

    t3_ijkabc = -P(ijk/abc) [ t2_ijae W_bcek - t2_imab W_mcjk ]/D_ijkabc

    Here the W quantity is either a two-electron integral <pq|rs> or a dressed
    intermediate (e.g., T1-similarity-transformed integrals) depending on the
    desired target.

    """
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
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(t3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= Fv[a] + Fv[b] + Fv[c]
        return t3/denom
    else:
        return t3


def t3d_ijk(o, v, i, j, k, t1, t2, Woovv, F, contract, WithDenom=True):
    """Build the disconnected contributions to the T3 amplitudes in batches
    for fixed i,j,k indices.

    Returns
    -------
    ndarray or torch.Tensor, shape (nv, nv, nv)

    Notes
    -----
    General expression:

    t3_ijkabc = W_ijab t1_kc + W_ikac t1_jb + W_jkbc t1_ia +
                t2_ijab F_kc + t2_ikac F_jb + t2_jkbc F_ia

    Here the W quantity is either a two-electron integral <pq|rs> or a dressed
    intermediate (e.g., T1-similarity-transformed integrals) depending on the
    desired target.

    """
    Fov = F[o,v]
    t3 = contract('ab,c->abc', Woovv[i,j], t1[k])
    t3 += contract('ac,b->abc', Woovv[i,k], t1[j])
    t3 += contract('bc,a->abc', Woovv[j,k], t1[i])
    t3 += contract('ab,c->abc', t2[i,j], Fov[k])
    t3 += contract('ac,b->abc', t2[i,k], Fov[j])
    t3 += contract('bc,a->abc', t2[j,k], Fov[i])

    if WithDenom is True:
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(t3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += Fo[i] + Fo[j] + Fo[k]
        return t3/denom
    else:
        return t3

def t3d_abc(o, v, a, b, c, t1, t2, Woovv, F, contract, WithDenom=True):
    """Build the disconnected contributions to the T3 amplitudes in batches
    for fixed a,b,c indices.

    Returns
    -------
    ndarray or torch.Tensor, shape (no, no, no)

    Notes
    -----
    General expression:

    t3_ijkabc = W_ijab t1_kc + W_ikac t1_jb + W_jkbc t1_ia +
                t2_ijab F_kc + t2_ikac F_jb + t2_jkbc F_ia

    Here the W quantity is either a two-electron integral <pq|rs> or a dressed
    intermediate (e.g., T1-similarity-transformed integrals) depending on the
    desired target.

    """
    t3 = contract('ij,k->ijk', Woovv[:,:,a,b], t1[:,c])
    t3 += contract('ik,j->ijk', Woovv[:,:,a,c], t1[:,b])
    t3 += contract('jk,i->ijk', Woovv[:,:,b,c], t1[:,a])
    Fov = F[o,v]
    t3 += contract('ij,k->ijk', t2[:,:,a,b], Fov[:,c])
    t3 += contract('ik,j->ijk', t2[:,:,a,c], Fov[:,b])
    t3 += contract('jk,i->ijk', t2[:,:,b,c], Fov[:,a])

    if WithDenom is True:
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(t3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= Fv[a] + Fv[b] + Fv[c]
        return t3/denom
    else:
        return t3


# Lee and Rendell's formulation
def t_tjl(ccwfn: "CCwfn") -> float:
    """Compute the (T) energy correction using the efficient formulation by Rendell
    and Lee, Chem. Phys. Lett. 178, 462-470 (1991).

    Returns
    -------
    float

    """
    contract = ccwfn.contract

    o = ccwfn.o
    v = ccwfn.v
    no = ccwfn.no
    nv = ccwfn.nv
    F = ccwfn.H.F
    ERI = ccwfn.H.ERI
    t1 = ccwfn.t1
    t2 = ccwfn.t2

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

                Fo = diag(F)[o]
                Fv = diag(F)[v]
                denom = zeros_like(W3)
                denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
                denom += Fo[i] + Fo[j] + Fo[k]

                for a in range(nv):
                    for b in range(a+1):
                        for c in range(b+1):
                            ET += (
                                (Y3[a,b,c] - 2.0 * Z3[a,b,c]) * (W3[a,b,c] + W3[b,c,a] + W3[c,a,b])
                                + (Z3[a,b,c] - 2.0 * Y3[a,b,c]) * (W3[a,c,b] + W3[b,a,c] + W3[c,b,a])
                                + 3.0 * X3[a,b,c]) * (2.0 - (int(i == j) + int(i == k) + int(j == k)))/denom[a,b,c]

    return ET


# Vikings' formulation
def t_vikings(ccwfn: "CCwfn") -> float:
    """Compute the (T) energy correction using the formulation by Helgaker,
    Jørgensen, and Olsen, Molecular Electronic Structure Theory, Wiley & Sons, 
    New York, 2000, Ch.14, pp. 794-795, Eqs. (14.6.62)-(14.6.64).  This algorithm
    batches triples oveer fixed i,j,k indices.

    Returns
    -------
    float

    """
    contract = ccwfn.contract

    o = ccwfn.o
    v = ccwfn.v
    no = ccwfn.no
    F = ccwfn.H.F
    ERI = ccwfn.H.ERI
    L = ccwfn.H.L
    t1 = ccwfn.t1
    t2 = ccwfn.t2
    X1 = zeros_like(ccwfn.t1)
    X2 = zeros_like(ccwfn.t2)

    Loovv = L[o,o,v,v]
    ERIvovv = ERI[v,o,v,v]
    ERIooov = ERI[o,o,o,v]
    Fov = F[o,v]
    for i in range(no):
        for j in range(no):
            for k in range(no):

                t3 = t3c_ijk(o, v, i, j, k, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, contract)

                X1[i] += contract('abc,bc->a',(t3 - t3.swapaxes(0,2)), Loovv[j,k])
                X2[i,j] += contract('abc,dbc->ad', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERIvovv[:,k])
                X2[i] -= contract('abc,lc->lab', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERIooov[j,k])

                X2[i,j] += contract('abc,c->ab',(t3 - t3.swapaxes(0,2)), Fov[k])


    ET = 2.0 * contract('ia,ia->', t1, X1)
    ET += contract('ijab,ijab->', (4.0*t2 - 2.0*t2.swapaxes(2,3)), X2)

    return ET


# Vikings' formulation – inverted algorithm
def t_vikings_inverted(ccwfn: "CCwfn") -> float:
    """Compute the (T) energy correction using the formulation by Helgaker,
    Jørgensen, and Olsen, Molecular Electronic Structure Theory, Wiley & Sons, 
    New York, 2000, Ch.14, pp. 794-795, Eqs. (14.6.62)-(14.6.64).  This algorithm
    batches triples oveer fixed a,b,c indices.

    Returns
    -------
    float

    """
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

    Loovv = L[o,o,v,v]
    ERIvovv = ERI[v,o,v,v]
    ERIooov = ERI[o,o,o,v]
    Fov = F[o,v]
    for a in range(nv):
        for b in range(nv):
            for c in range(nv):
                t3 = t3c_abc(o, v, a, b, c, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, contract, True)

                X1[a] += contract('ijk,jk->i',(t3 - t3.swapaxes(0,2)), Loovv[:,:,b,c])
                X2[a] += contract('ijk,dk->dij', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERIvovv[:,:,b,c])
                X2[a,b] -= contract('ijk,jkl->il', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERIooov[:,:,:,c])

                X2[a,b] += contract('ijk,k->ij',(t3 - t3.swapaxes(0,2)), Fov[:,c])


    ET = 2.0 * contract('ia,ia->', t1, X1.T)
    ET += contract('ijab,ijab->', (4.0*t2 - 2.0*t2.swapaxes(2,3)), X2.T)

    return ET

def l3_ijk(i, j, k, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    Loovv = L[o,o,v,v]
    l3 = contract('ab,c->abc', Loovv[i,j], l1[k]) - contract('ac,b->abc', Loovv[i,j], l1[k])
    l3 += contract('ac,b->abc', Loovv[i,k], l1[j]) - contract('ab,c->abc', Loovv[i,k], l1[j])
    l3 += contract('ba,c->abc', Loovv[j,i], l1[k]) - contract('bc,a->abc', Loovv[j,i], l1[k])
    l3 += contract('ca,b->abc', Loovv[k,i], l1[j]) - contract('cb,a->abc', Loovv[k,i], l1[j])
    l3 += contract('bc,a->abc', Loovv[j,k], l1[i]) - contract('ba,c->abc', Loovv[j,k], l1[i])
    l3 += contract('cb,a->abc', Loovv[k,j], l1[i]) - contract('ca,b->abc', Loovv[k,j], l1[i])

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
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(l3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += Fo[i] + Fo[j] + Fo[k]
        return l3/denom
    else:
        return l3

def l3_abc(a, b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    Loovv = clone(L[o,o,v,v])
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
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(l3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= Fv[a] + Fv[b] + Fv[c]
        return l3/denom
    else:
        return l3


# Efficient algorithm for l3
# Need further debugging
def l3_ijk_alt(i, j, k, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    Loovv = L[o,o,v,v]
    l3 = contract('ab,c->abc', Loovv[i,j], l1[k]) - contract('ac,b->abc', Loovv[i,j], l1[k])
    l3 += contract('ac,b->abc', Loovv[i,k], l1[j]) - contract('ab,c->abc', Loovv[i,k], l1[j])
    l3 += contract('ba,c->abc', Loovv[j,i], l1[k]) - contract('bc,a->abc', Loovv[j,i], l1[k])
    l3 += contract('ca,b->abc', Loovv[k,i], l1[j]) - contract('cb,a->abc', Loovv[k,i], l1[j])
    l3 += contract('bc,a->abc', Loovv[j,k], l1[i]) - contract('ba,c->abc', Loovv[j,k], l1[i])
    l3 += contract('cb,a->abc', Loovv[k,j], l1[i]) - contract('ca,b->abc', Loovv[k,j], l1[i])

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
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(l3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += Fo[i] + Fo[j] + Fo[k]
        return l3/denom
    else:
        return l3

def l3_abc_alt(a, b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    Loovv = clone(L[o,o,v,v])
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
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(l3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= Fv[a] + Fv[b] + Fv[c]
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
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(t3)
        denom += Fo.reshape(-1,1,1,1) + Fo.reshape(-1,1,1) + Fo.reshape(-1,1)
        denom -= Fv.reshape(1,1,1,-1)
        denom -= Fv[b] + Fv[c]
        return t3/denom
    else:
        return t3

def l3_bc(b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract, WithDenom=True):
    Loovv = clone(L[o,o,v,v])
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
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(l3)
        denom += Fo.reshape(-1,1,1,1) + Fo.reshape(-1,1,1) + Fo.reshape(-1,1)
        denom -= Fv.reshape(1,1,1,-1)
        denom -= Fv[b] + Fv[c]
        return l3/denom
    else:
        return l3

# Useful for RT-CC3
# Additional term in T3 equation when an external perturbation is present
def t3_pert_ijk(o, v, i, j, k, t2, V, F, contract, WithDenom=True):
    tmp = contract('ld,ad->al', V[o,v], t2[i,j])
    t3 = contract('al,lcb->abc', tmp, t2[k])

    if WithDenom is True:
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(t3)
        denom -= Fv.reshape(-1,1,1) + Fv.reshape(-1,1) + Fv
        denom += Fo[i] + Fo[j] + Fo[k]
        return t3/denom
    else:
        return t3

def t3_pert_abc(o, v, a, b, c, t2, V, F, contract, WithDenom=True):
    tmp = contract('ld,ijd->ijl', V[o,v], t2[:,:,a])
    t3 = contract('ijl,kl->ijk', tmp, t2[:,:,c,b])

    if WithDenom is True:
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(t3)
        denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
        denom -= Fv[a] + Fv[b] + Fv[c]
        return t3/denom
    else:
        return t3

def t3_pert_bc(o, v, b, c, t2, V, F, contract, WithDenom=True):
    tmp = contract('ld,ijad->ijal', V[o,v], t2)
    t3 = contract('ijal,kl->ijka', tmp, t2[:,:,c,b])

    if WithDenom is True:
        Fo = diag(F)[o]
        Fv = diag(F)[v]
        denom = zeros_like(t3)
        denom += Fo.reshape(-1,1,1,1) + Fo.reshape(-1,1,1) + Fo.reshape(-1,1)
        denom -= Fv.reshape(1,1,1,-1)
        denom -= Fv[b] + Fv[c]
        return t3/denom
    else:
        return t3


# ---- Spin-orbital (T) (open-shell UHF/ROHF references) -----------------------
#
# The spatial t_tjl / t_vikings(_inverted) drivers above are RHF-specific
# (spin-adapted). These are the spin-orbital "viking" (T) driver and its T3 batch
# builder, ported from ~/src/socc: they work directly off the antisymmetrized
# ERI = <pq||rs> (docs/archive/ENHANCEMENT_PLAN_2026-06.md, phase 4).

def t3c_ijk_so(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract, omega=0.0, WithDenom=True):
    """Spin-orbital connected T3 amplitudes for fixed occupied i, j, k.

    ``Wvvvo`` (<ab||ei>) and ``Wovoo`` (<ia||jk>) are passed explicitly: bare ERI
    slices for (T), or the T1-dressed CC3 intermediates for CC3.
    """
    abc = contract('ad,bcd->abc', t2[i,j], Wvvvo[:,:,:,k])
    abc = abc - contract('ad,bcd->abc', t2[k,j], Wvvvo[:,:,:,i])
    abc = abc - contract('ad,bcd->abc', t2[i,k], Wvvvo[:,:,:,j])
    t3 = abc - abc.swapaxes(0,1) - abc.swapaxes(0,2)

    abc = contract('lab,lc->abc', t2[i], Wovoo[:,:,j,k])
    abc = abc - contract('lab,lc->abc', t2[j], Wovoo[:,:,i,k])
    abc = abc - contract('lab,lc->abc', t2[k], Wovoo[:,:,j,i])
    t3 = t3 - (abc - abc.swapaxes(0,2) - abc.swapaxes(1,2))

    if WithDenom is True:
        occ = diag(F)[o]
        vir = diag(F)[v]
        denom = (occ[i] + occ[j] + occ[k] + omega
                 - (vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir))
        return t3/denom
    return t3

def t3d_ijk_so(o, v, i, j, k, t1, t2, Woovv, F, contract, omega=0.0, WithDenom=True):
    """Spin-orbital disconnected T3 amplitudes for fixed occupied i, j, k.

    ``Woovv`` (<ab||ei>) are passed explicitly: bare ERI slices for (T) 
    or the T1-dressed CC3 intermediates for CC3.
    """
    Fov = F[o,v]
    abc = contract('a,bc->abc', t1[i], Woovv[j,k])
    abc -= contract('a,bc->abc', t1[j], Woovv[i,k])
    abc -= contract('a,bc->abc', t1[k], Woovv[j,i])
    abc += contract('a,bc->abc', Fov[i], t2[j,k])
    abc -= contract('a,bc->abc', Fov[j], t2[i,k])
    abc -= contract('a,bc->abc', Fov[k], t2[j,i])
    t3 = abc - abc.swapaxes(0,1) - abc.swapaxes(0,2)

    if WithDenom is True:
        occ = diag(F)[o]
        vir = diag(F)[v]
        denom = (occ[i] + occ[j] + occ[k] + omega
                 - (vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir))
        return t3/denom
    return t3


def l3_ijk_so(o, v, i, j, k, l1, l2, F, Fov, Woovv, Wvovv, Wooov, contract, WithDenom=True):
    """Spin-orbital connected lambda-L3 amplitudes for fixed occupied i, j, k.

    Mirrors :func:`t3c_ijk_so` (spin-orbital convention, antisymmetrized
    ``<pq||rs>`` integrals) for the lambda equations. ``Woovv`` is ``ERI[o,o,v,v]``;
    ``Wvovv`` (<am||ef>) and ``Wooov`` (<mn||ie>) are the T1-dressed CC3
    intermediates. The result is antisymmetric in a,b,c.
    """
    abc = contract('ad,dbc->abc', l2[i,j], Wvovv[:,k,:,:])
    abc = abc - contract('ad,dbc->abc', l2[k,j], Wvovv[:,i,:,:])
    abc = abc - contract('ad,dbc->abc', l2[i,k], Wvovv[:,j,:,:])
    l3 = abc - abc.swapaxes(0,1) - abc.swapaxes(0,2)

    abc = contract('lab,lc->abc', l2[j], Wooov[i,k])
    abc = abc - contract('lab,lc->abc', l2[i], Wooov[j,k])
    abc = abc + contract('lab,lc->abc', l2[k], Wooov[j,i])
    l3 = l3 + (abc - abc.swapaxes(0,2) - abc.swapaxes(1,2))

    abc = contract('a,bc->abc', l1[i], Woovv[j,k]) + contract('a,bc->abc', Fov[i], l2[j,k])
    abc = abc - (contract('a,bc->abc', l1[j], Woovv[i,k]) + contract('a,bc->abc', Fov[j], l2[i,k]))
    abc = abc - (contract('a,bc->abc', l1[k], Woovv[j,i]) + contract('a,bc->abc', Fov[k], l2[j,i]))
    l3 = l3 + (abc - abc.swapaxes(0,1) - abc.swapaxes(0,2))

    if WithDenom is True:
        occ = diag(F)[o]
        vir = diag(F)[v]
        denom = (occ[i] + occ[j] + occ[k]
                 - (vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir))
        return l3/denom
    return l3


def t3c_abc_so(o, v, a, b, c, t2, Wvvvo, Wovoo, F, contract, omega=0.0, WithDenom=True):
    """Spin-orbital connected T3 amplitudes for fixed virtual a, b, c.

    The virtual-batched companion to :func:`t3c_ijk_so` (same spin-orbital
    convention and ``Wvvvo`` (<ab||ei>) / ``Wovoo`` (<ia||jk>) inputs). Result is
    antisymmetric in i,j,k. Used by the CC3 response ABC loop.
    """
    ijk = contract('ijd,dk->ijk', t2[:,:,a], Wvvvo[b,c])
    ijk = ijk - contract('ijd,dk->ijk', t2[:,:,b], Wvvvo[a,c])
    ijk = ijk - contract('ijd,dk->ijk', t2[:,:,c], Wvvvo[b,a])
    t3 = ijk - ijk.swapaxes(0,2) - ijk.swapaxes(1,2)

    ijk = contract('il,ljk->ijk', t2[:,:,a,b], Wovoo[:,c])
    ijk = ijk - contract('il,ljk->ijk', t2[:,:,c,b], Wovoo[:,a])
    ijk = ijk - contract('il,ljk->ijk', t2[:,:,a,c], Wovoo[:,b])
    t3 = t3 - (ijk - ijk.swapaxes(0,1) - ijk.swapaxes(0,2))

    if WithDenom is True:
        occ = diag(F)[o]
        vir = diag(F)[v]
        denom = (occ.reshape(-1,1,1) + occ.reshape(-1,1) + occ
                 - (vir[a] + vir[b] + vir[c]) + omega)
        return t3/denom
    return t3

def t3d_abc_so(o, v, a, b, c, t1, t2, Woovv, F, contract, omega=0.0, WithDenom=True):
    """Spin-orbital disconnected T3 amplitudes for fixed virtual a, b, c.

    ``Woovv`` (<ab||ei>) are passed explicitly: bare ERI slices for (T) 
    or the T1-dressed CC3 intermediates for CC3.
    """
    Fov = F[o,v]
    ijk = contract('i,jk->ijk', t1[:,a], Woovv[:,:,b,c])
    ijk -= contract('i,jk->ijk', t1[:,b], Woovv[:,:,a,c])
    ijk -= contract('i,jk->ijk', t1[:,c], Woovv[:,:,b,a])
    ijk += contract('i,jk->ijk', Fov[:,a], t2[:,:,b,c])
    ijk -= contract('i,jk->ijk', Fov[:,b], t2[:,:,a,c])
    ijk -= contract('i,jk->ijk', Fov[:,c], t2[:,:,b,a])
    t3 = ijk - ijk.swapaxes(0,1) - ijk.swapaxes(0,2)

    if WithDenom is True:
        occ = diag(F)[o]
        vir = diag(F)[v]
        denom = (occ.reshape(-1,1,1) + occ.reshape(-1,1) + occ
                 - (vir[a] + vir[b] + vir[c]) + omega)
        return t3/denom
    return t3

def t_vikings_so(o, v, t1, t2, F, ERI, contract):
    """Spin-orbital (T) energy via the occupied-batched ("viking") algorithm.

    Builds the connected T3 in (i,j,k) batches and contracts each batch into the
    disconnected (x1) and connected (x2) intermediates, then E(T) = t1.x1 +
    1/4 t2.x2. The spatial RHF (T) drivers (t_tjl/t_vikings) are not used here.
    """
    x1 = zeros_like(t1)
    x2 = zeros_like(t2)
    no = t1.shape[0]
    Wvvvo = ERI[v,v,v,o]
    Wovoo = ERI[o,v,o,o]
    # Pre-slice the integrals to the active occupied space, so the (i,j,k,l) loop indices
    # are relative to it. The spin-orbital Hamiltonian is full-MO (frozen core included);
    # under frozen core the active occupied does not start at index 0, so the loop indices
    # must not be used as absolute Hamiltonian indices.
    Woovv = ERI[o,o,v,v]
    Wvovv = ERI[v,o,v,v]
    Wooov = ERI[o,o,o,v]
    Fov = F[o,v]

    for i in range(no):
        for j in range(no):
            for k in range(no):
                t3 = t3c_ijk_so(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                x1[i] += 0.25 * contract('bc,abc->a', Woovv[j,k], t3)
                # Doubles intermediate, built WITHOUT per-term antisymmetrization -- the
                # P(ij)P(ab) antisymmetrizer is applied to x2 after the loops.  Every term
                # carries a 1/4: the antisymmetrizer quadruples an already-antisymmetric
                # contribution.  The occ-vir Fock term (cf. the CC3 Fme[k] term) is nonzero
                # only for a non-canonical (semicanonical ROHF) reference, where f_kc != 0.
                x2[i,j] += (1/4) * contract('c,abc->ab', Fov[k], t3)
                x2[i,j] += (1/4) * contract('dbc,abc->ad', Wvovv[:,k], t3)
                x2[i]   -= (1/4) * contract('md,abd->mab', Wooov[j,k], t3)

    # P(ij)P(ab) antisymmetrization of the doubles intermediate (see the loop note)
    x2 = x2 - x2.swapaxes(0,1) - x2.swapaxes(2,3) + x2.swapaxes(0,1).swapaxes(2,3)

    et = contract('ia,ia->', t1, x1) + 0.25 * contract('ijab,ijab->', t2, x2)
    return et


def _t_energy_from_t3_so(o, v, t1, t2, F, ERI, t3, contract):
    """Spin-orbital (T) energy from a pre-built connected T3, via the "viking"
    contraction of :func:`t_vikings_so` written in full-array form (no per-(i,j,k)
    batching).  ``E(T) = <t1|x1> + 1/4 <t2|x2>`` with the disconnected (x1) and
    connected (x2) intermediates built from the whole T3.  Factored out so both the
    canonical check and the invariant driver share one energy expression."""
    Fov = F[o,v]
    x1 = 0.25 * contract('ijkabc,jkbc->ia', t3, ERI[o,o,v,v])
    x2 = 0.25 * contract('ijkabc,kc->ijab', t3, Fov)                 # f_kc (non-Brillouin only)
    x2 = x2 + 0.25 * contract('ijkabc,dkbc->ijad', t3, ERI[v,o,v,v])
    x2 = x2 - 0.25 * contract('ijkabd,jkmd->imab', t3, ERI[o,o,o,v])
    x2 = x2 - x2.swapaxes(0, 1) - x2.swapaxes(2, 3) + x2.swapaxes(0, 1).swapaxes(2, 3)
    return contract('ia,ia->', t1, x1) + 0.25 * contract('ijab,ijab->', t2, x2)


def t_invariant_so(o, v, t1, t2, F, ERI, contract, e_conv=1e-11, maxiter=100):
    """Spin-orbital rotation-INVARIANT (T) energy via an explicitly-stored, iterated T3.

    Solves ``<nu3|([F,T3] + [V,T2])|0> = 0`` for the full connected T3, splitting the
    Fock as ``F = diag(F)`` (the energy denominator) ``+ F_offdiag`` (the iterative
    ``[F,T3]`` commutator).  Because the off-diagonal Fock is carried explicitly rather
    than assumed zero, the resulting (T) energy is invariant to occupied-occupied and
    virtual-virtual MO rotations -- exactly like the HF and CCSD energies -- unlike the
    canonical batched driver (:func:`t_vikings_so`), whose diagonal-denominator T3
    changes under such a rotation.  At a canonical (diagonal-Fock) reference the
    off-diagonal Fock vanishes, the iteration terminates in one step, and this reduces
    to the standard (T) energy.

    Reference/testing instrument only: it stores the full ``O(o^3 v^3)`` T3 and iterates,
    far more costly than the batched canonical drivers.  Its purpose is to (a) exhibit
    the canonical-vs-non-canonical distinction and (b) serve, via a fixed-basis finite
    field of this energy, as an independent oracle for the (T) polarizability.  The
    ``[F,T3]`` coupling mirrors the CC3 store_triples field path
    (:meth:`CCwfn._so_cc3_t_residual_full`), with the field ``V`` replaced by the
    off-diagonal Fock and bare (non-T1-dressed) integrals."""
    Wvvvo = ERI[v, v, v, o]     # <ab||ci>
    Wovoo = ERI[o, v, o, o]     # <ia||jk>

    # source: <nu3|[V,T2]|0> connected (bare integrals), full-array + antisymmetrized
    tmp = contract('ijad,bcdk->ijkabc', t2, Wvvvo)
    source = permute_triples(tmp, 'k/ij', 'a/bc')
    tmp = -contract('ilab,lcjk->ijkabc', t2, Wovoo)
    source = source + permute_triples(tmp, 'i/jk', 'c/ab')

    # F = diag (-> denominator) + off-diagonal oo/vv (-> [F,T3] coupling)
    fo = diag(F)[o]
    fv = diag(F)[v]
    denom = (fo.reshape(-1, 1, 1, 1, 1, 1) + fo.reshape(-1, 1, 1, 1, 1) + fo.reshape(-1, 1, 1, 1)
             - fv.reshape(-1, 1, 1) - fv.reshape(-1, 1) - fv)
    Foo = F[o, o] - diag(fo)    # off-diagonal occupied Fock
    Fvv = F[v, v] - diag(fv)    # off-diagonal virtual Fock

    t3 = source / denom
    eold = _t_energy_from_t3_so(o, v, t1, t2, F, ERI, t3, contract)
    for _ in range(maxiter):
        # <nu3|[F_offdiag,T3]|0>: virtual (c->d via Fvv) + occupied (k->l via Foo)
        tmp = contract('ijkabc,dc->ijkabd', t3, Fvv)
        comm = tmp - tmp.swapaxes(3, 5) - tmp.swapaxes(4, 5)
        tmp = -contract('ijkabc,kl->ijlabc', t3, Foo)
        comm = comm + (tmp - tmp.swapaxes(0, 2) - tmp.swapaxes(1, 2))
        t3 = (source + comm) / denom
        et = _t_energy_from_t3_so(o, v, t1, t2, F, ERI, t3, contract)
        if abs(et - eold) < e_conv:
            return et
        eold = et
    raise RuntimeError("t_invariant_so: T3 iteration did not converge in %d cycles "
                       "(off-diagonal Fock too large for the Jacobi solve?)" % maxiter)


# ---- (T) density / Lambda intermediates --------------------------------------
# These build the same connected+disconnected T3 as the (T) energy, but contract it
# into the (T) contributions to the one-/two-electron densities and the Lambda-1/2
# residuals (needed for CCSD(T) gradients and properties).  They also return the (T)
# energy correction as a byproduct, so the T3 is built only once.  Housed here (rather
# than on CCwfn) so the wavefunction does not compute density components; solve_cc
# delegates and caches the returned intermediates on the wfn for cclambda/ccdensity.

def t3_density(o, v, no, nv, t1, t2, F, ERI, L, contract):
    """(T) density/Lambda intermediates, spatial-orbital spin-adapted closed-shell RHF.

    Returns (ET, intermediates), where ET is the (T) energy correction and
    intermediates is a dict of the (T) contributions {Doo, Dvv, Dov, Goovv, Gooov,
    Gvvvo, S1, S2} that the caller caches on the wavefunction.
    """
    dvv = np.zeros(nv)   # diagonal of the (T) vir-vir 1-PDM (see below)
    doo = np.zeros(no)   # diagonal of the (T) occ-occ 1-PDM
    Dov = np.zeros((no,nv))
    Goovv = np.zeros_like(t2)
    Gooov = np.zeros((no,no,no,nv))
    Gvvvo = np.zeros((nv,nv,nv,no))
    S1 = np.zeros_like(t1)
    S2 = np.zeros_like(t2)
    X2 = np.zeros_like(t2)

    for i in range(no):
        for j in range(no):
            for k in range(no):
                M3 = t3c_ijk(o, v, i, j, k, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, contract, True)
                N3 = t3d_ijk(o, v, i, j, k, t1, t2, ERI[o,o,v,v], F, contract, True)
                X3 = 8*M3 - 4*M3.swapaxes(0,1) - 4*M3.swapaxes(1,2) - 4*M3.swapaxes(0,2) + 2*np.moveaxis(M3, 0, 2) + 2*np.moveaxis(M3, 2, 0)
                Y3 = 8*N3 - 4*N3.swapaxes(0,1) - 4*N3.swapaxes(1,2) - 4*N3.swapaxes(0,2) + 2*np.moveaxis(N3, 0, 2) + 2*np.moveaxis(N3, 2, 0)

                # Doubles contribution (T) correction (Viking's formulation)
                X2[i,j] += contract('abc,c->ab',(M3 - M3.swapaxes(0,2)), F[o,v][k])
                X2[i,j] += contract('abc,dbc->ad', (2*M3 - M3.swapaxes(1,2) - M3.swapaxes(0,2)),ERI[v,o,v,v][:,k])
                X2[i] -= contract('abc,lc->lab', (2*M3 - M3.swapaxes(1,2) - M3.swapaxes(0,2)),ERI[o,o,o,v][j,k])

                # (T) diagonal one-electron density (occ-occ and vir-vir).  Only the
                # diagonal is a genuine density term (the off-diagonal <0|L3[E_pq,T3]|0>
                # blocks appear in neither Lee-Rendell nor Hald et al.; the oo/vv orbital
                # response is the dependent-pair kappa-bar in CCderiv.gradient), so contract
                # straight to it.  doo and dvv share this ijk build: the symmetrized
                # combination X3+Y3 is invariant under the simultaneous occ<->vir index swap
                # (t3 permutational symmetry), so the occ diagonal needs no separate abc loop
                # (Lee-Rendell, J. Chem. Phys. 94, 6229 (1991), 2/3 the cost of Scuseria's
                # off-diagonal, non-canonical-perturbed-MO formulation).
                dvv += 0.5 * contract('acd,acd->a', M3, (X3 + Y3))
                doo[i] -= 0.5 * contract('abc,abc->', M3, (X3 + Y3))

                # (T) contribution to occ-vir block of one-electron density
                Dov[i] += contract('abc,bc->a', (M3 - M3.swapaxes(0,2)), (4*t2[j,k] - 2*t2[j,k].T))

                # (T) contributions to two-electron density
                Z3 = 2*(M3 - M3.swapaxes(1,2)) - (M3.swapaxes(0,1) - np.moveaxis(M3, 2, 0))
                Goovv[i,j,:,:] += 4*contract('c,abc->ab', t1[k,:], Z3)
                Gooov[j,i] -= contract('abc,lbc->la', (2*X3 + Y3), t2[:,k])
                Gvvvo[:,:,:,j] += contract('abc,cd->abd', (2*X3 + Y3), t2[k,i,:,:])

                # (T) contribution to Lambda_1 residual
                S1[i] += contract('abc,bc->a', 2*(M3 - M3.swapaxes(0,1)), L[o,o,v,v][j,k])
                # (T) contribution to Lambda_2 residual
                S2[i] -= contract('abc,lc->lab', (2*X3 + Y3), ERI[o,o,o,v][j,k])
                S2[i,j] += contract('abc,dcb->ad', (2*X3 + Y3), ERI[o,v,v,v][k])

    S2 = S2 + S2.swapaxes(0,1).swapaxes(2,3)

    # (T) correction
    ET = contract('ia,ia->', t1, S1)  # NB: factor of two is already included in S1
    ET += contract('ijab,ijab->', (4.0*t2 - 2.0*t2.swapaxes(2,3)), X2)

    return ET, {'Doo': np.diag(doo), 'Dvv': np.diag(dvv), 'Dov': Dov,
                'Goovv': Goovv, 'Gooov': Gooov, 'Gvvvo': Gvvvo,
                'S1': S1, 'S2': S2}


def so_t3_density(o, v, no, nv, t1, t2, F, ERI, contract):
    """(T) density/Lambda intermediates, spin-orbital (UHF/ROHF references).

    Returns (ET, intermediates), where ET is the (T) energy correction and
    intermediates is a dict of the (T) contributions {Doo, Dvv, Dov, Goovv, Gooov,
    Gvovv, Govoo, Gvvvo, S1, S2} that the caller caches on the wavefunction.
    """
    x2 = np.zeros_like(t2)
    dvv = np.zeros(nv)
    doo = np.zeros(no)
    Dov = np.zeros((no,nv))
    Goovv = np.zeros_like(t2)
    Gooov = np.zeros((no,no,no,nv))
    Gvovv = np.zeros((nv,no,nv,nv))
    Govoo = np.zeros((no,nv,no,no))
    Gvvvo = np.zeros((nv,nv,nv,no))
    S1 = np.zeros_like(t1)
    S2 = np.zeros_like(t2)

    Wvvvo = ERI[v,v,v,o]
    Wovoo = ERI[o,v,o,o]
    Woovv = ERI[o,o,v,v]
    Wvovv = ERI[v,o,v,v]
    Wooov = ERI[o,o,o,v]
    Fov = F[o,v]

    for i in range(no):
        for j in range(no):
            for k in range(no):
                t3c = t3c_ijk_so(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                t3d = t3d_ijk_so(o, v, i, j, k, t1, t2, Woovv, F, contract)

                # Singles and doubles contributions to the (T) energy correction, built
                # WITHOUT per-term antisymmetrization -- the P(ij)P(ab) antisymmetrizer is
                # applied to x2 after the loops (same structure as S2).  Every term carries a
                # 1/4: the antisymmetrizer quadruples an already-antisymmetric contribution.
                x2[i,j] += (1/4) * contract('c,abc->ab', Fov[k], t3c)
                x2[i,j] += (1/4) * contract('dbc,abc->ad', Wvovv[:,k], t3c)
                x2[i] -= (1/4) * contract('md,abd->mab', Wooov[j,k], t3c)

                # (T) diagonal one-electron density (vv and oo).  Only the diagonal is a
                # genuine density term; the oo/vv orbital response is the dependent-pair
                # kappa-bar in CCderiv (canonical perturbed MOs).  doo and dvv are the same
                # t3c*(t3c+t3d) contraction over the ijk-built T3, differing only in which
                # index is left free, so the occ diagonal needs no separate abc loop
                # (Lee-Rendell, J. Chem. Phys. 94, 6229 (1991)).
                dvv += (1/12) * contract('abc,abc->a', (t3c + t3d), t3c)
                doo[i] -= (1/12) * contract('abc,abc->', t3c, (t3c + t3d))

                # (T) contribution to the ov block of the one-electron density.  The 1/4 is the
                # T2^dagger normalization (D_ia = 1/4 sum_{lm,ef} t3c^{aef}_{ilm} t^{ef}_{lm}); the
                # free (j,k)/(d,e) loop sums carry no combinatorial cancellation here.  This block
                # is invisible to any relaxed first-order property -- the Z-vector/orbital response
                # absorbs the ov 1-PDM exactly (Handy-Schaefer stationarity), so the CCSD(T) gradient
                # is blind to this factor -- but it enters unrelaxed one-electron properties and the
                # perturbed density of second derivatives (e.g. the (T) polarizability) directly.
                Dov[i] += (1/4) * contract('ade,de->a', t3c, t2[j,k])

                # (T) contributions to the two-electron density
                Goovv[i,j] += contract('abc,c->ab', t3c, t1[k])
                Gooov[i,j] -= (1/2) * contract('kbc,abc->ka', t2[k], t3c)
                Gvovv[:,i] += (1/2) * contract('ec,abe->cab', t2[j,k], t3c)
                Govoo[:,:,i,j] -= (1/2) * contract('kbc,abc->ka', t2[k], (t3c + t3d))
                Gvvvo[:,:,:,i] += (1/2) * contract('abe,ec->abc', (t3c + t3d), t2[j,k])

                # (T) contribution to the L1 residual
                S1[i] += (1/4) * contract('abc,bc->a', t3c, Woovv[j,k])
                # (T) contribution to the L2 residual
                S2[i] -= (1/4) * contract('md,abd->mab', Wooov[j,k], (2*t3c + t3d))
                S2[i,j] += (1/4) * contract('ade,bde->ab', (2*t3c+t3d), Wvovv[:,k])

    # P(ij)P(ab) antisymmetrization of the doubles intermediates (see the loop note)
    S2 = S2 - S2.swapaxes(0,1) - S2.swapaxes(2,3) + S2.swapaxes(0,1).swapaxes(2,3)
    x2 = x2 - x2.swapaxes(0,1) - x2.swapaxes(2,3) + x2.swapaxes(0,1).swapaxes(2,3)

    # (T) correction
    ET = contract('ia,ia->', t1, S1) + (1/4) * contract('ijab,ijab->', t2, x2)

    return ET, {'Doo': np.diag(doo), 'Dvv': np.diag(dvv), 'Dov': Dov,
                'Goovv': Goovv, 'Gooov': Gooov, 'Gvovv': Gvovv,
                'Govoo': Govoo, 'Gvvvo': Gvvvo, 'S1': S1, 'S2': S2}
