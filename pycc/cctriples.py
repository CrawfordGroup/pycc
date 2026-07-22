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
    r"""Build the connected T3 amplitudes in batches for fixed i,j,k indices.

    Returns
    -------
    ndarray or torch.Tensor, shape (nv, nv, nv)

    Notes
    -----
    ``W`` is either a two-electron integral <pq|rs> (for the (T) correction) or a T1-dressed
    CC3 intermediate.  ``P(ijk/abc)`` is the full antisymmetrizer over the three
    (occupied, virtual) pairs; ``D_ijkabc`` is the orbital-energy denominator::

        t3_ijkabc = -P(ijk/abc) [ t2_ijae W_bcek - t2_imab W_mcjk ] / D_ijkabc
        D_ijkabc = f_ii + f_jj + f_kk - f_aa - f_bb - f_cc

    .. math::

        \begin{aligned}
        D^{abc}_{ijk}\, t^{abc}_{ijk} &= -\mathcal{P}(ijk/abc)\left( t^{ae}_{ij} W_{bcek} - t^{ab}_{im} W_{mcjk} \right) \\
        D^{abc}_{ijk} &= f_{ii} + f_{jj} + f_{kk} - f_{aa} - f_{bb} - f_{cc}
        \end{aligned}
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
    """Build the connected T3 amplitudes in batches for fixed a,b,c indices -- the a,b,c-batched
    form of :func:`t3c_ijk` (same amplitude and equation; see there).

    Returns
    -------
    ndarray or torch.Tensor, shape (no, no, no)
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
    r"""Build the disconnected contributions to the T3 amplitudes in batches for fixed i,j,k
    indices.

    Returns
    -------
    ndarray or torch.Tensor, shape (nv, nv, nv)

    Notes
    -----
    ``W`` is either a two-electron integral <pq|rs> (for (T)) or a T1-dressed CC3 intermediate;
    ``F`` is the occupied-virtual Fock block (zero for canonical HF).  Divided by the
    orbital-energy denominator ``D_ijkabc``::

        t3_ijkabc = W_ijab t1_kc + W_ikac t1_jb + W_jkbc t1_ia
                  + t2_ijab F_kc + t2_ikac F_jb + t2_jkbc F_ia

    .. math::

        \begin{aligned}
        D^{abc}_{ijk}\, t^{abc}_{ijk} = W_{ijab} t^c_k + W_{ikac} t^b_j + W_{jkbc} t^a_i + t^{ab}_{ij} F_{kc} + t^{ac}_{ik} F_{jb} + t^{bc}_{jk} F_{ia}
        \end{aligned}
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
    """Build the disconnected contributions to the T3 amplitudes in batches for fixed a,b,c
    indices -- the a,b,c-batched form of :func:`t3d_ijk` (same amplitude and equation; see there).

    Returns
    -------
    ndarray or torch.Tensor, shape (no, no, no)
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
    """Compute the (T) energy correction (spatial, spin-adapted closed-shell RHF) using the
    efficient formulation of Lee and Rendell, Chem. Phys. Lett. 178, 462-470 (1991).

    Returns the same E(T) as :func:`t_vikings` / :func:`t_vikings_inverted`, but via the
    Lee-Rendell factorization: the connected T3 batch ``W3`` (:func:`t3c_ijk`) and the full
    numerator ``V3 = W3 + t3d`` are combined into the ``X3``/``Y3``/``Z3`` symmetry
    intermediates and summed over the *triangular* ranges i>=j>=k and a>=b>=c with the
    corresponding permutation multiplicity weights -- about a factor of 6 fewer batch
    contractions than the full-loop viking algorithm.

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
    r"""Compute the (T) energy correction (spatial, spin-adapted closed-shell RHF) using the
    Helgaker-Jorgensen-Olsen "vikings" formulation (Molecular Electronic Structure Theory,
    Wiley, 2000, Ch. 14, pp. 794-795, Eqs. (14.6.62)-(14.6.64)), batching the triples over
    fixed i,j,k indices.

    Returns
    -------
    float

    Notes
    -----
    Occupied-batched: the connected T3 ``t3`` (:func:`t3c_ijk`, canonical denominator) is built
    per-(i,j,k) and contracted into the X1/X2 intermediates -- the full T3 is never stored.
    ``L = 2 ERI - ERI.swap``; ``t3_cba`` etc. are index permutations of ``t3``; the f_kc term is
    nonzero only for a non-canonical reference.  Repeated indices summed::

        X1_ia = (t3_abc - t3_cba) L_jkbc
        X2_ijab += (t3_abc - t3_cba) f_kc
                +  (2 t3_aef - t3_afe - t3_fea) <bk|ef>
                -  (2 t3_imkabc - t3_imkacb - t3_imkcba) <mk|jc>
        E(T) = 2 t1_ia X1_ia + (4 t2 - 2 t2.swap)_ijab X2_ijab

    .. math::

        \begin{aligned}
        (X_1)^a_i &= (t^{abc}_{ijk} - t^{cba}_{ijk}) L_{jkbc} \\
        (X_2)^{ab}_{ij} &\mathrel{+}= (t^{abc}_{ijk} - t^{cba}_{ijk}) F_{kc} + (2 t^{aef}_{ijk} - t^{afe}_{ijk} - t^{fea}_{ijk}) \langle bk|ef \rangle - (2 t^{abc}_{imk} - t^{acb}_{imk} - t^{cba}_{imk}) \langle mk|jc \rangle \\
        E^{(T)} &= 2 t^a_i (X_1)^a_i + (4 t^{ab}_{ij} - 2 t^{ba}_{ij}) (X_2)^{ab}_{ij}
        \end{aligned}
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
    """Compute the (T) energy correction (spatial, spin-adapted closed-shell RHF) via the
    Helgaker-Jorgensen-Olsen "vikings" formulation -- the virtual-batched (fixed a,b,c) form of
    :func:`t_vikings` (same E(T) and X1/X2 expressions; see there), building the connected T3
    with :func:`t3c_abc`.

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
    """Build the connected lambda-L3 amplitudes in batches for fixed i,j,k indices -- the
    spin-adapted (closed-shell, L = 2*ERI - ERI.swap) analogue of :func:`l3_ijk_so` (same
    amplitude; see there for the equation), assembled in fully-antisymmetrized form.  The
    lambda counterpart of the connected T3 :func:`t3c_ijk`.

    Returns
    -------
    ndarray or torch.Tensor, shape (nv, nv, nv)
    """
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
    """Connected lambda-L3, the a,b,c-batched form of :func:`l3_ijk` (same amplitude; see there).

    Returns
    -------
    ndarray or torch.Tensor, shape (no, no, no)
    """
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
    """Alternative (i,j,k-batched) factorization of the connected lambda-L3 from the original
    CC3-Lambda development (git ``3363618``).  Currently **unused** and **does not reproduce**
    :func:`l3_ijk`: its W (ladder) block antisymmetrizes the contracted intermediate
    (``2W - W.swap(0,1) - W.swap(0,2)``) rather than spin-adapting the integrals inline, and the
    two disagree by ~4e-3 (H2O/STO-3G/CC3).  Retained pending review of its original intent;
    use :func:`l3_ijk` (the validated form)."""
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
    """a,b,c-batched companion of :func:`l3_ijk_alt` -- the same unused alternative CC3-Lambda
    factorization, which **does not reproduce** the validated :func:`l3_abc`/:func:`l3_ijk`
    (~4e-3 disagreement).  Retained pending review of its original intent."""
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
    """Connected T3 for fixed virtuals b,c (one free virtual + three occupied), the
    partial-batching of :func:`t3c_ijk` (same amplitude and equation; see there) used by the
    (T)-density virtual loop."""
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
    """Connected lambda-L3 for fixed virtuals b,c (one free virtual + three occupied), the
    partial-batching of :func:`l3_ijk` used by the (T)-density virtual loop (same amplitude)."""
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
    r"""Perturbation coupling of the T3 amplitudes (the ``[V,T2].T2`` term), fixed i,j,k.  Valid
    for any real (Hermitian) one-electron perturbation ``V`` -- electric field, nuclear
    displacement, etc. -- not electric fields specifically.  In the perturbed / real-time (T)
    the connected T3 is corrected by this contribution (``t3 -= t3_pert``).  Divided by the
    orbital-energy denominator D_ijkabc::

        t3_ijkabc = V_ld t2_ijad t2_klcb

    .. math::

        \begin{aligned}
        D^{abc}_{ijk}\, t^{abc}_{ijk} = V_{ld}\, t^{ad}_{ij}\, t^{cb}_{kl}
        \end{aligned}
    """
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
    """T3 perturbation coupling, a,b,c-batched form of :func:`t3_pert_ijk` (same term; see there)."""
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
    """T3 perturbation coupling, fixed-b,c batched form of :func:`t3_pert_ijk` (same term; see there)."""
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
    r"""Spin-orbital connected T3 amplitudes for fixed occupied i, j, k.

    ``Wvvvo`` (<ab||ei>) and ``Wovoo`` (<ia||jk>) are passed explicitly: bare ERI
    slices for (T), or the T1-dressed CC3 intermediates for CC3.

    Notes
    -----
    With the one-vs-pair antisymmetrizer P(p/qr) = 1 - P(pq) - P(pr) and the orbital-energy
    denominator D_ijkabc (shifted by ``omega`` when set)::

        D_ijkabc t3_ijkabc = P(k/ij) P(a/bc) t2_ijad Wvvvo_bcdk
                           - P(i/jk) P(c/ab) t2_ilab Wovoo_lcjk

    .. math::

        \begin{aligned}
        D^{abc}_{ijk}\, t^{abc}_{ijk} = \mathcal{P}(k/ij)\,\mathcal{P}(a/bc)\, t^{ad}_{ij} W^{vvvo}_{bcdk} - \mathcal{P}(i/jk)\,\mathcal{P}(c/ab)\, t^{ab}_{il} W^{ovoo}_{lcjk}
        \end{aligned}
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
    r"""Spin-orbital disconnected T3 amplitudes for fixed occupied i, j, k.

    ``Woovv`` (<ab||ei>) are passed explicitly: bare ERI slices for (T)
    or the T1-dressed CC3 intermediates for CC3.

    Notes
    -----
    With P(p/qr) = 1 - P(pq) - P(pr) and the orbital-energy denominator D_ijkabc (repeated
    indices summed)::

        D_ijkabc t3_ijkabc = P(i/jk) P(a/bc) (t1_ia Woovv_jkbc + F_ia t2_jkbc)

    .. math::

        \begin{aligned}
        D^{abc}_{ijk}\, t^{abc}_{ijk} = \mathcal{P}(i/jk)\,\mathcal{P}(a/bc)\left( t^a_i W^{oovv}_{jkbc} + F_{ia}\, t^{bc}_{jk} \right)
        \end{aligned}
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
    r"""Spin-orbital connected lambda-L3 amplitudes for fixed occupied i, j, k.

    Mirrors :func:`t3c_ijk_so` (spin-orbital convention, antisymmetrized
    ``<pq||rs>`` integrals) for the lambda equations. ``Woovv`` is ``ERI[o,o,v,v]``;
    ``Wvovv`` (<am||ef>) and ``Wooov`` (<mn||ie>) are the T1-dressed CC3
    intermediates. The result is antisymmetric in a,b,c.

    Notes
    -----
    With P(p/qr) = 1 - P(pq) - P(pr) and the orbital-energy denominator D_ijkabc::

        D_ijkabc l3_ijkabc = P(i/jk) P(a/bc) (l1_ia Woovv_jkbc + F_ia l2_jkbc)
                           + P(k/ij) P(a/bc) l2_ijad Wvovv_dkbc
                           - P(i/jk) P(c/ab) l2_ilab Wooov_jklc

    .. math::

        \begin{aligned}
        D^{abc}_{ijk}\, \lambda^{abc}_{ijk} &= \mathcal{P}(i/jk)\,\mathcal{P}(a/bc)\left( \lambda^a_i W^{oovv}_{jkbc} + F_{ia}\, \lambda^{bc}_{jk} \right) \\
        &\quad + \mathcal{P}(k/ij)\,\mathcal{P}(a/bc)\, \lambda^{ad}_{ij} W^{vovv}_{dkbc} - \mathcal{P}(i/jk)\,\mathcal{P}(c/ab)\, \lambda^{ab}_{il} W^{ooov}_{jklc}
        \end{aligned}
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
    r"""Spin-orbital (T) energy via the occupied-batched ("viking") algorithm -- the
    spin-orbital counterpart of :func:`t_vikings`.

    Builds the connected T3 in (i,j,k) batches (:func:`t3c_ijk_so`, canonical denominator) and
    contracts each batch into the disconnected (x1) and connected (x2) intermediates -- the full
    T3 is **never materialized** (O(v^3) memory).  Contrast :func:`_t_energy_from_t3_so`, which
    evaluates the *same* E(T) as whole-array einsums from a **pre-built** O(o^3 v^3) T3 (used by
    the iterated :func:`t_invariant_so`).  The spatial RHF drivers (:func:`t_tjl`/:func:`t_vikings`)
    are not used here.

    Notes
    -----
    With P(pq) X = X - X_swap (the doubles x2 antisymmetrized P(ij)P(ab) after the loop; the
    f_kc term is nonzero only for a non-canonical reference; repeated indices summed)::

        x1_ia   = 1/4 t3_ijkabc <jk||bc>
        x2_ijab = P(ij) P(ab) [ 1/4 t3_ijkabc f_kc + 1/4 t3_ijkaef <bk||ef> - 1/4 t3_inlabd <nl||jd> ]
        E(T)    = t1_ia x1_ia + 1/4 t2_ijab x2_ijab

    .. math::

        \begin{aligned}
        (x_1)^a_i &= \tfrac{1}{4}\, t^{abc}_{ijk} \langle jk||bc \rangle \\
        (x_2)^{ab}_{ij} &= \mathcal{P}(ij)\,\mathcal{P}(ab)\left( \tfrac{1}{4}\, t^{abc}_{ijk} F_{kc} + \tfrac{1}{4}\, t^{aef}_{ijk} \langle bk||ef \rangle - \tfrac{1}{4}\, t^{abd}_{inl} \langle nl||jd \rangle \right) \\
        E^{(T)} &= t^a_i (x_1)^a_i + \tfrac{1}{4}\, t^{ab}_{ij} (x_2)^{ab}_{ij}
        \end{aligned}
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
    """Spin-orbital (T) energy from a **pre-built, whole-array** connected T3 (``t3`` passed in,
    O(o^3 v^3) stored) -- the full-array form of the :func:`t_vikings_so` contraction (same
    ``E(T) = t1.x1 + 1/4 t2.x2``; see there for the x1/x2 expressions).  Where
    :func:`t_vikings_so` builds the T3 batch-by-batch and never stores it, this consumes a T3
    built elsewhere, so the iterated :func:`t_invariant_so` (whose T3 carries the
    off-diagonal-Fock coupling) and the canonical whole-array cross-check can share one energy
    expression."""
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
    r"""(T) density/Lambda intermediates, spatial-orbital spin-adapted closed-shell RHF.

    Returns (ET, intermediates), where ET is the (T) energy correction and
    intermediates is a dict of the (T) contributions {Doo, Dvv, Dov, Goovv, Gooov,
    Gvvvo, S1, S2} that the caller caches on the wavefunction.

    Notes
    -----
    Built per-(i,j,k) from the connected T3 ``M3`` (:func:`t3c_ijk`) and disconnected T3 ``N3``
    (:func:`t3d_ijk`), with the spin-adapted symmetrizers below (a,b,c the three virtual axes;
    repeated indices summed over the i,j,k loop).  Full derivation in
    ``docs/ccsdt_t_density.tex`` and the appendix Table:ccsdt_1pdm/2pdm of
    ``docs/cc_gradients_orbital_response.tex``::

        sym(A)_abc = 8 A_abc - 4 A_bac - 4 A_acb - 4 A_cba + 2 A_bca + 2 A_cab
        X3 = sym(M3),  Y3 = sym(N3)
        Z3_abc = 2 (M3_abc - M3_acb) - (M3_bac - M3_cab)

        # one-particle (T) increments (Doo/Dvv diagonal in the canonical basis)
        Dvv_aa =  1/2 M3_acd (X3+Y3)_acd
        Doo_ii = -1/2 M3_abc (X3+Y3)_abc
        Dov_ia = (M3_abc - M3_cba) (4 t2_jkbc - 2 t2_jkcb)

        # two-particle (T) increments
        Goovv_ijab = 4 t1_kc Z3_abc
        Gooov_jila = -(2 X3 + Y3)_abc t2_lkbc
        Gvvvo_abdj = (2 X3 + Y3)_abc t2_kicd

        # Lambda-1/2 (T) sources (S2 antisymmetrized S2 += S2.T afterward) and (T) energy
        S1_ia   = 2 (M3_abc - M3_bac) L_jkbc
        S2_ilab = -(2 X3 + Y3)_abc <jk|lc> + (2 X3 + Y3)_abc <dk|cb>
        ET      = t1_ia S1_ia + (4 t2 - 2 t2.swap)_ijab X2_ijab

    .. math::

        \begin{aligned}
        \mathrm{sym}(A)_{abc} &= 8 A_{abc} - 4 A_{bac} - 4 A_{acb} - 4 A_{cba} + 2 A_{bca} + 2 A_{cab},\quad X_3 = \mathrm{sym}(M_3),\ Y_3 = \mathrm{sym}(N_3) \\
        D^{(T)}_{aa} &= \tfrac{1}{2} M_{3,acd}(X_3+Y_3)_{acd}, \qquad D^{(T)}_{ii} = -\tfrac{1}{2} M_{3,abc}(X_3+Y_3)_{abc} \\
        D^{(T)}_{ia} &= (M_{3,abc} - M_{3,cba})(4 t^{bc}_{jk} - 2 t^{cb}_{jk}) \\
        \Gamma^{(T)}_{ijab} &= 4\, t^c_k Z_{3,abc}, \quad \Gamma^{(T)}_{jila} = -(2X_3+Y_3)_{abc} t^{bc}_{lk}, \quad \Gamma^{(T)}_{abdj} = (2X_3+Y_3)_{abc} t^{cd}_{ki} \\
        S^a_i &= 2 (M_{3,abc} - M_{3,bac}) L_{jkbc}
        \end{aligned}
    """
    dvv = np.zeros(nv, dtype=t2.dtype)   # diagonal of the (T) vir-vir 1-PDM (see below)
    doo = np.zeros(no, dtype=t2.dtype)   # diagonal of the (T) occ-occ 1-PDM; dtype-propagating for complex-step
    Dov = np.zeros((no,nv), dtype=t2.dtype)
    Goovv = np.zeros_like(t2)
    Gooov = np.zeros((no,no,no,nv), dtype=t2.dtype)
    Gvvvo = np.zeros((nv,nv,nv,no), dtype=t2.dtype)
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
                # response is the dependent-pair P_oo/P_vv in CCderiv.gradient), so contract
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


def dt3_density(o, v, no, nv, t1, t2, dt1, dt2, F, df, ERI, deri, L, dL, contract):
    r"""Analytic response of the spatial (closed-shell RHF) (T) intermediates to any real
    (Hermitian) perturbation (electric field, nuclear displacement, etc.) -- the product-rule
    derivative of :func:`t3_density` along ``(t1+dt1, t2+dt2, F+df, ERI+deri, L+dL)`` with
    **canonical** perturbed orbitals.  Returns ``d{Doo,Dvv,Dov,Goovv,Gooov,Gvvvo,S1,S2}``
    (the spatial route has no ``Gvovv``/``Govoo``).  Verified to ~1e-16 against a complex-step
    derivative of :func:`t3_density`.

    Notes
    -----
    Every :func:`t3_density` contraction is differentiated by the product rule.  The perturbed
    T3 batches use the differentiated numerators ``dNc``/``dNd`` (each contract of the
    :func:`t3c_ijk`/:func:`t3d_ijk` numerators with one factor replaced by its response, e.g.
    dW.t2 + W.dt2) and the differentiated denominator ``ddenom = d_x(f_ii+f_jj+f_kk -
    f_aa-f_bb-f_cc)``::

        dM3 = (dNc - M3 . ddenom) / D_ijkabc
        dN3 = (dNd - N3 . ddenom) / D_ijkabc

    .. math::

        \begin{aligned}
        dM_3 &= \left(dN^{c}_{ijk} - M_3\, d D^{abc}_{ijk}\right) / D^{abc}_{ijk}, \qquad
        dN_3 = \left(dN^{d}_{ijk} - N_3\, d D^{abc}_{ijk}\right) / D^{abc}_{ijk}
        \end{aligned}

    each intermediate of :func:`t3_density` then following as ``d(A B) = (dA) B + A (dB)`` with
    ``dX3 = sym(dM3)``, ``dY3 = sym(dN3)``.
    """
    ddvv = np.zeros(nv, dtype=t2.dtype); ddoo = np.zeros(no, dtype=t2.dtype)
    dDov = np.zeros((no,nv), dtype=t2.dtype)
    dGoovv = np.zeros_like(t2); dGooov = np.zeros((no,no,no,nv), dtype=t2.dtype)
    dGvvvo = np.zeros((nv,nv,nv,no), dtype=t2.dtype); dS1 = np.zeros_like(t1); dS2 = np.zeros_like(t2)

    Wvvvo, Wovoo, Woovv, Fov = ERI[v,v,v,o], ERI[o,v,o,o], ERI[o,o,v,v], F[o,v]
    dWvvvo, dWovoo, dWoovv, dFov = deri[v,v,v,o], deri[o,v,o,o], deri[o,o,v,v], df[o,v]
    occ, vir = diag(F)[o], diag(F)[v]
    docc, dvir = diag(df)[o], diag(df)[v]

    def dNc(i, j, k):   # product rule of the spatial t3c_ijk numerator (12 terms)
        def T(W, t): return (contract('bae,ce->abc', W[:,:,:,i], t[k,j]) + contract('cae,be->abc', W[:,:,:,i], t[j,k])
                             + contract('ace,be->abc', W[:,:,:,k], t[j,i]) + contract('bce,ae->abc', W[:,:,:,k], t[i,j])
                             + contract('cbe,ae->abc', W[:,:,:,j], t[i,k]) + contract('abe,ce->abc', W[:,:,:,j], t[k,i]))
        def U(W, t): return (contract('mc,mab->abc', W[:,:,j,k], t[i]) + contract('mb,mac->abc', W[:,:,k,j], t[i])
                             + contract('mb,mca->abc', W[:,:,i,j], t[k]) + contract('ma,mcb->abc', W[:,:,j,i], t[k])
                             + contract('ma,mbc->abc', W[:,:,k,i], t[j]) + contract('mc,mba->abc', W[:,:,i,k], t[j]))
        return (T(dWvvvo, t2) + T(Wvvvo, dt2)) - (U(dWovoo, t2) + U(Wovoo, dt2))

    def dNd(i, j, k):   # product rule of the spatial t3d_ijk numerator (6 terms)
        return (contract('ab,c->abc', dWoovv[i,j], t1[k]) + contract('ac,b->abc', dWoovv[i,k], t1[j]) + contract('bc,a->abc', dWoovv[j,k], t1[i])
                + contract('ab,c->abc', Woovv[i,j], dt1[k]) + contract('ac,b->abc', Woovv[i,k], dt1[j]) + contract('bc,a->abc', Woovv[j,k], dt1[i])
                + contract('ab,c->abc', dt2[i,j], Fov[k]) + contract('ac,b->abc', dt2[i,k], Fov[j]) + contract('bc,a->abc', dt2[j,k], Fov[i])
                + contract('ab,c->abc', t2[i,j], dFov[k]) + contract('ac,b->abc', t2[i,k], dFov[j]) + contract('bc,a->abc', t2[j,k], dFov[i]))

    def sym(A): return 8*A - 4*A.swapaxes(0,1) - 4*A.swapaxes(1,2) - 4*A.swapaxes(0,2) + 2*np.moveaxis(A,0,2) + 2*np.moveaxis(A,2,0)

    for i in range(no):
        for j in range(no):
            for k in range(no):
                M3 = t3c_ijk(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract, True)
                N3 = t3d_ijk(o, v, i, j, k, t1, t2, Woovv, F, contract, True)
                denom = occ[i] + occ[j] + occ[k] - (vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir)
                ddenom = docc[i] + docc[j] + docc[k] - (dvir.reshape(-1,1,1) + dvir.reshape(-1,1) + dvir)
                dM3 = (dNc(i,j,k) - M3*ddenom) / denom
                dN3 = (dNd(i,j,k) - N3*ddenom) / denom
                X3, Y3 = sym(M3), sym(N3); dX3, dY3 = sym(dM3), sym(dN3)
                ddvv += 0.5 * (contract('acd,acd->a', dM3, (X3+Y3)) + contract('acd,acd->a', M3, (dX3+dY3)))
                ddoo[i] -= 0.5 * (contract('abc,abc->', dM3, (X3+Y3)) + contract('abc,abc->', M3, (dX3+dY3)))
                dDov[i] += (contract('abc,bc->a', (dM3 - dM3.swapaxes(0,2)), (4*t2[j,k] - 2*t2[j,k].T))
                            + contract('abc,bc->a', (M3 - M3.swapaxes(0,2)), (4*dt2[j,k] - 2*dt2[j,k].T)))
                Z3 = 2*(M3 - M3.swapaxes(1,2)) - (M3.swapaxes(0,1) - np.moveaxis(M3,2,0))
                dZ3 = 2*(dM3 - dM3.swapaxes(1,2)) - (dM3.swapaxes(0,1) - np.moveaxis(dM3,2,0))
                dGoovv[i,j] += 4*(contract('c,abc->ab', dt1[k], Z3) + contract('c,abc->ab', t1[k], dZ3))
                dGooov[j,i] -= contract('abc,lbc->la', (2*dX3+dY3), t2[:,k]) + contract('abc,lbc->la', (2*X3+Y3), dt2[:,k])
                dGvvvo[:,:,:,j] += contract('abc,cd->abd', (2*dX3+dY3), t2[k,i]) + contract('abc,cd->abd', (2*X3+Y3), dt2[k,i])
                dS1[i] += (contract('abc,bc->a', 2*(dM3 - dM3.swapaxes(0,1)), L[o,o,v,v][j,k])
                           + contract('abc,bc->a', 2*(M3 - M3.swapaxes(0,1)), dL[o,o,v,v][j,k]))
                dS2[i] -= contract('abc,lc->lab', (2*dX3+dY3), ERI[o,o,o,v][j,k]) + contract('abc,lc->lab', (2*X3+Y3), deri[o,o,o,v][j,k])
                dS2[i,j] += contract('abc,dcb->ad', (2*dX3+dY3), ERI[o,v,v,v][k]) + contract('abc,dcb->ad', (2*X3+Y3), deri[o,v,v,v][k])

    dS2 = dS2 + dS2.swapaxes(0,1).swapaxes(2,3)
    return {'Doo': np.diag(ddoo), 'Dvv': np.diag(ddvv), 'Dov': dDov,
            'Goovv': dGoovv, 'Gooov': dGooov, 'Gvvvo': dGvvvo, 'S1': dS1, 'S2': dS2}


def so_t3_density(o, v, no, nv, t1, t2, F, ERI, contract):
    r"""(T) density/Lambda intermediates, spin-orbital (UHF/ROHF references).

    Returns (ET, intermediates), where ET is the (T) energy correction and
    intermediates is a dict of the (T) contributions {Doo, Dvv, Dov, Goovv, Gooov,
    Gvovv, Govoo, Gvvvo, S1, S2} that the caller caches on the wavefunction.

    Notes
    -----
    Built per-(i,j,k) from the connected T3 ``t3c`` (:func:`t3c_ijk_so`) and disconnected T3
    ``t3d`` (:func:`t3d_ijk_so`); a,b,c/d,e are virtual, repeated indices summed over the
    i,j,k loop.  The doubles intermediates ``x2``/``S2`` are antisymmetrized P(ij)P(ab) after
    the loop.  Full derivation in ``docs/ccsdt_t_density.tex`` and the appendix
    Table:ccsdt_1pdm/2pdm of ``docs/cc_gradients_orbital_response.tex``::

        # one-particle (T) increments (Doo/Dvv diagonal in the canonical basis)
        Dvv_aa =  1/12 (t3c + t3d)_abc t3c_abc
        Doo_ii = -1/12 t3c_abc (t3c + t3d)_abc
        Dov_ia =  1/4  t3c_ade t2_jkde

        # two-particle (T) increments
        Goovv_ijab =  t3c_abc t1_kc
        Gooov_ijla = -1/2 t2_klbc t3c_abc
        Gvovv_ciab =  1/2 t2_jkec t3c_abe
        Govoo_laij = -1/2 t2_klbc (t3c + t3d)_abc
        Gvvvo_abci =  1/2 (t3c + t3d)_abe t2_jkec

        # Lambda-1/2 (T) sources (S2 antisymmetrized P(ij)P(ab) afterward) and (T) energy
        S1_ia   = 1/4 t3c_abc <jk||bc>
        S2_ijab = P(ij) P(ab) [ -1/4 <jk||md> (2 t3c + t3d)_abd + 1/4 (2 t3c + t3d)_ade <bk||de> ]
        ET      = t1_ia S1_ia + 1/4 t2_ijab x2_ijab

    .. math::

        \begin{aligned}
        D^{(T)}_{aa} &= \tfrac{1}{12}\, (t_{3c}+t_{3d})_{abc}\, t_{3c,abc}, \qquad
        D^{(T)}_{ii} = -\tfrac{1}{12}\, t_{3c,abc}\, (t_{3c}+t_{3d})_{abc} \\
        D^{(T)}_{ia} &= \tfrac{1}{4}\, t_{3c,ade}\, t^{de}_{jk} \\
        \Gamma^{(T)}_{ijab} &= t_{3c,abc}\, t^c_k, \quad \Gamma^{(T)}_{ijla} = -\tfrac{1}{2}\, t^{bc}_{kl}\, t_{3c,abc}, \quad \Gamma^{(T)}_{ciab} = \tfrac{1}{2}\, t^{ec}_{jk}\, t_{3c,abe} \\
        \Gamma^{(T)}_{laij} &= -\tfrac{1}{2}\, t^{bc}_{kl}\, (t_{3c}+t_{3d})_{abc}, \quad \Gamma^{(T)}_{abci} = \tfrac{1}{2}\, (t_{3c}+t_{3d})_{abe}\, t^{ec}_{jk} \\
        S^a_i &= \tfrac{1}{4}\, t_{3c,abc}\, \langle jk||bc \rangle
        \end{aligned}
    """
    x2 = np.zeros_like(t2)
    dvv = np.zeros(nv, dtype=t2.dtype)          # dtype-propagating so complex-step works
    doo = np.zeros(no, dtype=t2.dtype)
    Dov = np.zeros((no,nv), dtype=t2.dtype)
    Goovv = np.zeros_like(t2)
    Gooov = np.zeros((no,no,no,nv), dtype=t2.dtype)
    Gvovv = np.zeros((nv,no,nv,nv), dtype=t2.dtype)
    Govoo = np.zeros((no,nv,no,no), dtype=t2.dtype)
    Gvvvo = np.zeros((nv,nv,nv,no), dtype=t2.dtype)
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
                # P_oo/P_vv in CCderiv (canonical perturbed MOs).  doo and dvv are the same
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


def so_dt3_density(o, v, no, nv, t1, t2, dt1, dt2, F, df, ERI, deri, contract):
    r"""Analytic response of the spin-orbital (T) intermediates to any real (Hermitian)
    perturbation (electric field, nuclear displacement, etc.) -- the product-rule
    derivative of :func:`so_t3_density` along ``(t1+dt1, t2+dt2, F+df, ERI+deri)`` with **canonical**
    perturbed orbitals (so the batched diagonal denominator's response is ``diag(df)``).  Returns
    ``d{Doo,Dvv,Dov,Goovv,Gooov,Gvovv,Govoo,Gvvvo,S1,S2}``.  Each per-ijk triple response is
    ``dt3 = (dN - t3 dD)/D``; every intermediate is then the product rule of its contraction.
    Verified to ~1e-16 against a complex-step derivative of :func:`so_t3_density`.

    Notes
    -----
    The perturbed connected/disconnected T3 batches (dN = the numerator of
    :func:`t3c_ijk_so`/:func:`t3d_ijk_so` differentiated by the product rule, one factor at a
    time; dD = the denominator response ``d_x(f_ii+f_jj+f_kk - f_aa-f_bb-f_cc) = diag(df)``)::

        dt3 = (dN - t3 dD) / D_ijkabc

    .. math::

        \begin{aligned}
        dt_3 = \left(dN_{ijk} - t_3\, dD^{abc}_{ijk}\right) / D^{abc}_{ijk}
        \end{aligned}

    and every :func:`so_t3_density` intermediate follows as ``d(A B) = (dA) B + A (dB)``.
    """
    ddvv = np.zeros(nv, dtype=t2.dtype); ddoo = np.zeros(no, dtype=t2.dtype)
    dDov = np.zeros((no,nv), dtype=t2.dtype)
    dGoovv = np.zeros_like(t2); dGooov = np.zeros((no,no,no,nv), dtype=t2.dtype)
    dGvovv = np.zeros((nv,no,nv,nv), dtype=t2.dtype); dGovoo = np.zeros((no,nv,no,no), dtype=t2.dtype)
    dGvvvo = np.zeros((nv,nv,nv,no), dtype=t2.dtype); dS1 = np.zeros_like(t1); dS2 = np.zeros_like(t2)

    Wvvvo, Wovoo, Woovv = ERI[v,v,v,o], ERI[o,v,o,o], ERI[o,o,v,v]
    Wvovv, Wooov, Fov = ERI[v,o,v,v], ERI[o,o,o,v], F[o,v]
    dWvvvo, dWovoo, dWoovv = deri[v,v,v,o], deri[o,v,o,o], deri[o,o,v,v]
    dWvovv, dWooov, dFov = deri[v,o,v,v], deri[o,o,o,v], df[o,v]
    occ, vir = diag(F)[o], diag(F)[v]
    docc, dvir = diag(df)[o], diag(df)[v]

    def dNc(i, j, k):   # product-rule derivative of the t3c numerator
        a = (contract('ad,bcd->abc', dt2[i,j], Wvvvo[:,:,:,k]) + contract('ad,bcd->abc', t2[i,j], dWvvvo[:,:,:,k])
             - contract('ad,bcd->abc', dt2[k,j], Wvvvo[:,:,:,i]) - contract('ad,bcd->abc', t2[k,j], dWvvvo[:,:,:,i])
             - contract('ad,bcd->abc', dt2[i,k], Wvvvo[:,:,:,j]) - contract('ad,bcd->abc', t2[i,k], dWvvvo[:,:,:,j]))
        t3 = a - a.swapaxes(0,1) - a.swapaxes(0,2)
        a = (contract('lab,lc->abc', dt2[i], Wovoo[:,:,j,k]) + contract('lab,lc->abc', t2[i], dWovoo[:,:,j,k])
             - contract('lab,lc->abc', dt2[j], Wovoo[:,:,i,k]) - contract('lab,lc->abc', t2[j], dWovoo[:,:,i,k])
             - contract('lab,lc->abc', dt2[k], Wovoo[:,:,j,i]) - contract('lab,lc->abc', t2[k], dWovoo[:,:,j,i]))
        return t3 - (a - a.swapaxes(0,2) - a.swapaxes(1,2))

    def dNd(i, j, k):   # product-rule derivative of the t3d numerator
        a = (contract('a,bc->abc', dt1[i], Woovv[j,k]) + contract('a,bc->abc', t1[i], dWoovv[j,k])
             - contract('a,bc->abc', dt1[j], Woovv[i,k]) - contract('a,bc->abc', t1[j], dWoovv[i,k])
             - contract('a,bc->abc', dt1[k], Woovv[j,i]) - contract('a,bc->abc', t1[k], dWoovv[j,i])
             + contract('a,bc->abc', dFov[i], t2[j,k]) + contract('a,bc->abc', Fov[i], dt2[j,k])
             - contract('a,bc->abc', dFov[j], t2[i,k]) - contract('a,bc->abc', Fov[j], dt2[i,k])
             - contract('a,bc->abc', dFov[k], t2[j,i]) - contract('a,bc->abc', Fov[k], dt2[j,i]))
        return a - a.swapaxes(0,1) - a.swapaxes(0,2)

    for i in range(no):
        for j in range(no):
            for k in range(no):
                t3c = t3c_ijk_so(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                t3d = t3d_ijk_so(o, v, i, j, k, t1, t2, Woovv, F, contract)
                denom = occ[i] + occ[j] + occ[k] - (vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir)
                ddenom = docc[i] + docc[j] + docc[k] - (dvir.reshape(-1,1,1) + dvir.reshape(-1,1) + dvir)
                dt3c = (dNc(i,j,k) - t3c*ddenom) / denom
                dt3d = (dNd(i,j,k) - t3d*ddenom) / denom
                ddvv += (1/12) * (contract('abc,abc->a', (dt3c+dt3d), t3c) + contract('abc,abc->a', (t3c+t3d), dt3c))
                ddoo[i] -= (1/12) * (contract('abc,abc->', dt3c, (t3c+t3d)) + contract('abc,abc->', t3c, (dt3c+dt3d)))
                dDov[i] += (1/4) * (contract('ade,de->a', dt3c, t2[j,k]) + contract('ade,de->a', t3c, dt2[j,k]))
                dGoovv[i,j] += contract('abc,c->ab', dt3c, t1[k]) + contract('abc,c->ab', t3c, dt1[k])
                dGooov[i,j] -= (1/2) * (contract('kbc,abc->ka', dt2[k], t3c) + contract('kbc,abc->ka', t2[k], dt3c))
                dGvovv[:,i] += (1/2) * (contract('ec,abe->cab', dt2[j,k], t3c) + contract('ec,abe->cab', t2[j,k], dt3c))
                dGovoo[:,:,i,j] -= (1/2) * (contract('kbc,abc->ka', dt2[k], (t3c+t3d)) + contract('kbc,abc->ka', t2[k], (dt3c+dt3d)))
                dGvvvo[:,:,:,i] += (1/2) * (contract('abe,ec->abc', (dt3c+dt3d), t2[j,k]) + contract('abe,ec->abc', (t3c+t3d), dt2[j,k]))
                dS1[i] += (1/4) * (contract('abc,bc->a', dt3c, Woovv[j,k]) + contract('abc,bc->a', t3c, dWoovv[j,k]))
                dS2[i] -= (1/4) * (contract('md,abd->mab', dWooov[j,k], (2*t3c+t3d)) + contract('md,abd->mab', Wooov[j,k], (2*dt3c+dt3d)))
                dS2[i,j] += (1/4) * (contract('ade,bde->ab', (2*dt3c+dt3d), Wvovv[:,k]) + contract('ade,bde->ab', (2*t3c+t3d), dWvovv[:,k]))

    dS2 = dS2 - dS2.swapaxes(0,1) - dS2.swapaxes(2,3) + dS2.swapaxes(0,1).swapaxes(2,3)
    return {'Doo': np.diag(ddoo), 'Dvv': np.diag(ddvv), 'Dov': dDov,
            'Goovv': dGoovv, 'Gooov': dGooov, 'Gvovv': dGvovv,
            'Govoo': dGovoo, 'Gvvvo': dGvvvo, 'S1': dS1, 'S2': dS2}
