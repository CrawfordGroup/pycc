# Triples class for (T) corrections, CC3, etc.

# Unfinished (Zhe, 0407)

import numpy as np
import torch


class cctriples(object):

    def __init__(self, ccwfn):
        self.ccwfn = ccwfn
        self.contract = self.ccwfn.contract

    # Vikings' formulation
    def t_vikings(self):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        L = self.ccwfn.H.L
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2
        if isinstance(t1, torch.Tensor):
            X1 = torch.zeros_like(self.ccwfn.t1)
            X2 = torch.zeros_like(self.ccwfn.t2)
        else:
            X1 = np.zeros_like(self.ccwfn.t1)
            X2 = np.zeros_like(self.ccwfn.t2)

        for i in range(no):
            for j in range(no):
                for k in range(no):
                    t3 = self.t3c_ijk(o, v, i, j, k, t2, ERI[v,v,v,o], ERI[o,v,o,o], F)

                    X1[i] += contract('abc,bc->a',(t3 - t3.swapaxes(0,2)), L[j,k,v,v])
                    X2[i,j] += contract('abc,c->ab',(t3 - t3.swapaxes(0,2)), F[k,v])
                    X2[i,j] += contract('abc,dbc->ad', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[v,k,v,v])
                    X2[i] -= contract('abc,lc->lab', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[j,k,o,v])

        ET = 2.0 * contract('ia,ia->', t1, X1)
        ET += contract('ijab,ijab->', (4.0*t2 - 2.0*t2.swapaxes(2,3)), X2)

        return ET

    # Vikings' formulation – inverted algorithm
    def t_vikings_inverted(self):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccwfn.nv
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        L = self.ccwfn.H.L
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2
        X1 = np.zeros_like(t1.T)
        X2 = np.zeros_like(t2.T)

        for a in range(nv):
            for b in range(nv):
                for c in range(nv):
                    t3 = self.t3c_abc(o, v, a, b, c, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, True)

                    X1[a] += contract('ijk,jk->i',(t3 - t3.swapaxes(0,2)), L[o,o,b+no,c+no])
                    X2[a,b] += contract('ijk,k->ij',(t3 - t3.swapaxes(0,2)), F[o,c+no])
                    X2[a] += contract('ijk,dk->dij', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[v,o,b+no,c+no])
                    X2[a,b] -= contract('ijk,jkl->il', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[o,o,o,c+no])

        ET = 2.0 * contract('ia,ia->', t1, X1.T)
        ET += contract('ijab,ijab->', (4.0*t2 - 2.0*t2.swapaxes(2,3)), X2.T)

        return ET

    # Lee and Rendell's formulation
    def t_tjl(self):
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccwfn.nv
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2

        ET = 0.0
        for i in range(no):
            for j in range(i+1):
                for k in range(j+1):
                    W3 = self.t3c_ijk(o, v, i, j, k, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, False)
                    V3 = self.t3d_ijk(o, v, i, j, k, t1, t2, ERI[o,o,v,v], F, False) + W3

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

    # Various triples formulations; useful for (T) corrections and CC3

    def t3c_ijk(self, o, v, i, j, k, t2, Wvvvo, Wovoo, F, WithDenom=True):
        contract = self.contract 
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


    def t3c_abc(self, o, v, a, b, c, t2, Wvvvo, Wovoo, F, WithDenom=True):
        contract = self.contract
        no = o.stop

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
            Fo = np.diag(F)[o]
            denom = np.zeros_like(t3)
            denom += Fo.reshape(-1,1,1) + Fo.reshape(-1,1) + Fo
            denom -= F[a+no,a+no] + F[b+no,b+no] + F[c+no,c+no]
            return t3/denom
        else:
            return t3


    def t3d_ijk(self, o, v, i, j, k, t1, t2, Woovv, F, WithDenom=True):
        contract = self.contract
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
