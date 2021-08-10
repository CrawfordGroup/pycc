# Triples class for (T) corrections, CC3, etc.

import numpy as np
from opt_einsum import contract


class cctriples(object):

    def __init__(self, ccwfn):
        self.ccwfn = ccwfn

    # Vikings' formulation
    def t_vikings(self):
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        L = self.ccwfn.H.L
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2
        X1 = np.zeros_like(self.ccwfn.t1)
        X2 = np.zeros_like(self.ccwfn.t2)

        for i in range(no):
            for j in range(no):
                for k in range(no):
                    t3 = self.t3c_ijk(o, v, i, j, k, t2, ERI, F)

                    X1[i] += contract('abc,bc->a',(t3 - t3.swapaxes(0,2)), L[j,k,v,v])
                    X2[i,j] += contract('abc,c->ab',(t3 - t3.swapaxes(0,2)), F[k,v])
                    X2[i,j] += contract('abc,dbc->ad', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[v,k,v,v])
                    X2[i] -= contract('abc,lc->lab', (2.0*t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)),ERI[j,k,o,v])

        ET = 2.0 * contract('ia,ia->', t1, X1)
        ET += contract('ijab,ijab->', (4.0*t2 - 2.0*t2.swapaxes(2,3)), X2)

        return ET

    # Vikings' formulation – inverted algorithm
    def t_vikings_inverted(self):

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
                    t3 = self.t3c_abc(o, v, a, b, c, t2, ERI, F, True)

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
                    W3 = self.t3c_ijk(o, v, i, j, k, t2, ERI, F, False)
                    V3 = self.t3d_ijk(o, v, i, j, k, t1, t2, ERI, F, False) + W3

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


    # Various RHF triples formulations; useful for (T) corrections and CC3
    def t3c_ijk(self, o, v, i, j, k, t2, ERI, F, WithDenom=True):

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


    def t3c_abc(self, o, v, a, b, c, t2, ERI, F, WithDenom=True):
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


    def t3d_ijk(self, o, v, i, j, k, t1, t2, ERI, F, WithDenom=True):
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
