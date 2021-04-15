from .hbar_eqs import build_Hov, build_Hvv, build_Hoo
from .hbar_eqs import build_Hoooo, build_Hvvvv, build_Hvovv, build_Hooov
from .hbar_eqs import build_Hovvo, build_Hovov, build_Hvvvo, build_Hovoo


class cchbar(object):
    def __init__(self, ccwfn):
        o = ccwfn.o
        v = ccwfn.v
        F = ccwfn.F
        ERI = ccwfn.ERI
        L = ccwfn.L
        t1 = ccwfn.t1
        t2 = ccwfn.t2

        self.Hov = build_Hov(o, v, F, L, t1)
        self.Hvv = build_Hvv(o, v, F, L, t1, t2)
        self.Hoo = build_Hoo(o, v, F, L, t1, t2)
        self.Hoooo = build_Hoooo(o, v, ERI, t1, t2)
        self.Hvvvv = build_Hvvvv(o, v, ERI, t1, t2)
        self.Hvovv = build_Hvovv(o, v, ERI, t1)
        self.Hooov = build_Hooov(o, v, ERI, t1)
        self.Hovvo = build_Hovvo(o, v, ERI, L, t1, t2)
        self.Hovov = build_Hovov(o, v, ERI, t1, t2)
        self.Hvvvo = build_Hvvvo(o, v, F, ERI, L, self.Hov, self.Hvvvv, t1, t2)
        self.Hovoo = build_Hovoo(o, v, F, ERI, L, self.Hov, self.Hoooo, t1, t2)
