# We will assume that the ccwfn and cclambda objects already
# contain the t=0 amplitudes we need for the initial step


class rtcc(object):
    def __init__(self, ccwfn, cclambda, field, method):
        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.field = field
        self.method = method

    def __call__(self, t):

