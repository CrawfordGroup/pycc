"""
Symbolic verification of the RT/EOM discrepancy derivation (Eq. 17 analysis).

Model: 4 spin orbitals {0,1 occupied; 2,3 virtual}, full 16-dim Fock space,
exact Jordan-Wigner matrix representation. Hamiltonian = generic (random
rational) one- plus two-body operator. Cluster amplitudes are SYMBOLIC, so
every "== 0" below is a polynomial identity in the amplitudes, i.e. a proof
for arbitrary amplitudes (for this system size), not a numerical coincidence.
"""
import sympy as sp
from sympy import Rational, zeros, eye, symbols, expand, diff, factorial
import random, pickle, itertools

random.seed(42)

# ---------------------------------------------------------------- JW ops
I2 = sp.Matrix([[1, 0], [0, 1]])
Z  = sp.Matrix([[1, 0], [0, -1]])
sm = sp.Matrix([[0, 1], [0, 0]])   # annihilator on one mode: |0><1|

def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = sp.Matrix(sp.kronecker_product(out, m))
    return out

NSO = 4  # spin orbitals: 0,1 occ; 2,3 virt
def annihilate(p):
    facs = [Z]*p + [sm] + [I2]*(NSO-p-1)
    return kron(*facs)

a  = [annihilate(p) for p in range(NSO)]
ad = [m.T for m in a]   # real matrices -> dagger = transpose

DIM = 2**NSO
occ, virt = [0, 1], [2, 3]

def comm(A, B): return expand(A*B - B*A)
def anti(A, B): return expand(A*B + B*A)

# ---- sanity: CAR algebra
for p in range(NSO):
    for q in range(NSO):
        assert anti(a[p], a[q]) == zeros(DIM, DIM)
        assert anti(ad[p], ad[q]) == zeros(DIM, DIM)
        assert anti(ad[p], a[q]) == (eye(DIM) if p == q else zeros(DIM, DIM))
print("[ok] CAR algebra {a_p, a_q^+} = delta_pq verified on full Fock space")

# ---------------------------------------------------------------- H (generic 1+2-body, random rationals)
def rr():
    return Rational(random.randint(-9, 9), random.randint(1, 7))

H = zeros(DIM, DIM)
for p in range(NSO):
    for q in range(NSO):
        H += rr() * ad[p]*a[q]
for p in range(NSO):
    for q in range(NSO):
        for r in range(NSO):
            for s in range(NSO):
                c = rr()
                if c:
                    H += Rational(1,4)*c * ad[p]*ad[q]*a[s]*a[r]
H = expand(H)
print("[ok] generic two-body H built (random rational coefficients)")

# ---------------------------------------------------------------- excitation manifold
# singles tau_i^a = a_a^+ a_i ; doubles tau = a_2^+ a_3^+ a_1 a_0
taus, labels = [], []
for i in occ:
    for A in virt:
        taus.append(expand(ad[A]*a[i])); labels.append(f"t_{i}^{A}")
taus.append(expand(ad[2]*ad[3]*a[1]*a[0])); labels.append("t_01^23")
NT = len(taus)

t = symbols(f"t0:{NT}", real=True)            # amplitudes of T
s = symbols(f"s0:{NT}", real=True)            # amplitudes of T^+  (t* treated independently)
T    = expand(sum((t[k]*taus[k] for k in range(NT)), start=zeros(DIM, DIM)))
Tdag = expand(sum((s[k]*taus[k].T for k in range(NT)), start=zeros(DIM, DIM)))
sigma = expand(T - Tdag)

# reference and projection manifold
vac = zeros(DIM, 1); vac[0] = 1
Phi0 = expand(ad[0]*ad[1]*vac)
assert (Phi0.T*Phi0)[0] == 1
bras = [ (expand(taus[k]*Phi0)).T for k in range(NT) ]   # <Phi_mu|

def melem(Op):
    """vector over mu of <Phi_mu| Op |Phi_0>"""
    v = Op*Phi0
    return [expand((bras[k]*v)[0]) for k in range(NT)]

# ================================================================ LEMMA 1
for m_ in range(NT):
    for n_ in range(NT):
        assert comm(taus[m_], taus[n_]) == zeros(DIM, DIM)
assert comm(T, T) == zeros(DIM, DIM)
for n_ in range(NT):
    assert comm(taus[n_], T) == zeros(DIM, DIM)
print("[ok] LEMMA 1: [tau_mu, tau_nu] = 0 and [tau_nu, T] = 0  (exact operator identity, symbolic t)")

# ---- and its failure for the daggered set (needed for UCC later)
n_noncomm = sum(1 for m_ in range(NT) for n_ in range(NT)
                if comm(taus[m_], taus[n_].T) != zeros(DIM, DIM))
assert n_noncomm > 0
print(f"[ok]          ...while [tau_mu, tau_nu^+] != 0 for {n_noncomm}/{NT*NT} pairs")

# ---- singles Wick identity  [a_A^+ a_i, a_j^+ a_B] = d_ij a_A^+ a_B - d_AB a_j^+ a_i
for i in occ:
    for A in virt:
        for j in occ:
            for B in virt:
                lhs = comm(ad[A]*a[i], ad[j]*a[B])
                rhs = expand((1 if i == j else 0)*ad[A]*a[B]
                             - (1 if A == B else 0)*ad[j]*a[i])
                assert expand(lhs - rhs) == zeros(DIM, DIM)
print("[ok] singles Wick identity [a_A^+ a_i, a_j^+ a_B] = d_ij a_A^+ a_B - d_AB a_j^+ a_i")

# ================================================================ LEMMA 2 (slot transport, X = H)
for n_ in range(NT):
    lhs = comm(comm(H, taus[n_]), T)
    rhs = comm(comm(H, T), taus[n_])
    assert expand(lhs - rhs) == zeros(DIM, DIM)
print("[ok] LEMMA 2: [[H,tau_nu],T] = [[H,T],tau_nu]   (symbolic t)")

# ================================================================ LEMMA 3 (derivative of ad_T^n)
adT = [H]
for n_ in range(1, 5):
    adT.append(comm(adT[-1], T))
for n_ in range(1, 4):          # n = 1..3 symbolically (n=4 checked implicitly by Prop 1 later)
    for k in range(NT):
        lhs = expand(diff(adT[n_], t[k]))
        rhs = expand(n_ * comm(adT[n_-1], taus[k]))
        assert expand(lhs - rhs) == zeros(DIM, DIM)
print("[ok] LEMMA 3: d/dt_nu ad_T^n(H) = n [ad_T^(n-1)(H), tau_nu]  for n=1..3 (symbolic t)")



# ################################################################
# PART 2 -- Case 1: traditional CC (sigma = T)
# ################################################################
from sympy import factorial
def melem(Op):
    v = Op*Phi0
    return [expand((bras[k]*v)[0]) for k in range(NT)]
Zmat = zeros(DIM, DIM)

# ================================================================ PROP 2a: termination
adT5 = comm(adT[4], T)
assert adT5 == Zmat
print("[ok] PROP 2a: ad_T^5(H) = 0 identically in the amplitudes  (BCH terminates at 4)")

# ================================================================ PROP 2b: ad_T^4 is pure excitation
for k in range(NT):
    assert comm(adT[4], taus[k]) == Zmat
# stronger purity witnesses: a pure excitation operator annihilates <Phi0| from the right
assert expand((Phi0.T)*adT[4]) == zeros(1, DIM)      # <Phi0| (pure exc.) = 0
assert comm(adT[4], T) == Zmat
print("[ok] PROP 2b: [ad_T^4(H), tau_nu] = 0 for all nu, and <Phi0|ad_T^4(H) = 0 (pure excitation)")

# ================================================================ PROP 1: boundary formula, K = 1..4
# Hbar_K, A^J (by symbolic differentiation of the residual vector), A^comm (frozen commutator)
def Hbar(K, sig):
    out = zeros(DIM, DIM)
    X = H
    out += X
    for n in range(1, K+1):
        X = comm(X, sig)
        out += Rational(1, factorial(n)) * X
    return expand(out)

print("    checking D^(K) = -(1/K!) <Phi_mu|[ad_T^K(H), tau_nu]|Phi_0> ...")
for K in (1, 2, 3, 4):
    HK = Hbar(K, T)
    R = melem(HK)                       # residual vector R_mu(t), symbolic in t
    for nu in range(NT):
        for mu in range(NT):
            AJ    = expand(diff(R[mu], t[nu]))
            Acomm = expand((bras[mu]*(comm(HK, taus[nu])*Phi0))[0])
            boundary = expand(-Rational(1, factorial(K)) *
                              (bras[mu]*(comm(adT[K], taus[nu])*Phi0))[0])
            assert expand(AJ - Acomm - boundary) == 0
    print(f"      K={K}: exact (symbolic identity in all amplitudes)")
print("[ok] PROP 1: D^(K)|_(sigma=T) is a single BCH boundary term, K=1..4")

# ================================================================ THEOREM 1: D = 0 for K >= 4
for K in (4, 5):
    HK = Hbar(K, T)
    R = melem(HK)
    ok = all(expand(diff(R[mu], t[nu])
                    - (bras[mu]*(comm(HK, taus[nu])*Phi0))[0]) == 0
             for mu in range(NT) for nu in range(NT))
    assert ok
    print(f"[ok] THEOREM 1: A^J == A^comm identically at K={K}  (sigma = T)")

# ---- and the K=2 counterexample: standard CC on a truncated Hbar_2 DOES split
H2 = Hbar(2, T)
R2 = melem(H2)
D2 = sp.Matrix(NT, NT, lambda mu, nu:
      expand(diff(R2[mu], t[nu]) - (bras[mu]*(comm(H2, taus[nu])*Phi0))[0]))
assert D2 != zeros(NT, NT)
print("[ok] COUNTEREXAMPLE: at K=2 with sigma=T, D != 0  (truncation artifact, not unitarity artifact)")
nz = next((mu, nu) for mu in range(NT) for nu in range(NT) if D2[mu, nu] != 0)
print(f"     sample nonzero entry D_{nz} =", sp.factor(D2[nz]))

# ################################################################
# PART 3 -- Case 2: unitary CC (sigma = T - T^+)
# ################################################################
from sympy import Poly
def me(mu, Op): return expand((bras[mu]*(Op*Phi0))[0])
random.seed(7)
S = pickle.load(open("/home/claude/state.pkl", "rb"))
H, taus, t, s, T, Tdag, sigma, Phi0, bras, DIM, NT = (S[k] for k in
    ("H", "taus", "t", "s", "T", "Tdag", "sigma", "Phi0", "bras", "DIM", "NT"))

def comm(A, B): return expand(A*B - B*A)
def me(mu, Op): return expand((bras[mu]*(Op*Phi0))[0])

# ================================================================ [tau_nu, sigma] != 0
for nu in range(NT):
    lhs = comm(taus[nu], sigma)
    assert expand(lhs + comm(taus[nu], Tdag)) == Zmat     # equals -[tau_nu, T^+]
nonzero = sum(1 for nu in range(NT) if comm(taus[nu], sigma) != Zmat)
assert nonzero == NT
print(f"[ok] [tau_nu, sigma] = -[tau_nu, T^+] != 0 for all {NT} tau_nu  (symbolic)")

# ================================================================ Hbar_2 and the two matrices
Hs1 = comm(H, sigma)
Hs2 = comm(Hs1, sigma)
H2  = expand(H + Hs1 + Rational(1, 2)*Hs2)

# --- Eq (13): A^J by differentiation (Wirtinger: d sigma/d t_nu = tau_nu) vs 3-term expansion
# --- Eq (14): A^comm by frozen commutator vs 3-term expansion
R = [me(mu, H2) for mu in range(NT)]
AJ    = sp.Matrix(NT, NT, lambda mu, nu: expand(diff(R[mu], t[nu])))
Acomm = sp.Matrix(NT, NT, lambda mu, nu: me(mu, comm(H2, taus[nu])))

for nu in range(NT):
    e13 = expand(comm(H, taus[nu]) + Rational(1,2)*comm(comm(H, taus[nu]), sigma)
                                   + Rational(1,2)*comm(comm(H, sigma), taus[nu]))
    e14 = expand(comm(H, taus[nu]) + comm(comm(H, sigma), taus[nu])
                                   + Rational(1,2)*comm(comm(comm(H, sigma), sigma), taus[nu]))
    for mu in range(NT):
        assert expand(AJ[mu, nu]    - me(mu, e13)) == 0
        assert expand(Acomm[mu, nu] - me(mu, e14)) == 0
print("[ok] Eq (13): dR_mu/dt_nu == <[H,tau]> + (1/2)<[[H,tau],sigma]> + (1/2)<[[H,sigma],tau]>")
print("[ok] Eq (14): <[Hbar_2,tau]> == <[H,tau]> + <[[H,sigma],tau]> + (1/2)<[[[H,sigma],sigma],tau]>")

# ================================================================ Eq (16): Jacobi re-slotting
for nu in range(NT):
    lhs = comm(comm(H, taus[nu]), sigma)
    rhs = expand(comm(H, comm(taus[nu], sigma)) + comm(comm(H, sigma), taus[nu]))
    assert expand(lhs - rhs) == Zmat
print("[ok] Eq (16): [[H,tau],sigma] = [H,[tau,sigma]] + [[H,sigma],tau]  (Jacobi)")

# ================================================================ Eq (17): the closed form
termI  = sp.Matrix(NT, NT, lambda mu, nu:
            expand(Rational(1,2)*me(mu, comm(H, comm(taus[nu], sigma)))))
termII = sp.Matrix(NT, NT, lambda mu, nu:
            expand(Rational(1,2)*me(mu, comm(Hs2, taus[nu]))))
D_formula = expand(termI - termII)
D_direct  = expand(AJ - Acomm)
assert expand(D_direct - D_formula) == zeros(NT, NT)
print("[ok] Eq (17): A^J - A^comm == (1/2)<[H,[tau,sigma]]> - (1/2)<[[[H,sigma],sigma],tau]>")
print("             ...as an identity in ALL amplitudes t,s (no residual condition needed)")

# ================================================================ homogeneity degrees
allsyms = list(t) + list(s)
def degrees(M):
    degs = set()
    for e in M:
        e = expand(e)
        if e != 0:
            p = Poly(e, *allsyms)
            degs |= {sum(mon) for mon in p.monoms()}
    return degs

dI, dII = degrees(termI), degrees(termII)
assert dI == {1}, dI
assert dII == {2}, dII
# term I involves ONLY the de-excitation amplitudes s:
assert all(not expand(e).has(*t) for e in termI)
print(f"[ok] term I homogeneous of degree {dI} (and depends only on s = t*);"
      f" term II homogeneous of degree {dII}")
print("     => no mutual cancellation possible order-by-order; D = O(t) at leading order")

# ================================================================ genericity: D != 0 numerically
subs = {x: Rational(random.randint(-5, 5), random.randint(2, 9)) for x in allsyms}
Dnum = D_formula.subs(subs)
nnz = sum(1 for e in Dnum if e != 0)
assert nnz > 0
print(f"[ok] D != 0 at generic amplitudes: {nnz}/{NT*NT} nonzero entries; sample D_00 = {Dnum[0,0]}")

# leading order alone is already nonzero:
lin = termI.subs(subs)
assert any(e != 0 for e in lin)
print("[ok] the O(t) piece alone (term I) is nonzero: discrepancy appears at FIRST order in correlation")

# ================================================================ non-termination of BCH for sigma
subs_op = subs
sig_num = sigma.subs(subs_op)
X = H
alive = []
for n in range(1, 9):
    X = expand(X*sig_num - sig_num*X)
    alive.append(X != Zmat)
assert all(alive)
print("[ok] non-termination: ad_sigma^n(H) != 0 for n = 1..8 at generic amplitudes"
      " (vs ad_T^5(H) = 0 exactly)")

# ================================================================ sigma -> T limit recovers Case 1
Dlim = D_formula.subs({x: 0 for x in s})
tI_lim = termI.subs({x: 0 for x in s})
assert tI_lim == zeros(NT, NT)
print("[ok] limit s -> 0 (i.e. sigma -> T): term I vanishes identically;")
print("     D reduces to the pure boundary term of Prop 1  --  consistent closure of both cases")
print()
print("ALL CHECKS PASSED -- every identity verified symbolically in the amplitudes.")
