"""Backend-aware (NumPy / torch) tensor helpers and a DIIS extrapolator, shared across pycc.

The free functions dispatch on the argument's runtime type -- a ``torch.Tensor`` routes to the torch
op, anything else to the NumPy op -- so the wavefunction / response / real-time code runs unchanged
on either backend, carrying the dtype/precision and (for torch) the device along automatically.
:class:`helper_diis` accelerates the amplitude iterations (CC, Lambda, response) with Pulay's DIIS.
"""
import numpy as np
from pycc.ccwfn import HAS_TORCH
if HAS_TORCH:
    import torch


def zeros_like(a):
    """Backend-aware ``zeros_like``: ``torch.zeros_like`` for a torch tensor,
    else ``np.zeros_like``.

    Collapses the ``if HAS_TORCH and isinstance(a, torch.Tensor): ... else: ...``
    allocation branch that recurs throughout the CC modules into a single call.
    """
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.zeros_like(a)
    return np.zeros_like(a)


def permute_triples(ijkabc, perm_ijk, perm_abc):
    r"""Antisymmetric permutation operator ``P(perm_ijk) P(perm_abc)`` on a connected six-index
    triples tensor ``ijkabc`` -- e.g. ``P(k/ij) P(a/bc)`` expanded to nine signed swapaxes terms::

        P(k/ij) = 1 - P(ik) - P(jk),   P(a/bc) = 1 - P(ab) - P(ac)
        result  = P(k/ij) P(a/bc) . ijkabc

    .. math::

        \begin{aligned}
        \mathcal{P}(k/ij) &= 1 - P(ik) - P(jk) \\
        \mathcal{P}(a/bc) &= 1 - P(ab) - P(ac) \\
        t' &= \mathcal{P}(k/ij)\,\mathcal{P}(a/bc)\, t
        \end{aligned}

    ``perm_ijk``/``perm_abc`` are strings like ``'k/ij'`` / ``'a/bc'`` denoting the one-vs-pair
    antisymmetrizers.  Backend-agnostic (only ``swapaxes``).  Used by the full-array (store_triples)
    CC3 T3/L3/X3 builds; the batched per-ijk builders fold this antisymmetry into their explicit
    index terms instead. Port of socc utils.permute_triples."""
    idx = {'i': 0, 'j': 1, 'k': 2, 'a': 3, 'b': 4, 'c': 5}

    char_list = list(perm_ijk)
    if char_list[1] != '/':
        raise Exception('String format must be, e.g., i/jk.')
    char = [char_list[0], char_list[2], char_list[3]]
    if set(char).issubset(idx) is False:
        raise Exception('Only allowed indices are i, j, k or a, b, c.')
    ijk_perm1 = [idx[char[0]], idx[char[1]]]
    ijk_perm2 = [idx[char[0]], idx[char[2]]]

    char_list = list(perm_abc)
    if char_list[1] != '/':
        raise Exception('String format must be, e.g., c/ab.')
    char = [char_list[0], char_list[2], char_list[3]]
    if set(char).issubset(idx) is False:
        raise Exception('Only allowed indices are i, j, k or a, b, c.')
    abc_perm1 = [idx[char[0]], idx[char[1]]]
    abc_perm2 = [idx[char[0]], idx[char[2]]]

    t3 = (ijkabc
          - ijkabc.swapaxes(abc_perm1[0], abc_perm1[1])
          - ijkabc.swapaxes(abc_perm2[0], abc_perm2[1])
          - ijkabc.swapaxes(ijk_perm1[0], ijk_perm1[1])
          + ijkabc.swapaxes(ijk_perm1[0], ijk_perm1[1]).swapaxes(abc_perm1[0], abc_perm1[1])
          + ijkabc.swapaxes(ijk_perm1[0], ijk_perm1[1]).swapaxes(abc_perm2[0], abc_perm2[1])
          - ijkabc.swapaxes(ijk_perm2[0], ijk_perm2[1])
          + ijkabc.swapaxes(ijk_perm2[0], ijk_perm2[1]).swapaxes(abc_perm1[0], abc_perm1[1])
          + ijkabc.swapaxes(ijk_perm2[0], ijk_perm2[1]).swapaxes(abc_perm2[0], abc_perm2[1]))

    return t3


def zeros(shape, like):
    """Allocate a zero tensor of ``shape`` matching ``like``'s backend/dtype/device.

    Dispatches on the runtime type of ``like`` (NumPy vs torch) and inherits its
    dtype — and, for torch, its device — so single/mixed precision and GPU
    placement ride along automatically. Use this in place of the
    ``zeros_like(x)`` + ``np.pad``/``torch.nn.functional.pad`` idiom when the
    target shape mixes occupied and virtual dimensions and so is not
    ``zeros_like`` of any single amplitude array.
    """
    if HAS_TORCH and isinstance(like, torch.Tensor):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)
    return np.zeros(shape, dtype=like.dtype)


def diag(a):
    """Backend-aware ``diag``: ``torch.diag`` for a torch tensor, else ``np.diag``.

    Collapses the ``if HAS_TORCH and isinstance(a, torch.Tensor): torch.diag(a)
    else: np.diag(a)`` branch (paired with ``zeros_like`` in the triples-denominator
    construction throughout cctriples).
    """
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.diag(a)
    return np.diag(a)


def clone(a, device=None):
    """Backend-aware deep copy: ``a.clone()`` for a torch tensor (optionally moved
    to ``device``), else ``a.copy()``.

    Collapses the ``if HAS_TORCH and isinstance(a, torch.Tensor): a.clone()
    [.to(self.device1)] else: a.copy()`` branch that recurs throughout the CC
    modules. ``device`` is ignored for NumPy arrays (always CPU).
    """
    if HAS_TORCH and isinstance(a, torch.Tensor):
        a = a.clone()
        return a.to(device) if device is not None else a
    return a.copy()


def real_zeros(shape, like):
    """Allocate a real-dtype zero tensor of ``shape`` on ``like``'s backend/device.

    Uses the *real-component* dtype of ``like`` (``like.real.dtype``), so a complex
    amplitude yields a real buffer of the matching precision (complex128->float64,
    complex64->float32) while a real amplitude is unchanged. Used for the DIIS B
    matrix / residual vector, which must stay real even when the amplitudes are
    complex (e.g. frequency-dependent response in ``ccresponse``); precision and,
    for torch, device placement ride along from the data.
    """
    if HAS_TORCH and isinstance(like, torch.Tensor):
        return torch.zeros(shape, dtype=like.real.dtype, device=like.device)
    return np.zeros(shape, dtype=like.real.dtype)


def dot(a, b):
    """Backend-aware 1-D dot product: ``torch.dot`` for torch tensors, else ``np.dot``.

    torch (unlike NumPy) refuses a real<->complex dot, so when the operands mix the
    real one is upcast to the other's complex dtype. This lets real ground-state
    integrals dot with complex RT densities (e.g. in rtcc's dipole/energy)."""
    if HAS_TORCH and isinstance(a, torch.Tensor):
        if a.is_complex() and not b.is_complex():
            b = b.to(a.dtype)
        elif b.is_complex() and not a.is_complex():
            a = a.to(b.dtype)
        return torch.dot(a, b)
    return np.dot(a, b)


def absolute(a):
    """Backend-aware elementwise absolute value: ``torch.abs`` for a torch tensor,
    else ``np.abs``."""
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.abs(a)
    return np.abs(a)


def conj(a):
    """Backend-aware complex conjugate: ``torch.conj`` for a torch tensor, else ``np.conj``."""
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.conj(a)
    return np.conj(a)


def solve(A, b):
    """Backend-aware dense linear solve: ``torch.linalg.solve`` for a torch tensor,
    else ``np.linalg.solve``."""
    if HAS_TORCH and isinstance(A, torch.Tensor):
        return torch.linalg.solve(A, b)
    return np.linalg.solve(A, b)


def sqrt(a):
    """Backend-aware square root: ``torch.sqrt`` for a torch tensor, else ``np.sqrt``."""
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.sqrt(a)
    return np.sqrt(a)


def reshape(a, shape):
    """Backend-aware reshape: ``torch.reshape`` for a torch tensor, else ``np.reshape``."""
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.reshape(a, shape)
    return np.reshape(a, shape)


def concatenate(arrays):
    """Backend-aware 1-D concatenation: ``torch.cat`` for torch tensors, else
    ``np.concatenate``. Dispatches on the first array in ``arrays``."""
    if HAS_TORCH and isinstance(arrays[0], torch.Tensor):
        return torch.cat(arrays)
    return np.concatenate(arrays)


# ---- iterative-solver output ----
# Single source for the text every iterative solve prints, so the format lives in one place and
# each solve is self-identifying.  A solve announces itself once with title(), then prints one
# iteration() row per cycle, and (if it tracks a wall time) a converged() footer.  Plain string
# builders -- the caller still print()s them; no logging, no verbosity state, no shared object.

def title(text):
    r"""A blank line then a solve title naming what an iterative sequence solves -- e.g.
    ``'T-amplitudes (CCSD)'``, ``'perturbed Lambda'``, ``'EOM-CCSD (RIGHT)'``.  Printed once above
    the iteration rows so a run that stacks many solves (the 3N perturbed solves behind a Hessian,
    say) stays legible: each block says what it is rather than being an anonymous table of numbers."""
    return "\n" + text


def iteration(niter, energy=None, de=None, rms=None, e_label="Ecorr", note=None):
    r"""One iteration row, e.g.::

        Iter   3: Ecorr =   -0.070680088785123  dE = -1.20000E-07  rms =  3.40000E-08

    Only the supplied columns print, so an energy-bearing solve (CC, CI) and a residual-only solve
    (the perturbed-amplitude / Z-vector equations, which have no energy) share one formatter:
    ``energy`` (labeled ``e_label``), ``de``, and ``rms`` are each shown only when given.  ``note``
    (e.g. ``'MP2'``) is appended -- used on an initial-guess line in place of ``rms``."""
    cols = []
    if energy is not None:
        cols.append("%s = %.15f" % (e_label, energy))
    if de is not None:
        cols.append("dE = % .5E" % de)
    if rms is not None:
        cols.append("rms = % .5E" % rms)
    if note is not None:
        cols.append(note)
    return "Iter %3d: %s" % (niter, "  ".join(cols))


def converged(name, elapsed):
    r"""A convergence footer for a timed solve, e.g. ``'CCwfn converged in 0.420 seconds.'``."""
    return "%s converged in %.3f seconds." % (name, elapsed)


def timing(label, seconds):
    r"""A build-timing line for a major component, e.g. ``'CCwfn built in 0.005 seconds.'`` --
    a uniform report (one verb, one format) so a run's build steps read consistently and give the
    user a sense of how long the calculation will take."""
    return "%s built in %.3f seconds." % (label, seconds)


def field(label, value, width=22):
    r"""A colon-aligned ``label : value`` row for a preamble block, e.g.
    ``'  SCF energy             : -74.959948270260 Eh'``."""
    return "  %-*s : %s" % (width, label, value)


def solve_params(method, e_conv, r_conv, maxiter, max_diis, start_diis):
    r"""The wfn-specific solve preamble: the convergence limits and DIIS options a solve runs with,
    printed above its iteration table."""
    return "\n".join(("\n%s solve" % method,
                      field("energy convergence", "%.2E" % e_conv),
                      field("residual convergence", "%.2E" % r_conv),
                      field("max iterations", "%d" % maxiter),
                      field("DIIS max / start", "%d / %d" % (max_diis, start_diis))))


class helper_diis(object):
    r"""DIIS (Pulay direct inversion of the iterative subspace) extrapolator for the CC / Lambda /
    response amplitude iterations.  Keeps a rolling window (``max_diis``) of amplitude iterates and
    their error vectors ``e_n = t_n - t_{n-1}``, and replaces the current amplitudes by the subspace
    combination that minimizes the residual.

    Parameters
    ----------
    t1, t2 : tensors
        the initial amplitude pair (seeds the iterate history)
    max_diis : int
        maximum subspace size (0 disables extrapolation)
    precision : str
        'DP' or 'SP' (default 'DP')
    """
    def __init__(self, t1, t2, max_diis, precision='DP'):
        """Seed the iterate and error-vector history from the initial amplitudes ``t1``/``t2``."""
        self.oldt1 = clone(t1)
        self.oldt2 = clone(t2)
        self.diis_vals_t1 = [clone(t1)]
        self.diis_vals_t2 = [clone(t2)]

        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis
        self.precision = precision

    def add_error_vector(self, t1, t2):
        r"""Append the current amplitudes ``t1``/``t2`` to the history and record their error vector
        ``e = (t - t_prev)`` (flattened, ``t1`` and ``t2`` concatenated)."""
        # Add DIIS vectors
        self.diis_vals_t1.append(clone(t1))
        self.diis_vals_t2.append(clone(t2))
        # Add new error vectors
        error_t1 = (self.diis_vals_t1[-1] - self.oldt1).ravel()
        error_t2 = (self.diis_vals_t2[-1] - self.oldt2).ravel()
        self.diis_errors.append(concatenate((error_t1, error_t2)))
        self.oldt1 = clone(t1)
        self.oldt2 = clone(t2)

    def extrapolate(self, t1, t2):
        r"""Extrapolate the amplitudes from the stored subspace: build the bordered error-overlap
        matrix ``B``, solve the Pulay equations for the coefficients ``c``, and form the new
        amplitudes ``sum_i c_i t_i`` (returns ``t1, t2`` unchanged if ``max_diis == 0``)::

            [ B   -1 ] [ c      ]   [ 0  ]        B_mn = <e_m | e_n>
            [ -1   0 ] [ lambda ] = [ -1 ]
            t_new = sum_i c_i t_i

        .. math::

            \begin{pmatrix} B & -\mathbf{1} \\ -\mathbf{1}^{T} & 0 \end{pmatrix}
            \begin{pmatrix} c \\ \lambda \end{pmatrix}
            = \begin{pmatrix} \mathbf{0} \\ -1 \end{pmatrix},
            \qquad B_{mn} = \langle e_m | e_n\rangle, \qquad t^\mathrm{new} = \sum_i c_i\, t_i
        """
        if (self.max_diis == 0):
            return t1, t2

        # Limit size of DIIS vector
        if (len(self.diis_errors) > self.max_diis):
            del self.diis_vals_t1[0]
            del self.diis_vals_t2[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)

        # Build error matrix B. B/resid are real even when the amplitudes are complex
        # (e.g. ccresponse), so they are seeded from real_zeros (real-component dtype of
        # oldt1), which also carries the precision and, for torch, the device.
        B = real_zeros((self.diis_size + 1, self.diis_size + 1), self.oldt1) - 1.0
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            # For complex amplitudes (e.g. ccresponse) the error overlaps are complex; B is the
            # real overlap matrix by construction (see above), so take the real part explicitly
            # rather than let the assignment truncate it (which numpy flags as a ComplexWarning).
            B[n1, n1] = dot(e1, e1).real
            for n2, e2 in enumerate(self.diis_errors):
                if n1 >= n2:
                    continue
                B[n1, n2] = dot(e1, e2).real
                B[n2, n1] = B[n1, n2]

        B[:-1, :-1] /= absolute(B[:-1, :-1]).max()

        # Build residual vector
        resid = real_zeros((self.diis_size + 1,), self.oldt1)
        resid[-1] = -1

        # Solve pulay equations
        ci = solve(B, resid)

        # Calculate new amplitudes
        t1 = zeros_like(self.oldt1)
        t2 = zeros_like(self.oldt2)
        for num in range(self.diis_size):
            t1 += ci[num] * self.diis_vals_t1[num + 1]
            t2 += ci[num] * self.diis_vals_t2[num + 1]

        # Save extrapolated amplitudes to old_t amplitudes
        self.oldt1 = clone(t1)
        self.oldt2 = clone(t2)

        return t1, t2
