#!/usr/bin/env python3
"""Generate the supplementary validation table from an instrumented test run.

    rm -f /tmp/val.json          # the flush APPENDS; a stale file silently mixes old rows in
    PYCC_VALIDATION_LOG=/tmp/val.json pytest pycc/tests/test_0{83,84,87,88,89,90}*.py \
                                              pycc/tests/test_104*.py
    python docs/validation_table.py /tmp/val.json > validation_table.tex

Two tables are emitted: externally anchored properties (CFOUR, finite difference) and the
coupled-cluster AATs, which have no external implementation and are anchored on exact identities
and a second route within PyCC.  Keeping them in one table overflows the page, and REVTeX drops
the trailing footnotes without reporting an error.

Rows are aggregated to one line per (property, method, molecule, route, oracle); the
reported agreement is the WORST case over the all-electron / frozen-core variants, so the
table states a bound PyCC meets everywhere rather than its best showing.
"""
import json, sys, collections

ORACLES = {
    'cfour':     ('CFOUR',          r"Analytic CFOUR result: \texttt{VIB=EXACT} for the Hessians and APTs "
                                    r"and \texttt{PROP=SECOND\_ORDER} for the polarizabilities, in both cases "
                                    r"with \texttt{SCF\_CONV=13}, \texttt{CC\_CONV=12} and "
                                    r"\texttt{LINEQ\_CONV=12}.  The 6-31G and STO-3G cases use CFOUR's own "
                                    r"GENBAS basis, transcribed into PyCC so the AO bases agree to all printed "
                                    r"digits; cc-pVDZ requires no transcription, PyCC's and CFOUR's agreeing to "
                                    r"$4\times10^{-13}$ in the SCF energy."),
    'fd-energy': ('Finite difference', r"Central finite difference of PyCC's own correlation energy, "
                                    r'which shares no density, orbital-response, or perturbed-amplitude '
                                    r'machinery with the analytic code.'),
    'identity':  ('Exact identity', r'An identity the quantity must satisfy exactly: two-electron '
                                    r'exactness of CCSD against CISD, the translational sum rule, or '
                                    r'invariance to a redundant orbital-rotation gauge.'),
    'internal':  ('Internal',       r'Two independent routes to the same quantity within PyCC '
                                    r'(spin-orbital against spin-adapted spatial).'),
}
LIMITS = {
    'oracle-print':  r'Set by the precision to which the reference provides the data, not by PyCC.',
    'fd-truncation': r'Set by the finite difference, not by PyCC: the residual is the $O(h^4)$ '
                     r'truncation of the stencil.',
    'gauge-conditioning': r'Set by the conditioning of the canonical perturbed-MO gauge, not by '
                     r"PyCC's formulation: this system has an accidental near-degeneracy between "
                     r'virtual orbitals of different irreducible representations '
                     r'($\Delta\epsilon = 3.5\times10^{-5}$~$E_\mathrm{h}$), and the canonical '
                     r'gauge divides by that gap.  The value is reproducible within one execution '
                     r'context but shifts by roughly 10\% between them, so it is quoted to one '
                     r'figure.  The non-canonical rows for the same case are unaffected.',
}


_CASE_FIX = {
    'spinorbital': 'spin-orbital',
    'two-electron exactness': 'two-electron limit',
    'magnetic / perturbed-MO gauge': 'gauge invariance',
    'keystone': 'SO vs.\\ spatial',
}

# The tests record one free-text field; it means the orbital route for the CFOUR and
# finite-difference rows, and the identity being tested for the AAT rows.  Split it so
# neither column mixes the two.
_ROUTE = {
    'spatial':                        'spatial',
    'spin-orbital':                   'spin-orbital',
    'spinorbital':                    'spin-orbital',
    'two-electron exactness':         'spatial',
    'magnetic / perturbed-MO gauge':  '---',
    'keystone':                       '---',
}


def _route(note):
    """Orbital route for a row.  The identity being tested lives in the Reference column, so the
    free-text note only has to yield the route; the Case column carries all-electron / frozen core
    and the Basis column the AO basis."""
    return _ROUTE.get(note, note or '---')

_REF_FIX = {
    'CISD AAT (CCSD exact for two electrons)': 'CISD AAT',
    'spin-orbital vs spatial': 'two routes within PyCC',
}

_MOL_TEX = {'H2O': r'\ce{H2O}', 'H2O+': r'\ce{H2O+}', 'HOF': r'\ce{HOF}',
            'C2H4+': r'\ce{C2H4+}', 'H2O2': r'\ce{H2O2}', 'CH4': r'\ce{CH4}',
            'NH2': r'\ce{NH2}', 'H4_2PLUS': r'\ce{H4^2+}', 'CH2': r'\ce{CH2}',
            # test_084 parameterizes its CFOUR cases with lowercase keys
            'water': r'\ce{H2O}', 'hof': r'\ce{HOF}', 'nh2': r'\ce{NH2}',
            'h2o': r'\ce{H2O}', 'h2o2': r'\ce{H2O2}', 'ch2': r'\ce{CH2}'}


def _mol_tex(mol):
    """TeX for a molecule label.  An unmapped label used to fall through as a bare string and
    typeset unformatted (CH2 rather than CH_2), silently -- so say so instead."""
    if not mol:
        return '---'
    if mol not in _MOL_TEX:
        print('validation_table: WARNING no _MOL_TEX entry for %r; emitting \\ce{%s}'
              % (mol, mol), file=sys.stderr)
        return r'\ce{%s}' % mol
    return _MOL_TEX[mol]


def _tidy(value, table):
    return table.get(value, value)

def main(path):
    rows = json.load(open(path))
    agg = collections.OrderedDict()
    for r in rows:
        key = (r['property'], r['method'], r.get('molecule') or '', r.get('note') or '',
               r['oracle'], r.get('reference') or '', r.get('basis') or '',
               r.get('ao_basis') or '')
        prev = agg.get(key)
        if prev is None or r['achieved'] > prev['achieved']:
            agg[key] = r

    order = ['Gradient', 'Hessian', 'APT (length gauge)', 'APT (velocity gauge)',
             'Static polarizability', 'AAT', 'DBOC']
    def sort_key(k):
        return (order.index(k[0]) if k[0] in order else 99, k[1], k[4], k[2], k[7], k[3], k[6])

    # One longtable spanning however many pages it needs.  A REVTeX `table` float cannot break,
    # and when it overflows the page REVTeX silently DROPS the trailing \footnotetext entries
    # without a warning, so the notes are set as ordinary text after the table instead.
    LETTERS = 'abcdefghij'
    # longtable repeats the column header on a continuation page but cannot carry the current
    # Property label across, so a break inside a group leaves the continuation's first rows
    # unlabelled.  Rather than force a break after a named property (which goes stale the moment
    # rows are added -- it once left a page holding eight rows), end every row of a group with
    # `\\*`, which forbids a break there, so TeX must break at a group boundary.  Self-maintaining,
    # provided no single property group is taller than a page.
    HEADER = (r'Property & Method & Molecule & Basis & Route & Case & Reference & Agreement \\')

    used_oracles, used_limits = [], []
    body, last_prop = [], None
    for key in sorted(agg, key=sort_key):
        r = agg[key]
        prop, meth, mol, route, oracle, ref, basis, ao = key
        label, _ = ORACLES[oracle]
        if oracle not in used_oracles:
            used_oracles.append(oracle)
        marks = [('oracle', oracle)]
        lim = r.get('limited_by')
        if lim:
            if lim not in used_limits:
                used_limits.append(lim)
            marks.append(('limit', lim))
        shown_prop = '' if prop == last_prop else prop
        last_prop = prop
        route = _route(route)
        case = basis if basis else '---'
        ref = _tidy(ref, _REF_FIX)
        refcol = ref if ref else label
        mant, expo = ('%.1e' % r['achieved']).split('e')
        approx = r'\sim' if r.get('approximate') else ''
        body.append((marks, prop, r'%s & %s & %s & %s & %s & %s & %s@MARK@ & $%s\times10^{%d}$ \\' % (
            shown_prop, meth, _mol_tex(mol), ao or '---',
            route, case, refcol, approx + mant, int(expo))))

    def letter(kind, key):
        i = (used_oracles.index(key) if kind == 'oracle'
             else len(used_oracles) + used_limits.index(key))
        return LETTERS[i]

    out = []
    out.append(r'\footnotesize')
    out.append(r'\setlength{\LTcapwidth}{\textwidth}')
    out.append(r'\begin{longtable}{lllllllr}')
    out.append(r'\caption{Measured agreement between PyCC and its validation references. Each entry '
               r'is $\max|\mathrm{PyCC}-\mathrm{reference}|$ in atomic units. All-electron and '
               r'frozen-core cases are listed separately. Agreement is the measured result, not the '
               r"assertion tolerance used in the test suite. ``NCP'' in the Route column marks "
               r'rows computed with non-canonical perturbed molecular orbitals; the remaining '
               r"CCSD(T) rows use the canonical perturbed-MO gauge, which is PyCC's default for "
               r'that method. Lettered notes follow the table.}'
               r'\label{tab:validation}\\')
    out.append(r'\hline\hline')
    out.append(HEADER)
    out.append(r'\hline')
    out.append(r'\endfirsthead')
    out.append(r'\multicolumn{8}{l}{\textsc{Table}~\thetable\ (continued)}\\')
    out.append(r'\hline\hline')
    out.append(HEADER)
    out.append(r'\hline')
    out.append(r'\endhead')
    out.append(r'\hline')
    out.append(r'\multicolumn{8}{r}{\footnotesize\itshape continued on the next page}\\')
    out.append(r'\endfoot')
    out.append(r'\hline\hline')
    out.append(r'\endlastfoot')
    for i, (marks, prop, line) in enumerate(body):
        sup = r'\textsuperscript{%s}' % ','.join(letter(*m) for m in marks)
        line = line.replace('@MARK@', sup)
        last_of_group = (i + 1 == len(body)) or body[i + 1][1] != prop
        if not last_of_group:
            line = line[:-2] + r'\\*'     # no page break within a property group
        out.append(line)
        if last_of_group and i + 1 != len(body):
            out.append(r'\hline')         # separate the property groups

    out.append(r'\end{longtable}')
    out.append(r'\normalsize')
    out.append('')
    # The marker leads the note text directly, with no separating colon and no hanging label
    # (a description list puts the label in its own column and REVTeX appends a colon).
    out.append(r'\noindent\footnotesize\textbf{Notes.}\par\smallskip')
    notes = ([(letter('oracle', o), ORACLES[o][1]) for o in used_oracles]
             + [(letter('limit', l), LIMITS[l]) for l in used_limits])
    for mark, text in notes:
        out.append(r'\noindent\textsuperscript{%s}%s\par\smallskip' % (mark, text))
    out.append(r'\normalsize')

    return '\n'.join(out)

if __name__ == '__main__':
    print(main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/val.json'))
