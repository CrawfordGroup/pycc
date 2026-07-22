# CC linear-response reformulation — design plan

**Status:** design/discussion (not started). Living document.

## Goal

Recast CC linear-response properties (polarizability, optical rotation) in the same
density-based framework as the `correlatedderivs` second-derivative properties, reusing the
perturbed-amplitude/multiplier machinery on `CCderiv` rather than building on a parallel code path.
Target: numbers identical to `ccresponse` at matched convergence, with the response layer expressed
as a configuration of the shared derivative machinery.

**Ground rule:** do **not** modify `ccresponse.py`. It is the reference oracle
(`linresp_asym`, `solve_right`/`solve_left`), not code we change or import from.

## The correspondence

CC linear response is the same perturbed-amplitude machinery as the `correlatedderivs`
second-derivative properties, in a different configuration along two orthogonal axes — orbital
relaxation (on/off) and frequency (0 vs omega):

| axis | relaxed static derivative (`ccderiv`) | unrelaxed dynamic response (new) |
|---|---|---|
| perturbed T/L residual | `A · dT = -xi` | `(A -/+ omega) · dX = -xi` |
| omega in the residual | none | **-omega * X** (right/T), **+omega * Y** (left/L) |
| density | `Drel` (`D` + Z-vector + dependent pairs) | `D` (unrelaxed) |
| final tensor | `d_a Drel . mu_b + Drel . (U^aT mu_b + mu_b U^a)` | `d_a D . mu_b` (no `U.mu` terms) |
| reference (SCF) block | `HFwfn` CPHF value | **zero** — no orbital response |

The omega signs are verified against `ccresponse`:
- right (`solve_right`/`r_X1`,`r_X2`): `r_X1 = pertbar.Avo.T - omega*X1`, `r_X2 = pertbar.Avvoo - omega*X2` (both `-omega`).
- left (`solve_left`/`r_Y1`,`r_Y2`): `r_Y1 += omega*Y1`, `r_Y2 += 0.5*omega*Y2` (both `+omega`).

So the frequency enters as `-omega*X` on the right and `+omega*Y` on the left, with the
`(D + omega)`-shifted denominators.

## Mechanism (what the perturbed T/L equations actually use)

Two places the field perturbation enters, and they are treated differently:

1. **In the perturbed T/L equations — the perturbation DOES enter as the similarity-transformed
   dipole.** For a field, the perturbed-HBAR intermediates that the derivative code builds from the
   perturbed integrals reduce to the `pertbar` A-intermediates: `dHoo -> Aoo`, `dHvv -> Avv`,
   `dHov -> Aov`, etc. **Only seven A-blocks are non-zero:** `Aoo`, `Aov`, `Avo`, `Avv`, `Avvvo`,
   `Aovoo`, `Avvoo` (the blocks used in `ccresponse.linresp_asym`). These substitute for the
   `dHBAR` intermediates in the perturbed-amplitude/multiplier residuals.
2. **In the final polarizability — the similarity transform is NOT needed.** It is folded
   implicitly into the density, so the tensor is `alpha_ab(omega) = Tr(d_b D(omega) . mu_a)` with
   the **bare** MO dipole `mu_a`. No `pertbar`/`A_bar` in the contraction, no `U.mu` orbital terms.

The tensor comes out **naturally symmetric** (a consequence of the response-function symmetry + the
`(1+Lambda)` structure) — we verify it, we do not impose `0.5*(alpha + alpha^T)`.

Because there is no orbital (CPHF) response, the whole quantity lands in the **correlation** block;
in `PropertyComponents` the reference block is **zero** and `total == correlation`. (The missing
HF-CPHF orbital-relaxation piece is exactly the difference from the relaxed derivative.)

The frozen-orbital choice is forced, not merely convenient: the orbital-relaxed dynamic response
has spurious poles at the SCF excitation frequencies.

## Architecture

- **Entry: on `CCderiv`** — a general linear-response engine plus thin property-named wrappers
  (the `ccresponse` pattern: `linresp_asym` engine + `polarizability`/`optrot` wrappers). Not in
  `ccresponse`, not in the `pycc.polarizability` derivative facade (different physical quantity).
  Each returns a bare `(3,3)` correlation tensor (`reference` block is zero; see above).

  ```python
  def linear_response(self, a, b, omega=0.0):
      """Orbital-unrelaxed CC linear response function <<a; b>>_omega, shape (3,3), via
      Tr(d_b D(omega) . a). Operator keys a, b follow ccresponse's pertkey idiom
      ('mu' = electric dipole, 'm' = magnetic dipole)."""

  def response_polarizability(self, omega=0.0):
      return self.linear_response('mu', 'mu', omega)   # -<<mu; mu>>_omega

  def optical_rotation(self, omega):
      return self.linear_response('mu', 'm', omega)    # <<mu; m>>_omega, omega != 0
  ```

  Naming: `response_polarizability` (not `unrelaxed_polarizability`) — it is the frequency-dependent
  response quantity, distinct from the inherited relaxed static `polarizability`. Operator handling
  and the single-`omega`-per-call convention mirror `ccresponse` (a frequency sweep loops at the
  call site). Internally both wrappers ride one helper, the perturbed unrelaxed density at frequency
  omega:

  ```python
  def _response_density(self, op, omega):
      """Orbital-unrelaxed perturbed 1-PDM d_op D(omega): solve dT (residual -omega X) and
      dL (residual +omega Y) driven by operator `op`'s seven pertbar A-intermediates (no CPHF
      folding), then _perturbed_unrelaxed_densities. No Z-vector, no dependent pairs."""
  ```
- **Shared engine:** add an `omega` argument (default `0.0`, so the existing derivative path is
  unchanged) to `_perturbed_amplitudes`, `_perturbed_lambda`, and `_perturbed_unrelaxed_densities`,
  applying the `-/+ omega` residual shift and the `(D + omega)` denominators. Supply the seven
  field A-intermediates as the perturbed-HBAR substitute.
- **Assembly:** contract the perturbed unrelaxed 1-PDM `d_b D` (from `_perturbed_unrelaxed_densities`,
  the leaf hook — no Z-vector, no dependent pairs) with the bare MO dipole.

## Phasing

1. **CCSD static unrelaxed polarizability (omega = 0).** Isolates the unrelaxed-density step with no
   frequency. Validate three ways:
   - **amplitude-level:** our `dT`/`dL` vs `ccresponse`'s `X`/`Y` from `solve_right`/`solve_left`
     at omega = 0 (the earliest, sharpest check — before any density contraction);
   - **tensor:** vs `ccresponse.linresp_asym` / `linresp_sym`;
   - **finite difference of the unrelaxed dipole** (`d mu_unrelaxed / dF`).
2. **Dynamic (omega != 0).** Reproduce `ccresponse` at several frequencies; same amplitude-level +
   tensor checks.
3. **Optical rotation.** Operators are the electric dipole `mu` and the magnetic dipole `m` (the
   `<<mu; m>>` tensor) — not angular momentum. **omega != 0 only** (there is no static optical
   rotation). Antisymmetric combination + mixed `(1+Lambda)` terms.
4. **CC3.** Deferred; scope TBD.

## Validation

- Against `ccresponse` as the oracle: `test_057`/`test_055` (spatial/SO polarizability),
  `test_058`/`test_056` (optrot), and — later — `test_059` (CC3).
- Symmetric and asymmetric formulations give exactly the same tensor, so either `ccresponse`
  formulation is a valid check.
- With convergence thresholds matched, results should be **identical** to `ccresponse`.

## Open items

1. ~~**Entry-point signature** on `CCderiv`.~~ **Settled** (see Architecture): a general
   `linear_response(a, b, omega=0.0)` engine with `response_polarizability(omega)` and
   `optical_rotation(omega)` wrappers, operator keys and single-`omega` per `ccresponse`.
2. **CC3** — whether/how it fits this framework.

## Code references

- `ccresponse.py`: `linresp_asym` (:1627, the asymmetric response function this mirrors),
  `linresp_sym` (:641), `solve_right` (:217) / `r_X1` (:325) / `r_X2` (:384),
  `solve_left` (:1694) / `r_Y1` (:1908) / `r_Y2` (:2102), `_build_pertbar` (:74).
- `ccderiv.py`: `_perturbed_amplitudes` (:209), `_perturbed_lambda` (:498),
  `_perturbed_unrelaxed_densities` (:188).
- `correlatedderivs.py`: `_perturbed_relaxed_density` (:439), `polarizability` (:667).
- `docs/cc_gradients_orbital_response.tex`: eq. `eq:polarizability` (the derivative assembly the
  response strips the orbital terms from); pertbar per Crawford, cc_response.pdf eqn 78.
