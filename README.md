# PyStratify

Light scattering by **multilayered (stratified) spheres** in Python: any number of concentric
shells, absorbing, magnetic or gain media, and electric or magnetic dipole emitters inside or
outside the particle. The physics is the recursive transfer-matrix method of Moroz (2005) as used
by [STRATIFY](https://gitlab.com/iliarasskazov/stratify) (Rasskazov, Carney & Moroz,
[OSA Continuum 3, 2290 (2020)](https://doi.org/10.1364/OSAC.399979)). The numerics are rebuilt
so that every quantity stays finite and accurate to the orders it needs, and the defects found in
the MATLAB code are fixed ([AUDIT.md](AUDIT.md)).

| quantity | function |
|---|---|
| solution of the sphere, batched over wavelengths | `solve` → `Solution` |
| scattering / absorption / extinction, per multipole | `cross_sections` |
| angular scattering amplitudes | `scattering_amplitudes` |
| E and H near fields | `near_field` |
| orientation-averaged intensities and energy density | `energy_density` |
| energy stored in each shell | `shell_energy` |
| energy prefactors (Loudon, for Drude metals) | `electric_prefactor`, `energy_prefactors` |
| radiative / nonradiative / total decay rates | `decay_rates` → `DecayRates` |
| thin-shell electron free-path correction | `free_path_correction`, `DRUDE` |
| multipole truncation | `truncation_order` |
| decay rates from normalized quantities (reference formulation) | `normalized_decay_rates` |
| extended-precision and classical references | `pystratify.references` |

No optical constants are shipped: pass n + ik from a database such as
[refractiveindex.info](https://refractiveindex.info). The Drude fits in `DRUDE` exist only for
what tables cannot supply, the free-electron damping.

## Install

```bash
pip install git+https://github.com/multilayerapp/pystratify.git
```

Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10.

## Use

Lengths are unit-agnostic (give radii and wavelengths in the same unit); the Drude helpers take
nanometres. Shells are indexed from 0 (the core) to N (the host).

```python
import numpy as np
import pystratify as ps

wavelength = np.linspace(500, 900, 401)                              # nm
radii = [50.0, 55.0]                                                 # SiO2 core, Au shell
n = np.stack([np.full(401, 1.45), n_gold, np.full(401, 1.33)], 1)   # (W, N+1), host last
sol = ps.solve(radii, n, wavelength)                                 # mu defaults to 1

cs = ps.cross_sections(sol)                 # cs.q_ext, cs.q_sca, cs.q_abs: arrays over wavelength
cs.ext_by_order[:, ps.TM, 0]                # electric dipole contribution
field = ps.near_field(sol, x, y, z, wavelength_index=200)  # fields at the very surface:
                                            # ps.solve(..., regime="near") converges them
stored = ps.shell_energy(sol, wavelength_index=200)

rates = ps.decay_rates(radii, n[200], wavelength[200], r=[60.0, 75.0])
rates.radiative, rates.nonradiative, rates.total, rates.balance_error, rates.quantum_yield()
```

`examples/` has runnable scripts (multipole spectra, free-path correction, scattering pattern,
near-field maps, energy density, stored energy, electric and magnetic dipole decay); with
matplotlib installed each saves a PNG beside itself.

## Numerics

The transfer matrices of the method are products of Riccati–Bessel functions ψ_l, ξ_l at
different arguments. Past the size parameter they over- and underflow, and inside thin metal
shells their differences cancel long before that: in the 3-nm gold shell of OSAC Fig. 2(b) a
direct product violates the interface conditions by ~50 % at l = 14, while an emitter a few nm
from a metal needs l ≈ 500–1000. PyStratify keeps the physics and changes the representation.

* **Special functions as complex logarithms** (`riccati.py`). Orders up to |x| + 32 come from
  the exponentially scaled AMOS routines, which are exact near the zeros of ψ; higher orders from
  the stable recurrences, downward for ψ and upward for ξ, so the cost does not grow with the
  truncation order. In gain media AMOS's `hankel1e` returns spurious zeros and `yve` is
  mis-scaled; there the unscaled `hankel1` is used. Off the real axis ψ has no zeros, so a zero
  from AMOS is underflow (scaled `jve` underflows below n = |x| once Im x ≳ 1000) and the
  recurrence takes over. Accuracy is ≤ 1e-10 against extended-precision mpmath for orders
  0–2500, |x| from 1e-6 to 3000, Im x up to 700 and gain, and 1e-12 at 1500 + 1500i; spheres
  with |mx| = 3·10⁴ and Im(mx) = 8·10⁴ match BHMIE to 2e-12.
* **Scaled two-sided recursion** (`solver.py`). The regular solution is swept outwards through
  ρ = Rξ/ψ and the outgoing solution inwards through σ = Sψ/ξ, both carried as logarithms, so a
  ratio that falls to e⁻⁷⁰⁰ across a thick shell keeps full precision instead of turning
  subnormal. At high order ψ'/ψ ≈ (l+1)/x on both sides of an interface, so the mismatch that
  sets ρ is a small difference of large numbers; it is never formed by subtraction but
  assembled from the material contrast and the small ratios ψ_{l+1}/ψ_l, ξ_{l−1}/ξ_l. Each
  amplitude uses the better-conditioned of the value and derivative matching equations. The
  approach follows Yang (2003), Peña & Pal (2009), Ladutenko et al. (2017), Majic & Le Ru (2020)
  and Zhang (2025).
* **Decay rates from the Green's function.** The radiative rate, the Ohmic loss and the total
  rate (LDOS) are three independent sums, and `DecayRates.balance_error` checks
  total = radiative + nonradiative. The loss includes magnetic loss, Im(μ)|H|², as well as
  Im(ε)|E|². Sums are truncated from the geometry and accepted only when
  a remainder estimate meets `tol`; an unconverged position is flagged, never silently
  truncated.
* **Ohmic loss from boundary terms.** The absorption in a shell, ∫|A j_l + B h_l|² r² dr, is
  a Lommel integral: it needs the fields only at the shell's boundaries, so its cost is O(l)
  rather than the O(l²) of quadrature (whose node count must grow with the order). Past the
  turning point the textbook boundary term Im(k* f_{l−1}* f_l) is a small difference of real
  numbers, losing ~log₁₀(2l²/Im x²) digits; the jj and hh parts are therefore carried by ratio
  recurrences (downward for j, upward for h) that add only positive terms there, and the jh part
  has no cancellation. Weakly lossy shells (|Im k| < 10⁻³|k|), where Im k² → 0, use quadrature,
  in chunks so memory stays bounded. An emitter 1 nm from a 1-µm silver sphere needs ~17000
  orders and takes 0.8 s (quadrature ran out of memory); the default order cap is 20000.
* **Incident field in closed form.** In the host, `near_field` and `energy_density` sum only
  the scattered series, which converges with the sphere's orders at any distance, and add the
  plane wave exactly (its own series needs l ~ kr: with Wiscombe's truncation |E|² eight radii
  from a gold sphere was off by a factor of two).
* **Vectorised.** `solve` takes a whole spectrum at once (array operations over wavelength ×
  order; Python loops only over interfaces). Near fields and energy densities evaluate the
  special functions once per distinct kr for both polarisations.

**Accuracy you can expect.** Coefficients agree with 60-digit transfer matrices to ~1e-12, and
the interface conditions hold to 1e-9 at every order. Where a result is less accurate than that,
the problem is ill-conditioned rather than the algorithm: large lossless particles with sharp
internal resonances can amplify input rounding by 10⁶ (perturbing the inputs by 1e-15 moves the
exact answer by as much as the solver's error), and the absorption of a nearly lossless
Rayleigh particle is known only to ~eps·|a_l|. Layers index-matched to within δ determine their
coefficients to ~l·eps/δ.

Typical timings on one core:

| task | time |
|---|---|
| 601-wavelength spectrum, Au nanoshell | 25 ms |
| 601-wavelength spectrum, 6-interface matryoshka | 70 ms |
| x = 500 sphere, all orders | 8 ms |
| 300 × 300 near-field map | 1.4 s |
| decay rates at 100 emitter positions 0.5–25 nm from an Au nanoshell | 0.4 s |
| the same from 0.1 nm (6700 orders) | 2.4 s |

## Benchmarks

`python benchmarks/convergence.py` reruns the test cases of the multilayered-sphere literature
against independent references and writes the report in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md): decay rates near silver and silicon spheres down
to 0.25-nm gaps (Majic & Le Ru 2020) and inside and outside gold nanoshells and a matryoshka
(against high-precision transfer matrices), a dipole in a layered magnetic sphere and a Luneburg lens
(Yuan, Zhu & Zhu 2023, 2024), 1000-layer spheres at x = 1000, a graded 250-µm droplet at the
rainbow angle and the Cauchy profile (Wu & Wang 1991; Wu et al. 1997), absorbing spheres to
x = 20000, and fields far from the particle.

## Tests

```bash
pip install -e ".[test]"
pytest
```

* `test_normalized.py`: the normalized formulation (`normalized.py`: ψ'/ψ, ψξ and normalized j̄ only,
  reflection ratios swept outwards and inwards) against the solver, extended precision and the
  unnormalized formulas.
* `test_riccati.py`: special functions against mpmath in every regime above, and the
  Wronskian for |x| up to 8·10⁴.
* `test_solver.py`: coefficients against 60-digit transfer matrices (including a 1-nm film,
  index-matched layers and a gain shell), the interface conditions at every order up to 400,
  batching, 50 shells, x = 1000, Im x ≈ 700, and BHMIE at x = 20000.
* `test_physics.py`: textbook Mie (SciPy's spherical Bessel functions), the Bohren–Huffman
  reference case, the optical theorem, energy conservation of lossless multilayers, passivity,
  the Rayleigh limit, plane-wave limits, field continuity and convergence far from the
  particle, quadrature, decay rates against the mpmath Mie sums of Majic & Le Ru and, for
  emitters inside and outside metal shells, against high-precision transfer matrices, closed-form
  against quadrature Ohmic loss, magnetic-dipole rates against transfer matrices (including
  μ ≠ 1 and magnetically lossy shells), and energy conservation of decay rates for electric and
  magnetic dipoles, up to 17000 orders.

`pystratify.references` holds the independent references (mpmath Mie decay rates and extinction,
layered-sphere decay rates from transfer matrices, BHMIE); nothing in it shares code with the package.

## Licence

GPL-3.0-or-later, as STRATIFY (this is a derivative work).
