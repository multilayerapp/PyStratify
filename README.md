# PyStratify

Light scattering by **multilayered (stratified) spheres** in Python: any number of concentric
shells, absorbing, magnetic, gain or chiral media, and electric or magnetic dipole emitters inside
or outside the particle, with angle-resolved far fields for both. The physics is the recursive transfer-matrix method of Moroz (2005) as used
by [STRATIFY](https://gitlab.com/iliarasskazov/stratify) (Rasskazov, Carney & Moroz,
[OSA Continuum 3, 2290 (2020)](https://doi.org/10.1364/OSAC.399979)). The numerics are rebuilt
so that every quantity stays finite and accurate to the orders it needs, and the defects found in
the MATLAB code are fixed ([AUDIT.md](AUDIT.md)).

| quantity | function |
|---|---|
| solution of the sphere, batched over wavelengths | `solve` → `Solution` |
| scattering / absorption / extinction, per multipole | `cross_sections` |
| angular scattering amplitudes | `scattering_amplitudes` |
| amplitude and Mueller matrices (S1..S4, 4 × 4) | `amplitude_matrix`, `mueller_matrix` |
| scattering pattern for any polarisation: dσ/dΩ, directivity, Stokes, helicity | `scattering_pattern` → `ScatteringPattern` |
| cross sections and circular dichroism for both helicities | `helicity_cross_sections` |
| far field, directivity and radiated power of a dipole emitter | `dipole_far_field` → `EmissionPattern` |
| chiral (Pasteur) shells: T-matrix with TM–TE coupling | `solve_chiral` → `ChiralSolution` |
| E and H near fields | `near_field` |
| orientation-averaged intensities and energy density | `energy_density` |
| energy stored in each shell | `shell_energy` |
| energy prefactors (Loudon, for Drude metals) | `electric_prefactor`, `energy_prefactors` |
| radiative / nonradiative / total decay rates | `decay_rates` → `DecayRates` |
| thin-shell electron free-path correction | `free_path_correction`, `DRUDE` |
| multipole truncation | `truncation_order` |

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
field = ps.near_field(sol, x, y, z, wavelength_index=200)
stored = ps.shell_energy(sol, wavelength_index=200)

rates = ps.decay_rates(radii, n[200], wavelength[200], r=[60.0, 75.0])
rates.radiative, rates.nonradiative, rates.total, rates.balance_error, rates.quantum_yield()

theta, phi = np.meshgrid(np.linspace(0, np.pi, 181), np.linspace(0, 2 * np.pi, 361), indexing="ij")
pattern = ps.scattering_pattern(sol, theta, phi, polarization=(1, 1j))  # helicity +1 along +z
pattern.directivity, pattern.differential_cross_section, pattern.stokes()   # (W, 181, 361)
emitter = ps.dipole_far_field(radii, n[200], wavelength[200], position=[0, 20, 60],
                              moment=[1, 0, 0], theta=theta, phi=phi)
emitter.directivity, emitter.power, emitter.circular_polarization          # power = P_rad / P_0

kappa = [0, 0, 1e-3, 0]                     # Pasteur chirality per shell, host achiral
chiral = ps.solve_chiral([50.0, 55.0, 60.0], [1.45, n_gold[200], 1.5, 1.33], kappa, wavelength[200])
cd = ps.helicity_cross_sections(chiral)    # .ext[:, 0 or 1] for helicity +1 / -1, .cd_ext, .g_ext
ps.scattering_pattern(chiral, theta, phi)  # far fields and dipole_far_field(..., kappa=...) work alike
```

Far-field conventions are Bohren & Huffman's: E_sca = e^{ikr}/(−ikr) X with
(X_∥, X_⊥) = [[S2, S3], [S4, S1]] (E_∥, E_⊥) and dσ/dΩ = |X|²/k². Helicity +1 is
(x̂ + iŷ)/√2 for a wave along +z (left-circular in the optics convention). Chiral media follow
D = εE + iκH, B = μH − iκE (Gaussian units, e^{−iωt}; the convention of
[treams](https://github.com/tfp-photonics/treams)), so helicity ±1 has index n ± κ.

`examples/` has runnable scripts (multipole spectra, free-path correction, scattering pattern,
scattering and emission directivity, near-field maps, energy density, stored energy, electric and
magnetic dipole decay, circular dichroism of a chiral shell); with matplotlib installed each saves
a PNG beside itself.

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
  mis-scaled; there the unscaled `hankel1` is used. Accuracy is ≤ 1e-10 against
  extended-precision mpmath for orders 0–2500, |x| from 1e-6 to 3000, Im x up to 700 and gain.
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
  total = radiative + nonradiative. Sums are truncated from the geometry and accepted only when
  a remainder estimate meets `tol`; an unconverged position is flagged, never silently
  truncated.
* **Dipole far fields from reciprocity.** The far-field amplitude of a dipole p at r₀ in
  direction n̂ with polarisation ê equals p·E(r₀) for the plane wave ê incident from n̂, which
  `solve` already provides for every shell. Rotating the emitter onto the z axis leaves only
  m = 0, ±1 multipoles, so any position and complex orientation costs one set of π_l, τ_l; the
  radiated power is summed from orthogonal multipole coefficients. For an emitter in the host
  the direct dipole field is added in closed form, so the truncation depends on the particle
  and not on the emitter's distance.
* **Chiral shells** (`chiral.py`). In a Pasteur medium the field splits into Beltrami waves of
  helicity ±1 with wavenumbers k₀(n ± κ), and an interface mixes them only through its impedance
  contrast. The regular solution is swept outwards as a 2 × 2 matrix ρ with the same scaling as
  the achiral solver: the matrix Möbius step is assembled from the index, chirality and impedance
  contrasts (no cancellation of the large l/x parts), and ρ is carried as element-wise
  logarithms so it survives falling below the double range across thick absorbing shells.
* **Vectorised.** `solve` takes a whole spectrum at once (array operations over wavelength ×
  order; Python loops only over interfaces). Near fields and energy densities evaluate the
  special functions once per distinct kr for both polarisations; absorption integrals reuse one
  quadrature rule per shell.

**Accuracy you can expect.** Coefficients agree with 60-digit transfer matrices to ~1e-12, and
the interface conditions hold to 1e-9 at every order. Where a result is less accurate than that,
the problem is ill-conditioned rather than the algorithm: large lossless particles with sharp
internal resonances can amplify input rounding by 10⁶ (perturbing the inputs by 1e-15 moves the
exact answer by as much as the solver's error), and the absorption of a nearly lossless
Rayleigh particle is known only to ~eps·|a_l|. Layers index-matched to within δ determine their
coefficients to ~l·eps/δ. Chiral T-matrices agree with 80-digit 4 × 4 transfer matrices to
1e-12 up to l = 160, and with the independent code treams to 1e-13.

Typical timings on one core:

| task | time |
|---|---|
| 601-wavelength spectrum, Au nanoshell | 25 ms |
| 601-wavelength spectrum, 6-interface matryoshka | 70 ms |
| x = 500 sphere, all orders | 8 ms |
| 300 × 300 near-field map | 1.4 s |
| decay rates at 100 emitter positions, up to l = 1200 | 0.4 s |
| 601-wavelength spectrum, Au nanoshell with a chiral shell, both helicities | 46 ms |
| scattering pattern on a 181 × 361 (θ, φ) grid | 10 ms |
| dipole emission pattern on a 181 × 361 grid | 80 ms |

## Tests

```bash
pip install -e ".[test]"
pytest
```

* `test_riccati.py`: special functions against mpmath in every regime above.
* `test_solver.py`: coefficients against 60-digit transfer matrices (including a 1-nm film,
  index-matched layers and a gain shell), the interface conditions at every order up to 400,
  batching, 50 shells, x = 1000 and Im x ≈ 700.
* `test_physics.py`: textbook Mie (SciPy's spherical Bessel functions), the Bohren–Huffman
  reference case, the optical theorem, energy conservation of lossless multilayers, passivity,
  the Rayleigh limit, plane-wave limits, field continuity, quadrature, and energy conservation
  of decay rates for electric and magnetic dipoles.
* `test_farfield.py`: the amplitude matrix against textbook Mie with independently computed
  angular functions, the pattern as the far-zone limit of the near field (residual falling as
  1/kr), Bohren & Huffman's backscattering value, Mueller-matrix identities, directivity and
  dσ/dΩ integrating to 4π and σ_sca. Dipoles: closed forms in a homogeneous medium for any
  position and complex orientation, reciprocity against the plane-wave near field for emitters
  in the core, in an absorbing shell, in a magnetic shell and in the host (every phase), radiated
  power equal to `decay_rates` to 1e-12, a distant dipole reproducing the plane-wave amplitude of
  a chiral sphere, and helicity conservation of a (p, −ip) source beside a dual chiral particle.
* `test_chiral.py`: 80-digit 4 × 4 transfer matrices (thin Au shell, gain, a 1e-6 chirality
  contrast, l up to 160), treams, κ = 0 against the achiral solver, energy conservation of
  lossless chiral multilayers, reciprocity (symmetric TM–TE block), mirror symmetry (κ → −κ
  exchanges the helicities), helicity conservation of dual particles, and a chiral core behind
  2 µm of metal.

## Licence

GPL-3.0-or-later, as STRATIFY (this is a derivative work).
