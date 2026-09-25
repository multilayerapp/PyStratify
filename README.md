# PyStratify

Light scattering by **multilayered (stratified) spheres** in Python. It re-implements
[STRATIFY](https://gitlab.com/iliarasskazov/stratify) (Rasskazov, Carney & Moroz,
[OSA Continuum 3, 2290 (2020)](https://doi.org/10.1364/OSAC.399979)) on an overflow-free
formulation of its recursive transfer-matrix method, and fixes the defects found in the MATLAB
code ([AUDIT.md](AUDIT.md)).

For any number of concentric shells (absorbing, magnetic or gain media) it computes:

| quantity | function | replaces | OSAC eqs. |
|---|---|---|---|
| solution of the sphere, batched over wavelengths | `solve` | `util/t_mat.m` | 10–18 |
| scattering / absorption / extinction, per multipole | `cross_sections` | `field/crs_sec.m` | 19 |
| angular scattering amplitudes | `scattering_amplitudes` | `field/far_fld.m` | 20 |
| E and H near fields | `near_field` | `field/near_fld.m` | 6–7, 25–26 |
| orientation-averaged intensities, energy density | `energy_density` | `energy/nrg_dns.m` | 21, 24 |
| energy stored in each shell | `total_energy` | `energy/nrg_tot.m` | 21, 27 |
| energy prefactors G_e, G_m | `g_prefactors`, `g_electric` | `energy/G_prefac.m` | 22–23 |
| radiative / nonradiative / **total** decay rates, electric or magnetic dipole | `decay_rates` | `decay/edcy.m`, `mdcy.m`, `I_abs.m` | 28–34 |
| thin-shell electron free-path correction | `free_path_correction` | `util/el_fr_pth.m` | 37 |
| multipole truncation | `l_max` | `util/l_conv.m` | 35–36 |
| tabulated Au, Ag, Al, Si | `refractive_index` | `materials/*.mat` | |

## Why not a line-by-line port

STRATIFY multiplies 2×2 transfer matrices whose entries are products of Riccati–Bessel
functions. Once the multipole order exceeds the size parameter they overflow, and differences of
them cancel catastrophically. Inside the 3-nm gold shell of the paper's own Fig. 2(b) matryoshka
the interface conditions are already violated by ~50 % at l = 14. Decay rates of an emitter a
few nm from a metal need l ≈ 500–1000, which the direct products cannot reach.

PyStratify keeps the physics of the RTMM but changes how it is represented:

* **Special functions as complex logarithms** (`riccati.py`). Values come from the exponentially
  scaled AMOS routines where representable, and from the stable recurrences beyond: downward for
  ψ, upward for ξ. They are accurate to ~1e-12 against 50-digit mpmath for orders 0–2000 and
  arguments from 10⁻³ to 150 + 2i, 5 + 60i, or gain media.
* **Scaled two-sided recursion** (`solver.py`). The regular solution is swept outwards through
  ρ = Rξ/ψ and the outgoing solution inwards through σ = Sψ/ξ. Amplitudes are carried as
  logarithms, and the better-conditioned of the value/derivative matching equations is used per
  element. This follows Yang (2003), Peña & Pal (2009), Ladutenko et al. (2017), Majic & Le Ru
  (2020) and Zhang (2025); references are in AUDIT.md.
* **Decay rates from the Green's function.** The radiative rate, the Ohmic loss and the total
  rate (LDOS) are three independent sums, and `DecayRates.balance_error` checks
  total = rad + nrad. It is ≤ 1e-11 in the test configurations, including an emitter 0.5 nm
  from gold at l = 1200.
* **Vectorised.** `solve` takes a whole spectrum at once (array ops over wavelength × multipole;
  Python loops only over layers). The near field evaluates special functions once per unique kr.
  Decay-rate sums are truncated from the geometry and verified by a remainder estimate, with a
  separate, shorter truncation for the Ohmic-loss series.

Typical timings on one core:

| task | time |
|---|---|
| 601-wavelength spectrum, Au nanoshell | 21 ms |
| 601-wavelength spectrum, 6-layer matryoshka | 0.12 s |
| x = 500 sphere, l = 534 | 6 ms |
| 300 × 300 near-field map | 0.36 s |
| decay rates at 100 emitter positions | 0.5 s |

## Install

```bash
pip install git+https://github.com/multilayerapp/PyStratify.git
```

Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10.

## Use

The solvers are unit-agnostic: give radii and wavelengths in the same unit. The material helpers
take nanometres.

```python
import numpy as np
import pystratify as ps

lam = np.linspace(500, 900, 401)                                 # nm
rad = [50.0, 55.0]                                               # SiO2 core, Au shell
ref = np.stack([np.full(lam.size, 1.45),
                ps.refractive_index("Au_JC", lam),
                np.full(lam.size, 1.33)], axis=1)                # (W, N+1), host last
sol = ps.solve(rad, ref, [1, 1, 1], lam, l_max=ps.l_max(rad[-1], 1.33, lam))

cs = ps.cross_sections(sol)            # cs.q_ext, cs.q_sca, cs.q_abs: arrays over lam
E = ps.near_field(sol, X, Y, Z, w=200) # fields at the 201st wavelength
W = ps.total_energy(sol, w=200)

r = ps.decay_rates(rad, ref[200], [1, 1, 1], lam[200], r_dip=[60.0, 75.0], norm="host")
r.radiative, r.nonradiative, r.total, r.balance_error, r.quantum_yield(q0=1.0)
```

`examples/` ports STRATIFY's `scripts/tst_*.m`; with matplotlib installed each saves a PNG next
to itself.

## Tests

```bash
pip install -e ".[test]"
pytest                    # 68 tests; the Octave ones are skipped without GNU Octave
```

* `test_riccati.py`: special functions against 50-digit mpmath.
* `test_solver.py`: amplitudes against a 60-digit evaluation of STRATIFY's products, interface
  conditions at every order, and batching.
* `test_physics.py`: independent physics. Textbook Mie, the Bohren–Huffman reference case, the
  optical theorem, plane-wave limits, continuity, quadrature, and energy conservation of decay
  rates for electric and magnetic dipoles.
* `test_octave.py`: runs the original MATLAB (`reference/stratify-matlab/`, unmodified) in GNU
  Octave. It checks that PyStratify matches wherever the MATLAB is right, and pins down each
  deviation, patching single lines to confirm root causes. `audit/impact.py` prints the size of
  the deviations on STRATIFY's own configurations.

The planar `1D/` folder of STRATIFY is not ported.

## Licence

GPL-3.0-or-later, as STRATIFY (this is a derivative work).
