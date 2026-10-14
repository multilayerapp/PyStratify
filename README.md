# PyStratify

[![PyPI](https://img.shields.io/pypi/v/pystratify)](https://pypi.org/project/pystratify/)
[![Python](https://img.shields.io/pypi/pyversions/pystratify)](https://pypi.org/project/pystratify/)
[![tests](https://github.com/multilayerapp/PyStratify/actions/workflows/tests.yml/badge.svg)](https://github.com/multilayerapp/PyStratify/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23287561.svg)](https://doi.org/10.5281/zenodo.23287561)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](https://github.com/multilayerapp/PyStratify/blob/main/LICENSE)

**Light scattering and emission by multilayered films, cylinders and spheres, in Python.** One modal-response core
solves all three geometries under plane waves, electric and magnetic point dipoles and focused (Cassegrain) beams,
with numerics rebuilt so that every quantity stays finite and accurate: thin metal shells, multipole orders in the
thousands, emitters a fraction of a nanometre from a surface.

![Reflectance of a Bragg mirror, scattering by a silicon nanowire, and the multipoles of a gold nanoshell, computed with PyStratify](https://raw.githubusercontent.com/multilayerapp/PyStratify/main/docs/images/geometries.png)

PyStratify is the engine of **[multilayer.app](https://multilayer.app)**, a free browser calculator for the same
three geometries, with a [theory guide](https://multilayer.app/learn/index.html) and every
[comparison with other codes](https://multilayer.app/verification.html) plotted.

## Why PyStratify

- **Three geometries, one method.** The transfer-matrix method for films (after Byrnes's
  [`tmm`](https://github.com/sbyrnes321/tmm), whose function names it keeps), its cylindrical analogue for coated
  wires and fibres, and for spheres the recursive transfer-matrix method of Moroz (2005) as used by
  [STRATIFY](https://gitlab.com/iliarasskazov/stratify) (Rasskazov, Carney & Moroz,
  [OSA Continuum 3, 2290 (2020)](https://doi.org/10.1364/OSAC.399979)).
- **Sources beyond the plane wave.** Electric and magnetic dipoles inside or outside the structure, with decay
  rates split into radiated, guided and absorbed power; focused beams as in FT-IR microspectroscopy.
- **Spheres go further.** Absorbing, magnetic, gain and chiral (Pasteur) shells; 2D materials and Feibelman
  d-parameters on the interfaces; electric, magnetic, chiral and quadrupole emitters; angle-resolved far fields,
  near fields, stored energy, Casimir–Polder potentials and two-point Green's dyadics.
- **Accurate where others break down.** Special functions as complex logarithms and a scaled two-sided recursion
  keep 1e-12 agreement with 60-digit references, energy balance to 1e-14 at 0.05 nm from a lossless interface, and
  absorbing spheres to size parameter 20 000 ([docs/NUMERICS.md](https://github.com/multilayerapp/PyStratify/blob/main/docs/NUMERICS.md)).
- **Checked in public.** Every comparison with another code is a script in `benchmarks/` that anyone can rerun
  ([below](#verified-against-independent-codes)).
- **Light to install.** Pure Python on NumPy and SciPy; whole spectra are vectorised.

## Install

```bash
pip install pystratify
```

Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10. The development version:
`pip install git+https://github.com/multilayerapp/PyStratify.git`.

## Quick start

Lengths share one unit of your choosing (here nanometres), angles are in radians, and indices are n + ik. No optical
constants are shipped: take them from a database such as [refractiveindex.info](https://refractiveindex.info).

**Films.** The planar module keeps the [`tmm`](https://github.com/sbyrnes321/tmm) interface: here a quarter-wave
Bragg mirror, 8 pairs of TiO₂ / SiO₂ on glass.

```python
import numpy as np
import pystratify as ps
from pystratify import planar

n_list = [1.0] + [2.35, 1.46] * 8 + [1.52]                    # incident medium, layers, substrate
d_list = [np.inf] + [600 / 4 / 2.35, 600 / 4 / 1.46] * 8 + [np.inf]
R = [planar.coh_tmm("s", n_list, d_list, 0.0, wl)["R"] for wl in np.linspace(400, 900, 501)]
```

**Cylinders.** A silicon nanowire of radius 80 nm at normal incidence, both polarisations.

```python
wire = ps.solve_cylinder([80.0], [3.6, 1.0], 775.0)            # radii, indices (host last), wavelength
ps.cross_widths(wire, "axial-electric")["efficiency_scattering"]
ps.cross_widths(wire, "axial-magnetic")["efficiency_scattering"]
```

**Spheres.** A silica–gold nanoshell in water over a whole spectrum, with its multipole decomposition.

```python
wavelength = np.linspace(450, 1100, 651)
n_gold = 0.2 + 3.5j                                           # use tabulated n, k for real work
n = np.stack([np.full(651, 1.45), np.full(651, n_gold), np.full(651, 1.33)], axis=1)
sol = ps.solve([60.0, 75.0], n, wavelength)                   # radii of every interface, n per shell
cs = ps.cross_sections(sol)
cs.q_ext, cs.q_sca, cs.q_abs                                  # efficiencies over wavelength
cs.ext_by_order[:, ps.TM, 0] / cs.geometric                   # the electric dipole's share
```

**Emitters.** Decay rates of a dipole beside a sphere, and above a silver mirror through the unified
`solve_problem` interface, which takes the same arguments for all three geometries.

```python
rates = ps.decay_rates([40.0], [n_gold, 1.0], 600.0, r=40.0 + np.geomspace(0.2, 100, 60))
rates.total, rates.radiative, rates.nonradiative               # (positions, [radial, tangential])

mirror = ps.Problem("films", [np.inf, 200.0, np.inf], [1.0, 0.135 + 3.99j, 1.52], 633.0,
                    ps.PointDipole(-100.0, "electric", layer=0))  # 100 nm above 200 nm of silver on glass
result = ps.solve_problem(mirror)
result["total"], result["escape"], result["absorbed"]          # perpendicular, parallel, isotropic
```

![Decay rates of a dipole beside a gold sphere and above a silver mirror](https://raw.githubusercontent.com/multilayerapp/PyStratify/main/docs/images/emitters.png)

**Focused beams.** An FT-IR objective (NA 0.5, central obscuration 0.23) on a polymer fibre.

```python
beam = ps.FocusedBeam(0.5, 0.23)                              # NA, obscuration NA
wavenumber = np.linspace(1000, 4000, 61)                      # cm⁻¹
fibre = ps.focused_cylinders([5.0], [1.49 + 0.01j, 1.0], 1e4 / wavenumber, beam)   # lengths in µm here
fibre.apparent_absorbance
```

Far more is in the **[user guide](https://github.com/multilayerapp/PyStratify/blob/main/docs/GUIDE.md)**:
conventions, every source and output, chiral media, 2D sheets and the full sphere toolbox. The runnable scripts in
[`examples/`](https://github.com/multilayerapp/PyStratify/tree/main/examples) (multipole spectra, scattering patterns,
near-field maps, decay rates, circular dichroism, quadrupole emitters, graphene-coated spheres…) each save a figure,
and [`crosscheck/`](https://github.com/multilayerapp/PyStratify/tree/main/crosscheck) recomputes figures of classic
papers from their stated parameters.

## Verified against independent codes

![PyStratify's largest differences from sixteen independent references, and a dipole beside a fibre against 3D FDTD](https://raw.githubusercontent.com/multilayerapp/PyStratify/main/docs/images/verification.png)

Codes that solve the same equations exactly agree with PyStratify to between 1e-15 and 2e-6, the larger differences
being the reference's own precision (float32 in tmmax, a contour integral in smuthi). Solvers that discretize space
agree to their mesh or grid error, 0.7–4%; for Meep and MNPBEM the difference was shown to shrink as the grid or mesh
is refined. Each point is a script in
[`benchmarks/`](https://github.com/multilayerapp/PyStratify/tree/main/benchmarks) that writes its own report, and
`python benchmarks/verify_all.py` reruns the whole study in pinned environments
([benchmarks/README.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/README.md)).
[MATRIX.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/MATRIX.md) maps every check by geometry,
source and quantity, and the [Verification page](https://multilayer.app/verification.html) plots each one.

## Speed

Typical timings on one core:

| task | time |
|---|---|
| 601-wavelength spectrum, gold nanoshell | 25 ms |
| sphere of size parameter 500, all orders | 8 ms |
| decay rates at 100 positions 0.5–25 nm from a gold nanoshell | 0.15 s |
| scattering pattern on a 181 × 361 (θ, φ) grid | 10 ms |
| 300 × 300 near-field map | 1.4 s |

More, and the accuracy to expect in every regime: [docs/NUMERICS.md](https://github.com/multilayerapp/PyStratify/blob/main/docs/NUMERICS.md).

## Documentation

| | |
|---|---|
| [User guide](https://github.com/multilayerapp/PyStratify/blob/main/docs/GUIDE.md) | the problem description, focused beams, the sphere toolbox, conventions |
| [API contract](https://github.com/multilayerapp/PyStratify/blob/main/docs/API.md) | which names are stable for 1.x, provisional, legacy or internal |
| [Numerics](https://github.com/multilayerapp/PyStratify/blob/main/docs/NUMERICS.md) | how accuracy is kept, accuracy to expect, known limitations, timings |
| [Benchmarks](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/README.md) | every comparison with another code, and how to rerun it |
| [Tests](https://github.com/multilayerapp/PyStratify/blob/main/tests/README.md) | what each test file checks |
| [Changelog](https://github.com/multilayerapp/PyStratify/blob/main/CHANGELOG.md) | what changed in each release |
| [Audit of STRATIFY](https://github.com/multilayerapp/PyStratify/blob/main/AUDIT.md) | the defects of the MATLAB code that PyStratify fixes |
| [Learn](https://multilayer.app/learn/index.html) | the theory, from Maxwell's equations to each geometry, with live figures |

PyStratify is before 1.0: the names listed as frozen in the API contract keep their meaning from 1.0 on.
Questions and bug reports are welcome in [issues](https://github.com/multilayerapp/PyStratify/issues).

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Citing

PyStratify implements the method of STRATIFY; please cite the paper,

> I. L. Rasskazov, P. S. Carney and A. Moroz, "STRATIFY: a comprehensive and versatile MATLAB code
> for a multilayered sphere," *OSA Continuum* **3**, 2290 (2020),
> [doi:10.1364/OSAC.399979](https://doi.org/10.1364/OSAC.399979),

and the software itself: [doi:10.5281/zenodo.23287561](https://doi.org/10.5281/zenodo.23287561)
names every version and resolves to the latest; each release also has its own DOI on that Zenodo
page, which is the one to cite for the exact code you ran
([CITATION.cff](https://github.com/multilayerapp/PyStratify/blob/main/CITATION.cff)).

## Licence

GPL-3.0-or-later, as STRATIFY (this is a derivative work). The planar routines derive from Byrnes's `tmm` and keep
its MIT notice (`src/pystratify/PLANAR-LICENSE.txt`).
