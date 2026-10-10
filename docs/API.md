# PyStratify public API — the 1.0 contract

**Status: ruled by Ilia on 2026-10-10** (roadmap WP0.4, `multilayerapp/docs/roadmap/README.md`);
it binds from 1.0. Every name in `pystratify.__all__` (72 at 0.10.2) sits in exactly one of the
four lists below — 56 frozen, 8 provisional, 4 legacy, 4 internal; `tests/test_api_contract.py`
fails if a name is added to `__all__` without being placed here, or placed twice. The judgement
calls and their reasons are under **Rulings** at the end.

## What the four lists promise

- **Frozen** — stable for every 1.x release: the name, the positional order, keyword names, the
  attribute and key names of what it returns, units and conventions. 1.x may *add* (a keyword with a
  default, an attribute, a dict key, a new source or geometry); it may not rename, reorder or change
  a meaning. A change of meaning waits for 2.0.
- **Provisional** — shipped, tested and documented, but its signature may change in a 1.x minor
  release, with a changelog note. For features whose natural home may move under `Problem` once
  films and cylinders gain the same quantity.
- **Legacy** — works unchanged through 1.x, stays tested and documented, but a frozen entry point
  does the same job with the same numbers. New features land only on the frozen path. A
  `DeprecationWarning` no earlier than 1.1; removal no earlier than 2.0.
- **Internal** — leaves `__all__` at 1.0. Still importable from its module (`pystratify.farfield`,
  …), but with no promise; a numerical building block, not an API.

How the lists were drawn: the unified entry points the roadmap names; every type those entry points
*return* and every function that *consumes* those types (a frozen function that hands back a type
you cannot use is not frozen); the documented sphere family (a README row, an example or the site's
backend calls it); and the helpers that build *inputs* (sheets, Drude damping). Usage evidence at
0.10.1, counted with word-boundary matches: README rows, `examples/`, `tests/`, and the backend
(`multilayer/backend`, which runs the live site).

## Frozen

### The unified contract (films, cylinders, spheres)

| name | kind | why frozen |
|---|---|---|
| `Problem` | class | the one problem description: geometry, dimensions, indices, wavelength, source |
| `PlaneWave` | class | source |
| `PointDipole` | class | source |
| `FocusedBeam` | class | source |
| `solve_problem` | function | the one entry point; the backend's cylinder and dipole jobs call it |
| `solve_focused` | function | what `solve_problem` dispatches a `FocusedBeam` to; `test_solve_problem_focused_and_validation` covers all three geometries |

### Geometry-native solutions and what reads them

`solve_problem` returns the geometry's native solution under `"solution"`, so these types and the
functions that consume them are part of its contract.

| name | kind | why frozen |
|---|---|---|
| `Solution` | class | the sphere solution (`solve`, and `solve_problem` for a sphere under a plane wave) |
| `TM` | constant | index into per-polarisation arrays of `Solution` results |
| `TE` | constant | as `TM` |
| `CylinderSolution` | class | the cylinder solution `solve_problem` returns |
| `solve_cylinder` | function | the cylinder analogue of `solve` (oblique incidence via `beta`, `m_max`): a direct, batched entry point to the native solution (ruled) |
| `cross_widths` | function | reads a `CylinderSolution`; the backend calls it per polarisation |
| `cylinder_pattern` | function | reads a `CylinderSolution` |
| `truncation_order` | function | the multipole order every solver uses; the backend picks `l_max` with it |

### Focused beams, batched

`solve_focused` is one wavelength and one beam. The site sweeps NA and offset, so it calls these.

| name | kind | why frozen |
|---|---|---|
| `focused_films` | function | many beams on a film in one call; backend |
| `focused_spheres` | function | the same for spheres; backend |
| `focused_cylinders` | function | the same for cylinders; backend |
| `FocusedResult` | class | what the three return |
| `focused_field_films` | function | near field under a focused beam; backend |
| `focused_field_spheres` | function | backend |
| `focused_field_cylinders` | function | backend |
| `FocusedField` | class | what the three return |

### Spheres: scattering, fields, energy

| name | kind | why frozen |
|---|---|---|
| `solve` | function | the batched sphere solver; README quick start, 14 example uses, backend |
| `cross_sections` | function | backend; README |
| `CrossSections` | class | returned by `cross_sections` |
| `scattering_amplitudes` | function | backend |
| `amplitude_matrix` | function | README row (S1..S4) |
| `mueller_matrix` | function | README row |
| `scattering_pattern` | function | README, three examples |
| `ScatteringPattern` | class | returned by `scattering_pattern` |
| `near_field` | function | backend |
| `NearField` | class | returned by `near_field` |
| `energy_density` | function | backend |
| `EnergyDensity` | class | returned by `energy_density` |
| `shell_energy` | function | backend |
| `ShellEnergy` | class | returned by `shell_energy` |
| `electric_prefactor` | function | builds the `prefactors` *input* of `energy_density`/`shell_energy`; two examples |
| `energy_prefactors` | function | the same for every shell at once |

### Spheres: emitters and rates

| name | kind | why frozen |
|---|---|---|
| `decay_rates` | function | named in the roadmap; backend |
| `DecayRates` | class | returned by `decay_rates` |
| `emission_rates` | function | named in the roadmap; any source in any lossless layer; backend |
| `EmissionRates` | class | returned by `emission_rates` |
| `dipole_far_field` | function | backend; seven examples |
| `EmissionPattern` | class | returned by `dipole_far_field` |
| `source_covariance` | function | orientation averages for `dipole_far_field`; README row |

### Spheres: chiral media

| name | kind | why frozen |
|---|---|---|
| `solve_chiral` | function | Pasteur shells; README, tested. Its convention, D = εE + iκH, B = μH − iκE (Gaussian, e^{−iωt}, treams'; helicity ±1 sees n ± κ), is **the package-wide one** (ruled): WP1.4's planar and cylindrical chirality adopt it, converting internally if a formulation prefers another |
| `ChiralSolution` | class | returned by `solve_chiral` |
| `helicity_cross_sections` | function | CD of a chiral shell; README |
| `HelicityCrossSections` | class | returned by `helicity_cross_sections` |

### Media and interfaces (inputs)

| name | kind | why frozen |
|---|---|---|
| `Sheet` | class | 2D materials on an interface; README, two examples |
| `Feibelman` | class | d-parameters; README |
| `graphene_conductivity` | function | README; example |
| `DRUDE` | constant | the Drude damping table; README |
| `DrudeModel` | class | the entries of `DRUDE`, and what `energy_prefactors(drude=)` accepts |
| `free_path_correction` | function | thin-shell damping; four examples |
| `surface_damping_wavelength` | function | the damping wavelength `electric_prefactor` takes; three examples |

## Provisional

Sphere-specific signatures (radii, n, wavelength, …) for quantities films and cylinders do not have
yet; when they do, the shared form may live under `Problem`.

| name | kind | why provisional |
|---|---|---|
| `shell_average` | function | emitters spread through a shell; README row, tested |
| `ShellAverage` | class | returned by `shell_average` |
| `spectral_density` | function | J(ω) and shift over wavelength; README row, tested |
| `SpectralDensity` | class | returned by `spectral_density` |
| `casimir_polder` | function | README row, tested |
| `CasimirPolder` | class | returned by `casimir_polder` |
| `green_dyadic` | function | two-point Green's dyadic; README row, tested |
| `GreenDyadic` | class | returned by `green_dyadic` |

## Legacy

| name | kind | superseded by |
|---|---|---|
| `normalized_decay_rates` | function | `decay_rates`, whose default `route="auto"` takes total, radiative and shift from the same normalized formulation and adds the losses |
| `NormalizedRates` | class | returned by `normalized_decay_rates` |
| `normalized_terms` | function | the per-order Green's forms inside the normalized route; kept reachable as a documented research hook |
| `NormalizedTerms` | class | returned by `normalized_terms` |

## Internal (leave `__all__` at 1.0)

| name | kind | where it stays importable | why |
|---|---|---|---|
| `angular_functions` | function | `pystratify.farfield` | Bohren–Huffman π_l, τ_l; no README row, one test |
| `log_riccati` | function | `pystratify.riccati` | the logarithmic Riccati–Bessel kernel; tests only |
| `locate_shell` | function | `pystratify.decay` | `searchsorted` on the radii; tests only |
| `focal_field` | function | `pystratify.focused` | the focal integrals under `focused_field_*`; one test |

## Submodules

| module | status | notes |
|---|---|---|
| `pystratify.planar` | **frozen** (its public functions) | Byrnes's `tmm` lineage (MIT notice in `PLANAR-LICENSE.txt`), so its signatures are the ones `tmm` users already know: `coh_tmm`, `inc_tmm`, `coh_tmm_wavelength_sweep`, `position_resolved`, `absorp_in_each_layer`, `inc_absorp_in_each_layer`, `pim_tmm`, `pim_tmm_direct`, `pim_absorp_in_each_layer`, `phase_average_tmm`, `layered_response`, `ellips`, `unpolarized_RT`, `coh_tmm_reverse`, `interface_r`/`interface_t`/`interface_R`/`interface_T`, `snell`, `list_snell`, `admittance_index`, `is_forward_angle`, `is_evanescent_angle`, `find_in_structure`, `find_in_structure_with_inf`, `layer_starts`, `inc_group_layers`, `inc_find_absorp_analytic_fn`, `power_entering_from_r`, `R_from_r`, `T_from_t`, `make_2x2_array`. The site's whole film calculator runs on it (`multilayer/backend/tmm/tmm_core.py`). **It has no `__all__`**, so the backend's `from pystratify.planar import *` also re-exports `numpy` and the module's imports; 1.0 should give it one. |
| `pystratify.references` | **frozen** (its `__all__`) | extended-precision and classical references (`bhmie`, `sphere_extinction`, …); needs `mpmath` |
| `pystratify.problem.check_focused` | internal | the backend imports it (`multilayer/backend/emission.py`) to validate a beam before a job; 1.0 should either export it as a frozen validator or let the backend construct a `Problem` and catch its error |
| everything else | internal | including the private planar helpers the backend's tests reach for (`_prepare_coherent_inputs`, `_coh_tmm_amplitudes_numpy`, `_coh_tmm_amplitudes_jax`) |

## Rulings (Ilia, 2026-10-10)

1. **Research-grade sphere features are provisional**, not frozen: `shell_average`, `spectral_density`,
   `casimir_polder`, `green_dyadic` and their result types may be reshaped in a 1.x minor (with a
   changelog note) when films and cylinders gain the same quantities.
2. **One chirality convention for the whole package**: the spheres' D = εE + iκH, B = μH − iκE
   (Gaussian, e^{−iωt}). WP1.4 adopts it for films and cylinders, so `solve_chiral` and
   `helicity_cross_sections` are frozen now.
3. **`solve_cylinder` is frozen**, with the same standing as the sphere `solve`.
4. **The normalized route is legacy**: `normalized_decay_rates`, `normalized_terms` and their types
   are kept and documented through 1.x, superseded by `decay_rates`/`emission_rates`.
5. The README's 0.9.0 wording "advanced legacy sphere APIs" is replaced: the sphere API is part of
   the stable surface, as listed here.

Still to do before 1.0 is tagged: give `pystratify.planar` an `__all__` (its public functions, as
listed above), and either export `check_focused` or have the backend validate by constructing a
`Problem`.
