# Tests

```bash
pip install -e ".[test]"
pytest
```

* `test_planar_emission_references.py`: film emitter rates (electric and magnetic, perpendicular and
  parallel) against the Sommerfeld reference of `benchmarks/emitter_references.py`, itself checked
  against the image dipole at a perfect mirror; sources outside lossless guides against the lossless
  limit and smuthi (the latter skipped unless smuthi is installed).
* `test_gdm_spheres.py`: pyGDM2's volume dipoles within a few percent of PyStratify for dielectric
  spheres (skipped unless the `benchmarks` extra is installed).
* `test_mie_codes.py`: spheres against scattnlay and treams (skipped unless the `benchmarks` extra is
  installed; the CI job `benchmarks` installs it).
* `test_mnpbem_spheres.py`: MNPBEM's boundary elements approach PyStratify's sphere cross sections and
  decay rates as the mesh is refined (skipped unless GNU Octave is installed).
* `test_planar_codes.py`: the planar solver, its fields and its guided modes against PyMoosh, pyElli
  and the slab dispersion relation (skipped unless the
  `benchmarks` extra is installed; the CI job `benchmarks` installs it).
* `test_api_contract.py`: every name in `__all__` has exactly one place in `docs/API.md`
  (frozen, provisional, legacy or internal).
* `test_focused.py`: focused illumination ([docs/GUIDE.md](https://github.com/multilayerapp/PyStratify/blob/main/docs/GUIDE.md#focused-illumination)): every semi-analytic form against plane-wave
  superpositions through the existing engines, energy balance, limits, Köhler and Parseval.
* `test_normalized.py`: the normalized formulation (`normalized.py`: ψ'/ψ, ψξ and normalized j̄ only,
  reflection ratios swept outwards and inwards) against the solver, extended precision and the
  unnormalized formulas, including emitters and interfaces at real zeros of ψ_l, the small real
  part Re ψξ = ψ² and strongly amplifying shells.
* `test_routes.py`: the normalized route of `decay_rates` and `emission_rates` against the
  logarithmic one for every source (dipoles, chiral p ∥ m, general p + m, all electric- and
  magnetic-quadrupole components, orientation averages), energy conservation 1 to 0.05 nm from
  lossless interfaces, the energy-balance flag of the logarithmic route, the shift of a chiral
  emitter as the weighted sum of its electric and magnetic shifts, and the magnetic quadrupole by
  duality (a dual multilayer gives it the rates of the electric one).
* `test_shift.py`: the per-order Green's forms of `normalized_terms` and the frequency shift
  against mpmath transfer matrices (`references.layered_green_forms`, `layered_green_sums`),
  the quasi-static limit −Re r/(2 Im r) of the shift near gold, the automatic truncation
  (which includes the reactive tail the shift needs) and the vectorized emitter side.
* `test_riccati.py`: special functions against mpmath in every regime of [docs/NUMERICS.md](https://github.com/multilayerapp/PyStratify/blob/main/docs/NUMERICS.md), and the
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
  magnetic dipoles, up to 17000 orders. Partial-wave selections of the near field, the far field
  and the energy add up to the whole, and an electric dipole alone has the dipole's pattern.
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
  2 µm of metal, the published small-sphere polarisabilities, a chiral metal with Re(n − κ) < 0.
* `test_emitters.py`: emitters inside chiral particles against the achiral solver (κ → 0), the
  Pasteur interface conditions (tangential E and H, D_r = εE_r + iκH_r, B_r = μH_r − iκE_r)
  seen by emitters on either side of every interface, a (q, −iZq) source inside a dual chiral
  particle emitting one helicity only, the free chiral emitter's g_lum in a medium, orientation
  averages against the exact averages over the 24 rotations of the cube and 8 about an axis,
  quadrature of total and helicity-resolved power, and the mirror symmetry of valley excitons
  beside spheres (g_lum = 0 alone, ±g on either side or valley).
* `test_rates.py`: `decay_rates` (κ → 0), the reciprocity far field for lossless chiral
  multilayers (total = radiated, per helicity), energy balance with absorbing chiral, magnetic
  and gold layers down to 0.1 nm from interfaces, mirror symmetry (κ, m) → (−κ, −m), exact
  orientation averages, batches of positions, the free power in a chiral medium, normalisations
  and quantum yield, and Guzatov & Klimov's Eq. 46 for dielectric, lossy, metallic and
  double-negative chiral spheres.
* `test_sheets.py`: explicit films (O(d²) with the out-of-plane term, O(d) without it),
  sheets between chiral layers, the chiral and achiral solvers against each other to l = 400,
  graphene-coated-sphere plasmons against the quasi-static condition, graphene conductivity
  limits, plane-wave absorption = Poynting-flux jump = sheet-loss formula = extinction −
  scattering, emitter energy balance and far fields with sheets.
* `test_quadrupoles.py`: the coupling against 40-digit finite differences, Jackson's free power,
  total = radiated and energy balance with quadrupoles, the dipole-pair limit in amplitude, the
  two far-field routes against each other, pattern quadrature, the icosahedral design, and
  current-loop moments.
* `test_pairs.py`: the two-point Green's dyadic (`green_dyadic`): the multipole series of the free
  field against the closed-form free dyadic, reciprocity, the coinciding limit (decay rates and
  shift), the radial two-point forms in one shell and across a gold shell (transmission chain)
  against mpmath order by order, and the interface conditions approached from either side.
* `test_feibelman.py`: Feibelman d-parameters against the mesoscopic Mie theory of Gonçalves et
  al. (2020), Eqs. (4a) and (13a); the per-order forms of a nanoshell with d-parameters on both
  surfaces against mpmath matrices built from the boundary conditions; energy balance including
  the surface absorption; the TE response as the sheet σ = ik₀d∥(ε_other − ε_metal).
* `test_chiral_normalized.py`: chiral layers in the normalized formulation: the achiral limit is
  the scalar sweep, the 2 × 2 reflection matrices against 80-digit 4 × 4 transfer matrices, energy
  balance, agreement with the logarithmic route and mirror symmetry for chiral emitters, and
  emitters inside a chiral layer routed to the logarithmic formulation.
* `test_ensemble.py`: shell averages (free space, graded nodes, the d_min⁻² cutoff next to gold,
  the thin-shell limit) and the spectral density J(ω).
* `test_casimir.py`: the Casimir–Polder sums at imaginary frequency against mpmath transfer
  matrices, the non-retarded limit −C₃/d³ next to a large Drude sphere, and the convergence of the
  frequency quadrature.

`pystratify.references` holds the independent references (mpmath Mie decay rates and extinction,
layered-sphere decay rates from transfer matrices, BHMIE); nothing in it shares code with the package.

What runs where: the `tests` workflow (`.github/workflows/tests.yml`) runs `pytest` on Python 3.10 and 3.12 on every
push and pull request, and its `benchmarks` job installs the `benchmarks` extra and runs the comparisons with PyMoosh,
pyElli, scattnlay, treams and pyGDM2 (`test_planar_codes.py`, `test_mie_codes.py`, `test_gdm_spheres.py`).
Comparisons whose reference needs its own environment (smuthi, Meep, NGSolve, MNPBEM under Octave, tmmax) skip
there and are rerun with `python benchmarks/verify_all.py` ([benchmarks/README.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/README.md)).
