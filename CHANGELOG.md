# Changelog

Notable changes to PyStratify. Every release is on [PyPI](https://pypi.org/project/pystratify/) and
[Zenodo](https://doi.org/10.5281/zenodo.23287561) (one DOI per version); `git log` has the full history.

## 0.11.0 — thin-film ellipsometry: dispersion models and fits

Two new modules; nothing already computed changes.

- **`pystratify.dispersion`**: parametric dielectric functions, wavelengths in nm and oscillator parameters in eV,
  composed with `+`: Cauchy (with an Urbach tail), Sellmeier, poles, Lorentz, Gaussian, Tauc-Lorentz, Cody-Lorentz
  (with an Urbach tail; ε₁ by a numerical Kramers–Kronig transform), Drude in energy and resistivity form,
  Bruggeman and Maxwell Garnett mixtures, tables (and `Table.from_refractiveindex` for a refractiveindex.info
  page), and `kramers_kronig_real`. The oscillator forms are multilayer.app's: pole, Lorentz and Tauc-Lorentz
  agree with its backend to 1e-15.
- **`pystratify.ellipsometry`**: Ψ, Δ and N, C, S of a film stack in the instrument (Nebraska) convention that
  Woollam, SENTECH and pyElli report (Δ = 180° − Δ of `planar.ellips`), vectorised over wavelength and angle on
  the planar solver's own kernel; incoherent backside reflection from a finite substrate, with the depolarisation
  it causes; readers for J.A. Woollam (CompleteEASE, WVASE) and SENTECH SpectraRay files; `fit`, a bounded
  least-squares fit reporting the instruments' MSE, 90 % confidence limits and parameter correlations, with a grid
  search for the starting point; and `fit_nk`, point-by-point optical constants.
- Checked against pyElli (`benchmarks/DISPERSION_CODES.md`, `benchmarks/ELLIPSOMETRY_CODES.md`): every
  dispersion model to 5e-15 (pyElli's two Drude models return ε₁ with the wrong sign, so they are compared
  reversed; its Cody-Lorentz ε₁ is a coarse 1000-point sum, 1.5e-3); Ψ, Δ of four dispersive stacks to 4e-12°
  with both of its solvers; both readers exactly; and a fit of pyElli's measured TiO₂ film, where parameters
  agree with pyElli + lmfit to 4e-7 and standard errors and MSE to all printed digits.
- `examples/ellipsometry_fit.py` fits two real measurements published with pyElli (a Woollam RC2 wafer and a
  SENTECH TiO₂ film, in `examples/data/ellipsometry/`).

## 0.10.6 — documentation and packaging

No change to any computed number. The README is a front page (figures, a quick start for every geometry and source,
the comparison with sixteen independent references); its former detail moved to this changelog, `docs/GUIDE.md`,
`docs/NUMERICS.md`, `benchmarks/README.md` and `tests/README.md`. PyPI now lists authors, keywords, classifiers and
links to the site, the guide, the issues and the DOI. The test that the committed docs manifest is current compares
check values to a tolerance (it failed on Linux on a 2e-16 rounding residual).

## 0.10.5 — cylinder emitters at a mode's cutoff

Next to a light line the reflected spectral density of a cylinder source loses digits: its m = ±1
TE/TM-coupled terms carry 1/q factors that cancel, ~1e-4 ε/(1 − b)² (1e-5 at 1e-7 from the line, noise
at 1e-10). A guided mode just above cutoff (HE₁ₙ modes approach it exponentially) puts a sharp feature
there, and the quadrature chased the noise until its budget ran out; with scipy 1.18 even a plain fibre
did so at tolerance 1e-7. Within 1e-3 of the host's and the source medium's light lines the real part now
comes from the power balance at that axial wavenumber, escape + absorbed − direct (direct = ¾(1 + b²)
transverse, 3/2(1 − b²) axial), which equals the reflected density to ~1e-14 wherever that is accurate;
elsewhere the balance stays an independent check. Converged results are unchanged.

## 0.10.4 — every guided mode of a cylinder

A point source beside or inside a lossless cylinder could lose the power it couples into some of the
cylinder's guided modes. The mode search minimised the dispersion determinant on a grid and kept a
minimum only if the minimiser's last value was below a fixed bar; at a root |det| is V-shaped, Brent's
parabolic steps stalled just above it, and real modes were dropped while the solve reported convergence.
Over 60 random lossless cylinders (electric and magnetic sources) the totals moved by more than 1e-3
in 28 of 97 comparable cases (up to 9.5%), and 11 that raised an error now solve. Roots are now
bracketed and judged at the root, against the determinant's own scale; the light line, where modes cut
off, is sampled geometrically and never evaluated within 1e-12 of; the residue's error estimate
measures the value used. For a step-index fibre the modes found are exactly
those of its closed-form dispersion relation, and the rates are smooth through a mode's cutoff. Found by
a 3D FDTD comparison with Meep
([benchmarks/FDTD_MEEP.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/FDTD_MEEP.md)).
Near fields are now exact at a sphere's centre ([docs/NUMERICS.md](https://github.com/multilayerapp/PyStratify/blob/main/docs/NUMERICS.md), *Exact field at the centre*).

## 0.10.3 — guided power from a source outside the stack

A point source in an exterior half-space of a film stack (above or below it) lost the power it couples
into the stack's guided modes: the transverse-resonance search ran at the source plane, where the
source side has no interface, and found no poles. Total rates near lossless guides were low (by 20%,
30 nm above 200 nm of TiO2 on glass) while the solve reported convergence; lossy stacks, and sources
inside a finite layer, were unaffected. The modes are now found inside the stack. smuthi and the
lossless limit of an absorbing guide confirm it
([benchmarks/EMITTER_REFERENCES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/EMITTER_REFERENCES.md)).

## 0.10.0 — focused illumination

A fourth source for all three geometries: a beam focused by an aplanatic objective, through an annulus (a
Cassegrain) or a disc, as in FT-IR microspectroscopy. What it computes, and how it is checked:
[docs/GUIDE.md, *Focused illumination*](https://github.com/multilayerapp/PyStratify/blob/main/docs/GUIDE.md#focused-illumination).

## 0.9.0 — films, cylinders and spheres unified

`solve_problem(Problem(...))` chooses the geometry's modal basis and uses one response core for films, cylinders and
spheres, under plane waves and electric and magnetic point dipoles:
[docs/GUIDE.md, *Films, cylinders and spheres*](https://github.com/multilayerapp/PyStratify/blob/main/docs/GUIDE.md#films-cylinders-and-spheres-one-problem-description).
