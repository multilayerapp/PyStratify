# Changelog

Notable changes to PyStratify. Every release is on [PyPI](https://pypi.org/project/pystratify/) and
[Zenodo](https://doi.org/10.5281/zenodo.23287561) (one DOI per version); `git log` has the full history.

## Unreleased — one stable film kernel, and one owner of the material physics

One new module (`stable`), and (assurance WP07-D2, 2026-11-03) the physics that multilayer.app's API and the
replication-corpus harvester each kept their own copy of: the films' H field, ellipsometric angles and
incoherent-layer field profile, and the material code. Every displaced implementation is kept as a frozen test
oracle (`tests/oracles/`) or a pinned benchmark reference. Three fixes: one changes results only at a layer's exact
critical angle, one returns Δ = NaN where a reflected amplitude vanishes, and one corrects the field profile of
stacks with incoherent layers (O-W7-12).

- **`pystratify.rii`**: a refractiveindex.info page as a `Dispersion` (`RIIPage`, `RIIDatabase`). Formulas 1–9
  (wavelength in µm), `tabulated n`, `k` and `nk`, a formula with a tabulated k; the first covering entry wins, a
  wavelength no entry covers raises `ValueError` (no extrapolation). It is the multilayer.app API's evaluator
  ported unchanged; where the pip `refractiveindex` package differs (unsorted repeated rows, a formula beside a
  table, an even-length formula 2) it follows the API. **Database identity**: `database_identity` returns the
  SHA-256 of a database (the digest the API pins for its a66ef88 snapshot, reproduced) and the git commit of a
  checkout; `RIIDatabase(root, expected_sha256=…)` and `verify_database` refuse any other database (closes
  O-W7-2: the harvester read its copy with no identity check). Needs PyYAML (`pystratify[rii]`).
  `benchmarks/REFRACTIVEINDEX_PAGES.md`: the pip package and the frozen API evaluator against it on every page of the
  database. `tests/test_rii.py` on small CC0 fixture pages.
- **`dispersion.TaucLorentzUrbach`, `ForouhiBloomer`, `DrudeCarrier`**: from the harvester. Tauc-Lorentz with an
  Urbach tail below `Ec` (Foldyna's TLU; the Urbach energy from derivative continuity, not the printed Eq. 37
  factor, and the corrected second line of the Urbach Kramers-Kronig term; ε₁(∞) is yours to add with
  `EpsilonInf`); the single-term Forouhi-Bloomer n,k model; free carriers from a density N (m⁻³), m* and ħγ.
  `tests/test_dispersion_families.py`.
- **`drude.DrudeEnergyModel`, `fit_drude`, `free_path_permittivity`, `layer_effective_path`,
  `free_electron_fermi_velocity`**: from the API: eps_inf − ωp²/(ω(ω+iγ)) fitted to a metal's own permittivity (the
  1/|ε|-weighted fit with ε∞ ≥ 1), and the surface-damping correction γ_s = γ + A v_F / L_eff. Lengths in nm
  (the API used µm); constants from `scipy.constants` (the API's truncated ħ differs by 7.7e-11). The module's
  `DrudeModel`, `DRUDE`, `effective_path` and `free_path_correction` are unchanged. `tests/test_drude_energy.py`.

- **`planar.coh_tmm` at a layer's own critical angle** (WP07, O-W7-10): where cos θ = 0 inside an interior layer
  the reflection recursion lost the 0·∞ limit between a perfectly reflecting interface and a vanishing phase, and
  returned R_s = 1 for an air gap between glass prisms whose R_s is 0.3304 (Byrnes's `tmm` does the same). Wherever
  an interior layer has |cos θ| < `CRITICAL_COS` = 10⁻⁶ (the recursion's ε/|cos θ| error passes 2e-10 there),
  `coh_tmm` and `coh_tmm_wavelength_sweep` now take r, t, R and T from `stable.stable_tmm`, and the per-layer
  outputs (`absorp_in_each_layer`, `position_resolved`, `absorp_analytic_fn`, the sweep's layer absorption) refuse
  or return NaN there, since no forward/backward split exists. Every other point is computed exactly as before.
  `tests/test_planar_critical.py`.

- **`pystratify.stable`**: the coherent planar kernel of multilayer.app's replication corpus and calculator (WP07,
  2026-11-03), which each kept their own copy until now. `stable_tmm` solves a stack over a whole grid of wavelengths
  and angles by Redheffer star products of bounded scattering matrices, so any layer count and any absorption stay
  finite; repeated blocks, written as nestable `Repeat` groups or detected in a flat stack, are raised by binary
  doubling (131072 layers × 401 wavelengths in under a millisecond). It adds uniaxial layers with the optic axis
  along the normal, a perfect-conductor exit and a Bloch-terminated exit (a semi-infinite periodic crystal), in
  `planar.coh_tmm`'s conventions, and refuses with `ValueError` what it cannot compute unambiguously. A layer at its
  own critical angle, where the plane-wave basis degenerates (`planar.coh_tmm` returns R_s = 1 at that exact angle
  for a stack whose R_s is 0.337), is taken from its characteristic matrix.
- Checked (`benchmarks/STABLE_CHECKS.md`): against `planar.coh_tmm` to ≤ 1.8e-14 (1.7e-12 on 8192 layers), uniaxial
  and critical-angle layers against a direct integration of Maxwell's equations to ≤ 3.3e-15, the PEC and Bloch
  exits against their closed forms and limits, and the harvester's own kernels to ≤ 3.4e-14 through a committed
  fixture (`benchmarks/stable_fixtures.py`). `benchmarks/byrnes_tmm.py` compares `planar.coh_tmm`, `inc_tmm` and
  `stable_tmm` with Byrnes's `tmm` 0.2.0 (≤ 1e-14; lineage), now in the `benchmarks` extra.

- **`planar.psi_delta(r_s, r_p)`** (WP07-D2): ψ = atan2(|r_p|, |r_s|) and Δ = arg(−r_p r_s*) in (−π, π], in
  radians, vectorised. An amplitude at or below 64 ε max(|r_s|, |r_p|, 1) is rounding, so ψ needs one amplitude above
  that floor and Δ needs both; an undefined angle is NaN. This is multilayer.app's rule: the API's
  `serialize_ellipsometry_amplitudes`, which `psi_delta` replaces, is frozen in `tests/oracles/`, and on 3000 random
  stacks the two serialize identically after the API's degree and [0°, 360°) formatting (1.4e-14° unrounded).
  `ellips` now calls it and takes `mu_list`, passed to `coh_tmm`. **Fix:** at a Brewster angle `ellips` returned
  Δ = −π, the phase of a rounding-level r_p; it now returns NaN. Everywhere else ψ and Δ move by at most 4e-16 rad
  (atan2 in place of the arctan of a ratio). `tests/test_planar_api_oracles.py`.
- **H in `planar.position_resolved`** (WP07-D2): `Hx`, `Hy`, `Hz` (complex), in units of the incident wave's
  |H₀| = |y₀| |E₀|, with y = n/μ_r. They come from the same forward and backward amplitudes as E (s: Hx = −(y/|y₀|)
  cos θ (E_f − E_b), Hz = (y/|y₀|) sin θ (E_f + E_b); p: Hy = (y/|y₀|)(E_f + E_b)). On 400 random stacks with absorbing
  and magnetic layers, Re(E × H*)_z equals `poyn` to 4.4e-16, and tangential E and H, normal B and normal D are
  continuous to 9e-16. On 2000 random stacks H is bit for bit the API's `magnetic_field_components` (frozen in
  `tests/oracles/`) at all 70 618 points. E, `poyn` and `absor` are unchanged (bit for bit
  against the previous code on 8820 points).
- **`planar.inc_field(layer, distance, inc_data)`** (WP07-D2): phase-averaged |E|²/|E₀|² and |H|²/|H₀|² at any depth
  of a stack with incoherent layers, from `inc_tmm`'s output, with `distance` a scalar or an array. `inc_tmm`'s
  intensities are powers (normal Poynting flux over the incident flux), so each is converted to the amplitude² of the
  medium it travels in through Re(n cos θ) (s) or Re(n cos θ*) (p), lossy media and complex angles included; a coherent
  layer weights the stack's forward and reversed solutions by F and B converted to the amplitude² of their own incident
  media (`docs/NUMERICS.md`). **Fixes O-W7-12**: multilayer.app's API plotted the powers themselves. Inside an
  incoherent layer that read |E|² too high by Re(n cos θ)/(n₀ cos θ₀) (×1.5 in glass, ×3.5 for n = 3.5, ×1.87 for p at
  45°) and |H|² too low (×0.67, ×0.29, ×0.83). A coherent film read 1–3 % off in front of a thick slab and 50 % off
  behind one. The check is a band-averaged coherent field (`coh_tmm` + `position_resolved` alone, Hann-weighted over
  whole cycles of every standing wave), and it agrees to 9e-14 for lossless slabs, 8e-10 and 5e-8 for absorbing ones,
  and 6e-6 for coherent films on a 4 mm slab. The closed form of a lossless slab agrees to 4e-16, and a coherent stack
  routed through `inc_tmm` gives `position_resolved`'s field to 3e-15 (`benchmarks/planar_field_checks.py` →
  `PLANAR_FIELD_CHECKS.md`, `tests/test_planar_fields.py`).
- `planar.inc_tmm` also returns `th_list`, `pol`, `n_list`, `d_list`, `c_list`, `th_0` and `lam_vac`, which
  `inc_field` reads. Nothing it already returned changes.

## 0.11.0 — thin-film ellipsometry: dispersion models and fits

Two new modules; nothing already computed changes.

- **`pystratify.dispersion`**: parametric dielectric functions, wavelengths in nm and oscillator parameters in eV,
  composed with `+`: Cauchy (with an Urbach tail), Sellmeier, poles, Lorentz, Gaussian, Tauc-Lorentz, Cody-Lorentz
  (with an Urbach tail; ε₁ by a numerical Kramers–Kronig transform), Drude in energy and resistivity form,
  Bruggeman (also `bruggeman_epsilon` on bare permittivities) and Maxwell Garnett mixtures, tables (and `Table.from_refractiveindex` for a refractiveindex.info
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
