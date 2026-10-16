# PyStratify user guide

The detail behind the [README](https://github.com/multilayerapp/PyStratify/blob/main/README.md)'s quick start: the problem description shared by the three geometries,
thin-film ellipsometry (dispersion models, measurement files and fits), the focused beam, and the full sphere toolbox with its conventions. Which names are the stable 1.x surface:
[docs/API.md](https://github.com/multilayerapp/PyStratify/blob/main/docs/API.md). How the numbers are made accurate: [docs/NUMERICS.md](https://github.com/multilayerapp/PyStratify/blob/main/docs/NUMERICS.md).

## Films, cylinders and spheres: one problem description

`solve_problem(Problem(...))` chooses the geometry's modal basis and uses the owned
response core. NumPy/SciPy supply numerical primitives; external optical engines
are benchmarks. The development contract accepts passive, isotropic, nonmagnetic
media, plane waves, and electric/magnetic **3D point dipoles** in lossless source
and exterior regions. Cylinder point sources integrate the continuous axial spectrum.

```python
import numpy as np
import pystratify as ps

film = ps.Problem("films", [np.inf, 0.2, np.inf], [1, 1.5, 1], 0.6,
                  ps.PointDipole(0.1, "electric", layer=1,
                                 intrinsic_quantum_yield=0.8))
rates = ps.solve_problem(film)
assert rates["diagnostics"]["converged"]

cylinder = ps.Problem("cylinders", [0.05, 0.1], [1.5, 2, 1], 0.6,
                      ps.PlaneWave(angle=np.pi / 3))
widths = ps.solve_problem(cylinder)
```

Lengths share one caller-selected unit; angles are radians. Film thickness arrays
include infinite exteriors, and source positions are depth within the specified
region (negative in the incident region). Radial arrays are increasing interface
radii with the exterior index last; source positions are absolute radii. Cylinder
plane-wave angles are from +z. Widths use total incident irradiance and have length
units; sphere cross sections have area units.

Rates include canonical orientations and their isotropic average, `total`, `escape`,
`guided`, `absorbed`, efficiencies and diagnostics. The reference is the same dipole
in its own homogeneous medium. `lifetime_factor = (1-q0) + q0*total` is the inverse
lifetime ratio; its reciprocal is the lifetime ratio. Material absorption is distinct
from intrinsic nonradiative loss. Check convergence before using results. Numerical
budget/precision failures raise exceptions; the HTTP adapter masks those samples.

Lossy cylinder spectra guard narrow intervals next to lossless-medium light lines,
where the vector basis loses floating-point precision. The endpoint continuation
uncertainty is reported as `diagnostics.grazing_error` and included in the integration
error; it must meet the requested tolerance alongside independent channel balance
and angular-tail checks. Lossless guided-mode integration is unchanged. Scaled
Hankel functions use direct evaluation with logarithmic high-order overflow recovery,
and pattern cuts at the same polar angle share their mode coefficients.

Patterns use `outputs=("rates", "pattern"), theta=..., phi=...` with broadcast angle
arrays in radians, and return normalized power per solid angle. Film theta zero
points outward into the incident medium; spheres/cylinders use the +z polar axis.
Cylinder waves interpret their `theta` pattern samples as azimuth. The sphere API
below is part of the stable 1.x surface, except `shell_average`, `spectral_density`,
`casimir_polder` and `green_dyadic`, which are provisional, and the normalized-route
functions, which are legacy ([docs/API.md](https://github.com/multilayerapp/PyStratify/blob/main/docs/API.md)).

`response.py` owns scalar, logarithmic and coupled block composition. Geometry bases,
source spectra and reconstruction are separate modules. The migrated planar routines
retain the extended API and upstream MIT notice in `src/pystratify/PLANAR-LICENSE.txt`.
Coherent propagation has no artificial attenuation clipping. Texture emission,
line sources, finite cylinders and absorbing-source regularization are deferred.
Run `pytest tests/test_unified.py` for analytic Fresnel/Bessel limits, independent
flux/Green closure, guided channels, patterns, ED/MD, metal-cylinder endpoint error
budgets, mode reuse, high-order special functions and treams cylinder benchmarks.

| quantity | function |
|---|---|
| solution of the sphere, batched over wavelengths | `solve` → `Solution` |
| scattering / absorption / extinction, per multipole | `cross_sections` |
| angular scattering amplitudes, whole or of chosen partial waves | `scattering_amplitudes`, `scattering_amplitudes(..., orders=, polarisations=)` |
| amplitude and Mueller matrices (S1..S4, 4 × 4) | `amplitude_matrix`, `mueller_matrix` |
| scattering pattern for any polarisation: dσ/dΩ, directivity, Stokes, helicity | `scattering_pattern` → `ScatteringPattern` |
| cross sections and circular dichroism for both helicities | `helicity_cross_sections` |
| far field, directivity and radiated power of a dipole emitter anywhere (core, any shell incl. chiral, host) | `dipole_far_field` → `EmissionPattern` |
| chiral emitters (coherent p + m), random or in-plane orientation averages, helicity-resolved power and g_lum | `dipole_far_field(..., magnetic_moment=, orientation=)`, `source_covariance` |
| chiral (Pasteur) shells: T-matrix with TM–TE coupling | `solve_chiral` → `ChiralSolution` |
| E and H near fields, whole or of chosen partial waves (orders; electric, magnetic) | `near_field`, `near_field(..., orders=, polarisations=)` |
| orientation-averaged intensities and energy density, whole or of chosen partial waves | `energy_density`, `energy_density(..., orders=, polarisations=)` |
| energy stored in each shell, whole or of chosen partial waves | `shell_energy`, `shell_energy(..., orders=, polarisations=)` |
| energy prefactors (Loudon, for Drude metals) | `electric_prefactor`, `energy_prefactors` |
| radiative / nonradiative / total decay rates and frequency shift, radial and tangential dipoles (normalized formulation by default, `route="log"` for the logarithmic one) | `decay_rates` → `DecayRates` |
| decay rates and frequency shift of any source (p, m, electric and magnetic quadrupoles; fixed or orientation-averaged) in any lossless layer, chiral included: total (Purcell), radiative per helicity, absorbed per layer and per sheet; normalized formulation whenever every source is in an achiral layer (sheets, d-parameters and chiral layers elsewhere included) | `emission_rates` → `EmissionRates` |
| electric-quadrupole emitters: far field and rates | `dipole_far_field(..., quadrupole=)`, `emission_rates(..., quadrupole=)` |
| 2D materials on interfaces: in-plane conductivity and out-of-plane response; thin films; graphene | `solve(..., sheets=)`, `solve_chiral(..., sheets=)`, `Sheet`, `Sheet.from_film`, `graphene_conductivity` |
| Feibelman d-parameters of metal surfaces (mesoscopic boundary conditions; the Mie theory of Gonçalves et al. 2020 for a sphere) | `solve(..., sheets={j: Feibelman(d_perp, d_par, metal=)})`, `emission_rates`, `decay_rates` |
| thin-shell electron free-path correction | `free_path_correction`, `DRUDE` |
| multipole truncation | `truncation_order` |
| decay rates and frequency (Lamb) shift of ED and MD emitters from normalized quantities, at any distance from an interface, with automatic truncation | `normalized_decay_rates` → `NormalizedRates` (`.shift`) |
| per-order Green's forms at the source (value, mixed and derivative forms; radiated amplitudes), for many emitter radii at once | `normalized_terms` → `NormalizedTerms` |
| emitters spread through a shell: volume- and orientation-averaged rates and quantum yield, with a cutoff next to absorbing layers | `shell_average` → `ShellAverage` |
| spectral density J(ω) and frequency shift over wavelength, for a dipole of fixed moment (Wigner–Weisskopf kernel) | `spectral_density` → `SpectralDensity` |
| Casimir–Polder potential of an atom at any distance (normalized series at imaginary frequency) | `casimir_polder` → `CasimirPolder` |
| two-point Green's dyadic between any two points in lossless shells (energy transfer, dipole–dipole coupling, collective decay), normalized | `green_dyadic` → `GreenDyadic` |
| extended-precision and classical references | `pystratify.references` |

No optical constants are shipped: pass n + ik from a database such as
[refractiveindex.info](https://refractiveindex.info). The Drude fits in `DRUDE` exist only for
what tables cannot supply, the free-electron damping.

## Thin-film ellipsometry

Two modules take a spectroscopic ellipsometry measurement from file to fitted thicknesses and optical
constants. Wavelengths and thicknesses are in nm, angles of incidence in degrees, oscillator
parameters in eV.

**Dispersion models** (`pystratify.dispersion`). Each model returns `epsilon(wavelength)` and
`index(wavelength)` (n + ik, k ≥ 0); models add, so a material is a sum of terms with its ε∞ written
out. Every oscillator's ε tends to 0 at high energy, so no ε∞ is hidden inside one.

| model | form | parameters |
|---|---|---|
| `EpsilonInf(value)`, `Constant(n, k)` | constant ε, or ε = (n + ik)² | — |
| `Cauchy(A, B, C, k_amplitude, k_exponent, band_edge_nm)` | n = A + B/λ² + C/λ⁴ (λ in µm), k = α e^{β(E − E_band)} (Woollam's form) | B in µm², C in µm⁴ |
| `Sellmeier(B, C, A=1)` | ε = A + Σ Bᵢ λ²/(λ² − Cᵢ) (λ in µm): glass-catalogue coefficients unchanged | Cᵢ in µm² |
| `Pole(A, E0=0)` | A/(E0² − E²); E0 = 0 is the IR pole −A/E² | A in eV² |
| `Lorentz(A, E0, B)` | A B E0/(E0² − E² − iEB) | |
| `Gaussian(A, E0, B)` | KK-consistent Gaussian ε₂ of full width at half maximum B | |
| `TaucLorentz(A, E0, C, Eg)` | Jellison & Modine (1996), closed-form ε₁ | needs C < √2 E0 |
| `CodyLorentz(A, E0, Gamma, Eg, Ep, Et, Eu)` | Ferlauto et al. (2002) with its Urbach tail; ε₁ by numerical Kramers–Kronig (`kramers_kronig_real`) | Et = 0: no tail |
| `DrudeEnergy(Ep, Gamma)`, `DrudeResistivity(rho, tau)` | −Ep²/(E² + iΓE); the same in ρ (Ω m) and τ (s) | |
| `Bruggeman(a, b, fraction_a)`, `MaxwellGarnett(host, inclusion, fraction)` | effective media of spherical inclusions; Bruggeman keeps the passive root | models or scalar indices |
| `Table(wavelength_nm, n, k)`, `Table.from_refractiveindex(path)` | linear interpolation, never extrapolated | |

The oscillator forms (pole, Lorentz, Gaussian, Tauc-Lorentz, resistivity Drude, Bruggeman) are the ones
multilayer.app's oscillator dialog offers, parameter for parameter.

**Forward model** (`pystratify.ellipsometry`). A `Sample` lists `Layer(material, thickness)` from the
ambient side down, then the substrate; `simulate(sample, wavelengths, angles)` returns Ψ, Δ, N, C, S
and the depolarisation on the angle × wavelength grid. Ψ and Δ are in the instruments' (Nebraska)
convention, tan Ψ e^{iΔ} = conj(r_p/r_s) in this package's n + ik fields, so a bare dielectric below its
Brewster angle has Δ = 180° (Woollam, SENTECH and pyElli agree; `planar.ellips` and multilayer.app's
Ellipsometry card keep the transfer-matrix phase, 180° − Δ). Δ is in [0°, 360°). Roughness is a
`Bruggeman(film, 1.0, 0.5)` layer. With `substrate_thickness` set, light returned from the back of a
finite substrate is added incoherently, which depolarises the reflection (a glass slide); N, C, S then
have length 1 − depolarisation.

**Data.** `read_woollam` reads CompleteEASE and WVASE exports (Ψ/Δ, their standard deviations, and the
`dPolE` depolarisation; nm, Å, µm or eV), `read_spectraray` SENTECH SpectraRay exports, and `Data(...)`
takes arrays. `data.select(wavelength=(min, max), angles=[...])` cuts a range.

**Fits.** The sample is a function of a dict of parameter values; `Parameters` gives each a start,
bounds and `vary`:

```python
import numpy as np
from pystratify import dispersion as dm, ellipsometry as el

def sample(p):
    film = dm.EpsilonInf(p["einf"]) + dm.TaucLorentz(A=p["A"], E0=p["E0"], C=p["C"], Eg=p["Eg"])
    return el.Sample([el.Layer(dm.Bruggeman(film, 1.0, 0.5), p["rough"]), el.Layer(film, p["d"])],
                     substrate=silicon)

params = (el.Parameters().add("d", 80, min=0, max=500).add("rough", 2, min=0, max=10)
          .add("einf", 1.0, min=0).add("A", 120, min=0).add("E0", 3.6, min=1).add("C", 2.0, min=0.1)
          .add("Eg", 1.6, min=0, max=3))
result = el.fit(sample, params, data, grid={"d": np.arange(10, 500, 10.0)})
print(result.report())          # values ± 90 % limits, correlations above 0.9, MSE
result.values, result.stderr, result.correlation, result.model_spectra()
```

`fit` minimises the instruments' MSE with SciPy's bounded trust-region least squares: on Ψ/Δ,
√(Σ[(ΔΨ/σ_Ψ)² + (ΔΔ/σ_Δ)²]/(2P − M)) with the file's standard deviations (1° without them, Δ residuals
wrapped to ±180°); with `quantity="ncs"`, 1000 √(Σ[ΔN² + ΔC² + ΔS²]/(3P − M)), the choice for depolarising
samples and for Δ near 0° or 180°. Standard errors come from the Jacobian at the minimum scaled by the
MSE, the 90 % limits are 1.645 of them, as ellipsometry software prints. Thickness fits have a minimum
per fringe order, so pass `grid`: every combination of the listed values is scored and the best starts
the local fit. `fit_nk(model, data, start)` then inverts one layer's n and k wavelength by wavelength
(all angles at each), each wavelength starting from the last, and `.table()` turns the answer into a
`Table`. `examples/ellipsometry_fit.py` fits two real measurements, a Woollam RC2 wafer (native oxide
2.25 nm) and a SENTECH TiO₂ film; [benchmarks/ELLIPSOMETRY_CODES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/ELLIPSOMETRY_CODES.md)
fits the latter with pyElli + lmfit as well (parameters agree to 4e-7).

Not modelled: anisotropic films, graded layers, and depolarisation from the instrument (bandwidth,
angular spread) or from thickness non-uniformity.

## Focused illumination

A fourth source for all three geometries: a beam focused by an aplanatic objective whose
pupil is filled uniformly, through an annulus (a Cassegrain, `obscuration` > 0) or a disc,
as in FT-IR microspectroscopy (Davis, Carney & Bhargava, *Anal. Chem.* 82, 3474 (2010) and
83, 525 (2011); van Dijk et al., *Appl. Spectrosc.* 67, 546 (2013)). Powers are fractions
of the beam's power; the detected one passes a condenser's annulus (transmission) or the
objective's (film transflection), and the apparent absorbance is −log₁₀ of its ratio to a
reference setup's.

```python
import numpy as np
import pystratify as ps

beam = ps.FocusedBeam(0.5, 0.23)                       # NA, obscuration NA (n sin θ in the host)
wavenumber = np.linspace(1000, 4000, 301)              # cm⁻¹
wavelength = 1e4 / wavenumber                          # µm
fibre = ps.focused_cylinders([5.0], [1.49 + 0.01j, 1.0], wavelength, beam)
fibre.apparent_absorbance, fibre["extinction"], fibre.by_polarization["y"]["detected"]
film = ps.focused_films([1, n_polymer, 1.43, 1], [np.inf, 2.0, 400.0, np.inf], wavelength, beam,
                        coherence=["i", "c", "i", "i"],
                        reference=([1, 1.43, 1], [np.inf, 400.0, np.inf], ["i", "i", "i"]))
sphere = ps.focused_spheres([5.0], [n_pmma, 1.0], wavelength,
                            [ps.FocusedBeam(0.5, 0.23, offset=(x, 0, 0)) for x in (0, 2, 4)])  # a line scan
field = ps.focused_field_spheres([5.0], [n_pmma[0], 1.0], wavelength[0], beam, points)    # |E|² = 1 at the focus
```

`FocusedBeam(na, obscuration, collection_na, collection_obscuration, mode, illumination,
field_stop, model, polarization, offset)`: the condenser defaults to the objective;
`illumination="kohler"` makes every pupil direction an independent plane wave filling a
field stop of diameter `field_stop` (a cylinder through it is lit over that chord);
`model="scalar"` is the corrected scalar theory of the coherent sphere and cylinder spot;
`polarization` is the pupil's (`"x"`, `"y"` or `"unpolarized"`; a cylinder's axis is y);
`offset` moves the focus (coherent only). `solve_problem(Problem(..., source=beam))` takes
one wavelength, with `reference=Problem(...)` for films and `points=` for `"field"`; the
`focused_*` functions batch wavelengths and take a list of beams (an NA sweep or a focus
scan), sharing every Mie, oblique-cylinder and planar solve between them.

* **Films**: the cone average, uniform in sin²θ, s and p equally, of plane-wave R and T and
  each layer's absorptance, by adaptive quadrature split at the condenser's edges and every
  lossless layer's critical angle, with the plane-wave solver the coherence column calls for
  (`planar.layered_response`). A 400-µm coherent window is resolved fringe by fringe.
  Coherent and Köhler illumination coincide.
* **Spheres**: the beam's vector multipole coefficients (m = ±1, a 1-D pupil integral, when the
  focus lies on the axis through the centre; every m, by FFT in the azimuth, otherwise), the
  T-matrix, P_ext = −4 Re Σ T|c|², P_sca = 4 Σ |Tc|² and the collected power over the condenser
  in closed form per order. Köhler uses the cross sections over the cone and the scattering
  landing in the condenser (Legendre addition theorem).
* **Cylinders** (axis y, beam z): each pupil row s_y is a 2-D problem; one oblique solve per row
  serves every transverse angle, the rows add in power, and the s_y integral is adaptive with an
  error estimate, split where rows cross an annulus edge. The scattered light is collected over
  the condenser at its true angle, asin(s_x/√(1−s_y²)) — never masked to the illumination.
* **Scalar model**: U and ∂U/∂r continuous; the sphere's order-l coefficient is the TE one plus a
  monopole, the cylinder's the axial-electric one at normal incidence with indices
  √(n² − (n_host s_y)²); the pupil weighting is the vector model's, so vector − scalar isolates
  polarization.
* **Near fields** (coherent): spheres from the beam's multipoles and the shells' radial
  functions, cylinders as the coherent sum over rows, films as every pupil direction's coh_tmm
  field summed over the azimuth in closed form (Bessel J₀, J₁, J₂); the beam alone in the host
  is the Richards–Wolf integral.

The ambient (films) or host must be lossless and NA < n there. `pytest tests/test_focused.py`
checks every semi-analytic form against a plane-wave superposition on a dense pupil grid through
`amplitude_matrix`, the oblique `solve_cylinder`, `coh_tmm`/`position_resolved`, `near_field`
and `CylinderSolution.field`; energy balance (lossless: absorption 0, per row for cylinders);
NA → 0 against the plane-wave cross section and cross width; an index-matched sphere and an
empty stack invisible; the aplanatic film path factor ⟨1/cos θ⟩ = 1.0357 (n = 1.5, NA 0.23–0.5);
Köhler against cone averages of plane-wave results; the scalar coefficients against closed
forms; and the film near field's lateral integral against the cone-averaged plane-wave
intensity (Parseval).

## Spheres in depth

Which names are the stable 1.x surface and which are provisional, legacy or internal:
[docs/API.md](https://github.com/multilayerapp/PyStratify/blob/main/docs/API.md) (ruled 2026-10-10; binding from 1.0).

Lengths are unit-agnostic (give radii and wavelengths in the same unit); the Drude helpers and
`graphene_conductivity` take nanometres. Shells are indexed from 0 (the core) to N (the host).

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

# a chiral molecule (m parallel to p, a quarter period out of phase) inside the chiral shell,
# randomly oriented; and a valley exciton of a 2D semiconductor, a circular in-plane dipole
molecule = ps.dipole_far_field([50.0, 55.0, 60.0], [1.45, n_gold[200], 1.5, 1.33], wavelength[200],
                               [0, 0, 57.5], [0, 0, 1], theta, phi, kappa=kappa,
                               magnetic_moment=[0, 0, 0.01j], orientation="isotropic")
molecule.dissymmetry, molecule.helicity_power, molecule.coherency   # g_lum, (P_+, P_-), <F F^H>
exciton = ps.dipole_far_field(radii, n[200], wavelength[200], [0, 0, 57.0], [1, 1j, 0], theta, phi)

# decay of a randomly oriented chiral molecule at two distances from a chiral sphere; the enantiomer
# has m -> -m.  (A lossy kappa needs magnetic loss too: passivity is eps'' mu'' >= kappa''^2.)
rates = ps.emission_rates([50.0], [1.6 + 0.02j, 1.0], 600.0, [[0, 0, 55.0], [0, 0, 70.0]], [0, 0, 1],
                          magnetic_moment=[0, 0, -0.1j], kappa=[0.03 + 0.003j, 0], mu=[1 + 1e-3j, 1],
                          orientation="isotropic")
rates.total, rates.radiative_helicity, rates.absorption, rates.balance_error, rates.dissymmetry

# an electric-quadrupole emitter 5 nm from gold: rates and far field
quad = ps.emission_rates([40.0], [n_gold[200], 1.33], wavelength[200], [0, 0, 45.0], [0, 0, 0],
                         quadrupole=np.diag([-1.0, -1.0, 2.0]))

# 2D materials on interface 0: a TMD monolayer as a sheet, graphene from its conductivity
tmd = ps.Sheet.from_film(eps_ws2, 0.618, wavelength, eps_background=1.0, eps_normal=6.5)
sol = ps.solve([78.0], [3.9, 1.0], wavelength, sheets={0: tmd})
sigma = ps.graphene_conductivity(wavelength_ir, fermi_energy=0.5, damping=0.005)  # sigma Z0; nm, eV
sol = ps.solve([25.0], [1.45, 1.0], wavelength_ir, sheets={0: sigma})
```

Far-field conventions are Bohren & Huffman's: E_sca = e^{ikr}/(−ikr) X with
(X_∥, X_⊥) = [[S2, S3], [S4, S1]] (E_∥, E_⊥) and dσ/dΩ = |X|²/k². Helicity +1 is
(x̂ + iŷ)/√2 for a wave along +z (left-circular in the optics convention). Chiral media follow
D = εE + iκH, B = μH − iκE (Gaussian units, e^{−iωt}; the convention of
[treams](https://github.com/tfp-photonics/treams)), so helicity ±1 has index n ± κ.

Emitters follow Klimov, Guzatov & Ducloy ([EPL 97, 47004 (2012)](https://arxiv.org/abs/1108.0497))
and Guzatov & Klimov ([New J. Phys. 14, 123009 (2012)](https://arxiv.org/abs/1203.5393)):
a chiral emitter is an electric dipole p and a magnetic dipole m radiating coherently, Gaussian
units (in SI pass m/c), enantiomers differing in the sign of Im(p·m*); their molecule
⟨e|d|g⟩ = d₀, ⟨e|m|g⟩ = −i m₀ is (p, m) = (d₀, −i m₀). With both moments the
far field is normalised to the electric dipole's, so m enters as (n_h/μ_h) m, and in a host of
index n a free emitter has g_lum = 2(P₊ − P₋)/(P₊ + P₋) = 4n Im(p·m*)/(|p|² + n²|m|²). m is the
dual of p (the magnetic current −iωm of the Maxwell equations), as in `decay_rates`;
`magnetic_convention="current"` takes a current-loop moment m_A instead, which in a layer with
μ_d, κ_d acts as the dual moment μ_d m_A together with the electric dipole iκ_d m_A. An electric
quadrupole is Jackson's Q_ij = ∫(3x_ix_j − r²δ_ij)ρ dV: it couples through (1/6)Q:∇E and alone
in the host radiates (k²/120)Σ|Q_ij|² in units of the power of |p| = 1. Rates are normalised to
the same source in the unbounded host (`normalization="layer"`: in its own layer's medium). A
lossy chirality (circular dichroism) is passive only with magnetic loss, ε″μ″ ≥ κ″²;
`emission_rates` warns otherwise, since one handedness then sees gain. A valley exciton of a monolayer semiconductor is a circular in-plane dipole (x̂ ± iŷ)/√2
about the layer normal (Gong et al., [Science 359, 443 (2018)](https://arxiv.org/abs/1709.00762));
dark excitons are out-of-plane. For an ensemble of incoherent emitters uniformly covering a
sphere of radius r₀ (a conformal monolayer, a molecular shell) with orientation fixed relative
to the local normal, the emission is isotropic and its dissymmetry is that of one emitter at
r₀ẑ, so `power` and `helicity_power` of a single call give the ensemble.

A 2D material on interface j enters as generalised sheet transition conditions with the fields
averaged over its two sides (Kuester et al., IEEE TAP 51, 2641 (2003)):
r̂ × (H⁺ − H⁻) = σ⟨E_t⟩ and E_t⁺ − E_t⁻ = −ζ∇_t⟨D_n⟩, with σ = σ_s Z₀ (SI) = 4πσ_s/c the
dimensionless sheet conductivity (graphene's universal value is πα) and ζ a length for the
out-of-plane response. `Sheet.from_film` gives the first-order limit of a film of thickness d,
σ = −ik₀d(ε_∥ − ε_b), ζ = d(1/ε_b − 1/ε_⊥), which absorbs (c/8π)[Re σ|⟨E_t⟩|² + k₀ Im ζ|⟨D_n⟩|²]
per area. `graphene_conductivity` is the local RPA conductivity (Falkovsky; Hanson), intraband
at any temperature, interband for k_BT ≪ E_F.

`examples/` has runnable scripts (multipole spectra, free-path correction, scattering pattern,
scattering and emission directivity, near-field maps, energy density, stored energy, electric and
magnetic dipole decay, circular dichroism of a chiral shell, circularly polarised emission of chiral
molecules and valley excitons beside spheres, enantioselective decay beside chiral particles,
quadrupole versus dipole emitters near gold, a TMD monolayer on a silicon sphere and
graphene-coated spheres); with matplotlib installed each saves a PNG beside itself.
`crosscheck/` recomputes published figures from their stated parameters for comparison with the
papers (Guzatov & Klimov 2012, Figs. 3–6).
