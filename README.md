# PyStratify

Light scattering by **multilayered (stratified) spheres** in Python: any number of concentric
shells, absorbing, magnetic, gain or chiral media, 2D materials (graphene, TMD monolayers) on the
interfaces, and emitters inside or outside the particle - electric, magnetic, chiral (p + m) and
electric-quadrupole - with angle-resolved far fields and decay rates. The physics is the
recursive transfer-matrix method of Moroz (2005) as used
by [STRATIFY](https://gitlab.com/iliarasskazov/stratify) (Rasskazov, Carney & Moroz,
[OSA Continuum 3, 2290 (2020)](https://doi.org/10.1364/OSAC.399979)). The numerics are rebuilt
so that every quantity stays finite and accurate to the orders it needs, and the defects found in
the MATLAB code are fixed ([AUDIT.md](AUDIT.md)).

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
| decay rates and frequency shift of any source (p, m, electric and magnetic quadrupoles; fixed or orientation-averaged) in any lossless layer, chiral included: total (Purcell), radiative per helicity, absorbed per layer and per sheet; normalized formulation for achiral layers without sheets | `emission_rates` → `EmissionRates` |
| electric-quadrupole emitters: far field and rates | `dipole_far_field(..., quadrupole=)`, `emission_rates(..., quadrupole=)` |
| 2D materials on interfaces: in-plane conductivity and out-of-plane response; thin films; graphene | `solve(..., sheets=)`, `solve_chiral(..., sheets=)`, `Sheet`, `Sheet.from_film`, `graphene_conductivity` |
| thin-shell electron free-path correction | `free_path_correction`, `DRUDE` |
| multipole truncation | `truncation_order` |
| decay rates and frequency (Lamb) shift of ED and MD emitters from normalized quantities, at any distance from an interface, with automatic truncation | `normalized_decay_rates` → `NormalizedRates` (`.shift`) |
| per-order Green's forms at the source (value, mixed and derivative forms; radiated amplitudes), for many emitter radii at once | `normalized_terms` → `NormalizedTerms` |
| emitters spread through a shell: volume- and orientation-averaged rates and quantum yield, with a cutoff next to absorbing layers | `shell_average` → `ShellAverage` |
| spectral density J(ω) and frequency shift over wavelength, for a dipole of fixed moment (Wigner–Weisskopf kernel) | `spectral_density` → `SpectralDensity` |
| Casimir–Polder potential of an atom at any distance (normalized series at imaginary frequency) | `casimir_polder` → `CasimirPolder` |
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
* **Dipole far fields from reciprocity.** The far-field amplitude of a source (p, m) at r₀ in
  direction n̂ with polarisation ê is p·E(r₀) − m·H(r₀) for the plane wave ê incident from n̂,
  which `solve` and `solve_chiral` provide in every layer (Pasteur media are reciprocal, so this
  holds inside chiral shells). Rotating the emitter onto the z axis leaves only m = 0, ±1
  multipoles, so any position and complex orientation costs one set of π_l, τ_l; total and
  helicity-resolved powers are summed from orthogonal multipole amplitudes. Every emitter is
  six basis sources (unit p and m along x, y, z), so fixed sources and orientation averages are
  one bilinear form in the source covariance ⟨ss†⟩, exact rather than sampled. For an emitter
  in the host the direct dipole field is added in closed form, so the truncation depends on the
  particle and not on the emitter's distance.
* **Chiral shells** (`chiral.py`). In a Pasteur medium the field splits into Beltrami waves of
  helicity ±1 with wavenumbers k₀(n ± κ), and an interface mixes them only through its impedance
  contrast. The regular solution is swept outwards as a 2 × 2 matrix ρ with the same scaling as
  the achiral solver: the matrix Möbius step is assembled from the index, chirality and impedance
  contrasts (no cancellation of the large l/x parts), and ρ is carried as element-wise
  logarithms so it survives falling below the double range across thick absorbing shells.
  The layer amplitudes follow inwards from the host through the inverse of the same step, in
  element-wise logarithms as well.
* **Decay rates of any source** (`rates.py`). In the emitter's layer a source (p, m, Q) drives
  each helicity channel s through q_s = p + i s m/Z plus (1/6)Q:∇; the direct field is the dyadic
  expansion of Tai, the reflected one follows from the 2 × 2 R, S of the layer evaluated scaled
  at the emitter, B = (1 − RS)⁻¹R(D_ψ + S D_ξ), A = S(B + D_ξ). The total rate is the free power
  (closed form) plus Re q*·(Aψ + Bξ); the radiated power, per helicity, comes from the outgoing
  amplitudes carried to the host by per-interface maps (`ChiralSolution.log_in`, `log_out`), so
  no ill-conditioned amplitude matrix is ever inverted; the absorption of each layer is the loss
  density ε″|E|² + μ″|H|² − 2κ″ Im(E*·H) integrated by Gauss–Legendre quadrature in the basis
  (ψ₊, ψ₋, ξ₊, ξ₋) with per-function logarithmic scales, and that of each sheet comes from the
  fields on its two sides. The three are independent, and `balance_error` checks energy
  conservation. Orientation averages are exact: the analytic ⟨ss†⟩ for dipoles, the 60
  rotations of the icosahedral group (exact to rank 4) once a quadrupole is present.
* **2D sheets** (`sheets.py`). The averaged sheet conditions are a unimodular 2 × 2 transfer on
  each pair of tangential components (E_M, H_N) and (E_N, H_M); they enter the Möbius steps of
  both solvers as additions to the interface matrices (the large l/x parts still cancel
  analytically in the bulk contrast), and the amplitude equations acquire the matching terms.
* **Quadrupoles.** With the emitter on the local z axis a quadrupole couples to the axial
  families m = 0, ±1, ±2; the gradients of M and N there are closed forms in ψ'/ψ, x and l.
  Their far field is synthesised from the outgoing host amplitudes (for dipoles this route
  agrees with the reciprocity route to 1e-14 in complex amplitude).
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
  from a gold sphere was off by a factor of two). `near_field(..., incident=False)` returns the
  scattered field alone. A partial-wave field, `near_field(..., orders=, polarisations=)`, is the
  exception: it sums the incident wave's own terms of those orders, which is what such a field is,
  so that disjoint selections add up to their union and all orders together to the field itself.
  `energy_density` and `shell_energy` take the same selection; averaged over a sphere the partial
  waves do not interfere, so there the selections add up in energy, not only in field.
  `scattering_amplitudes(..., orders=, polarisations=)` keeps chosen partial waves of the far
  field: their amplitudes add up, their |S|² does not.
* **Vectorised.** `solve` takes a whole spectrum at once (array operations over wavelength ×
  order; Python loops only over interfaces). Near fields and energy densities evaluate the
  special functions once per distinct kr for both polarisations.

**Accuracy you can expect.** Coefficients agree with 60-digit transfer matrices to ~1e-12, and
the interface conditions hold to 1e-9 at every order. Where a result is less accurate than that,
the problem is ill-conditioned rather than the algorithm: large lossless particles with sharp
internal resonances can amplify input rounding by 10⁶ (perturbing the inputs by 1e-15 moves the
exact answer by as much as the solver's error), and the absorption of a nearly lossless
Rayleigh particle is known only to ~eps·|a_l|. Layers index-matched to within δ determine their
coefficients to ~l·eps/δ. Chiral T-matrices agree with 80-digit 4 × 4 transfer matrices to
1e-12 up to l = 160 (the amplitudes in every layer to 6e-13), with the independent code treams
to 1e-13, and a small chiral sphere reproduces the published quasi-static polarisabilities
α_EE, α_HH, α_EH of Guzatov & Klimov (2012).

`emission_rates` equals `decay_rates` to ~1e-13 (total, radiative, nonradiative) for achiral
spheres; for lossless chiral multilayers its total rate equals the independent reciprocity-based
radiated power to 1e-11, per helicity to 1e-12; with absorbing, magnetic and chiral layers, gold
and sheets, total = radiative + absorbed holds to ~1e-12 even 0.1 nm from an interface; the
orientation-averaged rate of a chiral molecule beside a small chiral sphere converges to
Guzatov & Klimov's quasi-static Eq. 46 as (k₀a)² for both enantiomers. One limit: within ~1 nm of
a *lossless* interface the reflected part of the total rate is the small real part of large
evanescent terms, so rounding limits `total` to ~1e-9 (1 nm) … 1e-5 (0.1 nm) relative, while
`radiative` and `nonradiative` stay accurate (`balance_error` shows it). Quadrupole couplings
agree with 40-digit finite differences of the vector wave functions to 1e-14, and a pair of
opposite dipoles ±p at r₀ ± δ/2, computed by the reciprocity route, converges to the quadrupole
3(pδ + δp) − 2(p·δ)I plus the current loop (ik₀/2)p × δ to 1e-12 in amplitude, near and inside
chiral particles. A sheet is the O(d²)-accurate limit of the explicit film it replaces, and
graphene-coated-sphere resonances sit on the quasi-static condition
ε₁l + ε₂(l+1) + iσl(l+1)/(k₀R) = 0 within the (k₀R)² retardation shift.

Typical timings on one core:

| task | time |
|---|---|
| 601-wavelength spectrum, Au nanoshell | 25 ms |
| 601-wavelength spectrum, 6-interface matryoshka | 70 ms |
| x = 500 sphere, all orders | 8 ms |
| 300 × 300 near-field map | 1.4 s |
| decay rates at 100 emitter positions 0.5–25 nm from an Au nanoshell | 0.4 s |
| the same from 0.1 nm (6700 orders) | 2.4 s |
| 601-wavelength spectrum, Au nanoshell with a chiral shell, both helicities | 46 ms |
| scattering pattern on a 181 × 361 (θ, φ) grid | 10 ms |
| dipole emission pattern on a 181 × 361 grid | 90 ms |
| orientation-averaged chiral emitter inside a chiral shell, 181 × 361 grid | 0.12 s |
| `emission_rates`, 100 positions, chiral source, orientation average, chiral + Au multilayer | 0.4 s |
| `emission_rates`, one position 10 nm / 1 nm from gold (p + m, quadrupole) | 40 ms / 1.3 s |
| quadrupole emission pattern, 181 × 361 grid, emitter on / off the grid's axis | 0.15 s / 1.1 s |
| 601-wavelength spectrum, graphene-coated sphere | 10 ms |

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

`pystratify.references` holds the independent references (mpmath Mie decay rates and extinction,
layered-sphere decay rates from transfer matrices, BHMIE); nothing in it shares code with the package.

## Citing

PyStratify implements the method of STRATIFY; please cite the paper,

> I. L. Rasskazov, P. S. Carney and A. Moroz, "STRATIFY: a comprehensive and versatile MATLAB code
> for a multilayered sphere," *OSA Continuum* **3**, 2290 (2020),
> [doi:10.1364/OSAC.399979](https://doi.org/10.1364/OSAC.399979),

and this repository at the version you used ([CITATION.cff](CITATION.cff)). Defects of the
MATLAB code that PyStratify does not share are listed in [AUDIT.md](AUDIT.md).

## Licence

GPL-3.0-or-later, as STRATIFY (this is a derivative work).
