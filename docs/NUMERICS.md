# PyStratify numerics

How PyStratify keeps every quantity finite and accurate, what accuracy to expect, its known limitations and typical
timings. The comparisons with other codes are in [benchmarks/README.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/README.md).

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
* **Exact field at the centre.** At r = 0 only the l = 1 internal waves survive, so `near_field`
  evaluates the origin in closed form rather than at a small offset: E(0) = d₁ x̂ and H(0) = m c₁ ŷ
  for a homogeneous sphere (Bohren & Huffman), to ~1e-15
  ([benchmarks/MIE_CODES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/MIE_CODES.md)).
  scattnlay, which evaluates the series there, is off by up to 5e-3.
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
Guzatov & Klimov's quasi-static Eq. 46 as (k₀a)² for both enantiomers. On the logarithmic route
(`route="log"`), within ~1 nm of a *lossless* interface the reflected part of the total rate is the
small real part of large evanescent terms, so rounding limits `total` to ~1e-9 (1 nm) … 1e-5
(0.1 nm) relative (the route flags it through the energy balance); the normalized route, the
default, keeps full precision there (see *Normalized formulation* below). Quadrupole couplings
agree with 40-digit finite differences of the vector wave functions to 1e-14, and a pair of
opposite dipoles ±p at r₀ ± δ/2, computed by the reciprocity route, converges to the quadrupole
3(pδ + δp) − 2(p·δ)I plus the current loop (ik₀/2)p × δ to 1e-12 in amplitude, near and inside
chiral particles. A sheet is the O(d²)-accurate limit of the explicit film it replaces, and
graphene-coated-sphere resonances sit on the quasi-static condition
ε₁l + ε₂(l+1) + iσl(l+1)/(k₀R) = 0 within the (k₀R)² retardation shift.

**Normalized formulation (the default route).** Every per-order term at a source is an explicit
geometric factor times quantities of order unity built from ψ'/ψ, ψξ and the normalized Bessel
function j̄ (`normalized.py`): the reflection ratios ρ, σ are swept through Möbius maps whose
mismatches come from the material contrasts, and near a lossless interface the small real parts of
the nearly reactive high-order terms are carried multiplicatively, so total rate, radiative rate and
frequency shift keep full relative precision at any distance (energy balance ≤ 1e-14 down to
0.05 nm from lossless interfaces). Interface responses - sheets (averaged transition conditions)
and Feibelman d-parameters (linearized from the metal side) - add terms to the same maps through
one matching form, value′ = τ(value + t_v deriv)/c_v, deriv′ = τ(deriv − t_d value)/c_d. Chiral
layers enter as 2 × 2 maps in the (TM, TE) basis, with the leading mismatches written through a
closed-form contrast matrix (its TE element is exactly zero between achiral nonmagnetic media,
where the helicity basis would lose ε(l/x)² of the TE terms). Absorption is computed independently
- Lommel boundary terms for layers, the jump of the radial flux for interface responses - so the
energy balance remains a test.

**Known limitations.**

* *Absorption next to metal.* The absorbed power (per layer and per interface response, i.e.
  `EmissionRates.absorption`, `.sheet_absorption` and `DecayRates.nonradiative`) is computed from
  the amplitudes of the logarithmic solver, whose logarithms grow like l ln(l/x). Within a few
  tenths of a nanometre of an absorbing layer, where 10³–10⁴ orders are summed, this limits the
  absorption to ~1e-13 relative at 1 nm, ~1e-12 at 0.1 nm and ~1e-11 at 0.05 nm (visible as the
  energy balance total − radiative − absorbed). Total rate, radiative rate and frequency shift come
  from the normalized formulation and are not affected.
* *Quadrupoles deep in a subwavelength lossless core.* There the outgoing solution is nearly a
  standing wave and the radiative part 1 + Re S of the low orders is a small difference: ~1e-14 for
  dipoles and 5–8e-14 for quadrupoles in a core of index contrast 3.5, worse for smaller cores and
  higher multipoles.
* *Chiral layers.* The normalized route needs every source in an achiral layer and no sheets in a
  structure with chiral layers; otherwise `route="auto"` takes the logarithmic route, with its
  limit next to lossless interfaces. With absorbing layers in a chiral structure the absorption is
  still integrated by quadrature, O(L^2): within ~0.5 nm of an absorbing layer (10^3-10^4 orders)
  this becomes slow (minutes to hours per position), and its accuracy is ~1e-12 at 1 nm.
* *d-parameters.* The mesoscopic boundary conditions are first order in d; they are applied
  linearized from the metal side (exact for d∥ = 0, and exactly the sphere Mie coefficients of
  Gonçalves et al., Nat. Commun. 11, 366 (2020)); with both d⊥ and d∥ nonzero, other second-order
  readings differ by d⊥d∥ l(l+1)/R².

Typical timings on one core:

| task | time |
|---|---|
| 601-wavelength spectrum, Au nanoshell | 25 ms |
| 601-wavelength spectrum, 6-interface matryoshka | 70 ms |
| x = 500 sphere, all orders | 8 ms |
| 300 × 300 near-field map | 1.4 s |
| decay rates at 100 emitter positions 0.5–25 nm from an Au nanoshell | 0.15 s |
| the same from 0.1 nm (6700 orders) | 0.85 s |
| 601-wavelength spectrum, Au nanoshell with a chiral shell, both helicities | 46 ms |
| scattering pattern on a 181 × 361 (θ, φ) grid | 10 ms |
| dipole emission pattern on a 181 × 361 grid | 90 ms |
| orientation-averaged chiral emitter inside a chiral shell, 181 × 361 grid | 0.12 s |
| `emission_rates`, 100 positions, chiral source, orientation average, chiral + Au multilayer | 0.6 s |
| `emission_rates`, one position 10 nm / 1 nm from gold (p + m, quadrupole) | 4 ms / 25–85 ms |
| quadrupole emission pattern, 181 × 361 grid, emitter on / off the grid's axis | 0.15 s / 1.1 s |
| 601-wavelength spectrum, graphene-coated sphere | 10 ms |
