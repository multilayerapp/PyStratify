---
id: infinite-cylinder
title: "Scattering by an infinite cylinder: two polarizations"
geometry: cylinders
chapter: Scattering
topic: cylinder
sources: [plane_wave]
examples: [glass-cylinder]
classics: [bohren-huffman-1983, ruan-fan-2010]
summary: >-
  The plane-wave solution for an infinitely long circular cylinder: cylindrical waves, the two decoupled polarizations
  at normal incidence and their coefficients, efficiencies per unit length, the thin-wire limit with its ε = −1
  resonance, coated cylinders by the same recursion, oblique incidence, and the single-channel bound that
  superscattering beats.
references:
- cite: Lord Rayleigh, Phil. Mag. 36, 365 (1918)
  doi: 10.1080/14786441108635843
- cite: C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley, 1983), ch. 8
  doi: 10.1002/9783527618156
- cite: M. Kerker, The Scattering of Light and Other Electromagnetic Radiation (Academic Press, 1969), ch. 6
- cite: Z. Ruan and S. Fan, Phys. Rev. Lett. 105, 013901 (2010)
  doi: 10.1103/PhysRevLett.105.013901
- cite: L. Cao et al., Nat. Mater. 8, 643 (2009)
  doi: 10.1038/nmat2477
---

A cylinder is the two-dimensional cousin of a sphere, and it keeps a feature the sphere loses by symmetry: it tells the
two polarizations of the incident light apart. Rayleigh solved the problem for a dielectric cylinder at normal
incidence in 1918; Bohren and Huffman's chapter 8 is the standard modern account and the source of the notation
below. Everything on this page is for an infinitely long circular cylinder of radius $a$ in a host of index $n_h$,
lit by a plane wave of vacuum wavelength $\lambda$. The size parameter and relative index are as for a sphere,

$$
x = \frac{2\pi n_h a}{\lambda} = k a, \qquad m = \frac{n_c}{n_h}.
$$

## Cylindrical waves and the two cases

In cylindrical coordinates $(\rho, \varphi, z)$ the vector Helmholtz equation separates into modes $e^{i m\varphi}\,
e^{i k_z z}$ whose radial dependence obeys Bessel's equation (the [Foundations](learn/common/maxwell-and-boundary-conditions.html)
table). The regular solution is $J_m(q\rho)$ and the outgoing one $H^{(1)}_m(q\rho)$, with $q = \sqrt{k^2 - k_z^2}$ the
radial wavenumber. At **normal incidence**, $k_z = 0$, $q = k$, and the problem splits into two independent cases,
named after the field component along the axis:

- **Case I**: the incident electric field is parallel to the cylinder axis (TM with respect to the axis; the
  calculator's *E along the axis*). The axial component $E_z$ is the scalar field; $\mathbf{H}$ lies in the cross
  section.
- **Case II**: the incident magnetic field is parallel to the axis (TE; *H along the axis*). Now $H_z$ is the scalar,
  and $\mathbf{E}$ lies in the cross section.

For each order $m$ the two boundary conditions, continuity of the tangential $\mathbf{E}$ and $\mathbf{H}$ at
$\rho = a$, fix one scattering coefficient per case, exactly as for the [transfer matrix](learn/common/why-transfer-matrices.html)
of any stratified geometry. Bohren and Huffman's results for a non-magnetic cylinder are

$$
b_m^{\mathrm{I}} = \frac{m\,J_m(mx)\,J_m'(x) - J_m(x)\,J_m'(mx)}{m\,J_m(mx)\,H_m^{(1)\prime}(x) - H_m^{(1)}(x)\,J_m'(mx)},
\qquad
a_m^{\mathrm{II}} = \frac{J_m(mx)\,J_m'(x) - m\,J_m(x)\,J_m'(mx)}{J_m(mx)\,H_m^{(1)\prime}(x) - m\,H_m^{(1)}(x)\,J_m'(mx)},
$$

the relative index $m$ in the two expressions sitting where the sphere's $a_\ell$ and $b_\ell$ have it, swapped
between the cases. The resonances are the near-zeros of the denominators: for a dielectric they are the whispering-gallery
or **leaky modes** of the rod, which make a silicon nanowire of a given diameter absorb one colour preferentially
(Cao et al. 2009); for a metal they are the surface plasmons of the wire.

## Efficiencies per unit length

An infinite cylinder has no cross-sectional area to normalise with, so one works per unit length. The scattering
and extinction *widths* $C_\mathrm{sca}$, $C_\mathrm{ext}$ (a length) divided by the geometric width $2a$ give the
efficiencies

$$
Q_\mathrm{sca}^{\mathrm{I}} = \frac{2}{x}\left[|b_0^{\mathrm{I}}|^2 + 2\sum_{m=1}^{\infty} |b_m^{\mathrm{I}}|^2\right],
\qquad
Q_\mathrm{ext}^{\mathrm{I}} = \frac{2}{x}\,\mathrm{Re}\left[b_0^{\mathrm{I}} + 2\sum_{m=1}^{\infty} b_m^{\mathrm{I}}\right],
$$

and the same with $a_m^{\mathrm{II}}$ for case II; $Q_\mathrm{abs} = Q_\mathrm{ext} - Q_\mathrm{sca}$, and
unpolarized light averages the two cases. The orders $\pm m$ give equal contributions, hence the factor 2. The series
stops a few orders beyond $x$, as Mie's does. Bohren and Huffman plot instead the scattering per unit *volume*,
$C_\mathrm{sca}/V = 4 Q_\mathrm{sca}/(\pi d)$ with $d = 2a$, which tends to the geometric-optics asymptote $8/(\pi d)$
as $Q \to 2$: the quantity a chemist measuring turbidity cares about.

::: example glass-cylinder

The two polarizations differ most strongly at a few hundred nanometres of radius, where case I rises first and case
II lags: a resonance of case I at a given order $m$ needs the electric field to fit along the axis, that of case II
needs the magnetic field to, and for $m > 1$ these fall at different sizes. At large $x$ both tend to 2 with a
decaying ripple, and the polarization dependence fades.

::: classic bohren-huffman-1983

## The thin wire

For $x \ll 1$ the two cases behave very differently, and the difference is the whole story of metal nanowires. In
case I the field along the axis induces a current along the wire with no charges at any surface; the response is
proportional to $(m^2 - 1)$ and the scattering efficiency grows like $x^3$, with no resonance. In case II the field
across the wire polarises it like a two-dimensional dipole, and the leading coefficient is

$$
a_1^{\mathrm{II}} \approx -\frac{i\pi x^2}{4}\,\frac{m^2 - 1}{m^2 + 1} = -\frac{i\pi x^2}{4}\,\frac{\varepsilon - \varepsilon_h}{\varepsilon + \varepsilon_h}.
$$

The denominator is the two-dimensional analogue of the sphere's $\varepsilon + 2\varepsilon_h$: a metal wire in
vacuum resonates where $\varepsilon = -1$, which for a Drude metal is $\omega = \omega_p/\sqrt{2}$ rather than the
sphere's $\omega_p/\sqrt{3}$. Embedding the wire in a medium of permittivity $\varepsilon_h$ moves the resonance to
$\varepsilon = -\varepsilon_h$. The [wires and fibres](learn/cylinders/wires-and-fibres.html) section follows this
resonance down to a wire one nanometre thick, where the local description reaches its limit.

## Coated cylinders and oblique incidence

A cylinder with shells is solved by the same $2\times 2$ recursion as a multilayered sphere: one transfer matrix per
interface and per order $m$, multiplied inward from the host. PyStratify's cylinder solver does this for any number of
shells, with the amplitudes carried in logarithmic form so that a thick absorbing shell or a high order never
overflows. At **oblique incidence** $k_z = k\cos\zeta$ is no longer zero: $q$ shrinks, every Bessel argument changes,
and, except for $m = 0$, the two polarizations **couple** at each interface, so the transfer matrix becomes $4\times 4$
per order and a case-I wave scatters partly into case-II light. The calculator's incidence angle for cylinders is this
$\zeta$ (90° is normal incidence); the hybrid modes it then finds are the ones that guide light along an optical
fibre, which is the subject of the [wires and fibres](learn/cylinders/wires-and-fibres.html) section.

## How much can one channel scatter? Superscattering

Each order $m$ is a separate channel, and for a lossless cylinder its coefficient can at most reach the value that
makes $|b_m|^2$ (or $|a_m|^2$) equal to 1: the contribution of a single channel to the scattering width is bounded by
$2\lambda/\pi$ (counting $\pm m$ together). A subwavelength object that resonates in one channel can therefore never
scatter more than this, no matter how strong the resonance; it is the two-dimensional analogue of the
single-channel limit of a sphere. Ruan and Fan noticed that nothing stops *several* channels from resonating at the
same wavelength if the structure has enough free parameters, and designed a three-layer plasmonic rod whose $m = 1,
2, 3, 4$ resonances coincide, so that the scattering width exceeds the single-channel bound several times over:
**superscattering**. The design is sensitive by construction, because four resonances have to be tuned onto each
other.

::: classic ruan-fan-2010
