---
id: mie-scattering
title: "Mie theory: light scattered by a sphere"
geometry: spheres
chapter: Mie theory
topic: mie
sources: [plane_wave]
summary: >-
  How a plane wave scatters off a homogeneous sphere: the plane wave as a sum of multipoles, the coefficients a_ℓ and
  b_ℓ from two boundary conditions per order, the efficiencies and the amplitude functions of the far field, the
  Rayleigh and geometric limits and the extinction paradox, van de Hulst's anomalous diffraction, where the series
  stops, and the numerics of the Riccati–Bessel functions that every Mie code lives or dies by.
classics: [mie-1908, mohlenhoff-2005]
examples: [si-sphere-mie]
references:
- cite: G. Mie, Ann. Phys. 330, 377 (1908)
  doi: 10.1002/andp.19083300302
- cite: C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley, 1983), ch. 4
  doi: 10.1002/9783527618156
- cite: H. C. van de Hulst, Light Scattering by Small Particles (Wiley, 1957; Dover, 1981), ch. 9 and 11
- cite: M. Kerker, The Scattering of Light and Other Electromagnetic Radiation (Academic Press, 1969), ch. 3
- cite: W. J. Wiscombe, Appl. Opt. 19, 1505 (1980)
  doi: 10.1364/AO.19.001505
- cite: W. J. Lentz, Appl. Opt. 15, 668 (1976)
  doi: 10.1364/AO.15.000668
- cite: A. Moroz, Ann. Phys. 315, 352 (2005)
  doi: 10.1016/j.aop.2004.07.002
- cite: A. I. Kuznetsov et al., Sci. Rep. 2, 492 (2012)
  doi: 10.1038/srep00492
---

A sphere of radius $a$ and refractive index $n_s$ sits in a host of index $n_h$, lit by a plane wave of vacuum
wavelength $\lambda$. Two numbers decide everything that follows: the **size parameter** and the **relative index**,

$$
x = \frac{2\pi n_h a}{\lambda}, \qquad m = \frac{n_s}{n_h}.
$$

Gustav Mie solved this problem exactly in 1908, to explain why colloidal gold is red, violet or blue depending on the
size of its particles (Lorenz had the solution in 1890 and Debye in 1909, which is why the theory carries three
names in the older literature). The same solution describes a raindrop, a silicon nanoparticle and a biological cell
under an infrared microscope, and it is the single-interface case of the
[multilayered sphere](learn/spheres/recursive-transfer-matrix.html) that PyStratify solves in general.

::: classic mie-1908

## The plane wave as a sum of multipoles

In spherical coordinates the vector wave equation separates (the
[Foundations](learn/common/maxwell-and-boundary-conditions.html) table). The fields inside and outside the sphere
become sums of vector spherical harmonics, each labelled by an angular order $\ell = 1, 2, \dots$ (dipole,
quadrupole, …) and by its character: **electric** (transverse magnetic, TM) or **magnetic** (transverse electric,
TE). The first step, and the one that makes the whole method work, is to write the incident plane wave in the same
language. A plane wave polarized along $x$ and travelling along $z$ is (Bohren and Huffman Eq. 4.37)

$$
\mathbf{E}_\mathrm{inc} = E_0\sum_{\ell=1}^{\infty} i^\ell\,\frac{2\ell+1}{\ell(\ell+1)}\left(\mathbf{M}^{(1)}_{o1\ell} - i\,\mathbf{N}^{(1)}_{e1\ell}\right),
$$

a sum over orders of the regular ($j_\ell$-based) magnetic and electric multipoles with $m = 1$ only, with the
coefficients $E_\ell = i^\ell E_0 (2\ell+1)/[\ell(\ell+1)]$. Only $|m| = 1$ appears because a plane wave has one
axis; a [dipole source](learn/spheres/dipole-near-a-sphere.html) off the centre excites every $m$. The radial
dependence of every term obeys the spherical Bessel equation, a second-order linear equation, so in each region it
is a combination of two independent solutions. We use the Riccati–Bessel functions

$$
\psi_\ell(\rho) = \rho\, j_\ell(\rho), \qquad \xi_\ell(\rho) = \rho\, h^{(1)}_\ell(\rho),
$$

regular at the centre and outgoing at infinity respectively: inside the sphere only $\psi_\ell$ is allowed, outside the
scattered field is pure $\xi_\ell$, and the incident field is $\psi_\ell$.

## Two boundary conditions per order

At $r = a$ the tangential components of $\mathbf{E}$ and $\mathbf{H}$ are continuous. Because the harmonics are
orthogonal on the sphere, the conditions hold order by order and character by character: for each $\ell$ and each
of TM and TE this gives two linear equations in two unknowns (the internal amplitude and the scattered one), the
same $2\times 2$ structure as a single interface of a thin film. Solving them gives the scattering coefficients of a
non-magnetic sphere (time dependence $e^{-i\omega t}$):

$$
a_\ell = \frac{m\,\psi_\ell(mx)\,\psi_\ell'(x) - \psi_\ell(x)\,\psi_\ell'(mx)}{m\,\psi_\ell(mx)\,\xi_\ell'(x) - \xi_\ell(x)\,\psi_\ell'(mx)},
\qquad
b_\ell = \frac{\psi_\ell(mx)\,\psi_\ell'(x) - m\,\psi_\ell(x)\,\psi_\ell'(mx)}{\psi_\ell(mx)\,\xi_\ell'(x) - m\,\xi_\ell(x)\,\psi_\ell'(mx)}.
$$

$a_\ell$ is the electric multipole of order $\ell$, $b_\ell$ the magnetic one, and the two differ only in where the
relative index $m$ sits, exactly as $r_s$ and $r_p$ of a film differ in where $n$ sits. For a magnetic sphere ($\mu
\ne 1$) $m$ in the numerators and denominators is replaced by $m/\mu$ or $m\mu$ in the appropriate places (Kerker's
book), which is how a sphere with $\varepsilon = \mu$ comes to have $a_\ell = b_\ell$, the origin of the Kerker
conditions of the [next section](learn/spheres/multipoles.html). The resonances of the sphere live in the
denominators: where one nearly vanishes, its coefficient peaks and the sphere radiates strongly in that multipole.
Add shells and each interface contributes one more $2\times 2$ matrix; that is the transfer-matrix solution of the
multilayered sphere, and what PyStratify computes for any number of layers.

## Efficiencies and the far field

Dividing the cross sections by the geometric area $\pi a^2$ gives the efficiencies:

$$
Q_\mathrm{sca} = \frac{2}{x^2}\sum_{\ell=1}^{\infty}(2\ell+1)\left(|a_\ell|^2 + |b_\ell|^2\right),
\qquad
Q_\mathrm{ext} = \frac{2}{x^2}\sum_{\ell=1}^{\infty}(2\ell+1)\,\mathrm{Re}\,(a_\ell + b_\ell),
$$

and $Q_\mathrm{abs} = Q_\mathrm{ext} - Q_\mathrm{sca}$. Each term of the sums is the share of one multipole, which is
how the calculator splits a spectrum into its electric and magnetic orders. The extinction formula is the **optical
theorem** in disguise: $Q_\mathrm{ext}$ is the forward-scattering amplitude's imaginary part, so the light removed from
the beam is measured by what is scattered exactly forward, whether the sphere absorbs or not. The angular
distribution comes from the two **amplitude functions**

$$
S_1(\theta) = \sum_\ell \frac{2\ell+1}{\ell(\ell+1)}\left(a_\ell\,\pi_\ell + b_\ell\,\tau_\ell\right),
\qquad
S_2(\theta) = \sum_\ell \frac{2\ell+1}{\ell(\ell+1)}\left(a_\ell\,\tau_\ell + b_\ell\,\pi_\ell\right),
$$

with $\pi_\ell = P_\ell^1/\sin\theta$ and $\tau_\ell = \mathrm{d}P_\ell^1/\mathrm{d}\theta$ the angular functions; the
scattered intensity for light polarized perpendicular and parallel to the scattering plane is $|S_1|^2$ and
$|S_2|^2$ over $(ka)^2$, and the polarization, the asymmetry parameter $\langle\cos\theta\rangle$ and the Mueller
matrix all follow from them. Checks of the engine's $S_1$, $S_2$ against Bohren and Huffman's own BHMIE and against
two other Mie codes are on the [Verification](verification.html) page.

## A silicon sphere

Silicon has an index near 4 across the visible. Inside so dense a particle the wavelength shrinks to $\lambda/n_s$, and
when it becomes comparable to the diameter the field inside circulates: a loop of displacement current, the
**magnetic dipole** $b_1$. It is the lowest resonance of a high-index sphere, longer in wavelength than the electric
dipole $a_1$ that follows it, and it gives such particles their colours (Kuznetsov et al., 2012).

::: example si-sphere-mie

Move the radius from 50 to 135 nm and the magnetic dipole walks across the visible spectrum; the electric dipole and
then the magnetic quadrupole follow it at shorter wavelengths.

## Where the series stops

The sums converge once $\ell$ exceeds the size parameter by a few terms: a partial wave of order $\ell$ has its
angular momentum at an impact parameter $\ell/k$, and orders with $\ell/k \gg a$ miss the sphere. Wiscombe's rule

$$
\ell_\mathrm{max} = x + 4x^{1/3} + 2
$$

is accurate for far-field quantities to about $10^{-8}$; the $x^{1/3}$ term is the width of the edge region where
grazing partial waves are diffracted. For a high-index core the internal size parameter $|m|x$ matters as well, so
PyStratify, and the calculator through it, take the larger of the two; a field evaluated at the surface, or a
[dipole close to it](learn/spheres/dipole-near-a-sphere.html), needs more still, and the engine extends the order
until the quantity converges.

## The numerics every Mie code lives by

Mie's formulas are exact and innocent-looking, and most of the published Mie codes that give wrong answers fail in
the Riccati–Bessel functions, not in the physics. Three facts matter. $\psi_\ell(z)$ computed by *upward* recurrence
in $\ell$ is unstable for $\ell > |z|$ (the errors grow like the ratio of the two solutions, which is huge there), so
the standard trick since Deirmendjian is to compute the **logarithmic derivative** $D_\ell(mx) = \psi_\ell'(mx)/\psi_\ell(mx)$
by *downward* recurrence from a starting order well above $\ell_\mathrm{max}$, or by Lentz's continued fraction
(1976), and to write $a_\ell$ and $b_\ell$ in terms of $D_\ell$ alone; this is BHMIE's algorithm. For complex $mx$ the
function $\psi_\ell(mx)$ itself grows like $e^{|\mathrm{Im}\,mx|}$, and the coefficients are ratios of numbers that
overflow long before the sums converge, which is where a **logarithmic** representation of the amplitudes, carried
through the transfer-matrix recursion (Moroz 2005), keeps a metal sphere or a thin absorbing shell safe at any order.
And for $\ell \gg x$ the coefficients fall like $(x/\ell)^{4\ell}$, so a sum that is stopped early is wrong by the
first dropped term, not by a rounding error; a coefficient of $10^{-300}$ must be representable, or set to exactly
zero, never to garbage. The [Verification](verification.html) page compares the engine's coefficients with the same
formulas evaluated in 60-digit arithmetic over forty orders of a thin nanoshell, and with BHMIE to $x = 20\,000$; one of
those comparisons found BHMIE's own double-precision recurrence drifting at large $x$ for a weakly absorbing sphere.

## Two limits worth knowing

For a sphere much smaller than the wavelength ($x \ll 1$) only $a_1$ survives, with
$a_1 \approx -\tfrac{2i}{3}x^3\,\frac{m^2-1}{m^2+2}$, and

$$
Q_\mathrm{sca} \approx \frac{8}{3}\,x^4 \left|\frac{m^2 - 1}{m^2 + 2}\right|^2,
\qquad
Q_\mathrm{abs} \approx 4x\,\mathrm{Im}\,\frac{m^2 - 1}{m^2 + 2} ,
$$

Rayleigh's $\lambda^{-4}$ law for scattering, and an absorption that goes as the *volume*, so that small absorbing
particles absorb far more than they scatter. For a metal the denominator vanishes near $\varepsilon_s = -2\varepsilon_h$:
that is the localized surface plasmon of a small gold or silver particle, at 520 nm for gold in water, the colour of
the ruby glass that started Mie on the problem. For a sphere much larger than the wavelength, $Q_\mathrm{ext} \to 2$:
the sphere removes twice the light that falls on its cross section, half by absorption or reflection and half by
diffraction around its edge, the **extinction paradox**, resolved by noting that the diffracted half is scattered into
so narrow a forward cone that any real detector collects it and measures $Q \approx 1$. Between the two limits a weakly
refracting sphere ($m \to 1$, $x \gg 1$) follows van de Hulst's anomalous-diffraction approximation,
$Q_\mathrm{ext} \approx 2 - \frac{4}{\rho}\sin\rho + \frac{4}{\rho^2}(1 - \cos\rho)$ with the phase lag $\rho = 2x(m-1)$:
a slow oscillation in wavelength, with its first maximum near $\rho = 4$, that a cell or a nucleus of a few
micrometres imprints on its infrared spectrum, and which is routinely mistaken for a baseline.

::: classic mohlenhoff-2005
