---
id: mie-scattering
title: "Mie theory: light scattered by a sphere"
geometry: spheres
chapter: Mie theory
topic: mie
sources: [plane_wave]
summary: >-
  How a plane wave scatters off a homogeneous sphere: the multipole expansion, the coefficients a_ℓ and b_ℓ from
  two boundary conditions per order, the efficiencies, and where the series can stop.
classics: [mie-1908, mohlenhoff-2005]
examples: [si-sphere-mie]
references:
- cite: G. Mie, Ann. Phys. 330, 377 (1908)
  doi: 10.1002/andp.19083300302
- cite: C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley, 1983), ch. 4
  doi: 10.1002/9783527618156
- cite: W. J. Wiscombe, Appl. Opt. 19, 1505 (1980)
  doi: 10.1364/AO.19.001505
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
size of its particles. The same solution describes a raindrop, a silicon nanoparticle and a biological cell under an
infrared microscope.

::: classic mie-1908

## Fields as a sum of multipoles

In spherical coordinates the vector wave equation separates. The fields inside and outside the sphere become sums of
vector spherical harmonics, each labelled by an angular order $\ell = 1, 2, \dots$ (dipole, quadrupole, …) and by its
character: **electric** (transverse magnetic, TM) or **magnetic** (transverse electric, TE). The radial dependence of
every term obeys the spherical Bessel equation, a second-order linear equation, so in each region it is a combination
of two independent solutions. We use the Riccati–Bessel functions

$$
\psi_\ell(\rho) = \rho\, j_\ell(\rho), \qquad \xi_\ell(\rho) = \rho\, h^{(1)}_\ell(\rho),
$$

regular at the centre and outgoing at infinity respectively.

## Two boundary conditions per order

At $r = a$ the tangential components of $\mathbf{E}$ and $\mathbf{H}$ are continuous. For each order $\ell$ and each
character this gives two linear equations in two unknowns, the same $2\times 2$ structure as a single interface of a
thin film. Solving them gives the scattering coefficients of a non-magnetic sphere (time dependence
$e^{-i\omega t}$):

$$
a_\ell = \frac{m\,\psi_\ell(mx)\,\psi_\ell'(x) - \psi_\ell(x)\,\psi_\ell'(mx)}{m\,\psi_\ell(mx)\,\xi_\ell'(x) - \xi_\ell(x)\,\psi_\ell'(mx)},
\qquad
b_\ell = \frac{\psi_\ell(mx)\,\psi_\ell'(x) - m\,\psi_\ell(x)\,\psi_\ell'(mx)}{\psi_\ell(mx)\,\xi_\ell'(x) - m\,\xi_\ell(x)\,\psi_\ell'(mx)}.
$$

$a_\ell$ is the electric multipole of order $\ell$, $b_\ell$ the magnetic one. The resonances of the sphere live in the
denominators: where one nearly vanishes, its coefficient peaks and the sphere radiates strongly in that multipole. Add
shells and each interface contributes one more $2\times 2$ matrix; that is the transfer-matrix solution of the
multilayered sphere, and what PyStratify computes for any number of layers.

## Efficiencies

Dividing the cross sections by the geometric area $\pi a^2$ gives the efficiencies:

$$
Q_\mathrm{sca} = \frac{2}{x^2}\sum_{\ell=1}^{\infty}(2\ell+1)\left(|a_\ell|^2 + |b_\ell|^2\right),
\qquad
Q_\mathrm{ext} = \frac{2}{x^2}\sum_{\ell=1}^{\infty}(2\ell+1)\,\mathrm{Re}\,(a_\ell + b_\ell),
$$

and $Q_\mathrm{abs} = Q_\mathrm{ext} - Q_\mathrm{sca}$. Each term of the sums is the share of one multipole, which is
how the calculator splits a spectrum into its electric and magnetic orders.

## A silicon sphere

Silicon has an index near 4 across the visible. Inside so dense a particle the wavelength shrinks to $\lambda/n_s$, and
when it becomes comparable to the diameter the field inside circulates: a loop of displacement current, the
**magnetic dipole** $b_1$. It is the lowest resonance of a high-index sphere, longer in wavelength than the electric
dipole $a_1$ that follows it, and it gives such particles their colours (Kuznetsov et al., 2012).

::: example si-sphere-mie

Move the radius from 50 to 135 nm and the magnetic dipole walks across the visible spectrum; the electric dipole and
then the magnetic quadrupole follow it at shorter wavelengths.

## Where the series stops

The sums converge once $\ell$ exceeds the size parameter by a few terms. Wiscombe's rule

$$
\ell_\mathrm{max} = x + 4x^{1/3} + 2
$$

is accurate for far-field quantities. For a high-index core the internal size parameter $|m|x$ matters as well, so
PyStratify, and the calculator through it, take the larger of the two.

## Two limits worth knowing

For a sphere much smaller than the wavelength ($x \ll 1$) only $a_1$ survives, and

$$
Q_\mathrm{sca} \approx \frac{8}{3}\,x^4 \left|\frac{m^2 - 1}{m^2 + 2}\right|^2,
$$

Rayleigh's $\lambda^{-4}$ law. For a metal the denominator vanishes near $\varepsilon_s = -2\varepsilon_h$: that is the
localized surface plasmon of a small gold or silver particle. For a sphere much larger than the wavelength,
$Q_\mathrm{ext} \to 2$: the sphere removes twice the light that falls on its cross section, half by absorption or
reflection and half by diffraction around its edge. Between the two limits a weakly refracting sphere ($m \to 1$,
$x \gg 1$) follows van de Hulst's anomalous-diffraction approximation, $Q_\mathrm{ext} \approx 2 - \frac{4}{\rho}\sin\rho
+ \frac{4}{\rho^2}(1 - \cos\rho)$ with the phase lag $\rho = 2x(m-1)$: a slow oscillation in wavelength that a cell
or a nucleus of a few micrometres imprints on its infrared spectrum, and which is routinely mistaken for a baseline.

::: classic mohlenhoff-2005
