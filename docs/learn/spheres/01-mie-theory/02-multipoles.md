---
id: multipoles-and-kerker
title: "Multipoles, directional scattering and the Kerker conditions"
geometry: spheres
chapter: Mie theory
topic: multipoles
sources: [plane_wave]
description: "Reading a scattering spectrum by its multipoles: electric and magnetic dipoles, the Kerker conditions, Huygens metasurfaces and the anapole."
reviewed:
summary: >-
  Reading a spectrum by its electric and magnetic multipoles: each order's share of the scattering and the
  single-channel bound, the small-sphere expansions of a_1 and b_1, how a magnetic dipole arises in a dielectric and
  where it sits, the forward and backward amplitudes, the first and second Kerker conditions and their generalisation
  to quadrupoles, Huygens metasurfaces, and the anapole as a zero of a_1.
classics: [kuznetsov-2012, fu-2013]
examples: [si-sphere-multipoles]
references:
- cite: C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley, 1983), ch. 4
  doi: 10.1002/9783527618156
- cite: M. Kerker, D.-S. Wang and C. L. Giles, J. Opt. Soc. Am. 73, 765 (1983)
  doi: 10.1364/JOSA.73.000765
- cite: A. B. Evlyukhin, C. Reinhardt, A. Seidel, B. S. Luk'yanchuk and B. N. Chichkov, Phys. Rev. B 82, 045404 (2010)
  doi: 10.1103/PhysRevB.82.045404
- cite: A. I. Kuznetsov et al., Sci. Rep. 2, 492 (2012)
  doi: 10.1038/srep00492
- cite: J. M. Geffrin et al., Nat. Commun. 3, 1171 (2012)
  doi: 10.1038/ncomms2167
- cite: Y. H. Fu et al., Nat. Commun. 4, 1527 (2013)
  doi: 10.1038/ncomms2538
- cite: I. Staude et al., ACS Nano 7, 7824 (2013)
  doi: 10.1021/nn402736f
- cite: A. E. Miroshnichenko et al., Nat. Commun. 6, 8069 (2015)
  doi: 10.1038/ncomms9069
---

The Mie series is a sum over multipoles, and every term has a physical identity: $a_\ell$ is the electric multipole
of order $\ell$ (dipole, quadrupole, …), $b_\ell$ the magnetic one. Taking a spectrum apart term by term turns a
curve with many bumps into a short list of resonances, each with its own field pattern and its own radiation
diagram, and the interference of those diagrams is what makes a particle scatter forward, backward, or not at all.
This is the vocabulary of all-dielectric nanophotonics, and it is a few lines of algebra on the
[Mie coefficients](learn/spheres/mie-scattering.html).

## Each order's share

Because the vector spherical harmonics are orthogonal over the sphere, the scattered power is a plain sum over orders
and types, with no cross terms:

$$
Q_\mathrm{sca} = \sum_{\ell=1}^{\infty} \left(Q^{E}_{\ell} + Q^{M}_{\ell}\right),
\qquad
Q^{E}_{\ell} = \frac{2}{x^2}(2\ell+1)\,|a_\ell|^2,
\qquad
Q^{M}_{\ell} = \frac{2}{x^2}(2\ell+1)\,|b_\ell|^2 .
$$

Each term is bounded: a single channel cannot scatter more than $\frac{2}{x^2}(2\ell+1)$, the limit reached when
$|a_\ell| = 1$, which for a lossless sphere is where the coefficient's phase passes through $\pi/2$ at a resonance.
In cross-section terms the bound is $(2\ell+1)\lambda^2/(2\pi)$ per channel: a dipole, however resonant, cannot
scatter more than $3\lambda^2/2\pi$, which is why a 10 nm silver particle at its plasmon has a cross section a
hundred times its geometric one and no more. Extinction splits the same way, through $\mathrm{Re}\,a_\ell$ and
$\mathrm{Re}\,b_\ell$; the calculator's multipole panel draws these shares, one colour per order, electric dashed
and magnetic dot-dashed.

## The small sphere: what a_1 and b_1 are made of

Expanding the coefficients in the size parameter shows what each resonance is physically. To leading orders,

$$
a_1 \approx -\frac{2i}{3}\,x^3\,\frac{m^2-1}{m^2+2}\left[1 + \frac{3}{5}\,\frac{m^2 - 2}{m^2 + 2}\,x^2 + \dots\right],
\qquad
b_1 \approx -\frac{i}{45}\,x^5\,(m^2-1) ,
\qquad
a_2 \approx -\frac{i}{15}\,x^5\,\frac{m^2-1}{2m^2+3} .
$$

The electric dipole is the quasi-static polarizability of a sphere, with the Clausius–Mossotti denominator and its
plasmon pole at $m^2 = -2$; the magnetic dipole has no pole in the quasi-static limit and starts two powers of $x$
later, at the same order as the electric quadrupole. For a metal the first term is everything and the particle is an
electric dipole; for a dielectric the $x^2$ correction inside the bracket and the $x^5$ terms grow quickly with $m$,
and at $mx \sim \pi$ the expansion fails altogether, because the magnetic dipole resonates.

## A magnetic dipole without magnetism

Silicon has $\mu = 1$, yet a silicon sphere has a strong magnetic-dipole resonance. When the wavelength inside the
particle, $\lambda/n$, is close to its diameter, the electric field inside circulates around an axis perpendicular to
the incident field. That loop of displacement current is a magnetic dipole, and in a high-index sphere it is the
first resonance on the long-wavelength side: the condition is $\lambda_\mathrm{MD} \approx n_s\, d$, the first zero of
the magnetic denominator at $mx \approx \pi$ for large $m$ (Evlyukhin et al. 2010), so a 150 nm silicon sphere resonates
near 600 nm. The electric dipole sits at shorter wavelength ($mx \approx 4.5$), the magnetic quadrupole between them.
Kuznetsov et al. (2012) saw the magnetic dipole move across the visible as the particles grew, giving them their
colours, and Geffrin et al. (2012) measured the same resonances in the microwave range with centimetre-sized
ceramic spheres, where the far-field pattern can be mapped directly.

::: example si-sphere-multipoles

Move the radius and watch the order: magnetic dipole first, then electric dipole and magnetic quadrupole close
together, then the electric quadrupole. The electric dipole is broad because it also radiates from the particle's
polarization outside the resonance; the magnetic modes, confined inside the sphere, are narrower.

::: classic kuznetsov-2012

## Forward and backward

The far field in the forward and backward directions follows from the same coefficients, because at $\theta = 0$ and
$\pi$ the angular functions $\pi_\ell$ and $\tau_\ell$ both equal $\pm\ell(\ell+1)/2$:

$$
Q_\mathrm{f} = \frac{1}{x^2}\left|\sum_\ell (2\ell+1)(a_\ell + b_\ell)\right|^2,
\qquad
Q_\mathrm{b} = \frac{1}{x^2}\left|\sum_\ell (2\ell+1)(-1)^\ell (a_\ell - b_\ell)\right|^2 .
$$

Electric and magnetic multipoles of the same order add in the forward direction and subtract backwards: an electric
dipole and a magnetic dipole, crossed and in phase, radiate like a Huygens source, forward only. If only the dipoles
matter and $a_1 = b_1$, the backscattering vanishes: the **first Kerker condition**, found by Kerker, Wang and Giles
(1983) for spheres with $\varepsilon = \mu$, where every $a_\ell$ equals its $b_\ell$ and the sphere does not
backscatter at any size. A dielectric sphere has no $\mu$, but it reaches $a_1 = b_1$ at one wavelength where its
electric- and magnetic-dipole curves cross on the long-wavelength side of the magnetic resonance, and there it
scatters forward almost only (Fu et al. 2013; Geffrin et al. 2012 saw the same in the microwave). The **second Kerker
condition**, $a_1 = -b_1$, minimises forward scattering, but cannot null it: the optical theorem ties forward
scattering to the extinction, which is never zero. With quadrupoles the conditions generalise to the full sums,
$\sum_\ell (2\ell+1)(-1)^\ell(a_\ell - b_\ell) = 0$ for zero backscattering, which can be met at more than one
wavelength and with a sharper directivity; the forward-to-backward ratio of the classic card is the number to watch.

::: classic fu-2013

## From one sphere to a Huygens surface

A sphere at its first Kerker condition transmits all the light it intercepts, with a phase delay set by the two
resonances. An array of such spheres, one per sub-wavelength cell, is a **Huygens metasurface** (Staude et al. 2013): a
flat sheet with near-unit transmittance whose phase can be chosen from $0$ to $2\pi$ by the particle size, the
building block of flat lenses and holograms. Its design reduces to placing the electric and magnetic dipole
resonances of the single particle, which is the calculation above, with the lattice adding a correction that the
single-sphere calculation leaves out.

## Beyond the dipoles

When an electric dipole and a toroidal dipole of the same strength radiate in antiphase, their far fields cancel and
the electric-dipole scattering nearly vanishes while the field inside stays strong: the anapole (Miroshnichenko et al.,
2015). In the Mie picture it is simply a zero of $a_1$, visible in the decomposition as a dip of the electric dipole
between resonances: the full coefficient contains both the Cartesian electric dipole and the toroidal moment, and the
multipole expansion in vector spherical harmonics, being complete, needs no separate toroidal term. The distinction
matters when one wants to know what is going on *inside* the particle (a toroidal field pattern) rather than what it
radiates, and the near-field panel of the calculator shows exactly that circulating pattern at the anapole
wavelength.
