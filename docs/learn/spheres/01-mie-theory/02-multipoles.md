---
id: multipoles-and-kerker
title: "Multipoles, directional scattering and the Kerker conditions"
geometry: spheres
chapter: Mie theory
topic: multipoles
sources: [plane_wave]
summary: >-
  Reading a spectrum by its electric and magnetic multipoles: each order's share of the scattering, how a magnetic
  dipole arises in a dielectric, and how two multipoles interfering in phase cancel the backscattering.
examples: [si-sphere-multipoles]
references:
- cite: C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley, 1983), ch. 4
  doi: 10.1002/9783527618156
- cite: M. Kerker, D.-S. Wang and C. L. Giles, J. Opt. Soc. Am. 73, 765 (1983)
  doi: 10.1364/JOSA.73.000765
- cite: A. I. Kuznetsov et al., Sci. Rep. 2, 492 (2012)
  doi: 10.1038/srep00492
- cite: Y. H. Fu et al., Nat. Commun. 4, 1527 (2013)
  doi: 10.1038/ncomms2538
- cite: A. E. Miroshnichenko et al., Nat. Commun. 6, 8069 (2015)
  doi: 10.1038/ncomms9069
---

The Mie series is a sum over multipoles, and every term has a physical identity: $a_\ell$ is the electric multipole
of order $\ell$ (dipole, quadrupole, …), $b_\ell$ the magnetic one. Taking a spectrum apart term by term turns a
curve with many bumps into a short list of resonances, each with its own field pattern and its own radiation
diagram.

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
$|a_\ell| = 1$. Extinction splits the same way, through $\mathrm{Re}\,a_\ell$ and $\mathrm{Re}\,b_\ell$.

## A magnetic dipole without magnetism

Silicon has $\mu = 1$, yet a silicon sphere has a strong magnetic-dipole resonance. When the wavelength inside the
particle, $\lambda/n$, is close to its diameter, the electric field inside circulates around an axis perpendicular to
the incident field. That loop of displacement current is a magnetic dipole, and in a high-index sphere it is the
first resonance on the long-wavelength side. Kuznetsov et al. (2012) saw it move across the visible as the particles
grew, giving them their colours.

::: example si-sphere-multipoles

Move the radius and watch the order: magnetic dipole first, then electric dipole and magnetic quadrupole close
together, then the electric quadrupole. The electric dipole is broad because it also radiates from the particle's
polarization outside the resonance; the magnetic modes, confined inside the sphere, are narrower.

## Forward and backward

The far field in the forward and backward directions follows from the same coefficients:

$$
Q_\mathrm{f} = \frac{1}{x^2}\left|\sum_\ell (2\ell+1)(a_\ell + b_\ell)\right|^2,
\qquad
Q_\mathrm{b} = \frac{1}{x^2}\left|\sum_\ell (2\ell+1)(-1)^\ell (a_\ell - b_\ell)\right|^2 .
$$

Electric and magnetic multipoles of the same order add in the forward direction and subtract backwards. If only the
dipoles matter and $a_1 = b_1$, the backscattering vanishes: the **first Kerker condition**, found by Kerker, Wang and
Giles (1983) for spheres with $\varepsilon = \mu$. A silicon sphere reaches it where its electric- and magnetic-dipole
curves cross on the long-wavelength side of the magnetic resonance, and scatters forward almost only (Fu et al.,
2013). The **second Kerker condition**, $a_1 = -b_1$, minimises forward scattering, but cannot null it: the optical
theorem ties forward scattering to the extinction, which is never zero.

[Classics](classics.html#fu-2013) opens Fu et al.'s 150 nm silicon sphere with its scattering pattern; pick 660 nm to
see the forward lobe.

## Beyond the dipoles

When an electric dipole and a toroidal dipole of the same strength radiate in antiphase, their far fields cancel and
the electric-dipole scattering nearly vanishes while the field inside stays strong: the anapole (Miroshnichenko et al.,
2015). In the Mie picture it is simply a zero of $a_1$, visible in the decomposition as a dip of the electric dipole
between resonances.
