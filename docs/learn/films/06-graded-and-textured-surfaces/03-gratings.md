---
id: gratings
title: "Gratings: diffraction orders and the sub-wavelength limit"
geometry: films
chapter: Graded and textured surfaces
topic: gratings
sources: [plane_wave]
summary: >-
  A periodic surface couples light into discrete orders: the grating equation, the cutoff below which only the zeroth
  order survives, and the effective film a sub-wavelength grating becomes.
examples: [grating-effective-film]
references:
- cite: M. G. Moharam and T. K. Gaylord, J. Opt. Soc. Am. 71, 811 (1981)
  doi: 10.1364/JOSA.71.000811
- cite: S. M. Rytov, Sov. Phys. JETP 2, 466 (1956)
- cite: "E. Popov (ed.), Gratings: Theory and Numeric Applications (Institut Fresnel, 2012)"
---

A planar stack conserves the in-plane wave vector exactly. A grating of period $\Lambda$ conserves it only to within a
multiple of $2\pi/\Lambda$, so it couples the incident wave into a comb of orders $m$, each leaving at the angle given by
the grating equation

$$
n \sin\theta_m = n_0 \sin\theta_0 + m\,\frac{\lambda}{\Lambda} .
$$

## Propagating and evanescent orders

Order $m$ propagates in a medium of index $n$ only while $|n \sin\theta_m| < n$; beyond that it is evanescent and
carries no power away. As the period shrinks, or the wavelength grows, the orders cut off one by one, the last of them
at

$$
\lambda_c = \Lambda\,(n_\mathrm{max} + n_0 \sin\theta_0),
$$

where $n_\mathrm{max}$ is the larger of the two media's indices. A 300 nm grating on glass at normal incidence
diffracts into the glass below 456 nm and into air below 300 nm.

## Above cutoff: an effective film

Past $\lambda_c$ only the zeroth order survives, and to it the grating is a homogeneous but anisotropic layer. For a
line grating of fill fraction $f$ the lowest-order effective indices (Rytov, 1956) are

$$
n_\mathrm{TE}^2 = f\,n_1^2 + (1-f)\,n_2^2, \qquad \frac{1}{n_\mathrm{TM}^2} = \frac{f}{n_1^2} + \frac{1-f}{n_2^2},
$$

for light polarized along and across the lines. That birefringence is a design tool in its own right (form-birefringent
wave plates, polarizers), and for TE light the grating reflects exactly like the film below.

::: example grating-effective-film

Below the cutoff the effective film is wrong: energy leaves along the diffracted orders, which no planar calculation
produces. Gratings are solved there by rigorous coupled-wave analysis (Moharam and Gaylord, 1981), which the
calculator runs for its grating layers, outside PyStratify.
