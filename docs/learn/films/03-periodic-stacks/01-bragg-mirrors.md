---
id: bragg-mirrors
title: "Bragg mirrors: periodic stacks and stop bands"
geometry: films
chapter: Periodic stacks
topic: bragg
sources: [plane_wave]
summary: >-
  A quarter-wave stack of two materials reflects almost perfectly over a band of wavelengths: the peak reflectance
  from the admittance, the stop band from the Bloch condition, and how contrast and period count set both.
examples: [bragg-mirror]
references:
- cite: P. Yeh, A. Yariv and C.-S. Hong, J. Opt. Soc. Am. 67, 423 (1977)
  doi: 10.1364/JOSA.67.000423
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 5
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §1.6.5
  doi: 10.1017/CBO9781139644181
- cite: Y. Fink et al., Science 282, 1679 (1998)
  doi: 10.1126/science.282.5394.1679
---

Alternate a high index $n_H$ and a low index $n_L$, each a quarter wave thick at $\lambda_0$, and the reflections from
every interface return in phase. With enough periods the stack reflects nearly everything over a band around
$\lambda_0$: a dielectric mirror with no absorption, the basis of laser mirrors, distributed Bragg reflectors in
VCSELs and the one-dimensional photonic crystal.

## The peak: quarter waves in series

At $\lambda_0$ each quarter-wave layer maps an admittance $Y$ onto $n^2/Y$ (see
[the transfer matrix](learn/films/transfer-matrix.html)). Starting from the substrate $n_s$, a pair L then H
multiplies it by $(n_H/n_L)^2$, so $N$ pairs, high index outermost, present

$$
Y = \left(\frac{n_H}{n_L}\right)^{2N} n_s ,
\qquad
R(\lambda_0) = \left(\frac{n_0 - Y}{n_0 + Y}\right)^2 .
$$

$Y$ grows geometrically with $N$, so $1 - R$ falls geometrically: eight pairs of TiO₂-like and SiO₂-like layers
(2.3 and 1.45) already reach 99.84 %.

## The band: Bloch waves

An infinite periodic stack is solved by Bloch's theorem. With $M$ the characteristic matrix of one period, a wave
propagates through the crystal when $|\tfrac{1}{2}\,\mathrm{Tr}\,M| \le 1$ and decays when
$|\tfrac{1}{2}\,\mathrm{Tr}\,M| > 1$: there the Bloch wave is evanescent and a thick enough stack reflects it
completely. For a quarter-wave stack at normal incidence the decaying range, the **stop band**, has a full width in
the relative wavenumber $g = \lambda_0/\lambda$ of

$$
\Delta g = \frac{4}{\pi}\arcsin\!\left(\frac{n_H - n_L}{n_H + n_L}\right),
$$

about 0.29 for 2.3/1.45, or 524–702 nm around 600 nm. The width depends on the index contrast alone; the period count
only decides how close to 1 the reflectance gets inside it and how sharp its edges are. The same band repeats at
$\lambda_0/3$, $\lambda_0/5$, …, where the layers are three or five quarter waves thick.

::: example bragg-mirror

Outside the band the stack is nearly transparent, with ripples, one fewer minimum between bands than there are
periods. [Classics](classics.html#yeh-1977) opens Yeh, Yariv and Hong's 15-period GaAs/AlGaAs stack, where the 14
zeros and the closing of the TM gap at the Brewster angle can be counted.

## At oblique incidence

Tilting the beam shortens every layer's phase thickness, so the band moves to shorter wavelengths, and splits by
polarization: the p band narrows and closes at the Brewster angle of the H/L interface. Fink et al. (1998) showed that
with enough contrast a stop band survives for every angle and both polarizations, an omnidirectional reflector built
from dielectrics alone.
