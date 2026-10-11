---
id: fabry-perot-etalons
title: "Fabry–Pérot etalons: a spacer between two mirrors"
geometry: films
chapter: Filters and cavities
topic: fabry-perot
sources: [plane_wave]
summary: >-
  A transparent spacer between two partial mirrors transmits a comb of narrow peaks: the resonance condition, the free
  spectral range, the finesse, and what metal mirrors add.
examples: [fabry-perot-etalon]
references:
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §7.6
  doi: 10.1017/CBO9781139644181
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 7
- cite: G. Hernandez, Fabry–Perot Interferometers (Cambridge University Press, 1986)
---

Put a transparent layer between two partially reflecting mirrors and light bounces back and forth inside it. Where
the round trip adds a whole number of wavelengths, every pass adds in phase and the cavity transmits strongly, even
though each mirror alone reflects most of the light. Elsewhere the passes cancel and the cavity reflects.

## The resonance condition

For a spacer of index $n$ and thickness $d$, the round-trip phase at angle $\theta$ inside it is
$\delta = \frac{4\pi}{\lambda} n d\cos\theta$ plus the phases $\phi_1$, $\phi_2$ picked up on reflection at the two
mirrors. Transmission peaks where

$$
2 n d \cos\theta = \left(m - \frac{\phi_1 + \phi_2}{2\pi}\right)\lambda_m , \qquad m = 1, 2, \dots
$$

For dielectric mirrors the reflection phases are 0 or $\pi$; for metal mirrors they are neither, and they shift every
order by a fraction of a free spectral range.

## Free spectral range and finesse

Neighbouring orders are separated by

$$
\Delta\lambda_\mathrm{FSR} \approx \frac{\lambda^2}{2 n d \cos\theta},
$$

so a thicker cavity packs its peaks closer. The width of each peak relative to that spacing is set by the mirror
reflectance $R$, through the coefficient of finesse and the finesse:

$$
F = \frac{4R}{(1-R)^2}, \qquad \mathcal{F} = \frac{\pi\sqrt{R}}{1-R} = \frac{\Delta\lambda_\mathrm{FSR}}{\delta\lambda_{1/2}} .
$$

For an ideal lossless cavity with equal mirrors the peaks reach $T = 1$ whatever $R$ is. Metal mirrors are not
lossless: every pass absorbs a little, so better mirrors sharpen the peaks but lower them.

::: example fabry-perot-etalon

The plotted peaks stay below 1 because the silver absorbs; the check below the figure confirms that what is not
reflected or transmitted is exactly what the layers absorb. Tilting the beam shortens $d\cos\theta$ and moves every
peak to shorter wavelengths, which is how an etalon is angle-tuned.

## From etalons to filters

A Fabry–Pérot cavity is the core of almost every narrowband filter. Replacing the metal mirrors by
[quarter-wave stacks](learn/films/bragg-mirrors.html) removes the absorption and raises the reflectance at will, which
is the [all-dielectric bandpass filter](learn/films/bandpass-filters.html).
