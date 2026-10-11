---
id: bandpass-filters
title: "Bandpass filters: all-dielectric cavities"
geometry: films
chapter: Filters and cavities
topic: bandpass
sources: [plane_wave]
summary: >-
  A half-wave spacer between two quarter-wave mirrors passes one narrow band inside a broad stop band: the design
  (HL)^p 2L (LH)^p, what sets the passband width, and the notch filter that does the opposite.
examples: [bandpass-filter]
references:
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 7
- cite: P. W. Baumeister, Optical Coating Technology (SPIE Press, 2004)
---

An all-dielectric bandpass filter is a [Fabry–Pérot cavity](learn/films/fabry-perot-etalons.html) built entirely
from quarter-wave layers. Two [Bragg mirrors](learn/films/bragg-mirrors.html) of high- and low-index layers surround
a spacer that is a whole number of half waves thick at the design wavelength $\lambda_0$:

$$
(\mathrm{HL})^p \,(2\mathrm{L})^m\, (\mathrm{LH})^p ,
$$

where $p$ is the number of pairs per mirror and $m$ the cavity order. At $\lambda_0$ the spacer is absent (a half-wave
layer), the two mirrors face each other as a symmetric pair, and the stack transmits; a little away from $\lambda_0$
the mirrors reflect nearly everything and the cavity no longer resonates.

## What sets the width

The mirrors' reflectance $R$ grows geometrically with $p$, and the passband narrows with it,

$$
\Delta\lambda_{1/2} \propto (1 - R) ,
$$

roughly by the index ratio squared for every pair added. A higher cavity order $m$ narrows it further, because the
phase in a thicker spacer changes faster with wavelength. The rejection outside the passband is the mirrors' own
[stop band](learn/films/bragg-mirrors.html): its width comes from the index contrast, so TiO₂/SiO₂ blocks about
490–630 nm around a 550 nm design, and a practical filter adds broadband blockers beyond that.

::: example bandpass-filter

Moving the spacer away from a half wave shifts the passband through the stop band; it disappears into the edge once
the cavity's resonance leaves the band. Absorption in the layers, ignored in the ideal design, is what limits the peak
transmittance of very narrow filters.

## The notch filter

Swap the roles and a stack rejects one band while passing the rest: a single quarter-wave mirror with weak index
contrast (or a rugate profile) reflects a narrow band around $\lambda_0$ and transmits elsewhere. Laser-line
blocking filters in Raman spectroscopy are built this way.
