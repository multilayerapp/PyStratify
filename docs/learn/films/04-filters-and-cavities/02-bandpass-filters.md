---
id: bandpass-filters
title: "Bandpass filters: all-dielectric cavities"
geometry: films
chapter: Filters and cavities
topic: bandpass
sources: [plane_wave]
description: "All-dielectric bandpass filters: the half-wave cavity design, bandwidth, multi-cavity square tops, angle shift, blocking, notch and rugate filters."
reviewed:
summary: >-
  A half-wave spacer between two quarter-wave mirrors passes one narrow band inside a broad stop band: the design
  (HL)^p 2L (LH)^p and why it transmits, Macleod's half-width formula and what each parameter does, the Lorentzian
  single-cavity line and the square-topped multi-cavity filter, the angle shift and the converging-beam limit,
  blocking, absorption and the narrowest filters, and the notch and rugate filters that do the opposite.
examples: [bandpass-filter]
references:
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 7
- cite: P. W. Baumeister, Optical Coating Technology (SPIE Press, 2004), ch. 8
- cite: A. Thelen, Design of Optical Interference Coatings (McGraw-Hill, 1989), ch. 6
- cite: W. H. Southwell, Appl. Opt. 28, 5091 (1989)
  doi: 10.1364/AO.28.005091
- cite: P. W. Baumeister, J. Opt. Soc. Am. 48, 955 (1958)
  doi: 10.1364/JOSA.48.000955
---

An all-dielectric bandpass filter is a [Fabry–Pérot cavity](learn/films/fabry-perot-etalons.html) built entirely
from quarter-wave layers, so that nothing absorbs and the mirrors can be as good as the period count allows. Two
[Bragg mirrors](learn/films/bragg-mirrors.html) of high- and low-index layers surround a spacer that is a whole
number of half waves thick at the design wavelength $\lambda_0$:

$$
(\mathrm{HL})^p \,(2\mathrm{L})^m\, (\mathrm{LH})^p ,
$$

where $p$ is the number of pairs per mirror and $m$ the cavity order. The notation is Macleod's: each letter a
quarter wave at $\lambda_0$, read from the incident medium to the substrate.

## Why it transmits

At $\lambda_0$ the spacer is a half-wave layer, which is *absent* in the transfer-matrix sense
($M = -I$). What is left is $(\mathrm{HL})^p(\mathrm{LH})^p$: the two mirrors face each other, and because the
sequence is symmetric the admittance that the second mirror builds up from the substrate is exactly undone by the
first, layer by layer, back to the substrate's own admittance $n_s$. The whole filter at $\lambda_0$ is therefore
optically the bare substrate, with $R = \left(\frac{n_0 - n_s}{n_0 + n_s}\right)^2$, 4 % for glass, and $T \approx 96 \%$
(a single antireflecting layer on the back or an absentee-corrected design removes the rest). A little away from
$\lambda_0$ the spacer is no longer absent, the mirrors are not undone, and since each reflects nearly everything the
transmittance collapses: a line, inside a stop band.

## What sets the width

The single-cavity line has the Airy shape of the etalon, which near its peak is a Lorentzian whose half-width Macleod
derives in closed form. For a spacer of low index with $p$ pairs on each side and a substrate $n_s$,

$$
\frac{\Delta\lambda_{1/2}}{\lambda_0} \approx \frac{4\,n_L^{\,2p-1}\, n_s}{m\,\pi\, n_H^{\,2p}} ,
$$

and with a high-index spacer the roles of $n_H$ and $n_L$ in the prefactor exchange. Every design knob is visible in
it:

- each extra pair on both mirrors multiplies the width by $(n_L/n_H)^2$, 0.4 for TiO₂/SiO₂: the width falls
  geometrically with $p$, as the mirrors' $1 - R$ does;
- the cavity order $m$ narrows the line in proportion, because the phase in a thicker spacer changes faster with
  wavelength; second- and third-order spacers are the usual compromise between narrowness and the tolerances of
  thicker layers;
- a high-index spacer is narrower than a low-index one with the same mirrors by $(n_H/n_L)^2$, and also shifts less
  with angle (below).

A $p = 4$, $m = 1$ TiO₂/SiO₂ filter has a half-width of about 1 % of $\lambda_0$, 5.5 nm at 550 nm; $p = 6$ gives
0.9 nm. The rejection outside the passband is the mirrors' own [stop band](learn/films/bragg-mirrors.html): its width
comes from the index contrast, so TiO₂/SiO₂ blocks about 490–630 nm around a 550 nm design, and a practical filter
adds absorbing glass or further stacks as broadband blockers beyond that, where the mirrors are transparent again.

::: example bandpass-filter

Moving the spacer away from a half wave shifts the passband through the stop band; it disappears into the edge once
the cavity's resonance leaves the band. Absorption in the layers, ignored in the ideal design, is what limits the peak
transmittance of very narrow filters: by the potential-transmittance argument of the
[etalon section](learn/films/fabry-perot-etalons.html), what counts is the absorption per pass relative to $1 - R$,
and a filter of 0.5 nm width with layers that absorb $10^{-4}$ per pass already loses a tenth of its peak.

## Multiple cavities: a square top

A single cavity has a Lorentzian line: sharp at the top but with slowly falling wings, so that a 1 nm filter still
passes a thousandth of the light 10 nm away. Putting two or three cavities in series, coupled through a shared
quarter-wave layer, $(\mathrm{HL})^p 2\mathrm{L} (\mathrm{LH})^p\, \mathrm{L}\, (\mathrm{HL})^p 2\mathrm{L}(\mathrm{LH})^p$,
splits the resonance into as many coupled modes, which together make a flat top and steep sides; the shape is the
optical analogue of a Butterworth or Chebyshev electrical filter, and the design rules carry over (Baumeister 1958).
Nearly every commercial narrowband filter is a two- or three-cavity design, and the wavelength-division multiplexers
of optical fibre links use up to a hundred cavities' worth of layers to pass one 0.4 nm channel and reject its
neighbours by $10^{-4}$.

## Angle: tuning and the converging-beam limit

Tilting the filter moves its passband to shorter wavelengths, as every interference feature moves, by

$$
\frac{\Delta\lambda}{\lambda_0} \approx -\frac{\sin^2\theta_0}{2\,n_\mathrm{eff}^2} ,
$$

where $n_\mathrm{eff}$ is an effective index close to the spacer's (slightly modified by the mirrors). A few degrees
of tilt is the standard way to fine-tune a filter after it has been made; it also means that a filter in a cone of
half-angle $\alpha$ passes a band broadened by the same formula with $\theta_0 \to \alpha$, so a 1 nm filter in an
f/2 beam no longer has a 1 nm band. High-index spacers, with their larger $n_\mathrm{eff}$, are preferred where the beam
cannot be collimated, and a filter's data sheet always states the cone angle its curve was measured in.

## The notch filter

Swap the roles and a stack rejects one band while passing the rest. A quarter-wave mirror with very weak index
contrast (two materials differing by a few hundredths, or a mixture graded between them) has a stop band whose width,
by the Bragg formula $\Delta g = \frac{4}{\pi}\arcsin\frac{n_H - n_L}{n_H + n_L}$, is as narrow as one likes, and needs
correspondingly many periods to reach high reflectance. Southwell's **rugate** filters (1989) replace the discrete
layers by a sinusoidal index profile, whose single Fourier component produces one stop band with no higher-order
bands and no ripple; the laser-line blocking filters of Raman spectroscopy, which must reject the excitation by
$10^{-6}$ a few nanometres from a band they transmit, are built this way.
