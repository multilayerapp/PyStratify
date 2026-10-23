---
id: fabry-perot-etalons
title: "Fabry–Pérot etalons: a spacer between two mirrors"
geometry: films
chapter: Filters and cavities
topic: fabry-perot
sources: [plane_wave]
description: "The Airy function of a spacer between two mirrors: resonances, free spectral range, finesse, mirror absorption, angle tuning and cavity Q."
reviewed:
summary: >-
  A transparent spacer between two partial mirrors transmits a comb of narrow peaks: the Airy function derived from
  the multiple reflections, the resonance condition with the mirrors' phases, free spectral range, finesse and
  resolving power, what absorption in the mirrors costs, angle tuning and the Jacquinot advantage, the photon
  lifetime and Q of the cavity, and the Gires–Tournois phase.
examples: [fabry-perot-etalon]
references:
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §7.6
  doi: 10.1017/CBO9781139644181
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 7
- cite: G. Hernandez, Fabry–Perot Interferometers (Cambridge University Press, 1986)
- cite: J. M. Vaughan, The Fabry–Perot Interferometer (Adam Hilger, 1989)
- cite: F. Gires and P. Tournois, C. R. Acad. Sci. Paris 258, 6112 (1964)
---

Put a transparent layer between two partially reflecting mirrors and light bounces back and forth inside it. Where
the round trip adds a whole number of wavelengths, every pass adds in phase and the cavity transmits strongly, even
though each mirror alone reflects most of the light. Elsewhere the passes cancel and the cavity reflects. Fabry and
Pérot built it in 1897 as a spectroscope; it is the resonator of every laser, the spacer of every narrowband
[filter](learn/films/bandpass-filters.html), and the simplest case in which the multiple-reflection sum can be done
by hand.

## The Airy function

Let the two mirrors have amplitude coefficients $r_1, t_1$ and $r_2, t_2$ seen from inside the spacer, and let the
round-trip phase be $\delta = \frac{4\pi}{\lambda} n d\cos\theta$. The transmitted amplitude is the sum over the
number of round trips, a geometric series,

$$
t = t_1 t_2\, e^{i\delta/2}\sum_{m=0}^{\infty}\left(r_1 r_2\, e^{i\delta}\right)^m
= \frac{t_1 t_2\, e^{i\delta/2}}{1 - r_1 r_2\, e^{i\delta}} ,
$$

which is exactly what the [transfer matrix](learn/films/transfer-matrix.html) returns for one layer, now with its
multiple-beam origin visible. With equal lossless mirrors of reflectance $R = |r|^2$ the transmittance is the
**Airy function**

$$
T(\delta) = \frac{1}{1 + F\sin^2(\delta/2 + \phi)}, \qquad F = \frac{4R}{(1 - R)^2},
$$

where $\phi$ collects the mirrors' reflection phases. $F$ is the coefficient of finesse; for $R = 0.9$ it is 360, and
the peaks, of unit height, are separated by valleys of depth $1/(1 + F)$. The Verification page's first film row
compares the engine with this formula for a plain slab.

## The resonance condition

For a spacer of index $n$ and thickness $d$ transmission peaks where the total round-trip phase is a multiple of $2\pi$,

$$
2 n d \cos\theta = \left(m - \frac{\phi_1 + \phi_2}{2\pi}\right)\lambda_m , \qquad m = 1, 2, \dots
$$

For dielectric mirrors at their design wavelength the reflection phases are 0 or $\pi$, and the condition reduces
to the spacer being a whole number of half waves. For metal mirrors they are neither: silver's reflection phase in
the visible is around $-150°$ from the glass side, so every order sits a fraction of a free spectral range away from
the half-wave position and the spacer that resonates at a given wavelength is thinner than $m\lambda/2n$. Dielectric
mirrors have the complementary subtlety: their phase varies with wavelength as if the cavity were longer by the
[penetration depth](learn/films/bragg-mirrors.html) of the mirror, which shifts the mode spacing.

## Free spectral range, finesse, resolving power

Neighbouring orders are separated in wavelength by

$$
\Delta\lambda_\mathrm{FSR} \approx \frac{\lambda^2}{2 n d \cos\theta},
\qquad
\Delta\nu_\mathrm{FSR} = \frac{c}{2 n d\cos\theta} ,
$$

so a thicker cavity packs its peaks closer (a 1 mm air gap has 150 GHz between orders, a 1 cm laser cavity 15 GHz).
The width of each peak relative to that spacing is set by the mirrors alone, through the **finesse**

$$
\mathcal{F} = \frac{\pi\sqrt{R}}{1-R} = \frac{\Delta\lambda_\mathrm{FSR}}{\delta\lambda_{1/2}} ,
$$

30 for $R = 0.9$, 300 for $R = 0.99$. The resolving power of the etalon as a spectroscope is $\lambda/\delta\lambda =
m\mathcal{F}$, the order times the finesse, which is how a centimetre-thick etalon with modest mirrors resolves
hyperfine structure that a grating cannot. Real etalons fall short of $\mathcal{F}$ through the flatness of their
plates and the parallelism of the beam: a plate flat to $\lambda/N$ limits the finesse to about $N/2$, and a beam of
half-angle $\alpha$ smears $\cos\theta$ by $\alpha^2/2$.

## What absorption costs

For an ideal lossless cavity with equal mirrors the peaks reach $T = 1$ whatever $R$ is. Metal mirrors are not
lossless. If each mirror absorbs a fraction $A$ with $R + T_\mathrm{m} + A = 1$, the peak transmittance becomes

$$
T_\mathrm{peak} = \left(\frac{T_\mathrm{m}}{1 - R}\right)^2 = \left(1 - \frac{A}{1 - R}\right)^2 ,
$$

the square of the mirrors' **potential transmittance**: what matters is the absorption relative to what the mirror
lets through, not the absorption itself. A silver mirror with $R = 0.9$ and $A = 0.04$ limits the peak to 36 %; with
$R = 0.95$ and the same $A$, to 4 %. Better metal mirrors sharpen the peaks but lower them, which is the dead end that
led to all-dielectric filters.

::: example fabry-perot-etalon

The plotted peaks stay below 1 because the silver absorbs; the check below the figure confirms that what is not
reflected or transmitted is exactly what the layers absorb. Tilting the beam shortens $d\cos\theta$ and moves every
peak to shorter wavelengths, which is how an etalon is angle-tuned.

## Angle, étendue and the Jacquinot advantage

Since $\delta$ depends on $\cos\theta$, an etalon lit by a cone of light transmits each wavelength in a ring of
angles: the **Haidinger fringes** seen through a plate held to the eye, and the reason an etalon spectrometer records
a ring pattern. For a small tilt the peak shifts by

$$
\frac{\Delta\lambda}{\lambda} \approx -\frac{\theta_0^2}{2 n^2} ,
$$

always to shorter wavelengths, and it is this dependence on $\theta^2$ rather than $\theta$ that gives the etalon its
large throughput for a given resolution, the Jacquinot advantage over a slit instrument. The same formula says how far
a tilted [bandpass filter](learn/films/bandpass-filters.html) moves, and why a filter in a converging beam broadens.

## The cavity as a resonator: lifetime and Q

Seen in time rather than frequency, a photon in the cavity survives on average
$\tau = \frac{n d}{c\,(1 - R)}$ round trips' worth of time (for $R$ close to 1), and the quality factor
$Q = \omega\tau = m\mathcal{F}$ is again the order times the finesse. A 1 mm cavity with $R = 0.99$ stores light for
about a third of a nanosecond. The field inside at resonance is enhanced by $1/(1 - R)$ in intensity over the incident wave, which
is what a dye molecule or a nonlinear crystal placed in the spacer sees, and why cavity enhancement is a tool of
spectroscopy and of frequency conversion.

## The Gires–Tournois etalon

Make the back mirror perfect. Nothing is transmitted, $|r| = 1$ at every wavelength, and all the physics is in the
phase of the reflected wave, which sweeps through $2\pi$ around each resonance the more rapidly the higher the front
mirror's reflectance. Such a Gires–Tournois etalon (1964) is a pure phase element: a delay that depends on wavelength,
used to compensate dispersion in femtosecond lasers and fibre links, and the simplest demonstration that a lossless
structure can have a strongly frequency-dependent group delay with a perfectly flat reflectance.

## From etalons to filters

A Fabry–Pérot cavity is the core of almost every narrowband filter. Replacing the metal mirrors by
[quarter-wave stacks](learn/films/bragg-mirrors.html) removes the absorption and raises the reflectance at will, which
is the [all-dielectric bandpass filter](learn/films/bandpass-filters.html).
