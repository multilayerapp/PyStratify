---
id: surface-plasmon-resonance
title: "Surface plasmon resonance: the Kretschmann geometry"
geometry: films
chapter: Metal films
topic: spr
sources: [plane_wave]
summary: >-
  A thin gold film on a prism couples p-polarized light to the surface plasmon at the gold/analyte boundary: the
  plasmon as a pole of the Fresnel coefficient, its dispersion relation and decay lengths, prism coupling in the
  Kretschmann and Otto geometries, the Lorentzian dip and critical coupling, why only p light couples, the
  sensitivity and figure of merit of an SPR sensor, and the limits of the model.
examples: [kretschmann-spr]
references:
- cite: E. Kretschmann and H. Raether, Z. Naturforsch. A 23, 2135 (1968)
  doi: 10.1515/zna-1968-1247
- cite: A. Otto, Z. Phys. 216, 398 (1968)
  doi: 10.1007/BF01391532
- cite: H. Raether, Surface Plasmons on Smooth and Rough Surfaces and on Gratings (Springer, 1988), ch. 2
  doi: 10.1007/BFb0048317
- cite: J. Homola, S. S. Yee and G. Gauglitz, Sens. Actuators B 54, 3 (1999)
  doi: 10.1016/S0925-4005(98)00321-9
- cite: "S. A. Maier, Plasmonics: Fundamentals and Applications (Springer, 2007), ch. 2–3"
  doi: 10.1007/0-387-37825-1
- cite: P. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370 (1972)
  doi: 10.1103/PhysRevB.6.4370
---

A surface plasmon is a charge-density wave bound to the boundary between a metal (permittivity $\varepsilon_m$, with
$\mathrm{Re}\,\varepsilon_m < 0$) and a dielectric $\varepsilon_d$. It is also, and this is the useful way to think of
it in a thin-film calculation, a **pole of the Fresnel coefficient** $r_p$ of that boundary: a wave that exists with
no incident field at all.

## The plasmon as a pole

Write $r_p$ of the metal/dielectric interface in terms of the normal wavenumbers $k_{z,d} = \sqrt{\varepsilon_d k_0^2 - k_x^2}$
and $k_{z,m} = \sqrt{\varepsilon_m k_0^2 - k_x^2}$:

$$
r_p = \frac{\varepsilon_m k_{z,d} - \varepsilon_d k_{z,m}}{\varepsilon_m k_{z,d} + \varepsilon_d k_{z,m}} .
$$

The denominator vanishes where $\varepsilon_m/k_{z,m} = -\varepsilon_d/k_{z,d}$, which with both $k_z$ imaginary (a
field decaying on both sides) requires $\varepsilon_m$ and $\varepsilon_d$ of opposite sign, and gives the
**dispersion relation**

$$
k_\mathrm{sp} = \frac{2\pi}{\lambda}\,\sqrt{\frac{\varepsilon_m \varepsilon_d}{\varepsilon_m + \varepsilon_d}} ,
$$

a wavenumber along the boundary larger than that of light in the dielectric, $k_0\sqrt{\varepsilon_d}$, for every
$\varepsilon_m < -\varepsilon_d$. The field decays away from the boundary with $1/|k_{z,d}|$ in the dielectric and
$1/|k_{z,m}|$ in the metal: for gold at 633 nm ($\varepsilon_m \approx -11.7 + 1.2i$) against water, about 250 nm into
the water and 30 nm into the gold, so the wave lives mostly on the dielectric side and is sensitive to what is there.
The imaginary part of $\varepsilon_m$ makes $k_\mathrm{sp}$ complex; the plasmon propagates a distance
$1/(2\,\mathrm{Im}\,k_\mathrm{sp})$, some 10 µm for gold in the red, before it has dissipated. The s-polarized
coefficient $r_s$ has no such pole for non-magnetic media, which is why there is no s plasmon.

Because $k_\mathrm{sp} > k_0\sqrt{\varepsilon_d}$, light arriving from the dielectric side can never match it: a flat
metal surface in air does not glow with plasmons, and a plasmon on a flat surface does not radiate. Something has to
supply the missing momentum.

## Borrowing momentum from a prism

Kretschmann and Raether (1968) put the metal film on a prism of index $n_p > \sqrt{\varepsilon_d}$. Past the critical
angle of the prism/analyte pair the light's in-plane wavenumber $k_x = \frac{2\pi}{\lambda} n_p \sin\theta$ exceeds
what the analyte supports, its field is evanescent beyond the metal and tunnels through a film a few tens of
nanometres thick, and at the angle where

$$
n_p \sin\theta_\mathrm{res} = \mathrm{Re}\sqrt{\frac{\varepsilon_m \varepsilon_d}{\varepsilon_m + \varepsilon_d}}
$$

it couples into the plasmon on the far side. In the prism / metal / analyte three-layer formula the plasmon pole of
the metal/analyte interface sits just below the real $k_x$ axis, and the reflectance seen from the prism dips as
$k_x$ passes it. Otto's geometry (1968) keeps the prism *away* from the metal, with an air or liquid gap of a few
hundred nanometres through which the evanescent wave reaches a thick metal; it is the better choice for a bulk
metal and for infrared work, Kretschmann's for a sensor, because the analyte is on the free side of the film.

## A Lorentzian dip, and critical coupling

Near the resonance the reflectance of the three-layer system has the form (Raether)

$$
R(k_x) \approx 1 - \frac{4\,\Gamma_i\,\Gamma_\mathrm{rad}}{(k_x - k_\mathrm{sp}')^2 + (\Gamma_i + \Gamma_\mathrm{rad})^2} ,
$$

where $\Gamma_i = \mathrm{Im}\,k_\mathrm{sp}$ is the plasmon's internal (ohmic) damping and $\Gamma_\mathrm{rad}$ its
radiative damping back into the prism through the film, which falls exponentially with the film's thickness. The
dip is a Lorentzian in $k_x$ (and nearly so in angle) of width $\Gamma_i + \Gamma_\mathrm{rad}$, and its depth is
governed by the ratio of the two dampings: it reaches zero exactly when $\Gamma_\mathrm{rad} = \Gamma_i$, **critical
coupling**, where the energy leaking back out interferes destructively with the directly reflected wave. Too thick a
film and the evanescent field does not reach the plasmon ($\Gamma_\mathrm{rad} \to 0$, a shallow dip); too thin and
the plasmon leaks back into the prism faster than the metal absorbs it (a broad, shallow dip). Near 50 nm of gold at
633 nm the two balance and the dip nearly reaches zero. The finite film also shifts the dip by a fraction of a degree
from the semi-infinite condition above, by the real part of the radiative correction.

## Only p light, and what s light is good for

The plasmon's field has a component normal to the surface, which is what the surface charge needs; s-polarized
light has none, so it is reflected almost totally at every angle beyond the critical one, less only the gold's weak
absorption. This makes the s reflectance a reference: the ratio $R_p/R_s$ removes the source's intensity drift and
is what most instruments record.

::: example kretschmann-spr

## A sensor

The resonance angle depends on $\varepsilon_d$, the analyte's index right at the gold surface, within the plasmon's
decay length. Molecules binding to the gold change that index by a tiny amount and shift the dip; following the dip
in real time is how SPR biosensors measure binding kinetics without labels (Homola, Yee and Gauglitz 1999). The
sensitivity follows from the dispersion relation: differentiating the resonance condition,

$$
\frac{\mathrm{d}\theta_\mathrm{res}}{\mathrm{d}n_d} = \frac{\varepsilon_m^{3/2}}{n_p\cos\theta_\mathrm{res}\,(\varepsilon_m + n_d^2)^{3/2}}
\;\;\text{(real parts)},
$$

about 100° per refractive-index unit for gold on a BK7 prism in water, so a change of $10^{-6}$ in index, the
resolution of a good instrument, is a millidegree. Since the dip's width is set by $\Gamma_i + \Gamma_\mathrm{rad}$, the
**figure of merit** is sensitivity over width, which favours silver (sharper, but it corrodes) over gold, longer
wavelengths (where $|\varepsilon_m|$ is larger and the plasmon lives longer), and a film at critical coupling. A
thin adhesion layer of chromium or titanium under the gold, a few nanometres, broadens the dip noticeably and is
one of the first things a film calculation shows.

## Limits of the model

The calculation assumes a flat, continuous, local metal with bulk optical constants. A rough film scatters the
plasmon into light (Raether's book is largely about this), a granular film below 20 nm has island plasmons of its
own, and the constants of an evaporated gold film differ from the bulk by a few percent, which moves the resonance
by a fraction of a degree, more than a sensor's resolution. In a measurement all of these are absorbed into a fitted
$\varepsilon_m$ and thickness, which the transfer matrix then reproduces exactly; the model's remaining limit is the
analyte layer itself, a few nanometres of protein whose index and thickness are not separately measurable from a
single angular scan, which is why two wavelengths, or an angular and a spectral scan, are used when both are wanted.
