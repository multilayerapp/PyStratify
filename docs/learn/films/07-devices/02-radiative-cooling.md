---
id: radiative-cooling
title: "Radiative cooling: emitting into the sky"
geometry: films
chapter: Devices
topic: thermal-emission
sources: [plane_wave]
summary: >-
  A surface that reflects sunlight and emits in the 8–13 µm atmospheric window can cool below ambient: the power
  balance of a radiative cooler and the sky's own emission, Kirchhoff's law and emissivity from a film calculation,
  the phonon bands of silica and silicon nitride as Lorentz oscillators and the reststrahlen reflection they cause,
  the silver mirror underneath, directional emissivity, and what the ideal selective emitter looks like.
examples: [radiative-cooler]
references:
- cite: A. P. Raman, M. A. Anoma, L. Zhu, E. Rephaeli and S. Fan, Nature 515, 540 (2014)
  doi: 10.1038/nature13883
- cite: E. Rephaeli, A. Raman and S. Fan, Nano Lett. 13, 1457 (2013)
  doi: 10.1021/nl4004283
- cite: C. G. Granqvist and A. Hjortsberg, J. Appl. Phys. 52, 4205 (1981)
  doi: 10.1063/1.329270
- cite: Y. Zhai et al., Science 355, 1062 (2017)
  doi: 10.1126/science.aai7899
- cite: J. Kischkat et al., Appl. Opt. 51, 6789 (2012)
  doi: 10.1364/AO.51.006789
- cite: M. Planck, Ann. Phys. 309, 553 (1901)
  doi: 10.1002/andp.19013090310
---

The atmosphere is nearly transparent between 8 and 13 µm, where the room-temperature black-body spectrum peaks. A
surface that emits strongly in that window radiates heat straight to the cold of space, at an effective
temperature far below any terrestrial one; if it also reflects sunlight, it cools below the ambient air even in
daylight. The physics was known to Granqvist in 1981, who cooled at night with silicon-monoxide films; Raman, Fan
and co-workers (2014) first did it under the sun, with a seven-layer stack of hafnia and silica on silver, and
polymer and glass-bead films (Zhai et al. 2017) have since made it a material rather than a device. All of it is a
question of designing one reflectance spectrum.

## The power balance

A cooler at temperature $T$ facing the sky exchanges radiation in both directions. Its net cooling power per unit
area is

$$
P_\mathrm{cool}(T) = P_\mathrm{rad}(T) - P_\mathrm{atm}(T_\mathrm{amb}) - P_\mathrm{sun} - P_\mathrm{cond+conv} ,
$$

with the radiated power an integral of Planck's spectrum $I_\mathrm{BB}(\lambda, T)$ over wavelength and over the
hemisphere, weighted by the surface's emissivity $\varepsilon(\lambda, \theta)$:

$$
P_\mathrm{rad}(T) = 2\pi\int_0^{\pi/2}\!\!\sin\theta\cos\theta\,\mathrm{d}\theta \int_0^\infty I_\mathrm{BB}(\lambda, T)\,\varepsilon(\lambda, \theta)\,\mathrm{d}\lambda .
$$

The atmosphere radiates back with its own emissivity $\varepsilon_\mathrm{atm}(\lambda, \theta) = 1 - t(\lambda)^{1/\cos\theta}$,
nearly black outside the window (water vapour, CO₂) and nearly transparent inside it at the zenith, less so toward
the horizon; $P_\mathrm{atm}$ is the same integral with the product of the two emissivities. The sun adds its
irradiance times the surface's solar absorptance, up to 1000 W m⁻², and conduction and convection from the air add
roughly $h\,(T_\mathrm{amb} - T)$ with $h$ of a few W m⁻² K⁻¹, which is what a wind shield is for. A black body at 300 K
radiates 460 W m⁻²; the window carries about a third of that, and a surface that emits only in the window and
nothing elsewhere, while reflecting the sun entirely, reaches a net cooling power of order 100 W m⁻² at ambient and
a temperature 10–20 K below ambient at zero load. That is the target the spectrum has to meet: $\varepsilon \approx 1$
for 8–13 µm, $\varepsilon \approx 0$ (and $R \approx 1$) for 0.3–2.5 µm and, for a sub-ambient cooler, outside the window
too, where the atmosphere is warm and would heat the surface.

## Emissivity from a film calculation

By Kirchhoff's law a surface's spectral emissivity equals its absorptance at the same wavelength and direction: a body
in equilibrium with radiation must emit what it absorbs, wavelength by wavelength and direction by direction. On an
opaque mirror nothing is transmitted, so

$$
\varepsilon(\lambda, \theta) = A(\lambda, \theta) = 1 - R(\lambda, \theta),
\qquad
\bar\varepsilon_{8\text{–}13} = \langle A \rangle_{8\text{–}13\,\mu\mathrm{m}} ,
$$

and designing an emitter is designing a reflectance spectrum: low in the window, high everywhere the sun shines. The
transfer matrix gives $A$ at every angle for both polarizations, and the hemispherical integral above is a sum over
the calculator's angle sweep weighted by $\sin\theta\cos\theta$.

## Phonon bands and a mirror

Silica and silicon nitride absorb strongly in the window through their lattice vibrations, and are transparent in
the visible. Each infrared-active vibration is a **Lorentz oscillator**,

$$
\varepsilon(\omega) = \varepsilon_\infty + \frac{\Delta\varepsilon\,\omega_\mathrm{TO}^2}{\omega_\mathrm{TO}^2 - \omega^2 - i\gamma\omega} ,
$$

whose real part is negative between the transverse-optical frequency $\omega_\mathrm{TO}$ and the longitudinal one
$\omega_\mathrm{LO}$: in that **reststrahlen band** the material behaves like a metal and *reflects* instead of
absorbing. Silica's Si–O stretch has $\omega_\mathrm{TO}$ near 1070 cm⁻¹ (9.3 µm) and a reststrahlen band from
about 8 to 9.3 µm, right in the middle of the window, where a thick silica layer reflects 70 % and emits 30 %;
silicon nitride's Si–N band lies near 11.5 µm, and the two complement each other. A thin layer of either on silver
emits near its absorption peak, while the silver, seen through the transparent layer, reflects most of the solar
spectrum; Granqvist's SiO on aluminium was the first of these. Silica alone leaves a gap in the window where it
reflects; stacking it with silicon nitride, or using several thinner layers of alternating materials (Raman's
hafnia/silica stack, which also antireflects the silver in the infrared), fills it, and the design problem is to
place the layers' thicknesses so that the window is covered while the solar reflectance stays above 97 %.

::: example radiative-cooler

The window average depends on thickness: a micron of silica begins to emit across the band, while too thick a layer
pushes the reststrahlen reflection back up. Mid-infrared constants are from Kischkat et al. (2012); silver from
Ciesielski et al. (2017), both via refractiveindex.info, and the Lorentz form above is what Kischkat fitted to
measure them.

## Direction, and the ideal emitter

The atmosphere is most transparent toward the zenith, so a cooler that emitted only into a narrow cone around the
vertical would exchange with the coldest part of the sky and reach the lowest temperature, at the price of less total
power; a near-Lambertian emitter, which most films are, gets more power and a higher floor. The directional
emissivity of a film stack is a free design variable: a thin absorbing film on a metal emits strongly near the
Brewster angle of p light and weakly at grazing incidence, which can be tuned with the layer thicknesses, and a
micro-structured surface (gratings, the photonic crystals of Rephaeli, Raman and Fan's 2013 design) can shape it
further. The ideal selective emitter for sub-ambient cooling is a step function in wavelength, unity in the window
and zero outside it, and black only where the atmosphere is clear; for cooling a hot object, such as a solar cell
or a roof, a broadband emitter is better, since every wavelength then carries heat away. Which one a given stack
approximates is read off its computed $A(\lambda)$, and the calculator's example is a two-layer start on it.
