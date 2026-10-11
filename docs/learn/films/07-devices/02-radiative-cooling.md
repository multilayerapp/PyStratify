---
id: radiative-cooling
title: "Radiative cooling: emitting into the sky"
geometry: films
chapter: Devices
topic: thermal-emission
sources: [plane_wave]
summary: >-
  A surface that reflects sunlight and emits in the 8–13 µm atmospheric window can cool below ambient: Kirchhoff's
  law, the phonon bands of silica and silicon nitride, and the silver mirror underneath.
examples: [radiative-cooler]
references:
- cite: A. P. Raman, M. A. Anoma, L. Zhu, E. Rephaeli and S. Fan, Nature 515, 540 (2014)
  doi: 10.1038/nature13883
- cite: J. Kischkat et al., Appl. Opt. 51, 6789 (2012)
  doi: 10.1364/AO.51.006789
- cite: C. G. Granqvist and A. Hjortsberg, J. Appl. Phys. 52, 4205 (1981)
  doi: 10.1063/1.329270
---

The atmosphere is nearly transparent between 8 and 13 µm, where the room-temperature black-body spectrum peaks. A
surface that emits strongly in that window radiates heat straight to the cold of space; if it also reflects sunlight,
it cools below the ambient air even in daylight (Raman et al., 2014).

## Emissivity from a film calculation

By Kirchhoff's law a surface's spectral emissivity equals its absorptance at the same wavelength and direction. On an
opaque mirror nothing is transmitted, so

$$
\varepsilon(\lambda) = A(\lambda) = 1 - R(\lambda),
\qquad
\bar\varepsilon_{8\text{–}13} = \langle A \rangle_{8\text{–}13\,\mu\mathrm{m}} ,
$$

and designing an emitter is designing a reflectance spectrum: low in the window, high everywhere the sun shines.

## Phonon bands and a mirror

Silica and silicon nitride absorb strongly in the window through their lattice vibrations (the Si–O stretch near
9.3 µm, Si–N near 11.5 µm), and are transparent in the visible. A layer of either on silver emits in the window, while
the silver, seen through the transparent layer, reflects most of the solar spectrum. Silica alone leaves a gap in the
window where it reflects (its reststrahlen band); stacking it with silicon nitride, or using several thinner layers,
fills it. Selective emitters of this kind go back to silicon-monoxide films on aluminium (Granqvist and Hjortsberg,
1981).

::: example radiative-cooler

The window average depends on thickness: a micron of silica begins to emit across the band, while too thick a layer
pushes the reststrahlen reflection back up. Mid-infrared constants are from Kischkat et al. (2012); silver from
Ciesielski et al. (2017), both via refractiveindex.info.
