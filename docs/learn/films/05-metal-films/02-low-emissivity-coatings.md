---
id: low-emissivity-coatings
title: "Low-emissivity glazing: dielectric / silver / dielectric"
geometry: films
chapter: Metal films
topic: low-e
sources: [plane_wave]
summary: >-
  An ultrathin silver film between two dielectric layers lets daylight through and reflects heat: how the dielectrics
  antireflect the silver in the visible, and why the silver still reflects the infrared.
examples: [low-e-coating]
references:
- cite: R. J. Martín-Palma et al., Sol. Energy Mater. Sol. Cells (1998), silver-based low-emissivity coatings for architectural windows
  doi: 10.1016/S0927-0248(98)00007-5
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 9
- cite: Ciesielski et al. (2017), silver n,k to 20.9 µm, refractiveindex.info main/Ag/nk/Ciesielski.yml
  url: https://refractiveindex.info/?shelf=main&book=Ag&page=Ciesielski
---

A window should let daylight in and keep heat where it is. Silver does both, if it is thin enough and dressed in the
right dielectrics. Its free electrons make it highly reflecting wherever the light's frequency is well below the
plasma frequency, which for silver is everywhere past the ultraviolet; but a film of 10–15 nm is thin enough that
much of the visible gets through.

## Antireflecting a metal

Bare, a 12 nm silver film still reflects a large part of the visible. Two dielectric layers of index around 2 (ZnO, TiO₂,
SnO₂) on either side act as an induced-transmission filter: they match the silver's admittance to the glass and the
air, so that in the visible the reflections from the stack cancel and transmittance rises toward 80 %. In the infrared
the silver's admittance is far too large for any dielectric to match, and the stack reflects.

## Emissivity is absorptance

For an opaque coated pane, $T \approx 0$ in the thermal infrared and Kirchhoff's law gives the emissivity directly:

$$
\varepsilon = A = 1 - R_\mathrm{IR} .
$$

A low-E pane radiates a fraction of what bare glass (ε ≈ 0.84) does, which is what keeps a room's heat inside in
winter, while the high near-infrared reflectance rejects solar heat in summer.

::: example low-e-coating

Thicker silver raises the infrared reflectance and lowers the visible transmittance; commercial double- and
triple-silver stacks repeat the dielectric/silver unit to steepen the edge between the two. The silver data here
extend to 20 µm (Ciesielski et al., 2017); the classic study of this design is
Martín-Palma et al. (1998).
