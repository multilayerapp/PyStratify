---
id: surface-plasmon-resonance
title: "Surface plasmon resonance: the Kretschmann geometry"
geometry: films
chapter: Metal films
topic: spr
sources: [plane_wave]
summary: >-
  A thin gold film on a prism couples p-polarized light to the surface plasmon at the gold/analyte boundary: the
  matching condition, why only p light couples, and why the dip angle measures the analyte's index.
examples: [kretschmann-spr]
references:
- cite: E. Kretschmann and H. Raether, Z. Naturforsch. A 23, 2135 (1968)
  doi: 10.1515/zna-1968-1247
- cite: H. Raether, Surface Plasmons on Smooth and Rough Surfaces and on Gratings (Springer, 1988)
  doi: 10.1007/BFb0048317
- cite: J. Homola, S. S. Yee and G. Gauglitz, Sens. Actuators B 54, 3 (1999)
  doi: 10.1016/S0925-4005(98)00321-9
- cite: P. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370 (1972)
  doi: 10.1103/PhysRevB.6.4370
---

A surface plasmon is a charge-density wave bound to the boundary between a metal (permittivity $\varepsilon_m$, with
$\mathrm{Re}\,\varepsilon_m < 0$) and a dielectric $\varepsilon_d$. Its field decays away from the boundary on both
sides, and its wave vector along the boundary is

$$
k_\mathrm{sp} = \frac{2\pi}{\lambda}\,\sqrt{\frac{\varepsilon_m \varepsilon_d}{\varepsilon_m + \varepsilon_d}} ,
$$

larger than the wave vector of light in the dielectric. Light arriving from the dielectric side can never match it,
which is why a flat metal surface in air does not glow with plasmons.

## Borrowing momentum from a prism

Kretschmann and Raether (1968) put the metal on a prism of index $n_p > \sqrt{\varepsilon_d}$. Past the critical
angle of the prism/analyte pair the light's in-plane wave vector $\frac{2\pi}{\lambda} n_p \sin\theta$ exceeds what
the analyte supports, its field tunnels through a metal film a few tens of nanometres thick, and at the angle where

$$
n_p \sin\theta_\mathrm{res} = \mathrm{Re}\sqrt{\frac{\varepsilon_m \varepsilon_d}{\varepsilon_m + \varepsilon_d}}
$$

it couples into the plasmon on the far side. The energy is dissipated in the metal and the reflectance drops.

## Only p light, and only one thickness

The plasmon's field has a component normal to the surface; s-polarized light has none, so it is reflected almost
totally at every angle beyond the critical one, less only the gold's weak absorption. For p light the depth of the
dip depends on the film: too thick and the evanescent field does not reach the plasmon; too thin and the plasmon leaks
back into the prism. Near 50 nm of gold at 633 nm the two losses balance (critical coupling) and the dip nearly
reaches zero. The finite film also shifts the dip by a fraction of a degree from the semi-infinite condition above.

::: example kretschmann-spr

## A sensor

The resonance angle depends on $\varepsilon_d$, the analyte's index right at the gold surface. Molecules binding to
the gold change that index by a tiny amount and shift the dip; following the dip in real time is how SPR biosensors
measure binding kinetics without labels (Homola, Yee and Gauglitz, 1999).
