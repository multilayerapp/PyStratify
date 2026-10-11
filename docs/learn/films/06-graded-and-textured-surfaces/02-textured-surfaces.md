---
id: textured-surfaces
title: "Textured surfaces: pyramids and light trapping"
geometry: films
chapter: Graded and textured surfaces
topic: texture
sources: [plane_wave]
summary: >-
  Upright pyramids on silicon give reflected light a second chance: the facet geometry, the product of two
  reflectances, and the longer path that traps light inside the wafer.
examples: [pyramid-facet]
references:
- cite: P. Campbell and M. A. Green, J. Appl. Phys. 62, 243 (1987)
  doi: 10.1063/1.339189
- cite: E. Yablonovitch and G. D. Cody, IEEE Trans. Electron Devices 29, 300 (1982)
  doi: 10.1109/T-ED.1982.20700
- cite: M. A. Green, Sol. Energy Mater. Sol. Cells 92, 1305 (2008) (silicon optical constants)
  doi: 10.1016/j.solmat.2008.06.009
---

Anisotropic etching of a (100) silicon wafer exposes (111) planes and leaves a field of upright pyramids whose facets
make $\alpha = 54.74°$ with the wafer plane. A texture this large (several micrometres) does not change the Fresnel
coefficients of each facet; it changes where a reflected ray goes next.

## Two chances to enter

A vertical ray meets a facet at the local angle $\theta_1 = \alpha$. The specularly reflected ray leaves rotated by
$2\alpha$ from the vertical; for $\alpha > 45°$ it still points downward and strikes the opposite facet, at

$$
\theta_2 = |180° - 3\alpha| = 15.8° .
$$

Light escapes only if it is reflected both times, so the front-surface reflectance becomes the product

$$
R_\mathrm{eff} = R(\theta_1)\, R(\theta_2) ,
$$

and a few percent on each facet becomes well under one percent overall. Each factor is the reflectance of the coated
silicon surface at that angle, which is a film calculation.

::: example pyramid-facet

## Trapping inside

Once in the silicon, refracted rays travel obliquely, by $\alpha - \theta_t$ from the vertical. Weakly absorbed
near-infrared light reaches the back at an angle, reflects, and meets the front facets beyond their critical angle,
so it bounces many times. For a randomizing (Lambertian) surface the path enhancement reaches $4n^2 \approx 50$
(Yablonovitch and Cody, 1982); pyramids come close to it (Campbell and Green, 1987). Following those rays through the
whole wafer needs ray tracing rather than a transfer matrix; the calculator's surface textures do it with
RayFlare, a separate engine.
