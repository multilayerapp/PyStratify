---
id: moth-eye-antireflection
title: "Graded index: moth-eye antireflection"
geometry: films
chapter: Graded and textured surfaces
topic: graded-index
sources: [plane_wave]
summary: >-
  A sub-wavelength taper removes the index step instead of cancelling its reflection: the effective-medium picture, the
  Bruggeman mixing rule, and why height matters more than shape.
examples: [moth-eye-taper]
references:
- cite: P. B. Clapham and M. C. Hutley, Nature 244, 281 (1973)
  doi: 10.1038/244281a0
- cite: W. H. Southwell, Opt. Lett. 8, 584 (1983)
  doi: 10.1364/OL.8.000584
- cite: D. A. G. Bruggeman, Ann. Phys. 416, 636 (1935)
  doi: 10.1002/andp.19354160705
---

A [quarter-wave coating](learn/films/transfer-matrix.html) cancels one reflection at one wavelength. A gradual change
of index does something different: with no step left, there is nothing to reflect from, over a broad band and a wide
range of angles. Moth eyes do it with arrays of cones a few hundred nanometres tall (Clapham and Hutley, 1973).

## The effective medium

While the period $\Lambda$ is small enough that no diffraction order but the zeroth propagates,

$$
\Lambda < \frac{\lambda}{n_s + n_0 \sin\theta_0} ,
$$

light cannot resolve the individual features and sees their average. Slice the taper into thin layers and give each
the effective permittivity $\varepsilon_e$ of its local fill fraction $f$ of material $\varepsilon_1$ in air
$\varepsilon_2$; Bruggeman's rule treats both constituents symmetrically,

$$
f\,\frac{\varepsilon_1 - \varepsilon_e}{\varepsilon_1 + 2\varepsilon_e} + (1-f)\,\frac{\varepsilon_2 - \varepsilon_e}{\varepsilon_2 + 2\varepsilon_e} = 0 .
$$

The taper becomes an ordinary graded multilayer, which the transfer matrix solves exactly.

## Height first, then shape

A graded layer stops reflecting once it is thicker than about a third of the wavelength: below that, light still sees
an abrupt step and reflects much as from the bare surface. Above it, the residual reflection depends on how smoothly
the index starts and ends. A linear ramp leaves a kink at both ends; Southwell's quintic profile (1983), smooth in its
first and second derivatives, suppresses reflection further for the same height.

::: example moth-eye-taper

The slices' indices are fixed; only their thickness scales with the height, which keeps the same profile while the
taper grows. Real moth-eye structures add scattering where the period approaches the cutoff, which an effective
medium cannot describe; the [gratings](learn/films/gratings.html) section shows where that limit falls.
