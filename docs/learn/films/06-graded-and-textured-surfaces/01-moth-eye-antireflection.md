---
id: moth-eye-antireflection
title: "Graded index: moth-eye antireflection"
geometry: films
chapter: Graded and textured surfaces
topic: graded-index
sources: [plane_wave]
description: "Graded-index antireflection: effective-medium rules, why taper height matters more than shape, Southwell's quintic, and the moth's eye."
reviewed:
summary: >-
  A sub-wavelength taper removes the index step instead of cancelling its reflection: the effective-medium picture
  and where it holds, Maxwell Garnett and Bruggeman mixing rules, the WKB view of a graded profile as the Fourier
  transform of its index gradient, why height matters more than shape and why smoothness matters at all, Southwell's
  quintic, and what the moth's eye adds that a graded film cannot.
examples: [moth-eye-taper]
references:
- cite: P. B. Clapham and M. C. Hutley, Nature 244, 281 (1973)
  doi: 10.1038/244281a0
- cite: W. H. Southwell, Opt. Lett. 8, 584 (1983)
  doi: 10.1364/OL.8.000584
- cite: D. A. G. Bruggeman, Ann. Phys. 416, 636 (1935)
  doi: 10.1002/andp.19354160705
- cite: J. C. Maxwell Garnett, Phil. Trans. R. Soc. A 203, 385 (1904)
  doi: 10.1098/rsta.1904.0024
- cite: Lord Rayleigh, Proc. London Math. Soc. 11, 51 (1880)
  doi: 10.1112/plms/s1-11.1.51
- cite: D. G. Stavenga, S. Foletti, G. Palasantzas and K. Arikawa, Proc. R. Soc. B 273, 661 (2006)
  doi: 10.1098/rspb.2005.3369
---

A [quarter-wave coating](learn/films/transfer-matrix.html) cancels one reflection at one wavelength. A gradual change
of index does something different: with no step left, there is nothing to reflect from, over a broad band and a wide
range of angles. Rayleigh showed in 1880 that a sufficiently slow transition between two media reflects nothing;
moth eyes do it with arrays of cones a few hundred nanometres tall (Clapham and Hutley 1973), and the same idea, as
graded porous films or as etched "black silicon", is the broadband antireflection of choice wherever a hard coating
is not required.

## The effective medium, and where it holds

While the period $\Lambda$ of a structure is small enough that no diffraction order but the zeroth propagates in
either medium,

$$
\Lambda < \frac{\lambda}{n_s + n_0 \sin\theta_0} ,
$$

light cannot resolve the individual features and sees their average: an **effective medium**. The condition is the
[grating cutoff](learn/films/gratings.html); for cones on glass under visible light it means periods below about
250 nm, which the moth's 200 nm array satisfies. Well below the cutoff the average is a quasi-static one, a
homogenisation of the local permittivities, and there are two classical rules for it. **Maxwell Garnett** (1904)
treats one material as inclusions in a host of the other, exact for dilute spherical inclusions:

$$
\frac{\varepsilon_e - \varepsilon_h}{\varepsilon_e + 2\varepsilon_h} = f\,\frac{\varepsilon_i - \varepsilon_h}{\varepsilon_i + 2\varepsilon_h} .
$$

**Bruggeman** (1935) treats both constituents symmetrically, each embedded in the effective medium itself,

$$
f\,\frac{\varepsilon_1 - \varepsilon_e}{\varepsilon_1 + 2\varepsilon_e} + (1-f)\,\frac{\varepsilon_2 - \varepsilon_e}{\varepsilon_2 + 2\varepsilon_e} = 0 ,
$$

which is the better model for an interpenetrating structure such as a tapered array near its base, and the one the
example below and the calculator's roughness finish use. For a two-dimensional array of columns the exact answer is anisotropic and
differs from both; the difference is a few hundredths in index for dielectrics, and matters more for metals, whose
effective permittivity near the percolation threshold is a subject of its own. Slice the taper into thin layers,
give each the $\varepsilon_e$ of its local fill fraction $f(z)$, and the taper becomes an ordinary graded multilayer,
which the [transfer matrix](learn/films/transfer-matrix.html) solves exactly. A few tens of slices are enough for any
profile at visible wavelengths; the example below uses sixteen, each at the Bruggeman index of its local glass fraction.

## Why a graded profile reflects so little: the WKB view

For a slowly varying index $n(z)$ the reflection coefficient can be written, to first order in the gradient, as

$$
r \approx -\int \frac{1}{2n(z)}\,\frac{\mathrm{d}n}{\mathrm{d}z}\, e^{2i\phi(z)}\,\mathrm{d}z ,
\qquad \phi(z) = k_0\int_0^{z} n(z')\,\mathrm{d}z' :
$$

the reflected wave is the **Fourier transform of the index gradient**, taken at twice the local wavenumber. Three
consequences follow directly. An abrupt step has a gradient that is a delta function, whose transform is flat: it
reflects at every wavelength. A ramp of height $h$ has a transform that falls as $1/(k_0 h)$ with side lobes, so
reflection drops once $h$ exceeds a fraction of a wavelength and then oscillates. And the faster the transform
decays, the smoother the gradient must be: a profile whose first $m$ derivatives are continuous at both ends has a
reflectance falling as $(\lambda/h)^{2(m+1)}$.

## Height first, then shape

A graded layer stops reflecting once it is thicker than about a third of the wavelength: below that, light still sees
an abrupt step and reflects much as from the bare surface. Above it, the residual reflection depends on how smoothly
the index starts and ends. A linear ramp leaves a kink at both ends, where the gradient jumps from zero to its
value, and its reflectance falls only as $(\lambda/h)^2$; Southwell's **quintic** profile (1983),

$$
n(t) = n_0 + (n_s - n_0)\,(10t^3 - 15t^4 + 6t^5), \qquad t = z/h ,
$$

has zero first and second derivatives at both ends, and its reflectance falls as $(\lambda/h)^6$, suppressing
reflection by orders of magnitude for the same height. The fill-fraction profile of the taper, not its geometric
outline, is what has to be quintic: a cone of circular cross section has $f \propto t^2$, which is too abrupt at
the tip and too steep at the base, and the best moth-eye shapes are paraboloids or the slightly flared cones nature
grows.

::: example moth-eye-taper

The slices' indices are fixed; only their thickness scales with the height, which keeps the same profile while the
taper grows. Real moth-eye structures add scattering where the period approaches the cutoff, which an effective
medium cannot describe; the [gratings](learn/films/gratings.html) section shows where that limit falls.

## Angle and bandwidth

Because nothing is tuned to a wavelength, a graded layer is antireflecting from the ultraviolet to wherever its
height becomes a small fraction of the wavelength, and it stays so to large angles: the WKB integral only changes
its phase argument, and the reflectance of a 400 nm quintic taper on glass stays below 0.1 % to 60° for both
polarizations, where a four-layer coating is already at 2 %. The practical limits are mechanical (a porous or
structured surface is soft and collects dirt), and, for solar cells and displays, the slight haze from residual
scattering, which is the reason the effective-medium picture must be checked by a rigorous grating calculation
before a period is chosen.

## What the moth's eye really is

The corneal nipples of moths and butterflies (Stavenga et al. 2006) are paraboloids of 200–250 nm period and
similar height, which model and measurement agree reduce the reflectance of the cuticle from about 4 % to below
0.5 % across the visible, and more at grazing incidence, where a night-flying insect's eye would otherwise glint.
The same structure is now printed into polymer films for displays, and etched into silicon and glass, where the
"black silicon" of nanometre-scale needles reaches a reflectance below 0.1 % from 300 nm to 2 µm. It works as a
graded film, except at the tips and at the base, where the array is a periodic grating and the effective medium is
only the first term of the story.
