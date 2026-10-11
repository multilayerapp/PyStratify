---
id: fresnel-equations
title: "The Fresnel equations: one interface"
geometry: films
chapter: Interfaces
topic: fresnel
sources: [plane_wave]
summary: >-
  Reflection and transmission of a plane wave at a single planar interface: Snell's law, the Fresnel coefficients for
  s and p polarization, the Brewster angle and total internal reflection.
examples: [fresnel-air-glass, fresnel-glass-air]
references:
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §1.5
  doi: 10.1017/CBO9781139644181
- cite: J. D. Jackson, Classical Electrodynamics, 3rd ed. (Wiley, 1999), §7.3–7.4
- cite: S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720 (2016)
  url: https://arxiv.org/abs/1603.02720
---

Every thin-film calculation is built from one event repeated: a plane wave meets a flat boundary between two
homogeneous media and splits into a reflected and a transmitted wave. Everything in this chapter follows from the
continuity of the tangential components of $\mathbf{E}$ and $\mathbf{H}$ at that boundary.

## Snell's law

A wave in medium 1 (index $n_1$) arrives at angle $\theta_1$ from the normal. The tangential component of its wave
vector is the same on both sides, which fixes the direction of the transmitted wave:

$$
n_1 \sin\theta_1 = n_2 \sin\theta_2 .
$$

The quantity $n \sin\theta$ is conserved through any stack of parallel layers, which is why a multilayer can be solved
layer by layer at a single in-plane wave number.

## Two polarizations, two sets of coefficients

The field is split into the part polarized perpendicular to the plane of incidence (**s**, or TE) and the part
polarized in it (**p**, or TM). They do not mix at a planar interface. With $\cos\theta_2$ taken from Snell's law,
the amplitude coefficients are

$$
r_s = \frac{n_1\cos\theta_1 - n_2\cos\theta_2}{n_1\cos\theta_1 + n_2\cos\theta_2},
\qquad
r_p = \frac{n_2\cos\theta_1 - n_1\cos\theta_2}{n_2\cos\theta_1 + n_1\cos\theta_2},
$$

$$
t_s = \frac{2 n_1\cos\theta_1}{n_1\cos\theta_1 + n_2\cos\theta_2},
\qquad
t_p = \frac{2 n_1\cos\theta_1}{n_2\cos\theta_1 + n_1\cos\theta_2}.
$$

The sign of $r_p$ depends on the convention for the direction of the p field; this one follows Byrnes (2016), whose
transfer-matrix code PyStratify's planar engine extends. The power coefficients are

$$
R = |r|^2, \qquad T = \frac{\mathrm{Re}\,(n_2\cos\theta_2)}{\mathrm{Re}\,(n_1\cos\theta_1)}\,|t|^2 \;\;(s),
\qquad T = \frac{\mathrm{Re}\,(n_2\cos\theta_2^{*})}{\mathrm{Re}\,(n_1\cos\theta_1^{*})}\,|t|^2 \;\;(p),
$$

and $R + T = 1$ when neither medium absorbs. At normal incidence both polarizations reduce to
$R = \left(\frac{n_2 - n_1}{n_2 + n_1}\right)^2$: 4 % for glass in air.

::: example fresnel-air-glass

## The Brewster angle

$r_p$ vanishes when the reflected and transmitted directions are perpendicular. The dipoles driven in medium 2 would
then have to radiate along their own axis, which they cannot. This happens at

$$
\tan\theta_B = \frac{n_2}{n_1},
$$

56.3° for glass in air. Light reflected at the Brewster angle is purely s-polarized, which is how polarizing filters
cut glare and how Brewster windows keep a laser cavity lossless for one polarization.

## Total internal reflection

From the dense side ($n_1 > n_2$), Snell's law has no real solution beyond the critical angle

$$
\sin\theta_c = \frac{n_2}{n_1}.
$$

There $\cos\theta_2$ becomes imaginary: the transmitted field decays exponentially away from the interface instead of
propagating, $|r_s| = |r_p| = 1$, and all the power is reflected. The evanescent field still reaches a fraction of a
wavelength into the second medium. A second dense medium brought that close lets light tunnel through (frustrated
total reflection), and a metal film there couples it to surface plasmons, the principle of the Kretschmann
configuration.

::: example fresnel-glass-air

## Absorbing media

For an absorbing medium the index is complex, $\tilde n = n + ik$ (time dependence $e^{-i\omega t}$), and so is
$\cos\theta_2$. The square root then has two branches; the physical one is the wave that carries power away from the
interface and decays into the medium. PyStratify chooses that branch for every layer, which is what keeps the
formulas above valid for metals, semiconductors above their band gap, and evanescent waves alike.
