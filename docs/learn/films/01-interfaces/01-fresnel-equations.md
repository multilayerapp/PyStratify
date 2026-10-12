---
id: fresnel-equations
title: "The Fresnel equations: one interface"
geometry: films
chapter: Interfaces
topic: fresnel
sources: [plane_wave]
summary: >-
  Reflection and transmission of a plane wave at a single planar interface, derived from the two boundary conditions:
  Snell's law, the Fresnel coefficients for s and p polarization and their phases, the power coefficients and why
  T is not |t|², the Brewster angle, total internal reflection and the evanescent wave, absorbing media and the
  branch of the complex angle, and the ellipsometric ratio ρ = r_p / r_s.
examples: [fresnel-air-glass, fresnel-glass-air]
references:
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §1.5 and §14.2
  doi: 10.1017/CBO9781139644181
- cite: J. D. Jackson, Classical Electrodynamics, 3rd ed. (Wiley, 1999), §7.3–7.4
- cite: R. M. A. Azzam and N. M. Bashara, Ellipsometry and Polarized Light (North-Holland, 1977), ch. 4
- cite: F. Goos and H. Hänchen, Ann. Phys. 436, 333 (1947)
  doi: 10.1002/andp.19474360704
- cite: S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720 (2016)
  url: https://arxiv.org/abs/1603.02720
---

Every thin-film calculation is built from one event repeated: a plane wave meets a flat boundary between two
homogeneous media and splits into a reflected and a transmitted wave. Everything in this chapter follows from the
continuity of the tangential components of $\mathbf{E}$ and $\mathbf{H}$ at that boundary (the
[Foundations](learn/common/maxwell-and-boundary-conditions.html)), applied once. The algebra is short, and worth
doing once in full, because the same two equations, with Bessel functions in place of exponentials, give Mie's
coefficients for a sphere and the cylinder's.

## Snell's law

A wave in medium 1 (index $n_1$) arrives at angle $\theta_1$ from the normal. Its field varies along the interface as
$e^{i k_0 n_1 \sin\theta_1\, x}$, and the boundary conditions must hold at every $x$, so the reflected and transmitted
waves carry the same in-plane wavenumber $k_x = k_0 n_1\sin\theta_1$. That alone fixes the directions:

$$
n_1 \sin\theta_1 = n_2 \sin\theta_2 ,
$$

and the reflected wave leaves at $\theta_1$. The quantity $n \sin\theta$ is conserved through any stack of parallel
layers, which is why a multilayer can be solved layer by layer at a single $k_x$, and why an angle of incidence is best
thought of as an in-plane wavenumber: every layer's normal wavenumber follows as $k_{z,j} = k_0\sqrt{n_j^2 - n_1^2\sin^2\theta_1}$,
real or imaginary.

## Two polarizations, two sets of coefficients

The field is split into the part polarized perpendicular to the plane of incidence (**s**, or TE, from the German
*senkrecht*) and the part polarized in it (**p**, or TM). They do not mix at a planar interface, because the
boundary conditions never couple a field component along $y$ to one in the $xz$ plane. For s polarization,
$\mathbf{E} = E_y\hat{\mathbf{y}}$, and the tangential $\mathbf{H}$ is $H_x \propto n\cos\theta\, E_y$ (from
$\nabla\times\mathbf{E} = i\omega\mu_0\mathbf{H}$). Continuity of $E_y$ and of $H_x$ at $z = 0$ gives, with the
incident amplitude 1,

$$
1 + r_s = t_s, \qquad n_1\cos\theta_1\,(1 - r_s) = n_2\cos\theta_2\, t_s ,
$$

two equations for two unknowns. For p polarization the roles swap: $\mathbf{H} = H_y\hat{\mathbf{y}}$ is continuous,
and the tangential $E_x \propto (\cos\theta/n)\,H_y$. Solving both pairs,

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

Both have the same structure, $(\eta_1 - \eta_2)/(\eta_1 + \eta_2)$ in terms of a **tilted admittance**,
$\eta = n\cos\theta$ for s and $\eta = n/\cos\theta$ for p (up to the convention for $r_p$). That one observation is
the whole thin-film theory in a nutshell: the [transfer matrix](learn/films/transfer-matrix.html) carries nothing but
these admittances from layer to layer.

**The sign of $r_p$.** Two conventions are in use for the direction of the p-polarized electric field, and they
differ in the sign of $r_p$; the expression above, in which $r_p \to r_s$ at normal incidence, follows Byrnes (2016),
whose transfer-matrix code PyStratify's planar engine extends. In the other convention (Born and Wolf) $r_p$ changes
sign, so that $r_p = -r_s$ at normal incidence. Nothing physical depends on the choice, but the phase $\Delta$ of
ellipsometry does, by 180°, and comparing two codes or a code with a paper starts with settling it.

**Phases.** For external reflection ($n_1 < n_2$) below the Brewster angle, $r_s$ is negative: the s wave flips its
phase by $\pi$ on reflection from a denser medium, the origin of the "half-wave loss" of elementary optics, while
$r_p$ in this convention is positive and crosses zero at Brewster. Internal reflection reverses both signs.

## Power coefficients

The reflected and incident waves travel in the same medium, so the reflectance is simply $R = |r|^2$. The
transmitted wave travels in a different medium at a different angle, and the ratio of the power flows *normal to
the interface* carries the ratio of the admittances:

$$
R = |r|^2, \qquad
T_s = \frac{\mathrm{Re}\,(n_2\cos\theta_2)}{\mathrm{Re}\,(n_1\cos\theta_1)}\,|t_s|^2,
\qquad
T_p = \frac{\mathrm{Re}\,(n_2\cos\theta_2^{*})}{\mathrm{Re}\,(n_1\cos\theta_1^{*})}\,|t_p|^2 .
$$

$T = |t|^2$ is a common mistake; it fails already at normal incidence between two dielectrics, where
$|t|^2 = 4n_1^2/(n_1+n_2)^2$ is not $1 - R$. The conjugates in $T_p$ matter only for absorbing media, where the
Poynting vector of an inhomogeneous wave is not along the real part of the wavevector. With them, $R + T = 1$ holds
exactly for lossless media, which is the first of the [Verification](verification.html) page's checks. At normal
incidence both polarizations reduce to

$$
R = \left(\frac{n_2 - n_1}{n_2 + n_1}\right)^2 ,
$$

4 % for glass in air, 31 % for silicon ($n \approx 3.9$), 1.3 % for a water surface.

::: example fresnel-air-glass

## The Brewster angle

$r_p$ vanishes where $n_2\cos\theta_1 = n_1\cos\theta_2$, which with Snell's law is $\theta_1 + \theta_2 = 90°$: the
reflected and transmitted directions are perpendicular. The dipoles driven in medium 2 oscillate along the transmitted
$\mathbf{E}$, which is then exactly along the would-be reflected ray, and a dipole does not radiate along its own
axis. This happens at

$$
\tan\theta_B = \frac{n_2}{n_1},
$$

56.3° for glass in air, 75.6° for silicon. Light reflected at the Brewster angle is purely s-polarized, which is how
polarizing sunglasses cut glare and how Brewster windows keep a laser cavity lossless for one polarization. For an
absorbing medium $r_p$ never reaches zero; its minimum defines a **pseudo-Brewster angle**, which for metals lies
near 70–80° and is where ellipsometers measure, because the phase difference between $r_p$ and $r_s$ changes fastest
there.

## Total internal reflection and the evanescent wave

From the dense side ($n_1 > n_2$), Snell's law has no real solution beyond the critical angle

$$
\sin\theta_c = \frac{n_2}{n_1},
$$

41.8° for glass to air. There $\cos\theta_2 = i\sqrt{n_1^2\sin^2\theta_1/n_2^2 - 1}$ becomes imaginary: the
transmitted field is $e^{ik_x x}e^{-\kappa z}$, bound to the interface and decaying into medium 2 with

$$
\frac{1}{\kappa} = \frac{\lambda}{2\pi\sqrt{n_1^2\sin^2\theta_1 - n_2^2}} ,
$$

about a fifth of a wavelength a few degrees past the critical angle, growing without bound as $\theta_1 \to \theta_c$.
Both reflection coefficients have unit modulus, $|r_s| = |r_p| = 1$, so all the power returns; what is not zero is
their **phase**, which differs between s and p and depends on the angle. That phase is a design quantity in its own
right: the Fresnel rhomb turns linear into circular polarization with two total reflections at the angle where the
s–p phase difference is 45°, with no birefringent material at all. The phase also varies with $k_x$, and a beam of
finite width, a superposition of plane waves with slightly different $k_x$, is shifted sideways along the interface
by a fraction of a wavelength on reflection: the Goos–Hänchen shift (1947), measured with a multiply reflecting
glass block.

The evanescent field still reaches a fraction of a wavelength into the second medium, and it carries no power only as
long as nothing is there to take it. A second dense medium brought that close lets light tunnel through (frustrated
total reflection, the optical analogue of quantum tunnelling and the principle of the beam-splitter cube), a molecule
there is excited (total-internal-reflection fluorescence microscopy), and a metal film couples it to surface
plasmons, the principle of the [Kretschmann configuration](learn/films/surface-plasmon-resonance.html).

::: example fresnel-glass-air

## Absorbing media and the branch of the square root

For an absorbing medium the index is complex, $\tilde n = n + ik$ (time dependence $e^{-i\omega t}$, $k \ge 0$), and
so is $\cos\theta_2 = \sqrt{1 - n_1^2\sin^2\theta_1/\tilde n_2^2}$. The angle $\theta_2$ then has no geometric
meaning: the planes of constant phase and of constant amplitude in medium 2 are not parallel (an *inhomogeneous*
wave), and the transmitted wave decays along $z$ whatever its direction. The square root has two branches, and the
physical one is the wave that carries power away from the interface and decays into the medium, that is
$\mathrm{Re}(\tilde n_2\cos\theta_2) > 0$ or, at the boundary of that condition, $\mathrm{Im}(\tilde n_2\cos\theta_2) > 0$.
A code that takes the principal square root blindly gets this wrong for some metals and for every evanescent wave
beyond the critical angle in a weakly absorbing medium; Byrnes's implementation tests the branch explicitly, and
PyStratify's planar engine inherits the test. Once the branch is right, every formula above holds unchanged for
metals, for semiconductors above their band gap and for evanescent waves alike, which is why one engine serves a
glass lens, a silver mirror and a surface-plasmon sensor.

## Ellipsometry: the ratio of the two

The two Fresnel coefficients are rarely measured separately. An ellipsometer sends in light of known polarization
and measures the polarization of the reflected light, that is the complex ratio

$$
\rho = \frac{r_p}{r_s} = \tan\Psi\, e^{i\Delta} ,
$$

$\tan\Psi$ being the amplitude ratio and $\Delta$ the phase difference. For a bare substrate $\rho$ inverts in closed
form to the complex index,

$$
\tilde n_2 = n_1 \sin\theta_1\sqrt{1 + \left(\frac{1 - \rho}{1 + \rho}\right)^2 \tan^2\theta_1} ,
$$

which is how the optical constants of most materials in the databases were first obtained, and for a film on a
substrate $\rho$ follows from the transfer matrix and is fitted for thickness and index (Azzam and Bashara). The
calculator's Ψ and Δ are this ratio with the sign convention $\rho = -r_p/r_s$, which puts $\Delta$ in the range
ellipsometer software reports; the 180° difference between conventions is the one to remember when comparing with a
paper.
