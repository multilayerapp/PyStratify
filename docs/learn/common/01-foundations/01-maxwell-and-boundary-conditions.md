---
id: maxwell-and-boundary-conditions
title: "Maxwell's equations, boundary conditions and the wave equation"
geometry: common
chapter: Foundations
topic: foundations
sources: [plane_wave, dipole, focused]
description: "Maxwell's equations in layered media, the interface conditions, and how films, cylinders and spheres reduce to one scalar wave equation."
reviewed:
summary: >-
  The equations every page of this site rests on: Maxwell's equations in a piecewise homogeneous medium, the time
  convention, the continuity of tangential E and H at an interface, and how separation of variables turns the vector
  problem for a film, a cylinder or a sphere into one scalar second-order equation per channel.
references:
- cite: J. D. Jackson, Classical Electrodynamics, 3rd ed. (Wiley, 1999), ch. 7 and §9.7
- cite: J. A. Stratton, Electromagnetic Theory (McGraw-Hill, 1941), ch. VI–VII and IX
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §1.1–1.6
  doi: 10.1017/CBO9781139644181
- cite: C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley, 1983), ch. 2–4 and 8
  doi: 10.1002/9783527618156
- cite: A. Moroz, Ann. Phys. 315, 352 (2005)
  doi: 10.1016/j.aop.2004.07.002
- cite: I. L. Rasskazov, P. S. Carney and A. Moroz, OSA Continuum 3, 2290 (2020)
  doi: 10.1364/OSAC.399979
---

Everything the calculator computes, for a thin-film stack, a coated cylinder or a multilayered sphere, lit by a plane
wave, a point dipole or a focused beam, follows from the same four equations and the same two boundary conditions.
This section states them once, fixes the conventions used throughout the site, and shows the step that makes all
three geometries tractable: separation of variables, which turns the vector field problem into a scalar second-order
ordinary differential equation in one coordinate. The next section explains why that equation is solved, in every
geometry, by a $2\times 2$ transfer matrix.

## Maxwell's equations in a piecewise homogeneous medium

The media on this site are **linear, isotropic and piecewise homogeneous**: the stack, the cylinder and the sphere are
made of regions in which the relative permittivity $\varepsilon$ and permeability $\mu$ are constants, separated by
sharp interfaces. Inside each region, with no free charges or currents, Maxwell's equations read

$$
\nabla\cdot\mathbf{D} = 0, \qquad \nabla\cdot\mathbf{B} = 0, \qquad
\nabla\times\mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \qquad
\nabla\times\mathbf{H} = \frac{\partial \mathbf{D}}{\partial t},
$$

with the constitutive relations $\mathbf{D} = \varepsilon_0\varepsilon\,\mathbf{E}$ and
$\mathbf{B} = \mu_0\mu\,\mathbf{H}$.

**The time convention.** Every field is monochromatic with the time dependence $e^{-i\omega t}$, so a plane wave
travelling along $+z$ is $e^{i(kz - \omega t)}$. With this convention an absorbing medium has a refractive index with a
*positive* imaginary part, $n = n' + i n''$, $n'' \ge 0$, which is how the optical-constant tables on this site list
$n$ and $k$ (the calculator's $k$ is $n''$). Engineers often use $e^{+j\omega t}$ instead, where the same medium has
$n' - j n''$; results carry over by complex conjugation, and the sign of every phase and of $\Delta$ in ellipsometry
flips with it. For harmonic fields the curl equations become

$$
\nabla\times\mathbf{E} = i\omega\mu_0\mu\,\mathbf{H}, \qquad
\nabla\times\mathbf{H} = -i\omega\varepsilon_0\varepsilon\,\mathbf{E},
$$

so that $\mathbf{H}$ is fixed by $\mathbf{E}$ alone: $\mathbf{H} = (i\omega\mu_0\mu)^{-1}\nabla\times\mathbf{E}$.
In each homogeneous region, taking the curl once more gives the vector Helmholtz equation

$$
\nabla\times\nabla\times\mathbf{E} - k^2\mathbf{E} = 0, \qquad k = \frac{\omega}{c}\,n, \qquad n = \sqrt{\varepsilon\mu},
$$

and $\nabla\cdot\mathbf{E} = 0$. The vacuum wavenumber is $k_0 = \omega/c = 2\pi/\lambda$, with $\lambda$ the vacuum
wavelength, which is the wavelength every spectrum on this site is plotted against. The wave impedance of a region,
$Z = Z_0\sqrt{\mu/\varepsilon}$, is what distinguishes a magnetic medium from a dielectric of the same $n$; for the
non-magnetic materials that make up almost every example here, $\mu = 1$ and $n = \sqrt{\varepsilon}$.

## The two boundary conditions

At an interface between two regions, integrating Faraday's and Ampère's laws over a vanishing loop that straddles it
gives the conditions every solution must satisfy:

$$
\hat{\mathbf{n}}\times(\mathbf{E}_2 - \mathbf{E}_1) = 0, \qquad \hat{\mathbf{n}}\times(\mathbf{H}_2 - \mathbf{H}_1) = 0 :
$$

**the tangential components of $\mathbf{E}$ and of $\mathbf{H}$ are continuous** (there are no surface currents in a
dielectric or a metal described by a bulk $\varepsilon$). Continuity of the normal components of $\mathbf{D}$ and
$\mathbf{B}$ follows from these two together with the divergence equations and need not be imposed separately. Two
conditions is exactly the number a second-order equation needs, and that count is the whole reason the transfer
matrix exists.

One exception matters on this site. A **two-dimensional sheet**, such as graphene, carries a surface current
$\mathbf{K} = \sigma\,\mathbf{E}_\parallel$ and makes the tangential $\mathbf{H}$ jump by $\hat{\mathbf{n}}\times
\mathbf{K}$; PyStratify models such sheets as a jump condition at the interface rather than as a layer of finite
thickness. A **surface current** is also how an idealised metal is treated in the limit of a perfect conductor.

## Separation of variables: one scalar equation per channel

The vector Helmholtz equation separates in exactly the coordinate systems in which the interfaces of our three
geometries are coordinate surfaces: planes $z = \mathrm{const}$, cylinders $\rho = \mathrm{const}$ and spheres
$r = \mathrm{const}$. In each, the field is a superposition of modes labelled by the quantum numbers of the other two
coordinates, and the dependence on the coordinate normal to the interfaces is a scalar function $\psi$ obeying a
**second-order linear ordinary differential equation**. The three cases are:

| geometry | normal coordinate | channel labels | the equation for $\psi$ | its two independent solutions |
|---|---|---|---|---|
| films | $z$ | in-plane wavevector $\mathbf{k}_\parallel$; polarization s or p | $\psi'' + k_z^2\,\psi = 0$, $k_z = \sqrt{k^2 - k_\parallel^2}$ | $e^{\pm i k_z z}$ |
| cylinders | $\rho$ | azimuthal order $m$; axial wavenumber $k_z$ | Bessel's equation, $\rho^2\psi'' + \rho\psi' + (q^2\rho^2 - m^2)\psi = 0$, $q = \sqrt{k^2 - k_z^2}$ | $J_m(q\rho)$, $H^{(1)}_m(q\rho)$ |
| spheres | $r$ | angular momentum $\ell$ (and $m$); electric or magnetic | spherical Bessel's equation, $(r\psi)'' + \left[k^2 - \ell(\ell+1)/r^2\right] r\psi = 0$ | $j_\ell(kr)$, $h^{(1)}_\ell(kr)$ |

The pattern is the same in every row. The first solution is regular (a standing or incoming wave, finite at the centre);
the second is outgoing and singular at the origin, where $h^{(1)}_\ell$ and $H^{(1)}_m$ diverge. In a slab geometry the
same roles are played by waves running towards and away from an interface. Inside region $n$ the field of one channel
is therefore

$$
\psi(x) = A_n\,\psi_+(k_n x) + B_n\,\psi_-(k_n x),
$$

with two unknown amplitudes per region and per channel. The vector field is rebuilt from $\psi$ by the appropriate
vector harmonics: for a film, $\mathbf{E}$ is transverse to the plane of incidence (s, TE) or $\mathbf{H}$ is (p, TM);
for a sphere, the **vector spherical harmonics** $\mathbf{Y}^{(m)}_{\ell m}$, $\mathbf{Y}^{(e)}_{\ell m}$,
$\mathbf{Y}^{(o)}_{\ell m}$ of Jackson and Moroz (or Bohren and Huffman's $\mathbf{M}_{o m \ell}$,
$\mathbf{N}_{e m \ell}$; the two notations give identical results) carry the angular dependence, and each $\ell$
splits into an **electric** (TM) and a **magnetic** (TE) multipole; for a cylinder the orders $m$ carry
$e^{im\varphi}$, and at normal incidence the two polarizations, $\mathbf{E}$ along the axis and $\mathbf{H}$ along it,
decouple exactly as s and p do for a film. The channels are orthogonal, so **each channel is solved on its own**, and
the cross sections and powers are sums over channels with no cross terms.

In the boundary conditions, the continuity of tangential $\mathbf{E}$ becomes the continuity of $\psi$ (up to a known
factor) and the continuity of tangential $\mathbf{H}$, which is proportional to $\nabla\times\mathbf{E}$, becomes the
continuity of a first derivative of $\psi$ weighted by the region's admittance. At each of the $N$ interfaces, each
channel therefore gives **two linear equations**, exactly what is needed to pass from $(A_n, B_n)$ to
$(A_{n+1}, B_{n+1})$.

## Counting the unknowns

For $N$ interfaces there are $N + 1$ regions and $2(N+1)$ amplitudes per channel, and the interfaces supply $2N$
equations. The two missing conditions describe the **source** and the far field:

- **Regularity**: no singular outgoing wave may emerge from the centre of a sphere or the axis of a cylinder unless a
  source sits there, so $B = 0$ in the core. For a film, the equivalent statement is that nothing travels *towards* the
  stack from the far side of the substrate.
- **The incident field**: a plane wave fixes the regular amplitude in the outermost region (for a film, the amplitude
  of the wave arriving from the incident medium). A **point dipole** instead adds a known singular outgoing wave at its
  own position, which is why a dipole problem is the same $2\times 2$ problem with one amplitude shifted, and a
  **focused beam** is a plane wave problem integrated over the angles the objective fills.

The reflected and transmitted waves of a film, the scattering coefficients $a_\ell$, $b_\ell$ of a sphere, and the
$a_m$, $b_m$ of a cylinder are nothing but the remaining amplitudes once the system is solved. From them follow the
far field (reflectance and transmittance; cross sections and the scattering pattern), the near field in every region
(the coefficients *are* the field), the power absorbed in each layer or shell by Poynting's theorem, and, for a
dipole, the power it radiates compared with the same dipole in free space, which is its decay rate.

## Conservation laws as checks

Because the medium is linear and the equations are exact, a lossless structure must satisfy $R + T = 1$ for a film
and $Q_\mathrm{ext} = Q_\mathrm{sca}$ for a particle, and the reciprocity theorem relates a problem to its reverse.
These identities, together with the closed forms of a single interface (the Fresnel equations), a single sphere (Mie's
coefficients) and a single cylinder, are the first lines of the [Verification](verification.html) page: every solver on
this site is checked against them before it is checked against other codes and against papers.
