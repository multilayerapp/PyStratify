---
id: dipole-near-a-sphere
title: "A dipole near a sphere: decay rates, quenching and enhancement"
geometry: spheres
chapter: Emitters near spheres
topic: sphere-emitters
sources: [dipole]
classics: [ruppin-1982, anger-2006]
description: "A dipole beside or inside a sphere: radiative and non-radiative decay rates, quantum yield, and the trade-off between enhancement and quenching."
reviewed:
summary: >-
  A molecule or quantum dot beside, or inside, a sphere: how its field is expanded in multipoles, how the sphere's
  transfer matrices return it, the radiative and non-radiative decay rates that result, the quantum yield, and the
  competition between enhancement and quenching that fixes the best distance for enhanced fluorescence.
references:
- cite: R. Ruppin, J. Chem. Phys. 76, 1681 (1982)
  doi: 10.1063/1.443196
- cite: H. Chew, J. Chem. Phys. 87, 1355 (1987)
  doi: 10.1063/1.453317
- cite: Y. S. Kim, P. T. Leung and T. F. George, Surf. Sci. 195, 1 (1988)
  doi: 10.1016/0039-6028(88)90776-5
- cite: A. Moroz, Ann. Phys. 315, 352 (2005)
  doi: 10.1016/j.aop.2004.07.002
- cite: R. Carminati, J.-J. Greffet, C. Henkel and J. M. Vigoureux, Opt. Commun. 261, 368 (2006)
  doi: 10.1016/j.optcom.2005.12.009
- cite: P. Anger, P. Bharadwaj and L. Novotny, Phys. Rev. Lett. 96, 113002 (2006)
  doi: 10.1103/PhysRevLett.96.113002
- cite: L. Novotny and B. Hecht, Principles of Nano-Optics, 2nd ed. (Cambridge University Press, 2012), ch. 8
  doi: 10.1017/CBO9780511794193
---

A metal nanoparticle next to a fluorescent molecule can make it brighter by an order of magnitude or switch it off
completely, and the difference is a few nanometres of distance. Both effects come out of the same calculation: the
dipole's own field, reflected by the sphere, acts back on the dipole. The sphere's response is the set of
[transfer matrices](learn/spheres/recursive-transfer-matrix.html) already built for the plane wave; only the source
changes, from a wave arriving from infinity to a singular outgoing wave at the dipole's position. Ruppin's 1982 paper
did this for a molecule beside a small silver sphere, Chew's 1987 paper gave the closed expressions used ever since,
and Kim, Leung and George extended them to a dipole inside a coated sphere.

## The dipole's field as outgoing multipoles

An electric dipole $\mathbf{p}$ at $\mathbf{r}_d$ radiates the field of a point source, which, seen from the sphere's
centre, is a sum of outgoing multipoles $\mathbf{H}_{pL}(k, \mathbf{r})$ for $r > r_d$ and of regular ones for
$r < r_d$, with known coefficients (the expansion of the free-space Green dyadic). A dipole therefore enters the
transfer-matrix recursion as a **jump in the amplitudes** at the radius $r_d$: between the inside and the outside of
the shell that contains it, $A_{pL}$ and $B_{pL}$ change by the dipole's own multipole coefficients. The two closing
conditions are regularity at the centre, $B_{pL}(1) = 0$, and no incoming wave from infinity, $A_{pL}(N+1) = 0$. Solving
the $2\times 2$ system per channel gives the field everywhere, and in particular the reflected field at the dipole,
which is what decides its fate.

Unlike a plane wave, which excites only $m = \pm 1$, a dipole off the centre excites every $m$ for each $\ell$, and
a dipole close to the surface excites high $\ell$ strongly, because the near field of a point source is rich in
short-wavelength harmonics. The sums below therefore converge slowly at small distances, and the engine extends the
number of orders until they do.

## Radiative and non-radiative rates

The power the dipole radiates to the far field, relative to the same dipole in the unbounded host, is a sum over
orders of the squared total exterior field amplitudes at the dipole's radius. For a dipole outside a sphere at
distance $r_d$ from the centre, with $z = k_h r_d$ and the Riccati–Bessel functions $\psi_\ell$, $\xi_\ell$, Chew's
results read, for the **radial** (perpendicular) and **tangential** (parallel) orientations,

$$
\frac{\Gamma^\mathrm{rad}_\perp}{\Gamma_0} = \frac{3}{2 z^4}\sum_{\ell=1}^{\infty} \ell(\ell+1)(2\ell+1)\,\bigl|\psi_\ell(z) + a_\ell\,\xi_\ell(z)\bigr|^2,
\qquad
\frac{\Gamma^\mathrm{rad}_\parallel}{\Gamma_0} = \frac{3}{4 z^2}\sum_{\ell=1}^{\infty} (2\ell+1)\left[\bigl|\psi_\ell'(z) + a_\ell\,\xi_\ell'(z)\bigr|^2 + \bigl|\psi_\ell(z) + b_\ell\,\xi_\ell(z)\bigr|^2\right],
$$

where $a_\ell$, $b_\ell$ are the sphere's (composite) Mie coefficients in the convention in which the total exterior
field of order $\ell$ is $\psi_\ell + a_\ell\xi_\ell$. A radial dipole drives electric multipoles only; a tangential one
drives both. Far from the sphere ($z \to \infty$) the brackets oscillate about $\psi_\ell$ and the rates tend to
$\Gamma_0$ with a decaying ripple of period half a wavelength, the spherical version of Drexhage's oscillations.

An absorbing sphere also takes power that never reaches the far field: the dipole's near field drives currents in the
metal. The **non-radiative** rate is this absorbed power, obtained either from Poynting's theorem on a surface just
inside the sphere, or as the total rate minus the radiative one, where the total follows from the imaginary part of the
reflected field at the dipole, $\Gamma^\mathrm{tot}/\Gamma_0 = 1 + \tfrac{6\pi}{k^3}\,\mathrm{Im}\,[\hat{\mathbf{p}}\cdot
\mathbf{E}_\mathrm{refl}(\mathbf{r}_d)]/|\mathbf{p}|$ in the normalisation of Novotny and Hecht. PyStratify computes
the three rates separately and reports the total as their sum, which is a check in itself: for a lossless sphere the
non-radiative rate must vanish to machine precision.

## Quenching, enhancement and the best distance

Two opposite trends meet at the surface of a metal sphere. The **radiative** rate is enhanced near the particle's
plasmon resonance, because the sphere acts as an antenna that radiates the dipole's near field efficiently, and the
enhancement reaches further out for a larger sphere. The **non-radiative** rate grows without bound as the dipole
approaches the surface, like $d^{-3}$ in the quasi-static limit for a local metal, because the high-$\ell$ multipoles
that a nearby dipole excites are dark: they store the energy in the metal and dissipate it. The quantum yield of the
coupled system,

$$
q = \frac{\Gamma^\mathrm{rad}}{\Gamma^\mathrm{rad} + \Gamma^\mathrm{nr} + \Gamma_0(1 - q_0)/q_0},
$$

therefore collapses within the first few nanometres (this is quenching), rises to a maximum at a distance of the order
of the particle's radius, and returns to the intrinsic yield $q_0$ far away. The maximum sits further out for a larger
sphere, which Anger, Bharadwaj and Novotny measured by scanning a gold particle over a single molecule: an 80 nm
sphere enhances the fluorescence by a factor of several at about 5 nm, a 20 nm sphere mostly quenches. Fluorescence
is the product of two factors, the excitation rate, proportional to the local plane-wave intensity $|\mathbf{E}|^2$ at
the molecule, and the quantum yield; both are sphere calculations, one with a plane wave and one with a dipole, and
the calculator runs them with the same inputs.

::: classic anger-2006

::: classic ruppin-1982

## Inside a sphere, and in a shell

A dipole **inside** a sphere, or in one of its shells, is handled by the same recursion with the jump placed in that
shell; the radiative rate then involves the amplitudes transmitted to the host, and the formula gains the host's
and the shell's indices as prefactors (Chew; Kim, Leung and George). This is the model of a dye molecule inside a
silica bead, of an emitter in the dielectric spacer of a nanomatryoshka, and, with the sphere of the size of a cell,
of the autofluorescence seen through a microscope. An emitter in a lossless dielectric sphere shows no quenching at
all, only the oscillation of the radiative rate with position and the sharp enhancements at the sphere's
whispering-gallery resonances, which for a large sphere of high index can reach factors of thousands over a narrow
band: the Purcell effect in its original form.

## What the model leaves out

The emitter is a point dipole of fixed frequency and the metal a local, homogeneous medium with a bulk permittivity;
within a nanometre or two of the surface the electron gas's nonlocal response, electron spill-out and the discreteness of
the lattice change the quenching law, and the calculated rate at contact is a limit rather than a prediction. A
dielectric host surrounding the dipole must be lossless for the free-space rate $\Gamma_0$ to be defined. Within those
limits the formulas above are exact, and the [Verification](verification.html) page lists their checks against
independently written Sommerfeld-type integrals, against smuthi and treams, and against the finite-difference and
finite-element methods that know nothing about multipoles.
