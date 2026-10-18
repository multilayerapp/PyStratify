---
id: wires-and-fibres
title: "Wires and fibres: emitters beside a cylinder and beams through it"
geometry: cylinders
chapter: Emitters and beams
topic: cylinder-emitters
sources: [dipole, focused]
classics: [ruppin-2001, davis-2011]
description: "A dipole beside a nanowire or fibre: guided modes, radiated and absorbed power, the thin-wire plasmon, and a fibre under an FT-IR Cassegrain."
reviewed:
summary: >-
  A point dipole next to a nanowire or an optical fibre: the continuous spectrum of axial wavenumbers, the guided
  modes as poles, and how the radiated, guided and absorbed powers are separated. The thin metal wire's plasmon at
  the local limit. And a fibre under an infrared microscope's Cassegrain: why a fibre that absorbs nothing shows
  a spectrum.
references:
- cite: V. V. Klimov and M. Ducloy, Phys. Rev. A 69, 013812 (2004)
  doi: 10.1103/PhysRevA.69.013812
- cite: A. W. Snyder and J. D. Love, Optical Waveguide Theory (Chapman and Hall, 1983), ch. 12–13
  doi: 10.1007/978-1-4613-2813-1
- cite: R. Ruppin, Opt. Commun. 190, 205 (2001)
  doi: 10.1016/S0030-4018(01)01063-X
- cite: B. J. Davis, P. S. Carney and R. Bhargava, Anal. Chem. 82, 3474 (2010)
  doi: 10.1021/ac902067p
- cite: B. J. Davis, P. S. Carney and R. Bhargava, Anal. Chem. 83, 525 (2011)
  doi: 10.1021/ac102239b
- cite: C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley, 1983), ch. 8
  doi: 10.1002/9783527618156
---

The plane-wave problem of the [previous section](learn/cylinders/infinite-cylinder.html) has a single axial
wavenumber, fixed by the angle of incidence. Two other sources break that simplicity in instructive ways: the
calculator offers the first, PyStratify both. A **point dipole** radiates in every direction at once, so its field
contains every axial wavenumber $k_z$ and every order $m$; the cylinder's response must be integrated over $k_z$,
and the poles of that integrand are the guided modes of the wire or fibre. A **focused beam** is a bundle of plane
waves filling a cone, and a cylinder under an infrared microscope answers each of them differently; the spectrum the
instrument records is the cone average.

## A dipole beside a cylinder

Expand the dipole's field in cylindrical waves $e^{im\varphi} e^{ik_z z} H^{(1)}_m(q\rho)$: unlike the plane wave,
which has one $k_z$, the dipole needs all of them, with $q = \sqrt{k^2 - k_z^2}$ passing from real (waves that
propagate away from the axis) to imaginary (waves bound to it) as $|k_z|$ grows past $k$. For each $(m, k_z)$ the
cylinder's transfer matrices give the reflected cylindrical wave at the dipole's position, with the two polarizations
coupled whenever $k_z \ne 0$. The power the dipole radiates, relative to free space, is an integral over $k_z$ and a
sum over $m$ of the work the reflected field does on the dipole, the direct analogue of the Chance–Prock–Silbey
integral for a [dipole near a plane](learn/films/dipole-near-an-interface.html) (Klimov and Ducloy give the complete
formulas for an atom beside a nanofibre).

The integrand has three kinds of contribution, and PyStratify separates them because they are what an experiment
sees separately:

- $|k_z| < k_h$ in the host: cylindrical waves that **radiate** away to the far field;
- **poles** of the reflected amplitude at real $k_z = n_\mathrm{eff} k_0 > k_h$: the guided modes of the fibre
  (HE$_{11}$ first, which has no cutoff, then TE$_{01}$, TM$_{01}$, HE$_{21}$, … as the fibre grows), each taking a
  share of the dipole's power that then travels along the fibre without radiating. In a lossless fibre these poles sit
  on the real axis and the residues are taken analytically, so the guided fraction does not depend on a quadrature
  grid; PyStratify checks the modes it finds against the fibre's closed-form dispersion relations (Snyder and Love);
- the rest of the evanescent range, which against an absorbing wire deposits power in the metal (the quenching
  channel) and against a lossless one contributes nothing to the total but shapes the near field.

A dipole oriented radially, azimuthally or axially couples to different orders $m$ and different modes, so the three
orientations have different rates, and an isotropic ensemble averages them. Close to a metal wire the quenching rises
as $d^{-3}$, as it does near a plane, but the enhancement at the wire's plasmon is sharper, because a thin wire is a
nearly one-dimensional resonator.

## The thin metal wire, and where the local model stops

The case-II resonance of a thin wire sits where $\varepsilon(\omega) = -\varepsilon_h$, that is at $\omega_p/\sqrt{2}$
for a Drude metal in vacuum and $\omega_p/\sqrt{3}$ in a host of $\varepsilon_h = 2$. Ruppin computed the extinction of
a sodium wire one nanometre in radius in two ways: with the local Drude permittivity, which gives exactly this
resonance, and with a hydrodynamic (nonlocal) electron gas, in which the pressure of the electron fluid pushes the
resonance to higher frequency and lets longitudinal bulk plasmons appear above $\omega_p$. The calculator reproduces
the local curve; the nonlocal correction is a different constitutive relation, and it matters only when a dimension
shrinks to a few nanometres or an emitter approaches a surface that closely. For gold and silver particles above
about 10 nm the local model is quantitatively right, and the local curve is the one to compare an experiment with
first.

::: classic ruppin-2001

## A fibre under a Cassegrain

An infrared microscope illuminates its sample through a Cassegrain objective: a cone of plane waves between an inner
obscuration and the outer numerical aperture, each arriving at a different angle $\zeta$ to a fibre lying in the
focal plane. For every angle the oblique-incidence solution of the [previous section](learn/cylinders/infinite-cylinder.html)
gives the field transmitted towards the detector; the instrument's reading is the incoherent average over the cone of
the power collected within its own aperture, and dividing by the empty-field reading gives an apparent absorbance.
Davis, Carney and Bhargava showed that this apparent absorbance of a fibre that **absorbs nothing** is far from zero:
light is refracted and scattered out of the collection cone in a way that depends on wavelength through the size
parameter, so a textile fibre or a hair shows a broad, oscillating baseline with dips where the fibre acts as a lens.
Real absorption bands ride on this baseline, and a quantitative spectrum needs the baseline computed and removed, not
fitted by a polynomial. This is the physics PyStratify's `FocusedBeam` computes for films, cylinders and spheres
(the calculator does not offer it); the sphere version is the Mie-type scattering that distorts the infrared spectra
of cells, met in the [Mie theory](learn/spheres/mie-scattering.html) section.

::: classic davis-2011

Two parameters of the instrument matter and are often left out of a paper's caption: the numerical aperture, which
sets the widest angle in the cone, and the central obscuration of the Cassegrain, which removes the near-normal rays.
Both change the baseline's shape; Davis's Fig. 5 is reproduced with the aperture of the same paper's Fig. 2.
PyStratify's `FocusedBeam` takes both, as `na` and `obscuration`, so with the fibre's radius the three can be varied
against a measured spectrum.
