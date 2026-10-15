---
id: beam-splitters
title: "Thin metal films: beam splitters"
geometry: films
chapter: Metal films
topic: metal-films
sources: [plane_wave]
description: "Thin metal films as beam splitters: the Drude metal and skin depth, unavoidable absorption, potential transmittance and induced transmission."
reviewed:
summary: >-
  A few nanometres of silver divide a beam into reflected and transmitted halves: the Drude metal and the skin depth,
  why a metal film is partly transparent, the absorption that no design removes and Macleod's potential
  transmittance that bounds it, the induced-transmission trick of dielectric matching layers, the polarizing and
  phase-shifting behaviour at 45°, and what dielectric splitters trade for their losslessness.
examples: [metal-beam-splitter]
references:
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 4 and 7
- cite: P. H. Berning and A. F. Turner, J. Opt. Soc. Am. 47, 230 (1957)
  doi: 10.1364/JOSA.47.000230
- cite: P. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370 (1972)
  doi: 10.1103/PhysRevB.6.4370
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), ch. 14
  doi: 10.1017/CBO9781139644181
- cite: N. W. Ashcroft and N. D. Mermin, Solid State Physics (Holt, Rinehart and Winston, 1976), ch. 1
---

A metal's index is mostly imaginary: silver at 550 nm has $\tilde n \approx 0.06 + 3.6i$ (Johnson and Christy 1972),
so $\varepsilon = \tilde n^2 \approx -13 + 0.4i$. The negative real part is the signature of free electrons, and it is
what makes the metal reflect; the small imaginary part is what makes it absorb. Both come out of the simplest model
of a conductor, and both are needed to understand why a film of silver can be a beam splitter at all.

## The Drude metal and the skin depth

Free electrons of density $N$ and effective mass $m$ driven by a field $E e^{-i\omega t}$ and damped at a rate
$\gamma$ give the Drude permittivity

$$
\varepsilon(\omega) = \varepsilon_\infty - \frac{\omega_p^2}{\omega^2 + i\gamma\omega},
\qquad
\omega_p^2 = \frac{N e^2}{\varepsilon_0 m} ,
$$

with the plasma frequency $\omega_p$ in the ultraviolet for the noble metals ($\hbar\omega_p \approx 9$ eV for
silver) and $\gamma \ll \omega$ in the visible. Below $\omega_p$ the real part is large and negative: the electrons
respond out of phase, screen the field, and the wave cannot propagate. Instead it decays over the **skin depth**

$$
\delta = \frac{\lambda}{4\pi k} ,
$$

12 nm in intensity for silver in the visible ($k \approx 3.6$), 25 nm for aluminium's lower $k$ in the near
ultraviolet. The interband transitions of the bound electrons (the $d$ bands) add to $\varepsilon_\infty$ and give
silver its edge near 320 nm and gold its colour; the tabulated $n, k$ contain all of it, and the Drude form is the
reason for their shape rather than a substitute for them. A film much thicker than $\delta$ is a mirror; a film
comparable to it lets a useful fraction through. That is the whole design freedom of a metallic beam splitter: one
thickness trades reflection against transmission.

## Absorption is unavoidable, and how much is bounded

In a metal, the field that does get through drives currents that dissipate. However the film is chosen,

$$
R + T + A = 1 , \qquad A > 0 ,
$$

and a 50/50 silver splitter loses a few percent to absorption, more toward the blue, where silver's $k$ falls and
its $n$ rises. Macleod's **potential transmittance** says how much a given metal layer *can* transmit if the
dielectrics around it were chosen perfectly:

$$
\psi = \frac{T}{1 - R} ,
$$

the transmittance per unit of light that enters the film. It depends on the metal layer and on the admittance
presented to its exit face, not on what is in front of it, and it has a maximum over that exit admittance. For a
given thickness of silver the maximum $\psi$ is the best any coating can do; design is then the choice of layers
that realise it (Berning and Turner 1957). This is the principle of **induced transmission**: a 30 nm silver film
alone transmits a few percent in the visible, but between two matching dielectric layers it transmits 60 % over a
band, with the rest reflected rather than absorbed. The [low-emissivity coating](learn/films/low-emissivity-coatings.html)
is this idea applied to a window.

::: example metal-beam-splitter

## s and p at 45°

At oblique incidence the two polarizations see different tilted admittances, $n\cos\theta$ for s and $n/\cos\theta$
for p (see [the transfer matrix](learn/films/transfer-matrix.html)), and a metal film therefore reflects s more
strongly than p: at 45° a 50/50 silver splitter is typically 55/45 for s and 45/55 for p, mildly polarizing. Worse
for interferometers, the two polarizations also pick up **different phases** on reflection and transmission, so a
linearly polarized beam at 45° to the plane of incidence leaves slightly elliptical. Where that matters, in
interferometers that recombine the two beams or in polarization-sensitive imaging, a compensating plate, a
polarization-insensitive multilayer or a cube splitter with the film at a lower angle is used. The same difference
is the reason the s-polarized dip of a [surface-plasmon sensor](learn/films/surface-plasmon-resonance.html) is
absent while the p-polarized one is deep.

## Dielectric splitters

A quarter-wave stack of a few periods splits a beam with no absorption at all, at the price of a narrower band and a
stronger dependence on angle and polarization: its reflectance follows the [Bragg](learn/films/bragg-mirrors.html)
curve, 50 % only near one wavelength per design, and its s and p reflectances separate faster with angle than a
metal's. Broadband dielectric splitters are therefore non-quarter-wave designs with many layers, optimised
numerically, and for the wide band and modest accuracy of a visual instrument a metal film remains the simpler and
cheaper part. Hybrid designs, a metal layer with dielectric enhancement on one side, sit between the two.

## Measuring a metal film

The optical constants of a thin metal film are not those of the bulk: a film thinner than the electron mean free
path (about 50 nm for silver) scatters its electrons at the surfaces, which raises $\gamma$ and the absorption, and
below about 10 nm an evaporated film is not continuous but an island structure with its own plasmon resonance. The
databases' $n, k$ are for thick, well-annealed films; a 10 nm beam splitter is measured, by ellipsometry or by $R$
and $T$ together, and the fitted thickness and index are what a design should use. The calculator's uploaded-table
inputs exist for exactly this case.
