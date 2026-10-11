---
id: beam-splitters
title: "Thin metal films: beam splitters"
geometry: films
chapter: Metal films
topic: metal-films
sources: [plane_wave]
summary: >-
  A few nanometres of silver divide a beam into reflected and transmitted halves: why a metal film can be partly
  transparent, why it always absorbs, and why at 45° it treats s and p differently.
examples: [metal-beam-splitter]
references:
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 4 and 9
- cite: P. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370 (1972)
  doi: 10.1103/PhysRevB.6.4370
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), ch. 14
  doi: 10.1017/CBO9781139644181
---

A metal's index is mostly imaginary: silver at 550 nm has $\tilde n \approx 0.06 + 3.6i$. Light entering it decays
over the skin depth

$$
\delta = \frac{\lambda}{4\pi k},
$$

about 12 nm in intensity for silver in the visible. A film much thicker than that is a mirror; a film comparable to
it lets a useful fraction through. That is the whole design freedom of a metallic beam splitter: one thickness trades
reflection against transmission.

## Absorption is unavoidable

In a metal, the field that does get through drives currents that dissipate. However the film is chosen,

$$
R + T + A = 1 , \qquad A > 0 ,
$$

and a 50/50 silver splitter loses a few percent to absorption, more at shorter wavelengths, where silver's $k$ falls
and its $n$ rises. Dielectric splitters (a quarter-wave stack of a few layers) avoid the loss at the price of a
narrower band.

::: example metal-beam-splitter

## s and p at 45°

At oblique incidence the two polarizations see different tilted admittances, $n\cos\theta$ for s and $n/\cos\theta$
for p (see [the transfer matrix](learn/films/transfer-matrix.html)). A metal film therefore reflects s more strongly
than p at 45°, and a metallic splitter is mildly polarizing. Where that matters, in interferometers that recombine the
two beams, a compensating plate or a polarization-insensitive multilayer design is used.
