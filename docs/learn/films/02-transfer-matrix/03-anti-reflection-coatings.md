---
id: anti-reflection-coatings
title: "Anti-reflection coatings: single layer, V-coat and broadband designs"
geometry: films
chapter: The transfer matrix
topic: antireflection
sources: [plane_wave]
description: "Anti-reflection coatings from the transfer matrix: the quarter-wave layer and its ideal index, the two-layer V-coat, broadband three-layer designs, and angle."
reviewed:
summary: >-
  How a thin coating removes the 4 % reflection of glass: the single-layer reflectance at any wavelength and the
  quarter-wave condition with its ideal index, the two-layer quarter–quarter (V-coat) design that reaches zero with
  real materials, the quarter–half–quarter design that holds the reflectance low across the visible, what the half-wave
  layer does, how the minimum moves with the angle of incidence, high-index substrates such as silicon, and the limits
  of all of them.
examples: [v-coat-ar, broadband-ar]
references:
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 4
- cite: L. B. Lockhart and P. King, J. Opt. Soc. Am. 37, 689 (1947)
  doi: 10.1364/JOSA.37.000689
- cite: A. Musset and A. Thelen, Prog. Opt. 8, 201 (1970)
  doi: 10.1016/S0079-6638(08)70189-6
- cite: J. A. Dobrowolski, D. Poitras, P. Ma, H. Vakil and M. Acree, Appl. Opt. 41, 3075 (2002)
  doi: 10.1364/AO.41.003075
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §1.6
  doi: 10.1017/CBO9781139644181
---

An uncoated glass surface ($n_s = 1.52$) reflects $((1 - n_s)/(1 + n_s))^2 = 4.26$ % at normal incidence, and every
air–glass surface of a lens train, a display or a solar panel loses that much. A coating removes it by adding a
second reflection of equal amplitude and opposite phase. Both conditions, amplitude and phase, are what every design
below is built to meet, and the [transfer matrix](learn/films/transfer-matrix.html) is the tool that tracks them.

## One layer

A layer of index $n_1$ and thickness $d$ between the incident medium $n_0$ and the substrate $n_s$ has two
interfaces, with Fresnel coefficients $r_1 = (n_0 - n_1)/(n_0 + n_1)$ and $r_2 = (n_1 - n_s)/(n_1 + n_s)$ at normal
incidence, and the phase thickness $\delta = 2\pi n_1 d/\lambda$. Summing the multiple reflections (the Airy sum,
which the characteristic matrix of the layer performs in one step) gives

$$
R(\lambda) = \frac{r_1^2 + r_2^2 + 2 r_1 r_2 \cos 2\delta}{1 + r_1^2 r_2^2 + 2 r_1 r_2 \cos 2\delta} .
$$

The phase condition is $\cos 2\delta = -1$: a quarter-wave layer, $n_1 d = \lambda_0/4$, where the minimum is

$$
R(\lambda_0) = \left(\frac{r_1 - r_2}{1 - r_1 r_2}\right)^2 = \left(\frac{n_0 n_s - n_1^2}{n_0 n_s + n_1^2}\right)^2 ,
$$

the admittance form of [the transfer matrix](learn/films/transfer-matrix.html) section. The amplitude condition is
$r_1 = r_2$, which sets the ideal index $n_1 = \sqrt{n_0 n_s}$: 1.233 for glass in air. No hard, durable coating
material has so low an index. Magnesium fluoride (1.38) comes closest and leaves 1.26 % at $\lambda_0$, about
1.4 % averaged over 430–670 nm for a coating centred at 550 nm: a factor of three, not a cure.

The same rule decides high-index substrates, where it is far kinder. Silicon reflects about 35 % at 600 nm
($n \approx 3.9$); the ideal index is $\sqrt{3.9} = 1.97$, which silicon nitride provides, so one quarter-wave of
nitride brings the reflectance at the design wavelength to below 0.02 %. That single layer is the blue-black coating
of crystalline [solar cells](learn/films/solar-cell-absorption.html).

## Two layers: the V-coat

With two layers the amplitude condition no longer needs an index nobody makes. Two quarter-wave layers, $n_L$
outermost and $n_M$ against the substrate, transform the substrate admittance twice:

$$
Y = \frac{n_L^2}{n_M^2}\, n_s ,
\qquad
R(\lambda_0) = \left(\frac{n_0 - Y}{n_0 + Y}\right)^2 ,
$$

which vanishes when $n_M / n_L = \sqrt{n_s / n_0}$. With magnesium fluoride outside, glass needs $n_M = 1.70$ underneath,
in the range of aluminium oxide and similar oxides. The reflectance at $\lambda_0$ is then zero to the accuracy of the
indices, but the minimum is sharp: it rises on both sides as the two quarter waves drift away from a quarter wave
together, giving the V that names the design. Calculated for 550 nm, it stays below 0.5 % only from about 500 to
610 nm and reaches nearly 3 % at the edges of the visible, worse than a single layer there. A V-coat is the choice
for one laser line, not for white light.

The layers need not be quarter waves. The zero is a pair of phase thicknesses that makes the product of the two
characteristic matrices map $n_s$ onto $n_0$, and whether such a pair exists depends on the indices (Macleod ch. 4
gives the conditions). With magnesium fluoride outside on glass it exists once the inner index exceeds 1.70, the
quarter–quarter value, and the two thicknesses then part: with an inner index of 2.1 the zero needs an inner layer
only about 0.07 wave thick under a magnesium fluoride layer of about 0.32 wave. That thin high-index design is the
V-coat usually made, because high-index oxides are easier to deposit than an index of exactly 1.70.

::: example v-coat-ar

## Three layers: quarter–half–quarter

Lockhart and King (1947) inserted a half-wave layer of high index between the two quarter waves. A half-wave layer is
*absent* at $\lambda_0$ ($M = -I$, as shown in the transfer-matrix section), so the reflectance at the design
wavelength is exactly that of the V-coat beneath it. Away from $\lambda_0$ it is not absent: its phase error has the
opposite sign to that of the quarter waves around it and cancels most of their drift, which flattens the V into a
broad trough with two shallow minima on either side of $\lambda_0$. With 1.38 / 2.10 / 1.70 on glass, centred at
550 nm, the reflectance stays below 0.5 % from about 450 to 720 nm and averages 0.17 % over 430–670 nm, eight times
less than magnesium fluoride alone.

This is the pattern of every broadband design: layers that are absent or quarter waves at $\lambda_0$, chosen so that
their phase errors compensate over a band. Musset and Thelen (1970) generalised it to many layers. Dobrowolski and
coworkers (2002) mapped what is possible: the residual reflectance falls with the number of layers and the total
thickness, and rises with the bandwidth and the angle range asked for, and the low index of the outer layer is what
limits it most. Taken to the limit, a continuous profile from the substrate index down to that of air is the
[moth-eye](learn/films/moth-eye-antireflection.html) surface.

::: example broadband-ar

## Angle of incidence and polarization

At oblique incidence the phase thickness becomes $\delta = 2\pi n_1 d \cos\theta_1/\lambda$, with $\theta_1$ the angle
inside the layer, so every quarter-wave condition moves to a shorter wavelength, $\lambda_0 \cos\theta_1$. For the
magnesium fluoride layer designed at 550 nm the minimum moves to about 533 nm at 20°, 487 nm at 40° and 428 nm at 60°.
The amplitude condition changes too, because the effective indices of each layer become $n\cos\theta$ for s and
$n/\cos\theta$ for p ([Fresnel equations](learn/films/fresnel-equations.html)): the two polarizations separate, s
reflecting more than p, and the coating's minimum depth is lost. The three-layer design above averages 0.17 % over
the visible at normal incidence, about 0.5 % at 40° and 4 % at 60°. A coating for a wide-angle lens or a cover glass
viewed from the side is designed over the angles it will see, not at normal incidence.

## What these designs leave out

Each surface of a window needs its own coating. A coated front surface does nothing for the 4 % reflected at the
back, and a substrate thick enough to be [incoherent](learn/films/coherence.html) adds the two surfaces' intensities,
not their amplitudes. Real coating materials are dispersive (magnesium fluoride's index rises toward the blue), some
absorb in the blue and ultraviolet, and evaporated films are less dense than bulk, with a lower index that rises as
they take up water. In the calculator a constant index stands in for each layer here; a refractiveindex.info page in
its place gives the dispersive answer, and the minimum moves with it.
