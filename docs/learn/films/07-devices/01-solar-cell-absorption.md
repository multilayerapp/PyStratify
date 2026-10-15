---
id: solar-cell-absorption
title: "Solar cells: where the light is absorbed"
geometry: films
chapter: Devices
topic: layer-absorption
sources: [plane_wave]
description: "Where a solar cell absorbs light: layer-resolved absorptance, the standing wave in a thin stack, photocurrent from AM1.5, and parasitic losses."
reviewed:
summary: >-
  Total absorption is not what a solar cell needs: the absorbed power density from the field, the layer-resolved
  absorptance from the Poynting flux and its conservation check, the standing wave in a thin stack and how it moves
  with thickness, the photocurrent from the AM1.5 spectrum, parasitic absorption in contacts and transport layers,
  and the Lambertian ceiling that textures aim at.
examples: [perovskite-cell]
references:
- cite: L. A. A. Pettersson, L. S. Roman and O. Inganäs, J. Appl. Phys. 86, 487 (1999)
  doi: 10.1063/1.370757
- cite: G. F. Burkhard, E. T. Hoke and M. D. McGehee, Adv. Mater. 22, 3293 (2010)
  doi: 10.1002/adma.201000883
- cite: P. Peumans, A. Yakimov and S. R. Forrest, J. Appl. Phys. 93, 3693 (2003)
  doi: 10.1063/1.1534621
- cite: ASTM G173-03, Standard Tables for Reference Solar Spectral Irradiances (AM1.5 global)
  url: https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html
- cite: S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720 (2016)
  url: https://arxiv.org/abs/1603.02720
---

A photon makes current only if it is absorbed in the active layer. Light absorbed in the transparent electrode, the
transport layers or the metal contact is heat. A thin-film cell is therefore designed not for maximum absorption but
for maximum absorption *in one layer*, and that needs the absorptance of every layer separately, which the transfer
matrix provides from the fields it already computes. Pettersson, Roman and Inganäs (1999) wrote this out for organic
cells, Peumans and Forrest and then Burkhard, Hoke and McGehee turned it into the standard optical model of thin-film
photovoltaics, and the same calculation serves perovskite, CIGS and thin silicon cells alike.

## The absorbed power density

In an absorbing layer of complex index $\tilde n = n + ik$ the time-averaged power dissipated per unit volume is

$$
q(z) = \tfrac{1}{2}\,\omega\,\varepsilon_0\,\mathrm{Im}\,\varepsilon\,|E(z)|^2 = \tfrac{1}{2}\,\omega\,\varepsilon_0\, 2nk\,|E(z)|^2 ,
$$

proportional to the local intensity, with the absorption coefficient $\alpha = 4\pi k/\lambda$ hidden in $2nk$. The
field $E(z)$ in a layer is the sum of the forward and backward amplitudes the
[transfer matrix](learn/films/transfer-matrix.html) returns, $a\,e^{ik_z z} + b\,e^{-ik_z z}$, so $|E|^2$ contains
the two decaying exponentials and their interference term, and $q(z)$ integrated over the layer is a closed form in
$a$, $b$ and $k_z$ (Pettersson's equations), which is what PyStratify evaluates. The profile $|E(z)|^2$ is the
calculator's field panel; its integral layer by layer is the per-layer absorptance.

## Absorptance from the Poynting flux

Equivalently, energy conservation in each layer says that what flows in and does not flow out is absorbed. With
$S_j^\mathrm{in}$ and $S_j^\mathrm{out}$ the normal components of the time-averaged Poynting vector at the layer's two
faces and $S_0$ the incident flux,

$$
A_j(\lambda) = \frac{S_j^\mathrm{in} - S_j^\mathrm{out}}{S_0},
\qquad
\sum_j A_j = 1 - R - T .
$$

The two routes must agree, and the sum rule is the check: PyStratify computes $A_j$ for every layer, and the check
under the example confirms that the layers' absorptances, $R$ and $T$ add to one to $10^{-9}$, Poynting's theorem
applied to the whole stack. A layer with $k = 0$ gets exactly $A_j = 0$ whatever the field inside it, which a numerical
integration of $|E|^2$ would not reproduce; the flux difference does.

::: example perovskite-cell

## The standing wave, and thickness

In a stack this thin the field inside is a standing wave: the back contact reflects nearly everything, so the
intensity has a node at the metal and antinodes a quarter wave (in the material) away from it. An absorber placed at
an antinode absorbs strongly, at a node weakly, and since the node spacing is $\lambda/2n$, a change of absorber
thickness of 50–100 nm moves the active layer from one to the other for a given wavelength. This is why the
absorption of a thin organic or perovskite cell oscillates with thickness and why its external quantum efficiency
has interference fringes rather than the smooth Beer–Lambert rise of a thick absorber. Thickening the perovskite
absorbs more of the red, up to the point where the light is gone before it reaches the back contact; beyond that
the extra thickness only adds collection losses, and the optimum is found by the photocurrent, not by the
absorptance at one wavelength.

## The photocurrent

Weighting $A_\mathrm{active}(\lambda)$ by the solar photon flux gives the current a perfect collector would deliver:

$$
J_\mathrm{sc}^\mathrm{max} = q\int A_\mathrm{active}(\lambda)\,\Phi_\mathrm{AM1.5}(\lambda)\,\mathrm{d}\lambda ,
$$

with $\Phi$ the AM1.5 global photon flux (ASTM G173; about $4.4 \times 10^{21}$ photons m⁻² s⁻¹ below 1100 nm,
corresponding to 44 mA cm⁻² if every one were absorbed in silicon). This is the figure of merit the optical models of
organic and perovskite cells optimise (Burkhard, Hoke and McGehee 2010), layer thickness by layer thickness, and the
one the calculator's sweeps are for. A measured external quantum efficiency divided by the computed $A_\mathrm{active}$
is the internal quantum efficiency, the electrical part of the problem separated from the optical one, which is the
main practical use of the layer-resolved calculation: it tells whether a cell loses photons or carriers.

## Parasitic absorption

Every other layer's $A_j$ is a loss, and the model tells which. Indium tin oxide absorbs in the blue (its band edge)
and in the near infrared (its free electrons; the trade-off with sheet resistance of the
[low-emissivity section](learn/films/low-emissivity-coatings.html)), typically 5–10 % of the current in a thin-film
cell; a 20 nm PEDOT:PSS or a doped transport layer a few percent; the metal back contact a few percent of the
long-wavelength light, more if the contact is rough or the stack is designed so that an antinode sits at the metal.
Silver loses less than aluminium, and a thin dielectric spacer between the absorber and the metal (which pushes the
antinode out of the metal) is the standard fix for both organic and silicon cells. The totals are what the layer
absorptances add up to, and an optical loss budget of a cell is a bar chart of them.

## Beyond a planar stack

A planar calculation assumes a single pass per reflection. Thick silicon cells trap light with
[textured surfaces](learn/films/textured-surfaces.html), whose ceiling is Yablonovitch's $4n^2$, and thin cells with
gratings and scattering back reflectors; the per-layer absorption of a textured cell comes from a ray-tracing or
grating solver fed with the planar coefficients of each facet, which is how the calculator's textured surfaces
compute it. The planar stack remains the reference case: it is exact, it is fast, and it is what the textured
result must reduce to when the texture is removed.
