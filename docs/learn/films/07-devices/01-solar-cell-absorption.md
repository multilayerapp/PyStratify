---
id: solar-cell-absorption
title: "Solar cells: where the light is absorbed"
geometry: films
chapter: Devices
topic: layer-absorption
sources: [plane_wave]
summary: >-
  Total absorption is not what a solar cell needs: layer-resolved absorptance from the Poynting flux, useful absorption
  in the active layer against parasitic loss in the contacts, and how thickness trades them.
examples: [perovskite-cell]
references:
- cite: L. A. A. Pettersson, L. S. Roman and O. Inganäs, J. Appl. Phys. 86, 487 (1999)
  doi: 10.1063/1.370757
- cite: G. F. Burkhard, E. T. Hoke and M. D. McGehee, Adv. Mater. 22, 3293 (2010)
  doi: 10.1002/adma.201000883
- cite: S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720 (2016)
  url: https://arxiv.org/abs/1603.02720
---

A photon makes current only if it is absorbed in the active layer. Light absorbed in the transparent electrode, the
transport layers or the metal contact is heat. A thin-film cell is therefore designed not for maximum absorption but
for maximum absorption *in one layer*, and that needs the absorptance of every layer separately.

## Absorptance from the Poynting flux

The transfer matrix gives the field everywhere in the stack, and with it the time-averaged Poynting flux normal to
the layers. Energy conservation in each layer says that what flows in and does not flow out is absorbed:

$$
A_j(\lambda) = \frac{S_j^\mathrm{in} - S_j^\mathrm{out}}{S_0},
\qquad
\sum_j A_j = 1 - R - T .
$$

Equivalently, the absorbed power density is $\tfrac{1}{2}\,\omega\,\varepsilon_0\,\mathrm{Im}\,\varepsilon\,|E|^2$,
integrated over the layer (Pettersson, Roman and Inganäs, 1999). PyStratify computes $A_j$ for every layer; the check
under the example confirms that the layers' absorptances, $R$ and $T$ add to one.

::: example perovskite-cell

## Thickness and interference

In a stack this thin the field inside is a standing wave, and the active layer's absorption depends on where its peaks
fall. Thickening the perovskite absorbs more of the red, up to the point where the light is gone before it reaches the
back contact; beyond that the extra thickness only adds collection losses. Weighting $A_\mathrm{active}(\lambda)$ by
the solar photon flux gives the photocurrent a perfect collector would deliver, the figure of merit optical models of
organic and perovskite cells optimise (Burkhard, Hoke and McGehee, 2010).
