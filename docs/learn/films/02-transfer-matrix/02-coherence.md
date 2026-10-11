---
id: coherence-and-thick-substrates
title: Coherence and thick substrates
geometry: films
chapter: The transfer matrix
topic: coherence
sources: [plane_wave]
summary: >-
  Why a thin film draws interference fringes and a millimetre-thick substrate does not: the coherence length, the
  incoherent sum of intensities, and how a stack mixes coherent and incoherent layers.
examples: [film-on-thick-glass]
references:
- cite: C. C. Katsidis and D. I. Siapkas, Appl. Opt. 41, 3978 (2002)
  doi: 10.1364/AO.41.003978
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §7.6
  doi: 10.1017/CBO9781139644181
- cite: S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720 (2016)
  url: https://arxiv.org/abs/1603.02720
---

The transfer matrix adds amplitudes: every reflected wave interferes with every other. That is right for a film a few
hundred nanometres thick, and wrong for the glass it sits on. A 1 mm plate in a spectrometer shows no fringes, though
the matrix predicts them.

## Fringes too fine to see

Two reflections that differ by a path $2 n d \cos\theta$ interfere only if the light stays coherent over that path.
A source of bandwidth $\Delta\lambda$ has a coherence length

$$
\ell_c \approx \frac{\lambda^2}{\Delta\lambda}.
$$

Put the other way, the fringes of a slab are spaced by

$$
\Delta\lambda_\text{fringe} = \frac{\lambda^2}{2 n d \cos\theta},
$$

which for 1 mm of glass at 600 nm is 0.12 nm. Any instrument with a coarser resolution, nearly all of them, averages
over many fringes and records their mean. Real plates also vary in thickness by more than a wavelength across the beam,
which averages the phase just as well.

## Adding intensities

When the phase is averaged out, the multiply reflected waves add in power, not amplitude. For a bare, non-absorbing
plate with face reflectance $R_0 = \left(\frac{n - 1}{n + 1}\right)^2$, the geometric series gives

$$
R = R_0 + \frac{(1 - R_0)^2 R_0}{1 - R_0^2} = \frac{2R_0}{1 + R_0}, \qquad T = \frac{1 - R_0}{1 + R_0} ,
$$

8.2 % and 91.8 % for glass: twice the single-surface loss, less a little.

## Coherent films on incoherent plates

Katsidis and Siapkas (2002) generalised this to any stack: the layers are grouped into coherent sub-stacks, each
solved with the transfer matrix of amplitudes, and joined through incoherent layers by $2 \times 2$ matrices of
intensities. A thin film on a thick substrate is then exact for both: the film's fringes survive, and the substrate
contributes its two faces in power. PyStratify's planar engine implements this scheme; in the calculator each layer
carries a coherence flag (*c* or *i*), and the substrate behind the last layer is semi-infinite.

::: example film-on-thick-glass

At the film's half-wave wavelength it is absent, and the reflectance returns to the bare plate's $2R_0/(1 + R_0)$.
Between those points the film's fringes ride on the plate's incoherent
background, which is what a measured spectrum looks like.

## Partial coherence

Between the two limits sit rough interfaces, thickness gradients and finite spectral resolution, which damp fringes
without removing them. Katsidis and Siapkas treat these as phase averaging within a layer;
[Classics](classics.html#katsidis-2002) opens their coherent and incoherent wafer of Fig. 2 in the calculator.
