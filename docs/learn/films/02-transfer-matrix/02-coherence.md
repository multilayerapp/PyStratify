---
id: coherence-and-thick-substrates
title: Coherence and thick substrates
geometry: films
chapter: The transfer matrix
topic: coherence
sources: [plane_wave]
description: "Why thin films show fringes and thick substrates do not: coherence length, incoherent layers as a phase average, and partially coherent stacks."
reviewed:
summary: >-
  Why a thin film draws interference fringes and a millimetre-thick substrate does not: the coherence length and the
  fringe spacing, the incoherent sum of intensities derived as a phase average, the intensity transfer matrix that
  joins coherent sub-stacks through incoherent layers, the asymmetry R ≠ R′ of an absorbing film, partial coherence,
  and what a spectrometer actually records.
classics: [katsidis-2002]
examples: [film-on-thick-glass]
references:
- cite: C. C. Katsidis and D. I. Siapkas, Appl. Opt. 41, 3978 (2002)
  doi: 10.1364/AO.41.003978
- cite: B. Harbecke, Appl. Phys. B 39, 165 (1986)
  doi: 10.1007/BF00697414
- cite: C. L. Mitsas and D. I. Siapkas, Appl. Opt. 34, 1678 (1995)
  doi: 10.1364/AO.34.001678
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §7.6 and ch. 10
  doi: 10.1017/CBO9781139644181
- cite: S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720 (2016)
  url: https://arxiv.org/abs/1603.02720
---

The transfer matrix adds amplitudes: every reflected wave interferes with every other. That is right for a film a few
hundred nanometres thick, and wrong for the glass it sits on. A 1 mm plate in a spectrometer shows no fringes, though
the matrix predicts them, with a contrast of nearly 8 % for glass. The discrepancy is not a failure of Maxwell's
equations; it is a statement about the light and the instrument, and once it is made precise it becomes a second
$2\times 2$ formalism, this time for intensities, that joins seamlessly to the first.

## Fringes too fine to see

Two reflections that differ by a path $2 n d \cos\theta$ interfere only if the light stays coherent over that path.
A source of bandwidth $\Delta\lambda$ has a coherence length

$$
\ell_c \approx \frac{\lambda^2}{\Delta\lambda},
$$

30 µm for a 10 nm bandwidth at 550 nm. Put the other way, the fringes of a slab are spaced by

$$
\Delta\lambda_\text{fringe} = \frac{\lambda^2}{2 n d \cos\theta},
$$

which for 1 mm of glass at 600 nm is 0.12 nm. Any instrument with a coarser resolution, nearly all of them, averages
over many fringes and records their mean. Three other averages act in the same direction: a real plate varies in
thickness by more than a wavelength across the beam, the beam has a spread of angles (a cone of a few degrees
already smears $\cos\theta$ by more than a fringe), and a thermal source's phase is random on a timescale set by its
bandwidth. Each of them averages the phase $\delta = 2\pi n d\cos\theta/\lambda$ uniformly over $2\pi$, and that is
the operational definition of an **incoherent layer**: one whose round-trip phase is, as far as the measurement can
tell, uniformly random.

## Adding intensities: the phase average

Take the Airy sum for a slab, $t = t_{12}t_{21}e^{i\delta}/(1 - r_{21}^2 e^{2i\delta})$, and average $|t|^2$ over
$\delta$. The cross terms between passes with different numbers of round trips carry $e^{2im\delta}$, $m \ne 0$, and
average to zero; what survives is the sum of the powers of the individual passes,

$$
\langle T \rangle_\delta = T_{12}\,T_{21}\sum_{m=0}^{\infty} R_{21}^{2m} = \frac{T_{12}T_{21}}{1 - R_{21}^2},
$$

a geometric series in powers, not amplitudes. For a bare, non-absorbing plate with face reflectance
$R_0 = \left(\frac{n - 1}{n + 1}\right)^2$ this gives

$$
R = R_0 + \frac{(1 - R_0)^2 R_0}{1 - R_0^2} = \frac{2R_0}{1 + R_0}, \qquad T = \frac{1 - R_0}{1 + R_0} ,
$$

8.2 % and 91.8 % for glass: twice the single-surface loss, less a little. The phase average and the "add
intensities" rule are the same statement; PyStratify's [Verification](verification.html) page checks the engine's
incoherent path against an explicit average of coherent solves over the layer's phase, and they agree to $10^{-9}$.

## Coherent films on incoherent plates

Harbecke (1986), Mitsas and Siapkas (1995) and Katsidis and Siapkas (2002) generalised this to any stack. The layers
are grouped into **coherent sub-stacks**, each solved with the transfer matrix of amplitudes, which yields its
reflectance and transmittance from the front, $R$, $T$, and from the back, $R'$, $T'$. The sub-stacks are then joined
through the incoherent layers by $2\times 2$ **intensity matrices**: for a sub-stack between two incoherent regions

$$
\mathcal{I} = \frac{1}{T}\begin{pmatrix} 1 & -R' \\ R & T T' - R R' \end{pmatrix},
$$

which carries the forward and backward *powers* across it exactly as the characteristic matrix carries the fields,
and the propagation through an incoherent layer of absorptance is a diagonal matrix of its single-pass attenuation
$e^{\mp 2\,\mathrm{Im}\,\delta}$. A thin film on a thick substrate is then exact for both: the film's fringes survive,
and the substrate contributes its two faces in power. Two features of the formula deserve notice. **$R \ne R'$** for
an absorbing sub-stack (a dielectric film on an absorbing one reflects differently from the two sides, though
$T = T'$ always, by reciprocity), so the back-side coefficients cannot be dropped. And an incoherent layer's own
thickness enters only through its absorption: a lossless incoherent plate's thickness has no effect on $R$ and $T$ at
all, which is why a measured spectrum cannot tell a 1 mm plate from a 2 mm one.

PyStratify's planar engine implements this scheme (Byrnes's `inc_tmm`); in the calculator each layer carries a
coherence flag (*c* or *i*), the substrate behind the last layer is semi-infinite, and a thick substrate is written
as an incoherent layer followed by air.

::: example film-on-thick-glass

At the film's half-wave wavelength it is absent, and the reflectance returns to the bare plate's $2R_0/(1 + R_0)$.
Between those points the film's fringes ride on the plate's incoherent background, which is what a measured spectrum
looks like. A coherent calculation of the same stack would show the film's fringes modulated by a comb of 0.12 nm
fringes of nearly 8 % depth, and would have to be computed on a grid finer than 0.03 nm to resolve them at all,
which is the practical reason the incoherent treatment matters even when the instrument's resolution is known.

## Partial coherence

Between the two limits sit rough interfaces, thickness gradients and finite spectral resolution, which damp fringes
without removing them. Three models are in use, and each is a different average of the coherent result:

- **spectral resolution**: convolve the coherent spectrum with the instrument's line shape; exact for a perfect
  plane-parallel plate measured with a known slit, and the right model for a thin wafer in a high-resolution FTIR,
  where the fringes of 500 µm of silicon are resolved and routinely used to measure thickness;
- **thickness variation**: average the coherent result over the distribution of $d$ across the beam, which damps the
  fringes by the Fourier transform of that distribution; a Gaussian spread of $\sigma_d$ multiplies the fringe
  contrast by $\exp[-2(2\pi n\sigma_d/\lambda)^2]$;
- **phase averaging within a layer**, Katsidis and Siapkas' formulation, in which a layer is assigned a degree of
  coherence between 0 and 1 that interpolates between the amplitude and the intensity matrices.

Their infrared reflectance of a thin film on a thick wafer is the textbook case of fringes riding on an incoherent
background, with the semi-infinite and the finite incoherent substrate side by side.

::: classic katsidis-2002

## What the model is not

An incoherent layer is a model of the measurement, not of the light inside the plate: the fields in the glass are as
coherent as anywhere else, and a different instrument (a laser and a photodiode, say) would see the fringes. The
treatment also assumes the incoherent layer is thick enough for the multiply reflected beams to be laterally
separated or phase-randomised; a layer of a few wavelengths that is simply rough belongs to the scattering models of
the [textured surfaces](learn/films/textured-surfaces.html) section. And a stack with *several* incoherent layers is
handled exactly by the intensity matrices, but the engine's only independent check for that case is the explicit
phase average, as the Verification page states.
