---
id: transfer-matrix
title: "The transfer matrix: a stack of thin films"
geometry: films
chapter: The transfer matrix
topic: tmm
sources: [plane_wave]
summary: >-
  Abelès' characteristic matrix of a layer, how a stack multiplies them, the admittance that gives R and T, and the
  quarter-wave and half-wave layers that coating design is built on.
examples: [quarter-wave-ar]
references:
- cite: F. Abelès, Ann. Phys. (Paris) 12, 596 (1950)
  doi: 10.1051/anphys/195012050596
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 2
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §1.6
  doi: 10.1017/CBO9781139644181
- cite: S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720 (2016)
  url: https://arxiv.org/abs/1603.02720
---

Between two interfaces a wave travels back and forth, picking up phase on each pass, and every reflection spawns
another. Summing those multiple reflections term by term is possible for one layer, but impractical for twenty. The
transfer matrix does the bookkeeping instead: in each layer the field is a forward and a backward wave, and a $2\times
2$ matrix carries the two amplitudes, or the tangential fields, from one side of the layer to the other.

## The characteristic matrix of a layer

In layer $j$, of index $n_j$ and thickness $d_j$, the phase thickness at vacuum wavelength $\lambda$ is

$$
\delta_j = \frac{2\pi}{\lambda}\, n_j d_j \cos\theta_j ,
$$

with $\theta_j$ from Snell's law. The tangential electric and magnetic fields at the layer's two faces are related by
Abelès' characteristic matrix

$$
\begin{pmatrix} E \\ H \end{pmatrix}_{\text{front}} =
\begin{pmatrix} \cos\delta_j & \dfrac{i}{\eta_j}\sin\delta_j \\[4pt] i\,\eta_j \sin\delta_j & \cos\delta_j \end{pmatrix}
\begin{pmatrix} E \\ H \end{pmatrix}_{\text{back}},
$$

where $\eta_j$ is the layer's tilted admittance, in units of the admittance of free space:
$\eta_j = n_j\cos\theta_j$ for s polarization and $\eta_j = n_j/\cos\theta_j$ for p.

## A stack and its admittance

Because the tangential fields are continuous at every interface, the matrices of successive layers simply multiply.
For $q$ layers on a substrate of admittance $\eta_s$,

$$
\begin{pmatrix} B \\ C \end{pmatrix} = \left(\prod_{j=1}^{q} M_j\right) \begin{pmatrix} 1 \\ \eta_s \end{pmatrix},
\qquad Y = \frac{C}{B},
$$

and the whole stack behaves like a single surface of admittance $Y$ seen from the incident medium $\eta_0$:

$$
r = \frac{\eta_0 - Y}{\eta_0 + Y}, \qquad R = |r|^2, \qquad
T = \frac{4\,\eta_0\,\mathrm{Re}\,\eta_s}{|\eta_0 B + C|^2}.
$$

PyStratify's planar engine uses the equivalent form in amplitudes (an interface matrix and a propagation matrix per
layer, after Byrnes 2016), which also returns the fields inside the stack and the power absorbed in each layer.

## Quarter waves and half waves

Two thicknesses make the matrix trivial at the design wavelength $\lambda_0$, at normal incidence:

- a **half-wave** layer ($n d = \lambda_0/2$, $\delta = \pi$) has $M = -I$: it is *absent*, and the stack reflects
  as if it were not there;
- a **quarter-wave** layer ($n d = \lambda_0/4$, $\delta = \pi/2$) turns an admittance $Y$ into $n^2/Y$.

A single quarter-wave layer of index $n_1$ on a substrate $n_s$ therefore gives

$$
R = \left(\frac{n_0 n_s - n_1^2}{n_0 n_s + n_1^2}\right)^2 ,
$$

which vanishes for $n_1 = \sqrt{n_0 n_s}$, 1.23 for glass in air. No durable material has so low an index; magnesium
fluoride (1.38) comes closest, and leaves about 1.3 % instead of 4 %.

::: example quarter-wave-ar

The minimum sits at $\lambda_0 = 4 n_1 d$ and is broad, because $\delta$ changes slowly with $\lambda$. Two or three
layers (the V-coat and the quarter–half–quarter design) broaden it further, and a graded index takes the idea to its
limit in the [moth-eye antireflection](learn/films/moth-eye-antireflection.html) section.

## Oblique incidence and absorbing layers

Nothing above assumes normal incidence or real indices: at an angle, $\cos\theta_j$ shortens each layer's phase
thickness and the admittances differ between s and p. In an absorbing layer $\delta_j$ is complex and the wave decays
across it. For a thick metal or an evanescent layer the growing exponential in the matrix elements can exceed floating
point range; amplitude formulations that propagate decaying waves only avoid that.
