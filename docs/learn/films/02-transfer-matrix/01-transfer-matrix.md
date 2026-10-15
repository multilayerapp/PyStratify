---
id: transfer-matrix
title: "The transfer matrix: a stack of thin films"
geometry: films
chapter: The transfer matrix
topic: tmm
sources: [plane_wave]
description: "The characteristic matrix of a thin film, how a stack multiplies them, admittance, the field inside, and the numerics of thick or absorbing layers."
reviewed:
summary: >-
  Abelès' characteristic matrix of a layer, derived from the two boundary conditions; how a stack multiplies them;
  the admittance that gives R and T and Macleod's admittance locus; the amplitude form with interface and propagation
  matrices that also gives the field inside; quarter-wave and half-wave layers and Herpin's equivalent layer; and the
  numerics: growing exponentials, the scattering-matrix alternative, and what PyStratify does.
examples: [quarter-wave-ar]
references:
- cite: F. Abelès, Ann. Phys. (Paris) 12, 596 (1950)
  doi: 10.1051/anphys/195012050596
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 2–3
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §1.6
  doi: 10.1017/CBO9781139644181
- cite: P. Yeh, Optical Waves in Layered Media (Wiley, 1988), ch. 5
- cite: A. Herpin, C. R. Acad. Sci. Paris 225, 182 (1947)
- cite: L. Li, J. Opt. Soc. Am. A 13, 1024 (1996)
  doi: 10.1364/JOSAA.13.001024
- cite: S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720 (2016)
  url: https://arxiv.org/abs/1603.02720
---

Between two interfaces a wave travels back and forth, picking up phase on each pass, and every reflection spawns
another. Summing those multiple reflections term by term is possible for one layer (it gives the Airy formula of the
[Fabry–Pérot](learn/films/fabry-perot-etalons.html) section) and impractical for twenty. The transfer matrix does
the bookkeeping instead: in each layer the field is a forward and a backward wave, and a $2\times 2$ matrix carries
either the two amplitudes or the two tangential fields from one side of the layer to the other. The
[Foundations](learn/common/why-transfer-matrices.html) chapter explains why such a matrix must exist for any
stratified geometry; this section writes it out for films, the way Abelès did in 1950, and shows what it is good for.

## The characteristic matrix of a layer

Take s polarization (p differs only in the admittance) and a layer $j$ of index $n_j$ and thickness $d_j$ at the
in-plane wavenumber $k_x = k_0 n_0\sin\theta_0$ fixed by the incidence. Inside the layer the field is
$E_y(z) = a\,e^{ik_z z} + b\,e^{-ik_z z}$ with $k_z = k_0 n_j\cos\theta_j$, and the tangential magnetic field, in
units that make the admittance of free space 1, is $H_x(z) = \eta_j\,(a\,e^{ik_z z} - b\,e^{-ik_z z})$ with the
**tilted admittance** $\eta_j = n_j\cos\theta_j$. Eliminate $a$ and $b$ between the values at the two faces, a
distance $d_j$ apart, and the result is a matrix relation between the tangential fields themselves:

$$
\begin{pmatrix} E \\ H \end{pmatrix}_{\text{front}} =
\begin{pmatrix} \cos\delta_j & \dfrac{i}{\eta_j}\sin\delta_j \\[4pt] i\,\eta_j \sin\delta_j & \cos\delta_j \end{pmatrix}
\begin{pmatrix} E \\ H \end{pmatrix}_{\text{back}},
\qquad
\delta_j = \frac{2\pi}{\lambda}\, n_j d_j \cos\theta_j ,
$$

Abelès' **characteristic matrix** $M_j$, with the phase thickness $\delta_j$ and, for p polarization,
$\eta_j = n_j/\cos\theta_j$. Its determinant is 1 (it is a product of two matrices of unit Wronskian), its trace is
$2\cos\delta_j$, and it depends on the layer through $\delta_j$ and $\eta_j$ only: two numbers per layer per
wavelength, which is why a stack of forty layers is no harder than one.

## A stack and its admittance

The tangential fields are continuous at every interface, so what leaves the back of layer $j$ enters the front of
layer $j+1$ unchanged, and the matrices of successive layers simply multiply. In the substrate there is only a
transmitted wave, whose fields are in the ratio $H/E = \eta_s$, so for $q$ layers on a substrate

$$
\begin{pmatrix} B \\ C \end{pmatrix} = \left(\prod_{j=1}^{q} M_j\right) \begin{pmatrix} 1 \\ \eta_s \end{pmatrix},
\qquad Y = \frac{C}{B} .
$$

$Y$ is the **admittance of the whole stack**: seen from the incident medium of admittance $\eta_0$, the coated
substrate behaves exactly like a single surface of admittance $Y$, and the single-interface formulas of the
[Fresnel section](learn/films/fresnel-equations.html) apply to it:

$$
r = \frac{\eta_0 - Y}{\eta_0 + Y}, \qquad R = |r|^2, \qquad
T = \frac{4\,\eta_0\,\mathrm{Re}\,\eta_s}{|\eta_0 B + C|^2}, \qquad A = 1 - R - T .
$$

This is the formulation of Macleod's book and of coating design in general, and its power is that $Y$ can be followed
layer by layer. Macleod's **admittance locus** plots $Y$ in the complex plane as each layer is added: a lossless
layer moves $Y$ along a circle centred on the real axis, through an angle $2\delta$, and a design is a path from
$\eta_s$ to the target $Y = \eta_0$ (zero reflectance) or as far from it as possible (a mirror). A designer reads
off from the diagram which layer to add next and how thick, which is how the classic antireflection and filter
designs below were found long before computers.

## The amplitude form: the field inside

The same physics can be written for the wave amplitudes instead of the fields, which is what Byrnes's code and
PyStratify's planar engine do. Crossing an interface between media $i$ and $j$ maps the forward and backward
amplitudes by an **interface matrix** built from the Fresnel coefficients, and crossing a layer by a diagonal
**propagation matrix**:

$$
\begin{pmatrix} a_i \\ b_i \end{pmatrix} = \frac{1}{t_{ij}}
\begin{pmatrix} 1 & r_{ij} \\ r_{ij} & 1 \end{pmatrix}
\begin{pmatrix} a_j \\ b_j \end{pmatrix},
\qquad
P_j = \begin{pmatrix} e^{-i\delta_j} & 0 \\ 0 & e^{i\delta_j} \end{pmatrix} .
$$

The product $\tilde M = I_{01} P_1 I_{12} P_2 \cdots I_{q s}$ relates the incident-medium amplitudes $(1, r)$ to the
substrate's $(t, 0)$, so $r = \tilde M_{21}/\tilde M_{11}$ and $t = 1/\tilde M_{11}$. The two forms are equivalent
(a change of basis in each layer), but the amplitude form hands over the forward and backward amplitudes in every
layer as a by-product. From them the field at any depth,

$$
E_y(z) = a_j e^{ik_{z,j}(z - z_j)} + b_j e^{-ik_{z,j}(z - z_j)} ,
$$

is a one-line evaluation, and the power absorbed in layer $j$ is the difference of the Poynting flux through its two
faces, the per-layer absorptance of the [solar-cell section](learn/films/solar-cell-absorption.html). The
calculator's field profile and layer absorptances come from exactly these amplitudes.

## Quarter waves and half waves

Two thicknesses make the characteristic matrix trivial at the design wavelength $\lambda_0$, at normal incidence:

- a **half-wave** layer ($n d = \lambda_0/2$, $\delta = \pi$) has $M = -I$: it is *absent*, and the stack reflects
  as if it were not there (which is why the spacer of a [bandpass filter](learn/films/bandpass-filters.html) is a
  half wave);
- a **quarter-wave** layer ($n d = \lambda_0/4$, $\delta = \pi/2$) has $M = \begin{pmatrix} 0 & i/\eta \\ i\eta & 0
  \end{pmatrix}$ and turns an admittance $Y$ into $\eta^2/Y$: an *admittance transformer*, the optical
  quarter-wave line.

A single quarter-wave layer of index $n_1$ on a substrate $n_s$ therefore presents $Y = n_1^2/n_s$ and gives

$$
R = \left(\frac{n_0 n_s - n_1^2}{n_0 n_s + n_1^2}\right)^2 ,
$$

which vanishes for $n_1 = \sqrt{n_0 n_s}$, 1.23 for glass in air. No durable material has so low an index; magnesium
fluoride (1.38) comes closest, and leaves about 1.3 % instead of 4 %.

::: example quarter-wave-ar

The minimum sits at $\lambda_0 = 4 n_1 d$ and is broad, because $\delta$ changes slowly with $\lambda$. Two or three
layers (the V-coat and the quarter–half–quarter design) broaden it further, and a graded index takes the idea to its
limit in the [moth-eye antireflection](learn/films/moth-eye-antireflection.html) section.

**Herpin's equivalent layer.** Any symmetric combination of layers, $p\,q\,p$ say, has a characteristic matrix of the
same form as a single layer, with an *equivalent* index and phase thickness that depend on wavelength (Herpin 1947;
Macleod ch. 6). A combination of two available materials can therefore stand in for an index that no material
offers, including the 1.23 above, and a periodic stack is a periodic repetition of one equivalent layer, which is the
shortest route to the stop band of the [Bragg mirror](learn/films/bragg-mirrors.html). The equivalent index becomes
imaginary exactly inside the stop band.

## Oblique incidence and absorbing layers

Nothing above assumes normal incidence or real indices. At an angle, $\cos\theta_j$ shortens each layer's phase
thickness, so every feature of a spectrum moves to shorter wavelengths as the coating is tilted, and the admittances
of s and p separate, so the two polarizations respond differently (the p admittance $n/\cos\theta$ grows with angle,
the s admittance shrinks). In an absorbing layer $\delta_j$ is complex, $\cos\delta_j$ and $\sin\delta_j$ become
hyperbolic, and the wave decays across the layer; the formulas stay valid, and $A = 1 - R - T$ is now positive.

## Numerics: where the matrix overflows, and what to do

For a thick absorbing layer, or an evanescent one beyond the critical angle, $e^{i\delta_j}$ contains a real
exponential $e^{\mathrm{Im}\,\delta_j}$ that grows without bound: 10 µm of a metal at visible wavelengths gives
$e^{+700}$, beyond double precision. The characteristic matrix carries both $e^{+}$ and $e^{-}$ and overflows, and
even before that the reflected amplitude is the small difference of two huge numbers. Three remedies are in use:

1. **Propagate only decaying waves.** Byrnes's formulation orders the exponentials so that the growing one never
   appears in a product that is later subtracted, and clamps a layer beyond which nothing returns as opaque; this is
   what `pystratify.planar` does, and it is why the calculator can hold a millimetre of metal.
2. **The scattering matrix.** Instead of transfer matrices, combine the reflection and transmission matrices of
   sub-stacks with Redheffer's star product. Every quantity is then a reflection or transmission coefficient, bounded
   by one for a passive stack, and no exponential ever grows (Li 1996, who proved the stability). PyStratify's
   [Verification](verification.html) page compares the planar engine with an independent scattering-matrix solver on
   lossy, oblique and opaque stacks to $10^{-10}$.
3. **Logarithmic amplitudes.** Carry $\log a_j$ and $\log b_j$ rather than $a_j$ and $b_j$, so that a factor
   $e^{700}$ is the number 700; this is how PyStratify's sphere and cylinder solvers stay stable to any thickness and
   order, as the [multilayered sphere](learn/spheres/recursive-transfer-matrix.html) section describes.

The second trap is subtler: the **branch of $\cos\theta_j$** in an absorbing or evanescent layer. Taking the wrong
sign makes the "forward" wave grow into the layer, and the error does not announce itself; it shows up as $R > 1$ or
as $T$ failing to vanish beyond the critical angle. The forward-decaying branch of the
[Fresnel section](learn/films/fresnel-equations.html) must be chosen in every layer, and the energy-conservation
checks ($R + T = 1$ below and beyond the critical angle for lossless stacks) are the tests that catch a wrong branch.
