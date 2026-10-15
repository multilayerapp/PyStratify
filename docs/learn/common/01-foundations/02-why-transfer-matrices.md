---
id: why-transfer-matrices
title: "Why a 2×2 transfer matrix solves every stratified geometry"
geometry: common
chapter: Foundations
topic: foundations
sources: [plane_wave, dipole, focused]
description: "Why two boundary conditions always give an invertible 2×2 matrix, and how multiplying them solves films, cylinders and spheres alike."
reviewed:
summary: >-
  The one idea behind this site's engine: a second-order wave equation has two independent solutions, so the two
  boundary conditions at an interface always give an invertible 2×2 matrix. Multiplying those matrices across the
  interfaces solves a film, a coated cylinder or a multilayered sphere with any number of layers, for plane waves and
  dipoles alike, and the same construction shows where the numerics bite and how PyStratify keeps them stable.
references:
- cite: F. Abelès, Ann. Phys. (Paris) 12, 596 (1950)
  doi: 10.1051/anphys/195012050596
- cite: P. Yeh, A. Yariv and C.-S. Hong, J. Opt. Soc. Am. 67, 423 (1977)
  doi: 10.1364/JOSA.67.000423
- cite: A. Moroz, Ann. Phys. 315, 352 (2005)
  doi: 10.1016/j.aop.2004.07.002
- cite: I. L. Rasskazov, P. S. Carney and A. Moroz, OSA Continuum 3, 2290 (2020)
  doi: 10.1364/OSAC.399979
- cite: W. Yang, Appl. Opt. 42, 1710 (2003)
  doi: 10.1364/AO.42.001710
- cite: S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720 (2016)
  url: https://arxiv.org/abs/1603.02720
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 2
---

The [previous section](learn/common/maxwell-and-boundary-conditions.html) reduced each geometry to a scalar equation
of the form

$$
\frac{\mathrm{d}^2\Psi}{\mathrm{d}x^2} + k^2(x)\,\Psi = 0
$$

(or its Bessel cousins), with a **piecewise constant** $k$. This section is about the single consequence that makes
the whole site possible: such an equation is solved, in every geometry, by multiplying $2\times 2$ matrices. The
argument takes a paragraph, and it is worth having it clearly in mind, because every later section (the Abelès
matrix of a film, the Mie coefficients of a sphere, the coefficients of a coated cylinder) is one instance of it.

## Two solutions, two conditions, one invertible matrix

A linear second-order equation has exactly two linearly independent solutions in each region, call them $\psi_+$ and
$\psi_-$ (the outgoing and incoming, or regular and singular, waves of the table in the previous section). The field
in region $n$ is

$$
\Psi(x) = A_n\,\psi_+(k_n x) + B_n\,\psi_-(k_n x).
$$

At the interface $x_n$ between regions $n$ and $n+1$ the boundary conditions require continuity of $\Psi$ and of its
"derivative" $\Psi'$ (for electromagnetic waves, of the tangential $\mathbf{E}$ and of the tangential $\mathbf{H}$,
the latter being $\nabla\times\mathbf{E}$ up to an admittance). In matrix form,

$$
\begin{pmatrix} \psi_+(k_n x_n) & \psi_-(k_n x_n) \\ \psi_+'(k_n x_n) & \psi_-'(k_n x_n) \end{pmatrix}
\begin{pmatrix} A_n \\ B_n \end{pmatrix}
=
\begin{pmatrix} \psi_+(k_{n+1} x_n) & \psi_-(k_{n+1} x_n) \\ \psi_+'(k_{n+1} x_n) & \psi_-'(k_{n+1} x_n) \end{pmatrix}
\begin{pmatrix} A_{n+1} \\ B_{n+1} \end{pmatrix}.
$$

The determinant of either matrix is the **Wronskian** $W(\psi_+, \psi_-) = \psi_+\psi_-' - \psi_+'\psi_-$, and the
Wronskian of two independent solutions of a second-order linear equation is never zero (for $e^{\pm ikx}$ it is
$-2ik$; for the Riccati–Bessel pair $\psi_\ell = x j_\ell$, $\xi_\ell = x h^{(1)}_\ell$ it is $i$). Both matrices are
therefore invertible, by nothing more than Cramer's rule for a $2\times 2$ matrix,

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix},
$$

and the amplitudes on one side of an interface determine those on the other side **uniquely**, in either direction:

$$
\begin{pmatrix} A_{n+1} \\ B_{n+1} \end{pmatrix} = T^+_n \begin{pmatrix} A_n \\ B_n \end{pmatrix},
\qquad
\begin{pmatrix} A_n \\ B_n \end{pmatrix} = T^-_n \begin{pmatrix} A_{n+1} \\ B_{n+1} \end{pmatrix},
\qquad T^-_n = (T^+_n)^{-1}.
$$

$T^\pm_n$ is the **transfer matrix** of interface $n$. Nothing in this argument depends on the geometry: the moment a
scalar linear second-order wave equation has been identified, a $2\times 2$ transfer-matrix description is not a
convenient choice but the intrinsic structure of the problem.

## Composition: any number of layers for the price of a product

Because the amplitudes of region $n+1$ feed the matrix of interface $n+1$, the matrices of successive interfaces
multiply. **Raising** carries the core's amplitudes outward,

$$
\begin{pmatrix} A_{n} \\ B_{n} \end{pmatrix} = \mathcal{T}(n) \begin{pmatrix} A_1 \\ B_1 \end{pmatrix},
\qquad \mathcal{T}(n) = T^+_{n-1}\,T^+_{n-2}\cdots T^+_1,
$$

and **lowering** carries the outer medium's amplitudes inward with the inverse product. For $N$ interfaces the cost is
$N$ matrix products per channel: linear in the number of layers, with no matrix larger than $2\times 2$ ever formed.
That is why the calculator treats a 50-period Bragg mirror, a nanomatryoshka with four shells and a bare interface
with the same code and the same speed, and why a multilayered sphere is formally **reduced to a homogeneous one**:
the composite matrix $\mathcal{T}(N+1)$ plays, for the whole onion, the role the single-interface matrix plays for
Mie's sphere, and the cross sections keep Mie's form with composite coefficients.

In the planar case the product is Abelès' 1950 construction. Written for the tangential fields rather than for
the amplitudes it is the **characteristic matrix** of a layer, $\begin{pmatrix}\cos\delta & (i/\eta)\sin\delta \\
i\eta\sin\delta & \cos\delta\end{pmatrix}$, whose product over the layers gives the admittance of a coating; the
[transfer-matrix section](learn/films/transfer-matrix.html) for films develops it. For a periodic stack Yeh, Yariv
and Hong showed that the eigenvalues of the unit cell's matrix decide whether a wave propagates or is stopped, which
is the band structure of a Bragg mirror. For spheres the same products are the recursive algorithm of the
[multilayered sphere](learn/spheres/recursive-transfer-matrix.html); for cylinders, of the
[coated cylinder](learn/cylinders/infinite-cylinder.html).

## Closing the system: the two conditions that are not at an interface

The products relate the $2(N+1)$ amplitudes by $2N$ equations. The remaining two are the physics of the source:

- **Regularity at the centre** (or no wave from beyond the substrate): $B_1 = 0$ for a sphere or cylinder with the
  source outside. The field in the core is then a regular wave of unknown amplitude $A_1$.
- **The incident field**: the regular amplitude in the outermost region is the known plane wave, $A_{N+1} = 1$ in the
  natural normalisation. The outgoing amplitude $B_{N+1}$ is the scattered wave: it is what $\mathcal{T}$ returns, and
  it *is* the Mie coefficient of order $\ell$ (or the Fresnel reflection coefficient, or the cylinder's $a_m$).

A **point dipole** changes only this last step: it adds a known outgoing wave in the region where it sits, so the
amplitudes jump by a prescribed amount at the dipole's radius, and the two conditions become regularity at the centre
and *no incoming wave from infinity*. The same matrices, built once for the structure, then give the fields of a plane
wave, of a dipole anywhere inside or outside, and, by integrating plane waves over a cone, of a focused beam. The
calculator's three sources share one solver for exactly this reason.

## What the matrix gives for free

Since the amplitudes $(A_n, B_n)$ in every region are produced along the way, the transfer matrix returns not just the
far field but the **field everywhere**: the standing-wave pattern inside a filter, the field at the surface of a
sphere where a molecule sits, the power dissipated in each shell (from Poynting's theorem applied region by region).
The channels remain separate, so each multipole's share of the scattering, each polarization's reflectance and each
guided mode's power are read off without any further work. These are the per-layer absorptance, the multipole
decomposition and the emission channels the calculator displays.

## Where it bites, and how PyStratify stays stable

The construction is exact in arithmetic of infinite precision. In floating point two things go wrong, and both are
visible in the matrices above.

1. **Growing exponentials.** In an absorbing layer, or where a wave is evanescent (beyond the critical angle, inside a
   metal, in a sphere's high-order channels where $|m|x \gg \ell$), one of the two solutions grows exponentially across
   the region. The transfer matrix carries both, so its elements grow like $e^{+|k''|d}$ and overflow for a thick metal
   or a large sphere, long before the physics does anything unusual; worse, subtracting two huge numbers to recover a
   small reflected amplitude loses every significant digit. The remedies are all variations on one theme: never
   propagate the growing solution explicitly. Scattering-matrix (Redheffer star-product) formulations only ever combine
   decaying exponentials; Byrnes' planar code, which `pystratify.planar` descends from, does the same for films; for
   spheres and cylinders PyStratify carries the amplitudes together with their **logarithms**, so that the
   Riccati–Bessel functions of complex argument, which grow like $e^{|\mathrm{Im}\, z|}$, are scaled out before any
   product is formed. The sections on the [multilayered sphere](learn/spheres/recursive-transfer-matrix.html) describe
   the recursion that results (Moroz 2005; Yang 2003; Rasskazov, Carney and Moroz 2020).
2. **Truncation.** The sum over channels is infinite. For spheres the series converges once $\ell$ exceeds the size
   parameter by a few terms (Wiscombe's $\ell_\mathrm{max} = x + 4x^{1/3} + 2$), but a high-index core, a thin metal
   shell or a dipole close to the surface needs more, because the field it radiates is rich in high orders;
   PyStratify therefore raises the order to cover the largest internal size parameter and, for emitters, until the
   rate converges. Keeping orders is cheap; dropping them silently is how a code reproduces Mie's figures and still
   gets a quenching curve wrong.

Both points are tested, not assumed: the [Verification](verification.html) page lists the extended-precision
references (the same recursion in 60-digit arithmetic), the independent codes and the closed forms against which every
geometry's transfer matrices are checked.
