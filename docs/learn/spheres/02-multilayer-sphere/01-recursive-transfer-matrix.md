---
id: recursive-transfer-matrix
title: "The multilayered sphere: a recursive transfer matrix"
geometry: spheres
chapter: The multilayered sphere
topic: multilayer-sphere
sources: [plane_wave, dipole]
classics: [prodan-2003]
description: "Mie theory for any number of concentric shells: a 2×2 matrix per interface, stable numerics for thin shells, and plasmon hybridization in nanoshells."
reviewed:
summary: >-
  Mie's solution extended to any number of concentric shells: the fields of each shell as regular and outgoing
  multipoles, the 2×2 transfer matrix of one interface written in Riccati–Bessel functions, raising and lowering
  through the shells, the two conditions that close the system, and the numerical discipline (logarithmic amplitudes,
  ratios of Bessel functions) that makes thousands of orders and nanometre shells safe. Plasmon hybridization in a
  nanoshell as the first application.
references:
- cite: A. L. Aden and M. Kerker, J. Appl. Phys. 22, 1242 (1951)
  doi: 10.1063/1.1699834
- cite: A. Moroz, Ann. Phys. 315, 352 (2005)
  doi: 10.1016/j.aop.2004.07.002
- cite: I. L. Rasskazov, P. S. Carney and A. Moroz, OSA Continuum 3, 2290 (2020)
  doi: 10.1364/OSAC.399979
- cite: W. Yang, Appl. Opt. 42, 1710 (2003)
  doi: 10.1364/AO.42.001710
- cite: O. Peña and U. Pal, Comput. Phys. Commun. 180, 2348 (2009)
  doi: 10.1016/j.cpc.2009.07.010
- cite: E. Prodan, C. Radloff, N. J. Halas and P. Nordlander, Science 302, 419 (2003)
  doi: 10.1126/science.1089171
- cite: C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley, 1983), §8.1
  doi: 10.1002/9783527618156
---

Aden and Kerker solved the coated sphere in 1951 by writing out the boundary conditions at both interfaces and
inverting the resulting $4\times 4$ system. For a nanomatryoshka with four shells that route gives an $8\times 8$
system, for a graded profile sampled in a hundred steps it is hopeless, and in every case the direct inversion is
numerically fragile. The transfer-matrix formulation of the [Foundations](learn/common/why-transfer-matrices.html)
chapter does better: one $2\times 2$ matrix per interface and per channel, multiplied inward or outward, with the
same code for a plane wave and for a dipole inside or outside any shell. This section writes that recursion down in the
form PyStratify implements, following Moroz (2005) and the STRATIFY paper.

## The fields in each shell

Number the regions from the core outward, $n = 1, \dots, N + 1$, the host being $N + 1$; region $n$ has index $n_n$,
wavenumber $k_n = n_n k_0$ and permeability $\mu_n$, and the interface between $n$ and $n + 1$ sits at $r_n$. In each
region the electric field is a sum over the composite angular index $L = (\ell, m)$ and the two polarizations
$p = M$ (magnetic, TE) and $p = E$ (electric, TM) of a regular and an outgoing multipole,

$$
\mathbf{E}(\mathbf{r}) = \sum_{p, L}\left[A_{pL}(n)\,\mathbf{J}_{pL}(k_n, \mathbf{r}) + B_{pL}(n)\,\mathbf{H}_{pL}(k_n, \mathbf{r})\right],
$$

where $\mathbf{J}$ and $\mathbf{H}$ are the vector multipole fields built on $j_\ell$ and $h^{(1)}_\ell$ with the
normalised vector spherical harmonics $\mathbf{Y}^{(m)}_L$, $\mathbf{Y}^{(e)}_L$, $\mathbf{Y}^{(o)}_L$ (Jackson's; Bohren
and Huffman's $\mathbf{M}$, $\mathbf{N}$ give the same results). The magnetic field follows from
$\mathbf{H} = -i\sqrt{\varepsilon_n/\mu_n}\,\nabla\times\mathbf{E}/k_n$, which swaps $\mathbf{J}_{ML}$ and
$\mathbf{J}_{EL}$: the magnetic multipole's $\mathbf{H}$ looks like an electric multipole's $\mathbf{E}$, and vice
versa. The amplitudes $A_{pL}(n)$, $B_{pL}(n)$ are the unknowns, two per region, per $\ell$, per polarization. Because
the harmonics are orthogonal and the problem has spherical symmetry, nothing couples different $L$ or different $p$:
every channel $(p, \ell)$ is a scalar problem of the Foundations type, and $m$ never appears in the matrices.

## The transfer matrix of one interface

Continuity of the tangential $\mathbf{E}$ and $\mathbf{H}$ at $r_n$ links the amplitudes on both sides. With the
Riccati–Bessel functions $\psi_\ell(z) = z\, j_\ell(z)$, $\zeta_\ell(z) = z\, h^{(1)}_\ell(z)$, the internal size
parameter $x_n = k_n r_n$, the external one $\tilde{x}_n = k_{n+1} r_n$, and the relative index and permeability
$\tilde{\eta}_n = n_n / n_{n+1}$, $\tilde{\mu}_n = \mu_n/\mu_{n+1}$, the **lowering** matrix that carries the outer
amplitudes to the inner ones for a magnetic multipole is

$$
\begin{pmatrix} A_{ML}(n) \\ B_{ML}(n) \end{pmatrix} = T^-_{M\ell}(n) \begin{pmatrix} A_{ML}(n+1) \\ B_{ML}(n+1) \end{pmatrix},
\qquad
T^-_{M\ell}(n) = -i
\begin{pmatrix}
\tilde{\eta}\,\zeta_\ell'(x)\,\psi_\ell(\tilde{x}) - \tilde{\mu}\,\zeta_\ell(x)\,\psi_\ell'(\tilde{x})
& \tilde{\eta}\,\zeta_\ell'(x)\,\zeta_\ell(\tilde{x}) - \tilde{\mu}\,\zeta_\ell(x)\,\zeta_\ell'(\tilde{x}) \\
-\tilde{\eta}\,\psi_\ell'(x)\,\psi_\ell(\tilde{x}) + \tilde{\mu}\,\psi_\ell(x)\,\psi_\ell'(\tilde{x})
& -\tilde{\eta}\,\psi_\ell'(x)\,\zeta_\ell(\tilde{x}) + \tilde{\mu}\,\psi_\ell(x)\,\zeta_\ell'(\tilde{x})
\end{pmatrix},
$$

with $x = x_n$, $\tilde{x} = \tilde{x}_n$ and the subscript $n$ dropped on $\tilde{\eta}$, $\tilde{\mu}$. The electric
matrix $T^-_{E\ell}$ is the same expression with $\tilde{\eta}$ and $\tilde{\mu}$ interchanged, and the **raising**
matrices $T^+ = (T^-)^{-1}$ have the same shape with the roles of $x$ and $\tilde{x}$ exchanged and $\tilde{\eta}$,
$\tilde{\mu}$ in the denominators. The prefactor $-i$ is the inverse Wronskian $\psi_\ell\zeta_\ell' -
\psi_\ell'\zeta_\ell = i$, which is what makes the matrix exactly invertible; $\det T^\pm = \tilde{\eta}\tilde{\mu}$
up to the normalisation of the fields, never zero.

## Raising, lowering, and the two closing conditions

Multiplying the matrices of successive interfaces moves the pair of amplitudes through the whole onion:

$$
\begin{pmatrix} A_{pL}(n) \\ B_{pL}(n) \end{pmatrix} = \mathcal{T}_{p\ell}(n)\begin{pmatrix} A_{pL}(1) \\ B_{pL}(1) \end{pmatrix},
\quad \mathcal{T}_{p\ell}(n) = \prod_{j = n-1}^{1} T^+_{p\ell}(j);
\qquad
\begin{pmatrix} A_{pL}(n) \\ B_{pL}(n) \end{pmatrix} = \mathcal{M}_{p\ell}(n)\begin{pmatrix} A_{pL}(N+1) \\ B_{pL}(N+1) \end{pmatrix},
\quad \mathcal{M}_{p\ell}(n) = \prod_{j = n}^{N} T^-_{p\ell}(j),
$$

with $\mathcal{M}_{p\ell}(1) = \mathcal{T}_{p\ell}(N+1)^{-1}$. For $N$ interfaces these are $2N$ equations for
$2(N+1)$ amplitudes per channel; two more conditions close the system:

- **regularity at the centre**: no outgoing (singular) wave emerges from the origin unless a source sits there, so
  $B_{pL}(1) = 0$;
- **the source**: for a plane wave, the regular amplitude in the host is the known expansion of the incident wave,
  $A_{pL}(N+1) = A^{\mathrm{inc}}_{pL}$, and the outgoing amplitude $B_{pL}(N+1)$ is the scattered wave.

Writing $\mathcal{T}_{p\ell}(N+1) = \begin{pmatrix} \mathcal{T}_{11} & \mathcal{T}_{12} \\ \mathcal{T}_{21} &
\mathcal{T}_{22}\end{pmatrix}$ and applying both conditions gives the scattered amplitude in one line,

$$
B_{pL}(N+1) = \frac{\mathcal{T}_{21}}{\mathcal{T}_{11}}\, A_{pL}(N+1) ,
$$

so the whole multilayered sphere has been reduced to a homogeneous one with **composite Mie coefficients**
$a_\ell, b_\ell = \mathcal{T}_{21}/\mathcal{T}_{11}$ (up to the sign convention of the outgoing wave). Every far-field
formula of [Mie theory](learn/spheres/mie-scattering.html), the efficiencies, the scattering pattern, the
[multipole decomposition](learn/spheres/multipoles.html), the Kerker conditions, holds unchanged with these
coefficients; the near field in any shell is read off from the amplitudes $\mathcal{T}(n)$ produced on the way. A
**dipole source** in shell $n_d$ changes only the second condition: the outgoing amplitude jumps by the dipole's own
multipole expansion at $r_d$, and "no incoming wave from infinity" replaces the plane wave; the matrices are the same.
That is the subject of the [next section](learn/spheres/dipole-near-a-sphere.html).

## Numerical discipline

The elegance of the recursion hides two traps that have broken many published codes.

**Growth of the Riccati–Bessel functions.** For complex argument $z$, $\zeta_\ell(z)$ grows like $e^{|\mathrm{Im}\,z|}$
and $\psi_\ell(z)$ like $e^{|\mathrm{Im}\,z|}$ too, while for $|z| \ll \ell$ the pair behaves as $z^{\ell+1}$ and
$z^{-\ell}$: inside a metal shell or at high orders, the matrix elements span hundreds of orders of magnitude and the
products lose every digit or overflow. Yang's "improved recursive algorithm" and scattnlay avoid the worst of it with
ratios and logarithmic derivatives of the Bessel functions. PyStratify goes one step further and carries each
amplitude together with its logarithm, so that the growing factors are divided out before any product is formed and a
sphere with a 2 nm silver shell at $\ell = 500$ is as safe as a glass bead at $\ell = 5$. The
[Verification](verification.html) page compares this recursion with the same formulas evaluated in 60-digit
arithmetic, and with scattnlay and treams, down to $10^{-11}$ in the cross sections and $10^{-13}$ in the fields.

**Truncation and the internal size parameter.** Wiscombe's rule for the number of orders, $\ell_\mathrm{max} = x +
4x^{1/3} + 2$ with $x$ the host size parameter, is designed for the far field of a homogeneous sphere. A high-index
core, where the internal wavelength is short, and a dipole close to a surface both need more orders, so the engine
takes the largest internal $|m| x$ into account and, for emitters, continues until the rates converge. The order
count the calculator reports is this adaptive one.

## The first application: plasmon hybridization

A metal shell on a dielectric core has two surfaces, and each supports a plasmon, that of a cavity in the metal and
that of a sphere of metal. In a thin shell the two couple and split into a low-energy **bonding** and a high-energy
**antibonding** mode, whose separation grows as the shell thins: the nanoshell's resonance can be tuned from the
visible deep into the near infrared with the shell thickness alone, which is the basis of its use in photothermal
therapy. Prodan, Radloff, Halas and Nordlander described this with a hybridization picture borrowed from molecular
orbital theory, and demonstrated it on a **nanomatryoshka**, a nanoshell inside a nanoshell, whose two shell plasmons
hybridize once more when the dielectric spacer between them is thin. In the transfer-matrix language all of this is
in the composite coefficient $a_1$: four interfaces, four matrices, and the zeros of $\mathcal{T}_{11}$ in the complex
frequency plane are the hybridized modes.

::: classic prodan-2003
