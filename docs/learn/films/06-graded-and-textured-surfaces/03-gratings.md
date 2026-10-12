---
id: gratings
title: "Gratings: diffraction orders and the sub-wavelength limit"
geometry: films
chapter: Graded and textured surfaces
topic: gratings
sources: [plane_wave]
summary: >-
  A periodic surface couples light into discrete orders: the grating equation from the conservation of in-plane
  momentum modulo 2π/Λ, propagating and evanescent orders and the cutoff, Rayleigh–Wood anomalies, the
  effective film a sub-wavelength grating becomes (Rytov's zeroth- and second-order indices and form birefringence),
  and how rigorous coupled-wave analysis solves the rest, with the Fourier factorization rules that make it converge.
examples: [grating-effective-film]
references:
- cite: M. G. Moharam and T. K. Gaylord, J. Opt. Soc. Am. 71, 811 (1981)
  doi: 10.1364/JOSA.71.000811
- cite: M. G. Moharam, E. B. Grann, D. A. Pommet and T. K. Gaylord, J. Opt. Soc. Am. A 12, 1068 (1995)
  doi: 10.1364/JOSAA.12.001068
- cite: L. Li, J. Opt. Soc. Am. A 13, 1870 (1996)
  doi: 10.1364/JOSAA.13.001870
- cite: S. M. Rytov, Sov. Phys. JETP 2, 466 (1956)
- cite: D. H. Raguin and G. M. Morris, Appl. Opt. 32, 1154 (1993)
  doi: 10.1364/AO.32.001154
- cite: "E. Popov (ed.), Gratings: Theory and Numeric Applications, 2nd ed. (Institut Fresnel, 2014)"
---

A planar stack conserves the in-plane wave vector exactly, because it is invariant under every translation along
the surface. A grating of period $\Lambda$ is invariant only under translations by $\Lambda$, so Bloch's theorem (the
same one that gives the [Bragg mirror](learn/films/bragg-mirrors.html) its band structure, now applied along the
surface) says the field is a sum of plane waves whose in-plane wavenumbers differ from the incident one by multiples
of $2\pi/\Lambda$. These are the diffraction orders $m$, each leaving at the angle given by the **grating equation**

$$
n \sin\theta_m = n_0 \sin\theta_0 + m\,\frac{\lambda}{\Lambda} ,
$$

in the medium of index $n$ on either side. The equation says where the orders go; it says nothing about how much
light each carries, which is the whole difficulty of grating theory.

## Propagating and evanescent orders

Order $m$ propagates in a medium of index $n$ only while $|n \sin\theta_m| < n$; beyond that its normal wavenumber is
imaginary, it is evanescent, bound to the surface, and carries no power away. As the period shrinks, or the wavelength
grows, the orders cut off one by one, the last of them ($m = -1$ into the denser medium) at

$$
\lambda_c = \Lambda\,(n_\mathrm{max} + n_0 \sin\theta_0),
$$

where $n_\mathrm{max}$ is the larger of the two media's indices. A 300 nm grating on glass at normal incidence
diffracts into the glass below 456 nm and into air below 300 nm. Exactly at a cutoff an order grazes the surface,
and the redistribution of its power among the others produces a sharp feature in every efficiency curve: the
**Rayleigh–Wood anomaly**, which Wood saw in 1902 as dark bands in a grating's spectrum and Rayleigh explained in
1907. Near a cutoff a grating is also most sensitive to its profile, which is where the simple models fail first.

## Above cutoff: an effective film

Past $\lambda_c$ only the zeroth order survives, and to it the grating is a homogeneous but **anisotropic** layer: the
field averages over the period, but it averages differently for the component along the lines and the component
across them, because the boundary conditions preserve $E_\parallel$ along the lines and $D_\perp$ across them. For a
line grating of fill fraction $f$ the lowest-order effective indices (Rytov 1956) are

$$
n_\mathrm{TE}^2 = f\,n_1^2 + (1-f)\,n_2^2, \qquad \frac{1}{n_\mathrm{TM}^2} = \frac{f}{n_1^2} + \frac{1-f}{n_2^2},
$$

the arithmetic mean of the permittivities for light polarized along the lines and the harmonic mean for light
polarized across them, the same two averages as capacitors in parallel and in series. Rytov's second-order
correction adds the first effect of the finite period,

$$
n_\mathrm{TE,2}^2 = n_\mathrm{TE}^2 + \frac{\pi^2}{3}\left(\frac{\Lambda}{\lambda}\right)^2 f^2(1-f)^2\,(n_1^2 - n_2^2)^2 ,
$$

and a corresponding term for TM, which is how far one can trust the zeroth-order picture: a few percent in index
at $\Lambda/\lambda = 0.3$ for a glass/air grating, more for silicon. The birefringence $n_\mathrm{TE} - n_\mathrm{TM}$
is a design tool in its own right, the **form birefringence** of wave plates and polarizers etched into a single
material (Raguin and Morris 1993), and for TE light the grating reflects exactly like the film below.

::: example grating-effective-film

Below the cutoff the effective film is wrong: energy leaves along the diffracted orders, which no planar calculation
produces, and the anomaly at the cutoff itself is invisible to it.

## Below cutoff: rigorous coupled-wave analysis

Gratings are solved exactly, for any period and profile, by **rigorous coupled-wave analysis** (Moharam and Gaylord
1981). The permittivity in the grating layer is expanded in its Fourier series, the field in a finite number of
Bloch harmonics (the orders, propagating and evanescent alike, $2N + 1$ of them), and Maxwell's equations in the
layer become a matrix eigenvalue problem whose eigenvectors are the layer's modes and whose eigenvalues their
propagation constants. Each mode is then an ordinary plane-wave-like solution, and the boundary conditions at the
top and bottom of the layer are a transfer-matrix problem of size $2(2N+1)$ rather than $2$: the
[Foundations](learn/common/why-transfer-matrices.html) construction, with the single channel replaced by a block of
coupled ones. A profile that is not a rectangle is staircased into slices, each with its own eigenproblem, and the
slices are joined as layers are.

Two refinements made the method practical. The products of the transfer matrices through a thick or absorbing
grating overflow exactly as they do for films, and the scattering-matrix formulation with Redheffer's star product
(the *enhanced transmittance matrix* of Moharam et al. 1995, and Li's stable algorithms) cured it. And the TM
polarization converged painfully slowly for metallic gratings until Li (1996) showed why: the product of two
discontinuous functions ($\varepsilon$ and $E_\perp$, whose product $D_\perp$ is continuous) must be Fourier
factorised by the inverse rule, $\left[\varepsilon\right]^{-1}$ of the Toeplitz matrix of $1/\varepsilon$, not by
Laurent's rule. With both, RCWA converges with a few tens of harmonics for dielectric gratings and a hundred or so
for metals, and it is what the calculator runs for its grating layers, outside PyStratify. The zero-patterning limit
of that solver reproduces the transfer matrix to $10^{-9}$, which is one of the [Verification](verification.html)
rows.

## What a grating is for

Below cutoff a grating is a spectrometer's dispersing element, a coupler into a waveguide (the $m = 1$ order matched
to a guided mode's wavenumber, the standard way light enters a photonic chip), a light-trapping structure on a solar
cell (the orders diffracted beyond the critical angle of the substrate stay inside it, as the
[textured surface](learn/films/textured-surfaces.html) does geometrically), and an anomaly-based sensor. Above cutoff
it is a film with a designable index and birefringence, the effective-medium route to the
[moth-eye antireflection](learn/films/moth-eye-antireflection.html) and to artificial wave plates. The same equation
and the same solver cover both, and the cutoff $\lambda_c$ is the line between them.
