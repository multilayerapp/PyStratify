---
id: bragg-mirrors
title: "Bragg mirrors: periodic stacks and stop bands"
geometry: films
chapter: Periodic stacks
topic: bragg
sources: [plane_wave]
description: "Quarter-wave stacks: peak reflectance, stop-band width from the index contrast, Bloch modes, N-period reflectance and the omnidirectional mirror."
reviewed:
summary: >-
  A quarter-wave stack of two materials reflects almost perfectly over a band of wavelengths: the peak reflectance
  from the admittance, Bloch's theorem and the dispersion relation of the one-dimensional crystal, the stop band
  width from the index contrast, the N-period reflectance through Chebyshev polynomials, penetration depth,
  higher-order bands and the ones that vanish, oblique incidence and the omnidirectional mirror.
classics: [yeh-1977]
examples: [bragg-mirror]
references:
- cite: P. Yeh, A. Yariv and C.-S. Hong, J. Opt. Soc. Am. 67, 423 (1977)
  doi: 10.1364/JOSA.67.000423
- cite: P. Yeh, Optical Waves in Layered Media (Wiley, 1988), ch. 6
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 5
- cite: M. Born and E. Wolf, Principles of Optics, 7th ed. (Cambridge University Press, 1999), §1.6.5
  doi: 10.1017/CBO9781139644181
- cite: D. I. Babic and S. W. Corzine, IEEE J. Quantum Electron. 28, 514 (1992)
  doi: 10.1109/3.123281
- cite: Y. Fink et al., Science 282, 1679 (1998)
  doi: 10.1126/science.282.5394.1679
- cite: J. N. Winn, Y. Fink, S. Fan and J. D. Joannopoulos, Opt. Lett. 23, 1573 (1998)
  doi: 10.1364/OL.23.001573
---

Alternate a high index $n_H$ and a low index $n_L$, each a quarter wave thick at $\lambda_0$, and the reflections from
every interface return in phase: the wave reflected at an H→L boundary has the opposite sign to the one reflected at
an L→H boundary, and the quarter-wave round trip adds another $\pi$, so all of them add. With enough periods the stack
reflects nearly everything over a band around $\lambda_0$: a dielectric mirror with no absorption, the basis of laser
mirrors, of distributed Bragg reflectors in VCSELs, and the one-dimensional photonic crystal. Yeh, Yariv and Hong's
1977 paper is the one that treats it as a crystal, and this section follows them.

## The peak: quarter waves in series

At $\lambda_0$ each quarter-wave layer maps an admittance $Y$ onto $n^2/Y$ (see
[the transfer matrix](learn/films/transfer-matrix.html)). Starting from the substrate $n_s$, a pair L then H
multiplies it by $(n_H/n_L)^2$, so $N$ pairs, high index outermost, present

$$
Y = \left(\frac{n_H}{n_L}\right)^{2N} n_s ,
\qquad
R(\lambda_0) = \left(\frac{n_0 - Y}{n_0 + Y}\right)^2 .
$$

$Y$ grows geometrically with $N$, so $1 - R$ falls geometrically, by the factor $(n_L/n_H)^4$ per pair once
$Y \gg n_0$: eight pairs of TiO₂-like and SiO₂-like layers (2.3 and 1.45) already reach 99.84 %, and eleven pairs
99.99 %. Ending the stack with a low-index layer instead divides $Y$ by $n_L^2$ and lowers the peak; the outermost
layer should be the one of higher admittance contrast with the incident medium. For a VCSEL mirror in GaAs/AlAs
(3.5/2.9, a contrast of only 1.2) the same 99.9 % needs about thirty pairs, which is why such mirrors are grown
rather than evaporated.

## The band: Bloch waves and the dispersion relation

An infinite periodic stack is a one-dimensional crystal, and Bloch's theorem says its eigenwaves reproduce
themselves from one period to the next up to a phase: $E(z + \Lambda) = e^{iK\Lambda}E(z)$, with $K$ the Bloch
wavenumber. The characteristic matrix $M$ of one period (the product $M_H M_L$) must therefore have the eigenvalue
$e^{iK\Lambda}$, and since $\det M = 1$ the two eigenvalues are $e^{\pm iK\Lambda}$, whose sum is the trace:

$$
\cos K\Lambda = \tfrac{1}{2}\,\mathrm{Tr}\,M
= \cos\delta_H\cos\delta_L - \tfrac{1}{2}\left(\frac{\eta_H}{\eta_L} + \frac{\eta_L}{\eta_H}\right)\sin\delta_H\sin\delta_L .
$$

This is the **dispersion relation** of the crystal. Where the right-hand side lies between $-1$ and $1$, $K$ is real
and the wave propagates; where it exceeds 1 in magnitude, $K$ is complex, the Bloch wave decays as
$e^{-|\mathrm{Im}\,K| z}$, and a thick enough stack reflects it completely. That decaying range is the **stop band**;
its edges are where $\cos K\Lambda = \pm 1$. For a quarter-wave stack at normal incidence, $\delta_H = \delta_L =
\delta$ and the condition $|\cos 2\delta - \ldots| = 1$ gives a band centred on $\delta = \pi/2$ with a full width, in the
relative wavenumber $g = \lambda_0/\lambda$, of

$$
\Delta g = \frac{4}{\pi}\arcsin\!\left(\frac{n_H - n_L}{n_H + n_L}\right),
$$

about 0.29 for 2.3/1.45, or 524–702 nm around 600 nm, and 0.11 for GaAs/AlAs. The width depends on the index contrast
alone; the period count only decides how close to 1 the reflectance gets inside the band and how sharp its edges
are. Inside the band the imaginary part of $K$ is largest at the centre, $|\mathrm{Im}\,K|\Lambda \approx
2(n_H - n_L)/(n_H + n_L)$ for small contrast, which gives the number of periods the field penetrates.

## N periods exactly: Chebyshev polynomials

For a finite stack of $N$ periods Yeh, Yariv and Hong evaluate $M^N$ in closed form through the Chebyshev
polynomials of the second kind, $U_N(\cos K\Lambda) = \sin[(N+1)K\Lambda]/\sin K\Lambda$:

$$
M^N = \begin{pmatrix} A\,U_{N-1} - U_{N-2} & B\,U_{N-1} \\ C\,U_{N-1} & D\,U_{N-1} - U_{N-2} \end{pmatrix},
\qquad
R_N = \frac{|C|^2\,|U_{N-1}|^2}{|C|^2\,|U_{N-1}|^2 + 1} ,
$$

with $A, B, C, D$ the elements of the single-period matrix. Outside the band $K\Lambda$ is real, $U_{N-1}$ oscillates,
and the reflectance has $N - 1$ zeros between two bands, one fewer than the number of periods, which is the count the
classic card below invites. Inside the band $U_{N-1}$ grows like $\sinh(N|\mathrm{Im}\,K|\Lambda)$, and $R_N \to 1$
exponentially in $N$, in agreement with the admittance result at the centre.

::: example bragg-mirror

Outside the band the stack is nearly transparent, with ripples, one fewer minimum between bands than there are
periods. Yeh, Yariv and Hong's 1977 paper treats the periodic stack as a one-dimensional crystal: the eigenvalues
of the unit cell's transfer matrix give the Bloch wavenumber, real in the pass bands and complex in the stop bands,
and the band edges, the number of zeros and the closing of the TM gap at the Brewster angle all follow from one
dispersion relation.

::: classic yeh-1977

## Penetration depth and the phase of a mirror

A Bragg mirror does not reflect at its surface. The field decays into the stack over a few periods, and a cavity
closed by such a mirror is effectively longer than its spacer by the **penetration depth**, which for a quarter-wave
stack of small contrast is (Babic and Corzine 1992)

$$
L_\mathrm{pen} \approx \frac{\lambda_0}{4\,(n_H - n_L)} ,
$$

about 180 nm for TiO₂/SiO₂ at 600 nm and 0.4 µm for GaAs/AlAs at 1 µm. The same quantity sets how fast the mirror's
reflection phase changes with wavelength, which decides the free spectral range and the mode spacing of a VCSEL or
of a dielectric [Fabry–Pérot cavity](learn/films/bandpass-filters.html), and it is the reason a cavity between two
Bragg mirrors resonates at a wavelength slightly off the naive $2nd = m\lambda$.

## Higher bands, and the bands that vanish

The dispersion relation is periodic in $\delta$: the stop band repeats wherever the layers are an odd number of
quarter waves thick, at $\lambda_0/3$, $\lambda_0/5$, …, with the same width in $g$ (so narrower in wavelength). At
$\lambda_0/2$, where both layers are half waves, $\cos K\Lambda = 1$ exactly: the second-order band of a quarter-wave
stack has **zero width**, and the stack is transparent there. In general the $m$-th band of a stack with optical-thickness
ratio $n_H d_H : n_L d_L$ vanishes when $m\,n_H d_H/(n_H d_H + n_L d_L)$ is an integer, a selection rule that is used to suppress unwanted
bands and, run backwards, to measure a stack's thickness ratio from which bands are missing.

## At oblique incidence

Tilting the beam shortens every layer's phase thickness, $\delta_j = k_0 n_j d_j\cos\theta_j$, so the band moves to
shorter wavelengths (a Bragg mirror seen at an angle turns bluer), and splits by polarization: the admittances
$\eta_H/\eta_L$ of the two polarizations diverge, so the s band widens and the p band narrows, closing entirely at
the Brewster angle of the H/L interface, $\tan\theta_B = n_L/n_H$ inside the stack, where successive interfaces no
longer reflect p light at all. In a stack lit from air the internal angle can never reach that Brewster angle if

$$
n_0 < \frac{n_H\, n_L}{\sqrt{n_H^2 + n_L^2}} ,
$$

and with enough contrast the stop bands of all angles and both polarizations then overlap over a common range:
Fink, Winn and co-workers (1998) built such an **omnidirectional reflector** from nine pairs of polystyrene and
tellurium, a dielectric mirror for every angle. The classic card's Yeh stack, lit from inside GaAs, does reach the
Brewster angle, which is where its TM gap closes.
