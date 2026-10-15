---
id: dipole-near-an-interface
title: "A dipole near a plane interface: emitters in thin films"
geometry: films
chapter: Emitters in films
topic: emitters
sources: [dipole]
classics: [lukosz-kunz-1977, neyts-1998]
description: "Emitters in thin films: decay rates by Chance–Prock–Silbey, escaping, substrate and guided power, quenching near metal, and OLED microcavities."
reviewed:
summary: >-
  How a molecule, a quantum dot or an OLED emitter radiates when it sits in a stratified medium: the plane-wave
  expansion of a dipole field, the Chance–Prock–Silbey integrals for its decay rate, the channels its power goes to
  (escaping light, substrate modes, guided modes, absorption), quenching near a metal and the microcavity effect
  that OLED design is built on.
references:
- cite: W. Lukosz and R. E. Kunz, J. Opt. Soc. Am. 67, 1607 (1977)
  doi: 10.1364/JOSA.67.001607
- cite: R. R. Chance, A. Prock and R. Silbey, Adv. Chem. Phys. 37, 1 (1978)
  doi: 10.1002/9780470142561.ch1
- cite: G. W. Ford and W. H. Weber, Phys. Rep. 113, 195 (1984)
  doi: 10.1016/0370-1573(84)90098-X
- cite: W. L. Barnes, J. Mod. Opt. 45, 661 (1998)
  doi: 10.1080/09500349808230614
- cite: K. A. Neyts, J. Opt. Soc. Am. A 15, 962 (1998)
  doi: 10.1364/JOSAA.15.000962
- cite: L. Novotny and B. Hecht, Principles of Nano-Optics, 2nd ed. (Cambridge University Press, 2012), ch. 10
  doi: 10.1017/CBO9780511794193
- cite: M. Furno, R. Meerheim, S. Hofmann, B. Lüssem and K. Leo, Phys. Rev. B 85, 115205 (2012)
  doi: 10.1103/PhysRevB.85.115205
---

An excited molecule is, to classical electrodynamics, an oscillating point dipole. In free space it radiates the
power $P_0$ and decays at the rate $\Gamma_0$. Put the same dipole a few hundred nanometres from a glass surface, inside
an organic layer on a mirror, or on top of a metal film, and both change: the field it radiates comes back to it,
reflected by every interface, and does work on it. This is Purcell's observation applied to a planar structure, and
it is the physics of OLED outcoupling, of surface-enhanced fluorescence and of the quenching that kills emission next
to a metal. It is computed with the same transfer matrix as a plane-wave spectrum, applied to a different source.

## A dipole as a sum of plane waves

The field of a dipole at height $z_0$ above a stack, in a medium of index $n_1$ and wavenumber $k_1 = n_1 k_0$, can be
written as a superposition of plane waves with every in-plane wavenumber $k_\parallel$, including $k_\parallel > k_1$,
for which $k_z = \sqrt{k_1^2 - k_\parallel^2}$ is imaginary and the wave is evanescent. Each plane wave meets the stack
below and returns multiplied by the stack's reflection coefficient $r^s(k_\parallel)$ or $r^p(k_\parallel)$, the
quantities the [transfer matrix](learn/films/transfer-matrix.html) already computes, now evaluated at complex angles.
Summing the reflected waves at the dipole's own position gives the reflected field there, and the work it does on the
dipole is the change in radiated power. With $s = k_\parallel / k_1$ and $s_z = \sqrt{1 - s^2}$, the result for a
dipole perpendicular and parallel to the interface is (Chance, Prock and Silbey; Lukosz and Kunz)

$$
\frac{P_\perp}{P_0} = 1 + \frac{3}{2}\,\mathrm{Re}\int_0^\infty \frac{s^3}{s_z}\, r^p(s)\, e^{2 i k_1 z_0 s_z}\,\mathrm{d}s,
\qquad
\frac{P_\parallel}{P_0} = 1 + \frac{3}{4}\,\mathrm{Re}\int_0^\infty \frac{s}{s_z}\left[r^s(s) - s_z^2\, r^p(s)\right] e^{2 i k_1 z_0 s_z}\,\mathrm{d}s .
$$

An emitter with intrinsic quantum yield $q_0 < 1$ also decays non-radiatively inside itself, so its total rate is
$\Gamma/\Gamma_0 = (1 - q_0) + q_0\, P/P_0$; a randomly oriented ensemble averages one perpendicular and two parallel
dipoles. The same integrals, with the transmission coefficients in place of $r$, give the power that crosses the
stack into the substrate. Everything about the structure enters through $r^s$ and $r^p$ as functions of $s$, which
is why a dipole calculation costs little more than an angle-resolved reflectance.

## Reading the integrand: where the power goes

The integration variable $s$ sorts the dipole's power into channels, and the calculator reports them separately:

- $0 \le s < 1$: plane waves that propagate in the dipole's own medium. Part escapes (into air above an OLED, into
  the objective of a microscope); the rest is reflected and interferes with the direct wave, which makes the rate
  **oscillate with $z_0$** with a period of half a wavelength in the medium. That oscillation is Lukosz and Kunz's
  curve, and Drexhage's 1970 experiment with europium layers above a silver mirror.
- $1 \le s < n_\mathrm{sub}/n_1$: evanescent in the emitter's medium but propagating in a denser substrate: light
  that enters the glass beyond its critical angle and is **trapped in the substrate** by total internal reflection
  unless the substrate is thin or textured.
- Discrete values of $s$ at which $r^s$ or $r^p$ has a pole: the **guided modes** of the stack (and the surface
  plasmon of a metal film), which take power without radiating it. In a lossless structure these are sharp poles on
  the real axis and PyStratify integrates around them analytically, which is why its guided fraction is exact and not
  a matter of grid resolution.
- $s \gg 1$: deeply evanescent waves that reach only a few nanometres. Against an absorbing medium they deposit the
  dipole's energy directly in it. For a dipole at distance $d$ from a metal this term grows like $(k_1 d)^{-3}$, the
  quasi-static image-dipole law, with the perpendicular dipole quenched twice as strongly as the parallel one: this
  is **quenching**, and it dominates below about 10 nm.

The sum of the four channels is the total rate; the escaping part divided by the total is the outcoupling efficiency
of an OLED or the collection efficiency of a single-molecule experiment.

::: classic lukosz-kunz-1977

## The microcavity effect and OLED design

In a thin-film light-emitting diode the emitter sits in an organic layer a fraction of a wavelength thick, between a
metal cathode and a transparent anode on glass. The cathode's reflection coefficient is close to $-1$, so the
reflected wave returns with a phase set almost entirely by the round-trip distance $2 n z_0 k_0$. At the distance where
it returns in phase the dipole radiates more, mostly towards the surface normal; a quarter wave away it radiates less,
and what it radiates goes sideways into guided modes. Neyts computed the two numbers an OLED engineer needs as the
emitter moves through the layer: the fraction of power that reaches air, and the total rate $F$ (the Purcell factor),
whose product with the intrinsic quantum yield is the effective yield. Furno and co-workers turned the same integrals
into the loss budget of a complete device, with each loss channel (substrate, waveguide, surface plasmon, absorption)
against the thickness of every layer, the analysis PyStratify's film emission reproduces.

::: classic neyts-1998

Two practical rules follow from the integrals. First, the optimum emitter position is **not** at the antinode of the
plane-wave standing wave but slightly closer to the mirror, because the evanescent channels grow monotonically as the
emitter approaches it. Second, a thick transparent substrate is a trap: without outcoupling structures roughly half of
what leaves the organic layers is lost in it, which is why the [coherence and thick substrate](learn/films/coherence.html)
section's incoherent treatment matters for devices and why textured surfaces appear in the
[graded and textured surfaces](learn/films/textured-surfaces.html) chapter.

## Limits of the model

The emitter is a classical point dipole with a fixed frequency: linewidths, saturation and the modification of the
emission *spectrum* by a strongly dispersive cavity are outside it, though the rates it gives are exactly what a
quantum treatment yields for a two-level system at weak coupling. The media are local and the interfaces sharp: at
distances below one or two nanometres from a metal, the electron gas's nonlocal response and the atomic structure of
the surface change the quenching law, which is why computed rates at contact are a limit rather than a prediction. The
calculator refuses an absorbing medium *around* the emitter (its power would not be defined without a cavity around
the point source); a metal electrode is therefore an opaque layer rather than a half-space, which changes nothing in
the physics once the layer is thicker than its skin depth.
