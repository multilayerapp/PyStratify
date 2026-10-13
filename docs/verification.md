---
title: How PyStratify is verified
summary: >-
  Every check behind multilayer.app's engine, geometry by geometry and source by source: closed forms,
  extended-precision references, independent methods, other groups' codes and published papers, each plotted
  against PyStratify, with the script that reruns it.
examples:
- {geometry: films, source: plane_wave, reference: "Fresnel at oblique incidence; Brewster zero; s = p at normal", example: fresnel-air-glass}
- {geometry: films, source: plane_wave, reference: "Airy slab; quarter-wave Bragg mirror", example: bragg-mirror}
- {geometry: spheres, source: plane_wave, reference: "textbook Mie (homogeneous); identical shells = one sphere", example: si-sphere-mie}
figures:
- {geometry: films, source: plane_wave, reference: "Fresnel at oblique incidence; Brewster zero; s = p at normal", figure: fresnel-oblique}
- {geometry: films, source: plane_wave, reference: "Airy slab; quarter-wave Bragg mirror", figure: airy-slab}
- {geometry: films, source: plane_wave, reference: "R + T = 1 for lossless stacks, below and beyond the critical angle; reciprocity under reversal", figure: energy-conservation}
- {geometry: films, source: dipole, reference: "Sommerfeld integrals over s = k_ρ/k₁ with Parratt-recursion r_s, r_p, written without `pystratify`", figure: sommerfeld-film-emitter}
- {geometry: films, source: plane_wave, reference: "Fresnel, single interface, incl. opaque", figure: opaque-film}
- {geometry: films, source: plane_wave, reference: "bare-substrate Fresnel ratio", figure: [bare-substrate-psi, bare-substrate-delta]}
- {geometry: films, source: plane_wave, reference: "scattering matrix (Redheffer star product)", figure: [smatrix-metal-stack, smatrix-beyond-critical]}
- {geometry: films, source: plane_wave, reference: "explicit phase average of coherent solves", figure: incoherent-phase-average}
- {geometry: films, source: plane_wave, reference: "RCWA with zero patterning", figure: rcwa-unpatterned}
- {geometry: films, source: plane_wave, reference: "position-resolved field against Parseval", figure: position-resolved-absorption}
- {geometry: films, source: plane_wave, reference: '`inc_tmm` ("GTMM" in the backend tests)', figure: inc-tmm-lineage}
- {geometry: films, source: plane_wave, reference: "**pyElli**, 2×2 and 4×4 solvers", figure: [pyelli-bragg-mirror, pyelli-gold-ellipsometry]}
- {geometry: films, source: plane_wave, reference: "**PyMoosh** (independent S-matrix code)", figure: [pymoosh-bragg-mirror, pymoosh-magnetic-layer]}
- {geometry: films, source: plane_wave, reference: "**PyMoosh** `field` (one Fourier order: a single plane wave)", figure: pymoosh-field}
- {geometry: films, source: plane_wave, reference: "**PyMoosh** `incoherent.py`", figure: pymoosh-incoherent}
- {geometry: films, source: plane_wave, reference: "**tmmax** (JAX transfer matrix, float32)", figure: [tmmax-single-layer, tmmax-bragg-mirror]}
- {geometry: films, source: dipole, reference: "dipole in a homogeneous film = free space", figure: film-dipole-homogeneous}
- {geometry: films, source: dipole, reference: "independent flux budget (escape + guided + absorbed = total)", figure: [film-coating-budget-electric, film-coating-budget-magnetic]}
- {geometry: films, source: dipole, reference: "guided-pole uncertainty within tolerance; 3D sources are not line sources", figure: [film-guided-pole-tolerance, film-guide-budget]}
- {geometry: films, source: dipole, reference: "emission pattern normalization; grazing limit", figure: film-pattern-free-space}
- {geometry: films, source: dipole, reference: "image dipole at a perfect mirror (Drexhage; Chance, Prock & Silbey)", figure: image-dipole-mirror}
- {geometry: films, source: dipole, reference: "the same reference on a weakly absorbing guide (k = 1e-3), quasi-poles as breakpoints", figure: weak-guide-sommerfeld}
- {geometry: films, source: dipole, reference: "lossless limit: a source outside a lossless guide equals the k → 0 limit of an absorbing one", figure: lossless-guide-limit}
- {geometry: films, source: dipole, reference: "**PyMoosh** `modes.guided_modes`", figure: pymoosh-guided-modes}
- {geometry: films, source: dipole, reference: "three-layer slab dispersion, κd = mπ + atan(ρ₁γ₁/κ) + atan(ρ₃γ₃/κ)", figure: slab-dispersion-modes}
- {geometry: films, source: dipole, reference: "**smuthi** `DipoleSource.dissipated_power`", figure: smuthi-film-guide}
- {geometry: films, source: dipole, reference: "**Meep**, cylindrical coordinates, dipole on the axis (m = 0 perpendicular, m = 1 parallel)", figure: meep-axial-film}
- {geometry: films, source: dipole, reference: "OLED outcoupling papers (e.g. Furno et al. 2012; Neyts 1998)", figure: [lukosz-kunz-1977, neyts-1998-microcavity]}
- {geometry: films, source: focused, reference: "cone quadrature against a pupil grid", figure: [film-cone-transmission, film-cone-reflection]}
- {geometry: films, source: focused, reference: "aplanatic path factor; transparent stack has zero absorbance; Köhler = coherent", figure: [film-path-factor, film-koehler-coherent]}
- {geometry: films, source: focused, reference: "near field against position-resolved plane waves and Parseval", figure: film-focused-near-field}
- {geometry: spheres, source: plane_wave, reference: "textbook Mie (homogeneous); identical shells = one sphere", figure: mie-homogeneous}
- {geometry: spheres, source: plane_wave, reference: "**BHMIE**", figure: bhmie-large-spheres}
- {geometry: spheres, source: plane_wave, reference: "60-digit transfer matrices", figure: sixty-digit-coefficients}
- {geometry: spheres, source: plane_wave, reference: "**treams**, achiral spheres and multilayers (x ≤ 15)", figure: treams-nanoshell}
- {geometry: spheres, source: dipole, reference: "normalized route against extended precision and classical formulas", figure: sphere-decay-mpmath}
- {geometry: spheres, source: dipole, reference: "**treams** (the dipole as an l = 1 singular wave, translated through the layered sphere's T-matrix and back)", figure: treams-dipole-nanoshell}
- {geometry: spheres, source: dipole, reference: "**smuthi** (dipole source and a sphere, by Mie T-matrix and its own translation of the dipole field)", figure: [smuthi-sphere-dielectric, smuthi-sphere-metal, smuthi-sphere-water, smuthi-sphere-large]}
- {geometry: spheres, source: plane_wave, reference: "Bohren & Huffman d₁, c₁ (only l = 1 survives at r = 0)", figure: centre-field}
- {geometry: spheres, source: plane_wave, reference: "optical theorem; Rayleigh limit; lossless absorbs nothing", figure: [optical-theorem, rayleigh-limit, lossless-absorbs-nothing]}
- {geometry: spheres, source: plane_wave, reference: "80-digit chiral transfer matrices", figure: chiral-80-digits}
- {geometry: spheres, source: plane_wave, reference: "graphene-coated sphere plasmons; thin-film limit of a sheet", figure: [graphene-sphere-plasmon, thin-film-sheet]}
- {geometry: spheres, source: plane_wave, reference: "near-field truncation against 120 orders", figure: near-field-truncation}
- {geometry: spheres, source: dipole, reference: "textbook Mie decay rates [MLR Eqs. 34–37, mpmath]", figure: silver-decay-mie}
- {geometry: spheres, source: dipole, reference: "layered-sphere transfer matrices (electric and magnetic dipoles)", figure: [nanoshell-decay-layers, silicon-magnetic-decay]}
- {geometry: spheres, source: dipole, reference: "dipole in a homogeneous medium; reciprocity with the plane-wave near field; radiated power = radiative rate", figure: radiated-power-rates}
- {geometry: spheres, source: dipole, reference: "two-point Green's dyadic against extended precision", figure: two-point-green}
- {geometry: spheres, source: dipole, reference: "quadrupole coupling against finite differences", figure: quadrupole-coupling}
- {geometry: spheres, source: dipole, reference: "Casimir–Polder non-retarded limit", figure: casimir-non-retarded}
- {geometry: spheres, source: focused, reference: "amplitude-matrix superposition of plane waves", figure: focused-sphere-superposition}
- {geometry: spheres, source: focused, reference: "axis path equals general path; index-matched sphere invisible; NA → 0 is the plane wave", figure: [focused-na-limit, focused-axis-path]}
- {geometry: spheres, source: focused, reference: "Köhler against a plane-wave cone average", figure: koehler-cone-average}
- {geometry: spheres, source: focused, reference: "near field against plane-wave near fields", figure: focused-near-field}
- {geometry: spheres, source: plane_wave, reference: "Bohren & Huffman reference values", figure: bh-reference-values}
- {geometry: spheres, source: plane_wave, reference: "many layers, graded profiles, rainbow, Luneburg lens [W97, WW91, Y24]", figure: [many-identical-layers, graded-droplet-rainbow]}
- {geometry: spheres, source: plane_wave, reference: "**treams**, chiral shells", figure: treams-chiral}
- {geometry: spheres, source: plane_wave, reference: "published small chiral sphere polarisabilities", figure: chiral-polarisabilities}
- {geometry: spheres, source: plane_wave, reference: "Gonçalves et al. 2020 (Feibelman d-parameters)", figure: goncalves-feibelman}
- {geometry: spheres, source: plane_wave, reference: "**treams**, near fields", figure: treams-near-field}
- {geometry: spheres, source: plane_wave, reference: "**scattnlay**", figure: [scattnlay-matryoshka, scattnlay-pattern]}
- {geometry: spheres, source: plane_wave, reference: "**scattnlay**, near fields", figure: scattnlay-near-field}
- {geometry: spheres, source: plane_wave, reference: "**MNPBEM** (boundary elements; MNPBEM17 from the CPC program library, run under GNU Octave)", figure: mnpbem-spectrum}
- {geometry: spheres, source: plane_wave, reference: "**pyGDM2** (a second different method: volume Green dyadic)", figure: gdm-spectrum}
- {geometry: spheres, source: plane_wave, reference: "thin (5 nm) nanoshell: MNPBEM against three Mie codes", figure: mnpbem-thin-nanoshell}
- {geometry: spheres, source: plane_wave, reference: "Oldenburg et al. 1998 nanoshells; Kerker conditions", figure: fu-2013-kerker}
- {geometry: spheres, source: dipole, reference: "layered magnetic sphere, Yuan, Zhu & Zhu 2024 Table 1", figure: layered-magnetic-balance}
- {geometry: spheres, source: dipole, reference: "Guzatov & Klimov 2012, Eq. 46 and Figs. 3–6", figure: guzatov-chiral-molecule}
- {geometry: spheres, source: dipole, reference: "Anger, Bharadwaj & Novotny 2006", figure: [anger-2006-excitation, anger-2006-fluorescence]}
- {geometry: spheres, source: dipole, reference: "**MNPBEM** `dipoleret`", figure: mnpbem-decay}
- {geometry: spheres, source: dipole, reference: "**pyGDM2** `decay_rate`", figure: gdm-decay}
- {geometry: spheres, source: dipole, reference: "**Meep**, cylindrical coordinates (dipole on the axis through the sphere)", figure: meep-axial-sphere}
- {geometry: cylinders, source: plane_wave, reference: "**treams**, oblique incidence", figure: treams-cylinder-oblique}
- {geometry: cylinders, source: plane_wave, reference: "Bessel boundary equations solved directly", figure: cylinder-bessel-direct}
- {geometry: cylinders, source: dipole, reference: "step-index fibre dispersion relations (TE₀₁, TM₀₁; Snyder & Love)", figure: [fibre-dispersion, fibre-rates-cutoff]}
- {geometry: cylinders, source: plane_wave, reference: "homogeneous cylinder fields; index-matched limit", figure: cylinder-index-matched}
- {geometry: cylinders, source: dipole, reference: "grazing light-line caps, bounded error", figure: [cylinder-grazing-caps-electric, cylinder-grazing-caps-magnetic]}
- {geometry: cylinders, source: plane_wave, reference: "canonical cylinder papers (Bohren & Huffman's infinite cylinder; coated fibres)", figure: [bohren-huffman-cylinder, ruan-fan-superscattering, ruan-fan-lossy, ruppin-sodium-wire]}
- {geometry: cylinders, source: dipole, reference: "homogeneous medium; independent flux budget; guided channels", figure: [cylinder-dipole-homogeneous, cylinder-coating-budget-electric, cylinder-coating-budget-magnetic, cylinder-guide-budget]}
- {geometry: cylinders, source: dipole, reference: "grazing endpoint continuation within the error budget", figure: cylinder-grazing-tolerance}
- {geometry: cylinders, source: dipole, reference: "**Meep**, 3D FDTD (the cylinder through the PML)", figure: meep-cylinder-fibre}
- {geometry: cylinders, source: dipole, reference: "**NGSolve** (3D finite elements; total field of a point dipole, Im(p·E) over the same mesh without the fibre)", figure: ngsolve-cylinder-fibre}
- {geometry: cylinders, source: focused, reference: "rows against plane-wave sums; whole beam against fixed quadrature", figure: [focused-cylinder-rows, focused-cylinder-whole-beam-x, focused-cylinder-whole-beam-y]}
- {geometry: cylinders, source: focused, reference: "Köhler against a plane-wave cone average; energy; NA → 0 is the plane wave", figure: [focused-cylinder-koehler, focused-cylinder-na-limit]}
- {geometry: cylinders, source: focused, reference: "near field against plane-wave fields", figure: focused-cylinder-near-field}
- {geometry: cylinders, source: focused, reference: "scalar model: coefficients, energy, brute-force sum", figure: [scalar-cylinder-coefficients, scalar-cylinder-rows]}
- {geometry: cylinders, source: focused, reference: "Davis, Carney & Bhargava, *Anal. Chem.* **83**, 525 (2011): FT-IR of intact fibres", figure: davis-2011-fibres}
---

The calculator computes every film, cylinder and sphere with PyStratify, an open-source Python package (GPL-3.0).
This page shows what PyStratify is checked against: closed forms and conservation laws, extended-precision
references, independent methods, other groups' codes and published papers.

Coverage at a glance places every test for the geometry the switch at the top picks: one circle per test, by the
kind of reference and the source of light. A circle opens the test's page: what exactly is calculated, the agreement
required, and a plot for each case, PyStratify as lines and the reference as circles on the same axes, with the
largest difference beneath it. The replication corpus is plotted record by record in the Replications.

A check counts as independent only when the reference shares no code with PyStratify. Closed forms and conservation
laws are independent of any code; extended-precision references prove the numerics; another group's code or a
different method (finite differences in time, finite elements, boundary elements, volume integrals) proves the
formulation. Every plot is computed by a script in the PyStratify repository, linked from its page, so anyone can
rerun it and see the numbers; one command, `benchmarks/verify_all.py`, reruns the whole study.
