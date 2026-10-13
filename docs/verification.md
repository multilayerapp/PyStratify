---
title: How PyStratify is verified
summary: >-
  Every check behind multilayer.app's engine, geometry by geometry and source by source: closed forms,
  extended-precision references, independent methods, other groups' codes and published papers, each with its
  script, its tolerance and whether it runs on every test.
examples:
- {geometry: films, source: plane_wave, reference: "Fresnel at oblique incidence; Brewster zero; s = p at normal", example: fresnel-air-glass}
- {geometry: films, source: plane_wave, reference: "Airy slab; quarter-wave Bragg mirror", example: bragg-mirror}
- {geometry: spheres, source: plane_wave, reference: "textbook Mie (homogeneous); identical shells = one sphere", example: si-sphere-mie}
figures:
- {geometry: films, source: plane_wave, reference: "Fresnel at oblique incidence; Brewster zero; s = p at normal", figure: fresnel-oblique}
- {geometry: films, source: plane_wave, reference: "Airy slab; quarter-wave Bragg mirror", figure: airy-slab}
- {geometry: films, source: plane_wave, reference: "R + T = 1 for lossless stacks, below and beyond the critical angle; reciprocity under reversal", figure: energy-conservation}
- {geometry: films, source: dipole, reference: "Sommerfeld integrals over s = k_ρ/k₁ with Parratt-recursion r_s, r_p, written without `pystratify`", figure: sommerfeld-film-emitter}
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
- {geometry: cylinders, source: dipole, reference: "step-index fibre dispersion relations (TE₀₁, TM₀₁; Snyder & Love)", figure: fibre-dispersion}
---

The calculator computes every film, cylinder and sphere with PyStratify, an open-source Python package (GPL-3.0).
This page lists what PyStratify is checked against. One row is one comparison: a reference, the quantity compared,
the script that runs it, the tolerance it must meet and whether the test suite asserts it on every change.

A check counts as independent only when the reference shares no code with PyStratify. Closed forms and conservation
laws are independent of any code; extended-precision references prove the numerics; another group's code or a
different method (finite differences in time, finite elements, boundary elements, volume integrals) proves the
formulation. Every comparison is a script in the PyStratify repository, so anyone can rerun it and see the numbers;
the rows marked *suite* run on every commit.

Rows the calculator can open show a worked example; the switch at the top shows one geometry at a time.
