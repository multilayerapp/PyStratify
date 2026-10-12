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
