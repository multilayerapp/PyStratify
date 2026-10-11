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
