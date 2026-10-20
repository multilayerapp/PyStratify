# Benchmarks

Every comparison of PyStratify with another code or an extended-precision reference is a script here that writes its
own report, so anyone can rerun it; the tests import them. [MATRIX.md](MATRIX.md) maps every check, existing and
planned, by geometry, source and quantity; `python benchmarks/verify_all.py` reruns the whole study in the pinned
environments of [envs/README.md](envs/README.md). Each page of the [Verification section](https://multilayer.app/verification.html)
is computed by one script, [pages/](pages/)`<page>.py`, which a reader runs from a clone to reproduce that page's figures
value for value; `python benchmarks/figures.py` writes them, with the calculator setup each page freezes
([figure_setups.py](figure_setups.py)), to `docs/figures/`.

`python benchmarks/convergence.py` reruns the test cases of the multilayered-sphere literature
against independent references and writes the report in
[benchmarks/RESULTS.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/RESULTS.md): decay rates near silver and silicon spheres down
to 0.25-nm gaps (Majic & Le Ru 2020) and inside and outside gold nanoshells and a matryoshka
(against high-precision transfer matrices), a dipole in a layered magnetic sphere and a Luneburg lens
(Yuan, Zhu & Zhu 2023, 2024), 1000-layer spheres at x = 1000, a graded 250-µm droplet at the
rainbow angle and the Cauchy profile (Wu & Wang 1991; Wu et al. 1997), absorbing spheres to
x = 20000, and fields far from the particle.

The planar solver descends from Byrnes's `tmm`, so it is checked against two codes outside that
lineage: `python benchmarks/planar_codes.py` (after `pip install -e ".[benchmarks]"`) compares it with
[PyMoosh](https://github.com/AnMoreau/PyMoosh) (scattering matrices) and
[pyElli](https://github.com/PyEllips/pyElli) (2×2 and Berreman 4×4) and writes
[benchmarks/PLANAR_CODES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/PLANAR_CODES.md): R, T, r, t, absorption per layer,
Ψ and Δ agree to 1e-14 in power and 6e-13° in angle, through opaque and 20-µm absorbing layers,
total internal reflection, a magnetic layer and an incoherent substrate; plane-wave fields agree with
PyMoosh to 2e-14, and the guided modes behind a film emitter's guided channel with PyMoosh's mode
search (1e-11) and the slab dispersion relation (7e-14).
`python benchmarks/dispersion_codes.py` compares every `pystratify.dispersion` model with pyElli's
([benchmarks/DISPERSION_CODES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/DISPERSION_CODES.md)),
and `python benchmarks/ellipsometry_codes.py` (with `lmfit`) the ellipsometry module: Ψ, Δ of dispersive stacks
with both of pyElli's solvers, the Woollam and SENTECH readers, and a fit of a real TiO₂ measurement by both codes
([benchmarks/ELLIPSOMETRY_CODES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/ELLIPSOMETRY_CODES.md)).
`python benchmarks/emitter_references.py` checks film emitters: decay rates against Sommerfeld
integrals written without `pystratify` (themselves pinned to the image dipole at a perfect mirror, to
5e-12), and near lossless guides against [smuthi](https://gitlab.com/AmosEgel/smuthi) and the
lossless limit of an absorbing guide
([benchmarks/EMITTER_REFERENCES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/EMITTER_REFERENCES.md)).
`python benchmarks/mnpbem_spheres.py` checks spheres by a different method, the boundary elements of
[MNPBEM](https://doi.org/10.17632/gbyj97hfnc.1) run under GNU Octave (`benchmarks/mnpbem_octave.py`
fetches the CPC program, verifies it and builds an Octave-ready copy): cross sections and decay rates
converge toward PyStratify as the mesh is refined, to 0.5–4% at the finest mesh
([benchmarks/MNPBEM_SPHERES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/MNPBEM_SPHERES.md)).
`python benchmarks/mie_codes.py` compares spheres with two independent Mie codes,
[scattnlay](https://github.com/ovidiopr/scattnlay) and [treams](https://github.com/tfp-photonics/treams):
cross sections, backscattering and amplitudes of ten spheres up to x = 1000 (nanoshells, a
matryoshka, a 10-layer graded sphere) and the T-matrices of chiral multilayers agree to 2e-11 once
20 orders beyond Wiscombe's truncation are kept, and to 4e-8 at the default truncation
([benchmarks/MIE_CODES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/MIE_CODES.md)).
`python benchmarks/gdm_spheres.py` adds a third method, the volume Green dyadic method of
[pyGDM2](https://gitlab.com/wiechapeter/pyGDM2): dielectric spheres agree within 3% in cross sections
and 0.2% in decay rates 40 nm from the surface, at the volume-equivalent radius
([benchmarks/GDM_SPHERES.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/GDM_SPHERES.md)). Every comparison is a
script in `benchmarks/` that writes its own report, so it can be rerun by anyone; the tests import
them. [benchmarks/MATRIX.md](https://github.com/multilayerapp/PyStratify/blob/main/benchmarks/MATRIX.md) maps every check, existing and planned, by
geometry, source and quantity.
