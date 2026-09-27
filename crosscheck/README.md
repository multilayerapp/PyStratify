# Cross-checks against published figures

Scripts that recompute figures of papers from their stated parameters, so that the
curves can be laid over the published ones. Each writes CSV files and a PNG to
`results/`.

| Script | Paper | What to compare |
|---|---|---|
| `guzatov_klimov_2012.py` | D. V. Guzatov & V. V. Klimov, *New J. Phys.* **14**, 123009 (2012), [arXiv:1203.5393](https://arxiv.org/abs/1203.5393) | Figs. 3–6: radiative decay of a chiral molecule at the surface of a chiral (Drude–Born–Fedorov) sphere |

Conventions (`_papers.py`):

* **Drude–Born–Fedorov → Pasteur.** D = ε(E + η∇×E), B = μ(H + η∇×H), χ = k₀η map to
  ε_P = ε/s, μ_P = μ/s, κ = n²χ/s with s = 1 − n²χ² (same impedance; n ± κ = n/(1 ∓ nχ),
  their k_L, k_R). The mapping reproduces their quasi-static polarisabilities (Eq. 48) in
  `tests/test_chiral.py`. Double-negative media need a small loss to select n < 0.
* **Molecule.** ⟨e|d|g⟩ = d₀ and ⟨e|m|g⟩ = −i m₀ with m₀ = ξ d₀ (ξ > 0 "right", ξ < 0
  "left") are the source (p, m) = (d₀, −iξ d₀). Rates over (4k₀³/3ħ)(|d₀|² + |m₀|²) are
  PyStratify's host normalisation.
* **Check already in the test suite.** Their quasi-static, orientation-averaged Eq. 46 is
  reproduced to O((k₀a)²) for dielectric, lossy, metallic and double-negative chiral spheres
  and both enantiomers (`tests/test_rates.py::test_chiral_molecule_near_small_chiral_sphere_guzatov_klimov`).

Open points, to settle against the PDF:

* The text gives no orientation for Figs. 4–6 and no k₀a for Fig. 6; the script uses
  orientation averages and k₀a = 0.1. If a caption says otherwise, change `rate(...)`.
* Fig. 5 at k₀a = 0.1: the exact rates have a second resonance near ε′ ≈ 2.9 — the magnetic
  quadrupole of a μ ≈ −3/2 sphere — that the dipole formula (Eq. 46) does not contain. If the
  published Fig. 5 shows it, it was computed exactly; if not, from Eq. 46.
* Fig. 6: the extreme discrimination depends strongly on k₀a and orientation, so the caption
  decides what to compare. Scans over ε′ ∈ [−1.5, 1.5], μ′ ∈ [−2.6, −1.4] (0.01i loss, χ = 0.2)
  give max(Γ_L/Γ_R, Γ_R/Γ_L) ≈ 14 (isotropic), 6 (tangential) and > 5000 (normal: one
  enantiomer's radiation nearly cancels) at k₀a = 0.1, and ≈ 50 (isotropic) at k₀a = 0.05.
  A secondary summary of the paper mentions factors of 25–30 — unverified.

Pending: the 2021 *J. Chem. Phys.* benchmark mentioned earlier — its details were not
recorded; add a script here once the PDF is available.
