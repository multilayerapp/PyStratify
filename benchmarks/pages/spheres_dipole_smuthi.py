"""Spheres, point dipole: PyStratify against smuthi.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh smuthi                # the smuthi environment (benchmarks/envs/README.md)          #
#   ~/.multilayer/bench/smuthi/bin/python benchmarks/pages/spheres_dipole_smuthi.py                                 #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-smuthi.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
spheres_dipole_smuthi.json (the curves the page draws, value for value) and spheres_dipole_smuthi.png (needs matplotlib).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "benchmarks", ROOT / "benchmarks" / "pages"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


from _page import figure, run, series, sig  # noqa: E402


# smuthi (benchmarks/smuthi_spheres.py): each of its four cases swept outward from the case's own gap. smuthi's
# truncation l_max converges geometrically, as (R / (R + gap))^(2 l_max) close to the sphere and above the size
# parameter of the sphere in its host further away; it is taken from that estimate for ~1e-9, and capped at 40
# (the report's ladder: 1.8e-6 for the metal-like sphere at 5 nm, 2.6e-5 for the large one at 50 nm).
SMUTHI_SWEEPS = (
    # slug, case, gaps [nm], logarithmic axis, description
    ("dielectric", "dielectric, n = 2, R = 150 nm, 100 nm away", (100, 1000), False,
     "a dielectric sphere, n = 2, R = 150 nm, in air, 800 nm"),
    ("metal", "metal-like, eps = -10 + i, R = 20 nm, 5 nm away", (5, 60), True,
     "a metal-like sphere, ε = −10 + i, R = 20 nm, in air, 600 nm"),
    ("water", "in water, n = 1.5, R = 100 nm, 20 nm away", (20, 300), False,
     "a sphere of n = 1.5, R = 100 nm, in water (n = 1.33), 633 nm"),
    ("large", "large high-index, n = 3.5 + 0.01i, R = 500 nm, 50 nm away", (50, 1000), False,
     "a large high-index sphere, n = 3.5 + 0.01i, R = 500 nm, in air, 800 nm"),
)


def smuthi_l_max(case, gap):
    from smuthi_spheres import CASES

    wavelength, radius, _, host, _, _ = CASES[case]
    x = 2 * np.pi * host * radius / wavelength
    near = 9 * np.log(10) / (2 * np.log1p(gap / radius))
    return int(min(40, max(10, np.ceil(near), np.ceil(x + 4 * x ** (1 / 3) + 2))))


def smuthi_sweep(case, gaps):
    """PyStratify as a line (151 gaps) and smuthi at every tenth of them (16): the reference series is null between."""
    import smuthi_spheres as ss

    import smuthi  # noqa: F401  (ImportError: the figure is skipped)
    dense = np.geomspace(*gaps, 151)
    ours = np.array([ss.stratify(case, g) for g in dense])
    sampled = dense[::10]
    theirs = np.array([ss.smuthi_rates(case, smuthi_l_max(case, g), g) for g in sampled])
    rel = float(np.max(np.abs(theirs / ours[::10] - 1)))
    sparse = np.full_like(ours, np.nan)
    sparse[::10] = theirs
    return dict(x=sig(dense), series=[
        series("radial", "radial, PyStratify", ours[:, 0], "pystratify"), series("tangential", "tangential, PyStratify", ours[:, 1], "pystratify"),
        series("radial_ref", "radial, smuthi", sparse[:, 0], "reference"), series("tangential_ref", "tangential, smuthi", sparse[:, 1], "reference")],
        worst=rel, worst_label="largest relative difference",
        note=f"smuthi's T-matrix truncated at l_max = {smuthi_l_max(case, sampled[0])} nearest the sphere, "
             f"{smuthi_l_max(case, sampled[-1])} farthest; PyStratify at tolerance 1e-10.")


for _slug, _case, _gaps, _log, _what in SMUTHI_SWEEPS:
    def _make(case=_case, gaps=_gaps):
        return smuthi_sweep(case, gaps)
    _make.__name__ = f"smuthi_{_slug}"
    figure(f"smuthi-sphere-{_slug}", {"dielectric": "Dielectric sphere", "metal": "Metal-like sphere", "water": "Sphere in water", "large": "Large sphere"}[_slug],
           "gap [nm]", "Γ / Γ₀", log=_log, env="smuthi")(_make)


if __name__ == "__main__":
    run(__file__)
