"""Spheres, point dipole: PyStratify against smuthi.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh smuthi                # the smuthi environment (benchmarks/envs/README.md)          #
#   ~/.multilayer/bench/smuthi/bin/python benchmarks/pages/spheres_dipole_smuthi.py                                 #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-smuthi.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, spheres_dipole_smuthi.json (the curves the page draws, value for value) and
spheres_dipole_smuthi.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run


def smuthi_l_max(case, gap):
    """smuthi's truncation for ~1e-9 at `gap` from the sphere: it converges geometrically, as (R / (R + gap))^(2 l_max)
    close to the sphere and above the size parameter of the sphere in its host further away; capped at 40 (the
    report's ladder: 1.8e-6 for the metal-like sphere at 5 nm, 2.6e-5 for the large one at 50 nm)."""
    from smuthi_spheres import CASES

    wavelength, radius, _, host, _, _ = CASES[case]
    x = 2 * np.pi * host * radius / wavelength
    near = 9 * np.log(10) / (2 * np.log1p(gap / radius))
    return int(min(40, max(10, np.ceil(near), np.ceil(x + 4 * x ** (1 / 3) + 2))))


def _smuthi_sweep(case, gaps):
    """A case of benchmarks/smuthi_spheres.py swept outward from its own gap: PyStratify at 151 gaps, smuthi at every
    tenth of them (16)."""
    import smuthi_spheres as ss

    gaps = np.geomspace(*gaps, 151)
    every = 10
    theirs = np.array([ss.smuthi_rates(case, smuthi_l_max(case, g), g) for g in gaps[::every]])
    ours = np.array([ss.stratify(case, g) for g in gaps])
    return result(
        gaps,
        [line("radial", "radial, PyStratify", ours[:, 0]), line("tangential", "tangential, PyStratify", ours[:, 1]),
         dots("radial_ref", "radial, smuthi", theirs[:, 0], every, size=gaps.size),
         dots("tangential_ref", "tangential, smuthi", theirs[:, 1], every, size=gaps.size)],
        worst=np.max(np.abs(theirs / ours[::every] - 1)),
        worst_label="largest relative difference",
        note=f"smuthi's T-matrix truncated at l_max = {smuthi_l_max(case, gaps[0])} nearest the sphere, "
             f"{smuthi_l_max(case, gaps[::every][-1])} farthest; PyStratify at tolerance 1e-10.")


@figure("smuthi-sphere-dielectric", "Dielectric sphere", "gap [nm]", "Γ / Γ₀", env="smuthi")
def plot_dielectric_sphere():  # a dielectric sphere, n = 2, R = 150 nm, in air, 800 nm
    return _smuthi_sweep("dielectric, n = 2, R = 150 nm, 100 nm away", (100, 1000))


@figure("smuthi-sphere-metal", "Metal-like sphere", "gap [nm]", "Γ / Γ₀", log=True, env="smuthi")
def plot_metal_like_sphere():  # a metal-like sphere, ε = −10 + i, R = 20 nm, in air, 600 nm
    return _smuthi_sweep("metal-like, eps = -10 + i, R = 20 nm, 5 nm away", (5, 60))


@figure("smuthi-sphere-water", "Sphere in water", "gap [nm]", "Γ / Γ₀", env="smuthi")
def plot_sphere_in_water():  # a sphere of n = 1.5, R = 100 nm, in water (n = 1.33), 633 nm
    return _smuthi_sweep("in water, n = 1.5, R = 100 nm, 20 nm away", (20, 300))


@figure("smuthi-sphere-large", "Large sphere", "gap [nm]", "Γ / Γ₀", env="smuthi")
def plot_large_sphere():  # a large high-index sphere, n = 3.5 + 0.01i, R = 500 nm, in air, 800 nm
    return _smuthi_sweep("large high-index, n = 3.5 + 0.01i, R = 500 nm, 50 nm away", (50, 1000))


if __name__ == "__main__":
    run(__file__)
