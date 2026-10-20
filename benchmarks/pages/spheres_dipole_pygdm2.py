"""Spheres, point dipole: PyStratify against pygdm2.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_dipole_pygdm2.py                                                                #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-pygdm2.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
spheres_dipole_pygdm2.json (the curves the page draws, value for value) and spheres_dipole_pygdm2.png (needs matplotlib).
"""

from __future__ import annotations

import json  # noqa: F401
import sys
from pathlib import Path

import numpy as np
from numpy import inf  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "benchmarks", ROOT / "benchmarks" / "pages"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pystratify as ps  # noqa: E402,F401
from pystratify import planar  # noqa: E402,F401

from _page import figure, run, series, sig, worst  # noqa: E402,F401


@figure("gdm-decay", "Dielectric sphere", "distance from the surface [nm]", "Γ / Γ₀")
def gdm_decay():
    from gdm_spheres import gdm_rates, gdm_simulation

    gaps = np.linspace(10, 100, 10)
    simulation, dipoles, r_eff = gdm_simulation(40.0, 2.0, 6.0)
    radial, tangential = gdm_rates(simulation, dipoles, 40.0, gaps)
    dense = np.linspace(10, 100, 91)
    rates = ps.decay_rates([r_eff], [2.0, 1.0], 600.0, 40.0 + dense)
    rad_ref, tan_ref = np.full(dense.size, np.nan), np.full(dense.size, np.nan)
    rad_ref[::10], tan_ref[::10] = radial, tangential
    rel = max(np.max(np.abs(radial / rates.total[::10, 0] - 1)), np.max(np.abs(tangential / rates.total[::10, 1] - 1)))
    return dict(x=sig(dense), series=[series("rad", "radial, PyStratify", rates.total[:, 0], "pystratify"), series("tan", "tangential, PyStratify", rates.total[:, 1], "pystratify"),
                                      series("rad_ref", f"radial, pyGDM2 ({len(dipoles)} dipoles)", rad_ref, "reference"),
                                      series("tan_ref", f"tangential, pyGDM2 ({len(dipoles)} dipoles)", tan_ref, "reference")],
                worst=float(rel), worst_label="largest relative difference",
                note="pyGDM2's decay_rate couples the emitter to the dipole lattice through the Green dyadic; distances from the centroid of "
                     "the lattice, PyStratify for the sphere of the same volume. The lattice is coarse close to the surface (the test: 10% at "
                     "20 nm, 0.5% at 40 nm).")


if __name__ == "__main__":
    run(__file__)
