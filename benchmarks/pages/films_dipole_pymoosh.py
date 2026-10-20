"""Thin films, point dipole: PyStratify against pymoosh.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_dipole_pymoosh.py                                                                 #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-dipole-pymoosh.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
films_dipole_pymoosh.json (the curves the page draws, value for value) and films_dipole_pymoosh.png (needs matplotlib).
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
from _shared import _guide_modes  # noqa: E402


@figure("pymoosh-guided-modes", "Slab modes", "thickness [nm]", "n_eff")
def pymoosh_guided_modes():
    t, ours, theirs, diff, width = _guide_modes("pymoosh")
    return dict(x=sig(t), series=[series(f"m{i}", f"mode {i + 1}, PyStratify", ours[:, i], "pystratify") for i in range(width)]
                + [series(f"m{i}_ref", f"mode {i + 1}, PyMoosh", theirs[:, i], "reference") for i in range(width)],
                worst=diff, worst_label="largest |n_eff − n_eff(PyMoosh)|",
                note="Every TE and TM mode, numbered from the highest index: the poles PyStratify finds behind a dipole's guided channel "
                     "against PyMoosh's guided_modes (its search stops at 1e-10). A mode appears at its cutoff, n_eff = 1. The test runs four guides.")


if __name__ == "__main__":
    run(__file__)
