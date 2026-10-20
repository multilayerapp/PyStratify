"""Spheres, plane wave: PyStratify against bhmie.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_plane_wave_bhmie.py                                                             #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-plane-wave-bhmie.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
spheres_plane_wave_bhmie.json (the curves the page draws, value for value) and spheres_plane_wave_bhmie.png (needs matplotlib).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "benchmarks", ROOT / "benchmarks" / "pages"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pystratify as ps  # noqa: E402

from _page import figure, run, series, sig, worst  # noqa: E402


@figure("bhmie-large-spheres", "Large absorbing spheres", "size parameter x", "Q_ext")
def bhmie_large_spheres():
    from pystratify.references import bhmie

    xs = np.geomspace(1, 5000, 120)
    m = 1.5 + 0.1j
    ours, ref = [], []
    for x in xs:
        R = x / (2 * np.pi)
        lmax = int(x + 4 * x ** (1 / 3) + 2) + 20  # both codes 20 orders beyond Wiscombe's truncation
        sol = ps.solve([R], np.array([[m, 1.0]]), np.array([1.0]), l_max=lmax)
        ours.append(ps.cross_sections(sol).q_ext[0])
        ref.append(bhmie(x, m, lmax)[0])
    return dict(x=sig(xs), series=[series("q", "Q_ext, PyStratify", ours, "pystratify"), series("q_ref", "Q_ext, BHMIE (Bohren & Huffman, Appendix A)", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="BHMIE's downward recurrence for the logarithmic derivative, ported from the book's Appendix A, against the logarithmic transfer-matrix recursion, both summed 20 orders beyond Wiscombe's truncation; the x axis is logarithmic and the extinction tends to 2 with the ripple of the edge waves. For a weakly absorbing or lossless sphere BHMIE's double-precision recurrence drifts at large x (3e-4 in Q_ext at x = 1481 for m = 1.33 + 1e-4 i, 6e-4 for m = 1.33), where PyStratify agrees with the same series summed in 40-digit arithmetic (pystratify.references.sphere_extinction) to 2e-13: the absorbing case here is the one the test suite asserts.")


if __name__ == "__main__":
    run(__file__)
