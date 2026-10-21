"""Thin films, plane wave: PyStratify against tmmax.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh tmmax                # the tmmax environment (benchmarks/envs/README.md)            #
#   ~/.multilayer/bench/tmmax/bin/python benchmarks/pages/films_plane_wave_tmmax.py                                 #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-plane-wave-tmmax.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, films_plane_wave_tmmax.json (the curves the page draws, value for value) and
films_plane_wave_tmmax.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

EVERY = 4  # tmmax every 20 nm, PyStratify every 5 nm


def _tmmax_figure(name, versions):
    """PyStratify's R of a STACKS entry at normal incidence, s, and tmmax's for each (label, scan step fixed?)."""
    from tmmax_films import STACKS, stratify_rt, tmmax_rt

    n, d = STACKS[name]
    lam = np.linspace(400, 900, 101)
    ours = stratify_rt(n, d, "s", lam, (0.0,))[0][:, 0]
    curves, diffs = [line("R", "R, PyStratify", ours)], []
    for label, fixed in versions:
        theirs = tmmax_rt(n, d, "s", fixed, lam[::EVERY], (0.0,))[0][:, 0]
        curves.append(dots(f"R_{int(fixed)}", f"R, tmmax 1.1.4 {label}", theirs, EVERY, size=lam.size))
        diffs.append(largest(theirs, ours[::EVERY]))
    return lam, curves, diffs


@figure("tmmax-single-layer", "Single layer", "λ [nm]", "R", env="tmmax")
def plot_single_layer():
    lam, curves, diffs = _tmmax_figure("single layer n = 2.1 in air", (("as shipped", False),))
    return result(
        lam, curves, worst=diffs[0], worst_label="largest |R − R_tmmax| (float32)",
        note="tmmax computes in single precision (its scan carries complex64), so agreement stops near 1e-7; single "
             "layers are asserted as shipped, to 2e-6.")


@figure("tmmax-bragg-mirror", "Quarter-wave mirror", "λ [nm]", "R", env="tmmax")
def plot_quarter_wave_mirror():
    lam, curves, diffs = _tmmax_figure("quarter-wave mirror, 5 pairs",
                                       (("with the scan step fixed", True), ("as shipped", False)))
    return result(
        lam, curves, worst=diffs[0], worst_label="largest |R − R_tmmax| (fixed step, float32)",
        note="tmmax 1.1.4, the version on PyPI, is wrong for two or more layers: its scan step broadcasts the 2×2 "
             "matrices as a stack of column vectors (reported upstream, bahremsd/tmmax #44, #45; fixed on its main "
             "branch, not released). With the step rebuilt (tmmax_films.py) it agrees to float32; as shipped it is "
             f"off by up to {diffs[1]:.2f}. The test asserts the fixed step on multilayers.")


if __name__ == "__main__":
    run(__file__)
