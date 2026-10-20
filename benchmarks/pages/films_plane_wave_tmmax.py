"""Thin films, plane wave: PyStratify against tmmax.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh tmmax                # the tmmax environment (benchmarks/envs/README.md)            #
#   ~/.multilayer/bench/tmmax/bin/python benchmarks/pages/films_plane_wave_tmmax.py                                 #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-plane-wave-tmmax.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
films_plane_wave_tmmax.json (the curves the page draws, value for value) and films_plane_wave_tmmax.png (needs matplotlib).
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


def _tmmax_figure(name, versions):
    from tmmax_films import STACKS, stratify_rt, tmmax_rt

    n, d = STACKS[name]
    lam = np.linspace(400, 900, 101)
    R0, _ = stratify_rt(n, d, "s", lam, (0.0,))
    out = [series("R", "R, PyStratify", R0[:, 0], "pystratify")]
    diffs = []
    for label, fixed in versions:
        R, _ = tmmax_rt(n, d, "s", fixed, lam, (0.0,))
        out.append(series(f"R_{int(fixed)}", f"R, tmmax 1.1.4 {label}", R[:, 0], "reference"))
        diffs.append(float(np.max(np.abs(R[:, 0] - R0[:, 0]))))
    return lam, out, diffs


@figure("tmmax-single-layer", "Single layer", "λ [nm]", "R", env="tmmax")
def tmmax_single_layer():
    lam, out, diffs = _tmmax_figure("single layer n = 2.1 in air", (("as shipped", False),))
    return dict(x=sig(lam), series=out, worst=diffs[0], worst_label="largest |R − R_tmmax| (float32)",
                note="tmmax computes in single precision (its scan carries complex64), so agreement stops near 1e-7; single layers are "
                     "asserted as shipped, to 2e-6.")


@figure("tmmax-bragg-mirror", "Quarter-wave mirror", "λ [nm]", "R", env="tmmax")
def tmmax_bragg_mirror():
    lam, out, diffs = _tmmax_figure("quarter-wave mirror, 5 pairs", (("with the scan step fixed", True), ("as shipped", False)))
    return dict(x=sig(lam), series=out, worst=diffs[0], worst_label="largest |R − R_tmmax| (fixed step, float32)",
                note="tmmax 1.1.4, the version on PyPI, is wrong for two or more layers: its scan step broadcasts the 2×2 matrices as a "
                     "stack of column vectors (reported upstream, bahremsd/tmmax #44, #45; fixed on its main branch, not released). With the "
                     f"step rebuilt (tmmax_films.py) it agrees to float32; as shipped it is off by up to {diffs[1]:.2f}. The test asserts the "
                     "fixed step on multilayers.")


if __name__ == "__main__":
    run(__file__)
