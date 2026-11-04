"""Thin films, plane wave: PyStratify's pim_tmm against inc_tmm, a lineage check.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_plane_wave_tmm.py                                                                 #
#                                                                                                                   #
#####################################################################################################################

Its one figure is on no multilayer.app page: inc_tmm keeps the incoherent ladder of S. J. Byrnes' tmm, from which
PyStratify's planar engine descends, so agreement is a regression check (a lineage row of benchmarks/MATRIX.md), not
an independent verification, and the page that drew it, verification/films-plane-wave-tmm.html, was retired on
2026-11-04 (its old address redirects to the films table). The backside-reflection figure it also drew, PyStratify's
own consistency check, is computed by films_plane_wave_consistency.py. This file runs from a PyStratify clone; it
prints the figure's RMSE and writes, beside itself, films_plane_wave_tmm.json (the curves, value for value) and
films_plane_wave_tmm.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run
from _shared import INCOHERENT

from pystratify import planar


@figure("inc-tmm-lineage", "Incoherent stack", "λ [nm]", "R, T, A")
def plot_incoherent_stack():
    n, d, c = INCOHERENT
    lam = np.linspace(450, 900, 91)
    every = 3  # inc_tmm every 15 nm

    def rta(solve, w):
        out = solve("s", n, d, c, np.radians(31), w * 1e-3)
        return [out["R"], out["T"], 1 - out["R"] - out["T"]]

    ours = np.array([rta(planar.pim_tmm, w) for w in lam])
    ref = np.array([rta(planar.inc_tmm, w) for w in lam[::every]])
    labels = ("R", "T", "A")
    return result(
        lam,
        [line(q, f"{q}, PyStratify (pim_tmm)", ours[:, i]) for i, q in enumerate(labels)]
        + [dots(f"{q}_ref", f"{q}, inc_tmm", ref[:, i], every, size=lam.size) for i, q in enumerate(labels)],
        worst=largest(ours[::every], ref),
        worst_label="largest |difference|",
        note="inc_tmm is the incoherent transfer matrix of S. J. Byrnes' tmm package, from which PyStratify's "
             "planar engine descends; it is kept as a lineage check, not an independent one. 31°, s polarization; "
             "the backend's tests run 120 random stacks as well.")


if __name__ == "__main__":
    run(__file__)
