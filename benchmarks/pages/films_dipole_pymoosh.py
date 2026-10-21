"""Thin films, point dipole: PyStratify against PyMoosh.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_dipole_pymoosh.py                                                                 #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-dipole-pymoosh.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, films_dipole_pymoosh.json (the curves the page draws, value for value) and
films_dipole_pymoosh.png (needs matplotlib).
"""

from __future__ import annotations

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run
from _shared import guide_modes


@figure("pymoosh-guided-modes", "Slab modes", "thickness [nm]", "n_eff")
def plot_slab_modes():
    t, ours, theirs, diff, width = guide_modes("pymoosh")
    return result(
        t,
        [line(f"m{i}", f"mode {i + 1}, PyStratify", ours[:, i]) for i in range(width)]
        + [dots(f"m{i}_ref", f"mode {i + 1}, PyMoosh", theirs[:, i]) for i in range(width)],
        worst=diff,
        worst_label="largest |n_eff − n_eff(PyMoosh)|",
        note="Every TE and TM mode, numbered from the highest index: the poles PyStratify finds behind a dipole's "
             "guided channel against PyMoosh's guided_modes (its search stops at 1e-10). A mode appears at its "
             "cutoff, n_eff = 1. The test runs four guides.")


if __name__ == "__main__":
    run(__file__)
