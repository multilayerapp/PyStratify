"""Thin films, point dipole: PyStratify against smuthi.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh smuthi                # the smuthi environment (benchmarks/envs/README.md)          #
#   ~/.multilayer/bench/smuthi/bin/python benchmarks/pages/films_dipole_smuthi.py                                   #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-dipole-smuthi.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, films_dipole_smuthi.json (the curves the page draws, value for value) and
films_dipole_smuthi.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, line, result, run


@figure("smuthi-film-guide", "TiO₂ guide", "height above the film [nm]", "Γ / Γ₀", env="smuthi")
def plot_tio2_guide():
    import emitter_references as er

    n, d = er.GUIDES["air above 200 nm TiO2 on glass"]
    h = np.linspace(30, 300, 28)
    every = 3  # smuthi every 30 nm
    ours = np.array([er.stratify_rates(n, d, hi, tolerance=1e-8) for hi in h])
    ref = np.array([er.smuthi_rates(n, d, hi) for hi in h[::every]])
    return result(
        h,
        [line("perp", "perpendicular, PyStratify", ours[:, 0]), line("par", "parallel, PyStratify", ours[:, 1]),
         dots("perp_ref", "perpendicular, smuthi", ref[:, 0], every, size=h.size),
         dots("par_ref", "parallel, smuthi", ref[:, 1], every, size=h.size)],
        worst=np.max(np.abs(ref / ours[::every] - 1)),
        worst_label="largest relative difference (smuthi's contour)",
        note="smuthi's DipoleSource.dissipated_power integrates along its own Sommerfeld contour deflected around "
             "the guide's poles (resolution 2e-3), a method independent of PyStratify's residues; agreement is set "
             "by that contour's resolution.")


if __name__ == "__main__":
    run(__file__)
