"""Thin films, point dipole: PyStratify against smuthi.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   sh benchmarks/envs/setup.sh smuthi                # the smuthi environment (benchmarks/envs/README.md)          #
#   ~/.multilayer/bench/smuthi/bin/python benchmarks/pages/films_dipole_smuthi.py                                   #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-dipole-smuthi.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
films_dipole_smuthi.json (the curves the page draws, value for value) and films_dipole_smuthi.png (needs matplotlib).
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


@figure("smuthi-film-guide", "TiO₂ guide", "height above the film [nm]", "Γ / Γ₀", env="smuthi")
def smuthi_film_guide():
    import emitter_references as er

    n, d = er.GUIDES["air above 200 nm TiO2 on glass"]
    h = np.linspace(30, 300, 28)
    ours = np.array([er.stratify_rates(n, d, hi, tolerance=1e-8) for hi in h])
    ref = np.full(ours.shape, np.nan)
    for i in range(0, h.size, 3):
        ref[i] = er.smuthi_rates(n, d, h[i])
    m = np.isfinite(ref[:, 0])
    return dict(x=sig(h), series=[series("perp", "perpendicular, PyStratify", ours[:, 0], "pystratify"), series("par", "parallel, PyStratify", ours[:, 1], "pystratify"),
                                  series("perp_ref", "perpendicular, smuthi", ref[:, 0], "reference"), series("par_ref", "parallel, smuthi", ref[:, 1], "reference")],
                worst=float(np.max(np.abs(ref[m] / ours[m] - 1))), worst_label="largest relative difference (smuthi's contour)",
                note="smuthi's DipoleSource.dissipated_power integrates along its own Sommerfeld contour deflected around the guide's poles "
                     "(resolution 2e-3), a method independent of PyStratify's residues; agreement is set by that contour's resolution.")


if __name__ == "__main__":
    run(__file__)
