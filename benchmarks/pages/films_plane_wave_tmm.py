"""Thin films, plane wave: PyStratify against tmm.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_plane_wave_tmm.py                                                                 #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-plane-wave-tmm.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
films_plane_wave_tmm.json (the curves the page draws, value for value) and films_plane_wave_tmm.png (needs matplotlib).
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
from _shared import INCOHERENT  # noqa: E402


@figure("inc-tmm-lineage", "Incoherent stack", "λ [nm]", "R, T, A")
def inc_tmm_lineage():
    n, d, c = INCOHERENT
    lam = np.linspace(450, 900, 91)
    ours, ref = [], []
    for w in lam:
        a = planar.pim_tmm("s", n, d, c, np.radians(31), w * 1e-3)
        b = planar.inc_tmm("s", n, d, c, np.radians(31), w * 1e-3)
        ours.append([a["R"], a["T"], 1 - a["R"] - a["T"]]), ref.append([b["R"], b["T"], 1 - b["R"] - b["T"]])
    ours, ref = np.array(ours), np.array(ref)
    labels = ("R", "T", "A")
    return dict(x=sig(lam), series=[series(q, f"{q}, PyStratify (pim_tmm)", ours[:, i], "pystratify") for i, q in enumerate(labels)]
                + [series(f"{q}_ref", f"{q}, inc_tmm", ref[:, i], "reference") for i, q in enumerate(labels)],
                worst=worst(ours, ref), worst_label="largest |difference|",
                note="inc_tmm is the incoherent transfer matrix of S. J. Byrnes' tmm package, from which PyStratify's planar engine "
                     "descends; it is kept as a lineage check, not an independent one. 31°, s polarization; the backend's tests run 120 random "
                     "stacks as well.")


@figure("backside-reflection", "Backside reflection", "λ [nm]", "R, Re r_s r̄_p")
def backside_reflection():
    from ellipsometry_codes import backside_phase_average
    from pystratify import dispersion as dm
    from pystratify import ellipsometry as el

    film, glass, thickness, angle = dm.Cauchy(2.0, 0.01), dm.Constant(1.52), 1e5, 65.0
    sample = el.Sample([el.Layer(film, 120.0)], substrate=glass, substrate_thickness=thickness)
    lam = np.linspace(400.0, 900.0, 51)
    ss, pp, sp, _, _ = el._response(sample, lam, np.full(lam.size, angle))
    inc_s, inc_p, avg_sp = [], [], []
    for w in lam:
        n = [1.0, film.index(w), 1.52, 1.0]
        d, c = [np.inf, 120.0, thickness, np.inf], ["i", "c", "i", "i"]
        inc_s.append(planar.inc_tmm("s", n, d, c, np.radians(angle), w)["R"])
        inc_p.append(planar.inc_tmm("p", n, d, c, np.radians(angle), w)["R"])
        avg_sp.append(backside_phase_average(n, d, w, angle)[2])
    inc_s, inc_p, avg_sp = map(np.array, (inc_s, inc_p, avg_sp))
    worst_value = max(worst(ss, inc_s), worst(pp, inc_p), float(np.max(np.abs(np.asarray(sp) - avg_sp))))
    return dict(x=sig(lam), series=[series("ss", "Σ|r_s|², PyStratify", np.real(ss), "pystratify"), series("pp", "Σ|r_p|², PyStratify", np.real(pp), "pystratify"),
                                    series("sp", "Re Σ r_s r̄_p, PyStratify", np.real(sp), "pystratify"),
                                    series("ss_ref", "R_s, inc_tmm", inc_s, "reference"), series("pp_ref", "R_p, inc_tmm", inc_p, "reference"),
                                    series("sp_ref", "Re r_s r̄_p, phase average", np.real(avg_sp), "reference")],
                worst=float(worst_value), worst_label="largest |difference|",
                note="Light returning from a thick substrate's back adds in power, so an ellipsometer sees sums over the partial waves: "
                     "Σ|r_s|² and Σ|r_p|² are inc_tmm's R, and the cross term Σ r_s r̄_p, which sets the depolarisation, equals r_s r̄_p of the "
                     "coherent stack averaged over one period of the substrate's round-trip phase (64 coherent solves per point, "
                     "benchmarks/ellipsometry_codes.py backside_phase_average).")


if __name__ == "__main__":
    run(__file__)
