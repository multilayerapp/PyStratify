"""Thin films, plane wave: PyStratify against tmm.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_plane_wave_tmm.py                                                                 #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-plane-wave-tmm.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, films_plane_wave_tmm.json (the curves the page draws, value for value) and
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


@figure("backside-reflection", "Backside reflection", "λ [nm]", "R, Re r_s r̄_p")
def plot_backside_reflection():
    from ellipsometry_codes import backside_phase_average

    from pystratify import dispersion as dm
    from pystratify import ellipsometry as el

    film, glass, thickness, angle = dm.Cauchy(2.0, 0.01), dm.Constant(1.52), 1e5, 65.0
    sample = el.Sample([el.Layer(film, 120.0)], substrate=glass, substrate_thickness=thickness)
    lam = np.linspace(400.0, 900.0, 51)
    every = 2  # the references every 20 nm
    ss, pp, sp, _, _ = el._response(sample, lam, np.full(lam.size, angle))
    inc_s, inc_p, avg_sp = [], [], []
    for w in lam[::every]:
        n = [1.0, film.index(w), 1.52, 1.0]
        d, c = [np.inf, 120.0, thickness, np.inf], ["i", "c", "i", "i"]
        inc_s.append(planar.inc_tmm("s", n, d, c, np.radians(angle), w)["R"])
        inc_p.append(planar.inc_tmm("p", n, d, c, np.radians(angle), w)["R"])
        avg_sp.append(backside_phase_average(n, d, w, angle)[2])
    inc_s, inc_p, avg_sp = map(np.array, (inc_s, inc_p, avg_sp))
    return result(
        lam,
        [line("ss", "Σ|r_s|², PyStratify", np.real(ss)), line("pp", "Σ|r_p|², PyStratify", np.real(pp)),
         line("sp", "Re Σ r_s r̄_p, PyStratify", np.real(sp)),
         dots("ss_ref", "R_s, inc_tmm", inc_s, every, size=lam.size),
         dots("pp_ref", "R_p, inc_tmm", inc_p, every, size=lam.size),
         dots("sp_ref", "Re r_s r̄_p, phase average", np.real(avg_sp), every, size=lam.size)],
        worst=max(largest(np.real(ss[::every]), inc_s), largest(np.real(pp[::every]), inc_p),
                  float(np.max(np.abs(np.asarray(sp)[::every] - avg_sp)))),
        worst_label="largest |difference|",
        note="Light returning from a thick substrate's back adds in power, so an ellipsometer sees sums over the "
             "partial waves: Σ|r_s|² and Σ|r_p|² are inc_tmm's R, and the cross term Σ r_s r̄_p, which sets the "
             "depolarisation, equals r_s r̄_p of the coherent stack averaged over one period of the substrate's "
             "round-trip phase (64 coherent solves per point, benchmarks/ellipsometry_codes.py "
             "backside_phase_average).")


if __name__ == "__main__":
    run(__file__)
