"""Size of the STRATIFY deviations on the configurations STRATIFY itself uses.

Runs the original MATLAB (GNU Octave) and PyStratify side by side and prints
a Markdown table.  Usage:  python audit/impact.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pystratify as ps  # noqa: E402
from tests import octave  # noqa: E402
from tests.octave import mat  # noqa: E402

LAM = 614e-9
AU614 = complex(ps.refractive_index("Au_JC", 614.0))


def edcy(rad, ref, rd, l, norm, rin=200):
    body = (
        f"rad={mat(rad)}; ref={mat(ref)}; mu=ones(1,{len(ref)}); lam={LAM:.17g}; l=1:{l};\n"
        f"T=t_mat(rad,ref,mu,lam,l); [gr,gnr]=edcy(rad,{mat(rd)},ref,mu,lam,l,{rin},T,'{norm}');"
    )
    o = octave.run(body, ["gr", "gnr"])
    return np.atleast_2d(o["gr"]), np.atleast_2d(o["gnr"])


def pct(a, b):
    return 100 * (np.asarray(a) / np.asarray(b) - 1)


def row(label, matlab, ours):
    m, p = DecayRatesAvg(matlab), DecayRatesAvg(ours)
    return f"| {label} | {m:.4g} | {p:.4g} | {pct(m, p):+.1f}% |"


def DecayRatesAvg(a):
    a = np.atleast_2d(a)
    return float((a[0, 0] + 2 * a[0, 1]) / 3)


def main():
    lines = ["| configuration | STRATIFY | PyStratify | STRATIFY error |", "|---|---|---|---|"]

    # 1. tst_dcy.m / OSAC Fig. 2(c): Au@SiO2 {50,70} nm in water, 614 nm, host normalisation, l = 1:100
    rad = np.array([50, 70]) * 1e-9
    ref = [AU614, 1.45, 1.33]
    for rd_nm, where in (
        (52, "SiO2 shell, 2 nm from Au"),
        (60, "SiO2 shell, 10 nm from Au"),
        (71, "water, 1 nm outside"),
    ):
        gr, gnr = edcy(rad, ref, [rd_nm * 1e-9], 100, "host")
        ours = ps.decay_rates(rad, ref, [1, 1, 1], LAM, [rd_nm * 1e-9], norm="host", tol=1e-8)
        lines.append(row(f"Au@SiO2 (tst_dcy), r_d={rd_nm} nm ({where}): Gamma_rad", gr, ours.radiative))
        lines.append(row(f"Au@SiO2 (tst_dcy), r_d={rd_nm} nm ({where}): Gamma_nrad", gnr, ours.nonradiative))

    # 2. SiO2@Au nanoshell {50,55} nm in water (absorbing shell -> M2)
    rad = np.array([50, 55]) * 1e-9
    ref = [1.45, AU614, 1.33]
    for rd_nm, where in ((30, "inside the SiO2 core"), (60, "water, 5 nm outside"), (70, "water, 15 nm outside")):
        gr, gnr = edcy(rad, ref, [rd_nm * 1e-9], 100, "host", rin=2000)
        ours = ps.decay_rates(rad, ref, [1, 1, 1], LAM, [rd_nm * 1e-9], norm="host", tol=1e-8)
        lines.append(row(f"SiO2@Au nanoshell, r_d={rd_nm} nm ({where}): Gamma_rad", gr, ours.radiative))
        lines.append(row(f"SiO2@Au nanoshell, r_d={rd_nm} nm ({where}): Gamma_nrad", gnr, ours.nonradiative))

    # 3. energy prefactor for Au (Ordal) across the visible
    src = (octave.STRATIFY / "energy" / "G_prefac.m").read_text(encoding="latin-1")
    case = src.split("case 'Au_Ord'")[1].split("case '")[0].split("\n        case")[0]
    case = case.replace("G.e(i)", "ge").replace("G.m(i)", "gm")
    p = ps.DRUDE["Au_Ord"]
    for lam_nm in (500, 700, 900):
        o = octave.run(f"lam={lam_nm * 1e-9:.17g};\n{case}", ["ge"])
        w, g, wp = 1 / lam_nm, 1 / p["lam_gamma"], 1 / p["lam_p"]
        eps = 1 - wp**2 / (w**2 + 1j * g * w)
        correct = ps.g_electric(eps, lam_nm, p["lam_gamma"])
        lines.append(
            f"| G_e for Au (Ordal Drude), {lam_nm} nm | {o['ge']:.4g} | {correct:.4g} | {pct(o['ge'], correct):+.0f}% |"
        )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
