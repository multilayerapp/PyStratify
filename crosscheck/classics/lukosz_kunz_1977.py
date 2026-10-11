"""Lukosz & Kunz, "Light emission by magnetic and electric dipoles close to a plane interface. I. Total
radiated power", J. Opt. Soc. Am. 67, 1607 (1977), doi:10.1364/JOSA.67.001607: Fig. 4 -- the total power
radiated by electric and magnetic dipoles, perpendicular and parallel to the interface, against their
distance z0 from it, for relative indices n = 1.5 and 3 (the normalized decay rate L/L_inf).

Inputs (dimensionless in the paper): the dipole in a lossless medium 1, a lossless medium 2 of relative
index n = n2/n1, distance z0 in units of λ1 = λ0/n1.  Concretely: n1 = 1 (air), λ0 = 600 nm, z0 from 3 to
600 nm.  The calculator's film decay rate is L/L_inf exactly (both relative to the dipole in medium 1).
Values read off Fig. 4 (±0.05).
"""

import numpy as np
from _common import peak, report, save_csv

import pystratify as ps

LAM = 0.600
z0 = np.linspace(0.003, 0.600, 200)


def rates(n2, kind):
    out = np.empty((z0.size, 2))
    for i, z in enumerate(z0):
        r = ps.solve_problem(ps.Problem("films", [np.inf, np.inf], [1.0, n2], LAM, ps.PointDipole(-z, kind, layer=0)))
        out[i] = r["total"][:2]  # perpendicular, parallel
    return out


curves = {(n2, kind): rates(n2, kind) for n2 in (1.5, 3.0) for kind in ("electric", "magnetic")}
x = z0 / LAM  # z0 / λ1 with n1 = 1
save_csv("lukosz_kunz_1977_fig4", "z0_over_lambda1," + ",".join(f"{k[1]} {o} n={k[0]}" for k in curves for o in ("perp", "par")),
         [x, *[c[:, j] for c in curves.values() for j in (0, 1)]])
e15, e3 = curves[1.5, "electric"], curves[3.0, "electric"]
ok = report("lukosz_kunz_1977", [
    ("e, perp, n = 1.5, at contact", 2.3, e15[0, 0], 0.1, "Fig. 4, read"),
    ("e, par, n = 1.5, at contact", 1.4, e15[0, 1], 0.1, "Fig. 4, read"),
    ("e, par, n = 3: first minimum", 0.9, -peak(x, -e3[:, 1], 0.05, 0.3)[1], 0.05, "Fig. 4, read"),
    ("e, par, n = 3: z0/λ1 of the first minimum", 0.13, peak(x, -e3[:, 1], 0.05, 0.3)[0], 0.04, "Fig. 4, read"),
    ("e, par, n = 3: second minimum near z0/λ1 = 0.6", 0.92, -peak(x, -e3[:, 1], 0.5, 0.75)[1], 0.05, "Fig. 4, read"),
    ("e, par, n = 3: maximum near z0/λ1 = 0.35", 1.15, peak(x, e3[:, 1], 0.25, 0.45)[1], 0.06, "Fig. 4, read"),
    ("e, perp, n = 1.5, z0/λ1 = 0.4: back to ~1", 1.0, np.interp(0.4, x, e15[:, 0]), 0.05, "Fig. 4, read"),
    ("m, perp, n = 1.5, at contact", 2.0, curves[1.5, "magnetic"][0, 0], 0.15, "Fig. 6, read"),
])
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, (kind, j, title) in zip(axes.flat, [("electric", 0, "e, perp"), ("electric", 1, "e, par"), ("magnetic", 0, "m, perp"), ("magnetic", 1, "m, par")]):
        for n2 in (1.5, 3.0):
            ax.plot(x, curves[n2, kind][:, j], label=f"n = {n2}")
        ax.set_ylim(0, 3)
        ax.set_title(title, fontsize=9)
        ax.axhline(1, color="0.8", lw=0.6)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("z0 / λ1")
    fig.suptitle("Lukosz & Kunz 1977, Fig. 4, recomputed: L/L_inf", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
