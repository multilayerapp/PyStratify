"""Bohren & Huffman, *Absorption and Scattering of Light by Small Particles* (Wiley, 1983),
doi:10.1002/9783527618156, section 8.4: Fig. 8.7 -- the scattering cross section per unit volume of a
nonabsorbing infinite cylinder in air against its diameter, for light polarized parallel (solid) and
perpendicular (dashed) to the axis: the textbook demonstration that a cylinder, unlike a sphere,
scatters the two polarizations differently.

Inputs (p. 207): n = 1.55 ("many silicates at visible wavelengths"), λ = 0.6328 µm, air, normal incidence;
diameter 0-6 µm.  C_sca / V of an infinite cylinder is its scattering width over its cross-sectional area,
= 4 Q_sca / (π d); the geometric-optics limit Q -> 2 gives the dotted asymptote 8 / (π d).  Parallel is the
calculator's axial-electric polarization, perpendicular its axial-magnetic one.  Maxima read off Fig. 8.7
at 300 dpi (±0.2 in units of 1e4 cm-1, ±0.03 µm).
"""

import numpy as np
from _common import peak, report, save_csv

import pystratify as ps

LAM, N = 0.6328, 1.55
d = np.linspace(0.01, 6.0, 1200)
width = np.array([[ps.cross_widths(ps.solve_cylinder([x / 2], [N, 1.0], LAM), p)["scattering"]
                   for p in ("axial-electric", "axial-magnetic")] for x in d])
per_volume = width / (np.pi * (d[:, None] / 2) ** 2)  # µm⁻¹, i.e. units of 1e4 cm⁻¹
par, perp = per_volume[:, 0], per_volume[:, 1]
save_csv("bohren_huffman_1983_fig8_7", "diameter_um,Csca_over_V_parallel_1e4_per_cm,Csca_over_V_perpendicular_1e4_per_cm,asymptote",
         [d, par, perp, 8 / (np.pi * d)])
ok = report("bohren_huffman_1983", [
    ("parallel: first maximum", 7.3, peak(d, par, 0.08, 0.25)[1], 0.2, "Fig. 8.7, read"),
    ("parallel: second maximum", 9.5, peak(d, par, 0.25, 0.40)[1], 0.2, "Fig. 8.7, read"),
    ("parallel: highest maximum", 10.0, peak(d, par)[1], 0.2, "Fig. 8.7, read"),
    ("parallel: diameter of the highest maximum (µm)", 0.47, peak(d, par)[0], 0.03, "Fig. 8.7, read"),
    ("parallel: shoulder after it", 8.8, peak(d, par, 0.55, 0.70)[1], 0.25, "Fig. 8.7, read"),
    ("perpendicular: maximum", 8.0, peak(d, perp)[1], 0.2, "Fig. 8.7, read"),
    ("perpendicular: diameter of the maximum (µm)", 0.50, peak(d, perp)[0], 0.03, "Fig. 8.7, read"),
])
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(d, par, "k-", lw=1, label="parallel")
    ax.plot(d, perp, "k--", lw=1, label="perpendicular")
    ax.plot(d, 8 / (np.pi * d), "k:", lw=1, label="asymptotic limit")
    ax.set_ylim(0, 10.5)
    ax.set_xlabel("diameter (µm)")
    ax.set_ylabel("C_sca / volume (1e4 cm$^{-1}$)")
    ax.legend(fontsize=8)
    ax.set_title("Bohren & Huffman 1983, Fig. 8.7, recomputed: n = 1.55, λ = 0.6328 µm", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
