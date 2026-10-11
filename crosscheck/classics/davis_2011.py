"""Davis, Carney & Bhargava, "Theory of infrared microspectroscopy for intact fibers", Anal. Chem. 83, 525
(2011), doi:10.1021/ac102239b: Fig. 5 -- the transmission an FT-IR microscope records through fibers of
radius 5 and 10 µm that do not absorb at all (constant index n0 = 1.477 and 1.478): a strong, wavenumber-
dependent "baseline" made by refraction and scattering alone, which the paper's inversion has to remove.

Inputs: homogeneous fiber, n = n0 + 0i (Fig. 5 caption), in air; matched Cassegrain objective and condenser,
illumination annulus NA 0.1-0.4 (the only pair the paper prints, Fig. 2; Fig. 5's is not stated, and this one
reproduces its shape); the calculator's focused beam, vector model, transmission.  Fig. 4's toluene fibers
need Bertie's 1994 toluene optical constants, which the paper only plots.  Values read off Fig. 5 at 300 dpi (±3%).
"""

import numpy as np
from _common import peak, report, save_csv

import pystratify as ps

WN = np.linspace(800, 4000, 161)
LAM = 1e4 / WN
BEAM = ps.FocusedBeam(0.4, 0.1)
T = {}
for R, n0 in ((5.0, 1.477), (10.0, 1.478)):
    r = ps.focused_cylinders([R], [n0, 1.0], LAM, BEAM)
    T[R] = 100 * 10 ** (-np.asarray(r.apparent_absorbance))
save_csv("davis_2011_fig5", "wavenumber_cm-1,T_percent_R5um_n1.477,T_percent_R10um_n1.478", [WN, T[5.0], T[10.0]])
ok = report("davis_2011", [
    ("R = 5 µm: transmission minimum (%)", 10, -peak(WN, -T[5.0], 900, 1400)[1], 4, "Fig. 5a, read"),
    ("R = 5 µm: wavenumber of the minimum (cm-1)", 1100, peak(WN, -T[5.0], 900, 1400)[0], 80, "Fig. 5a, read"),
    ("R = 5 µm: maximum near the fit range, 1900-2600 cm-1 (%)", 76, peak(WN, T[5.0], 1900, 2600)[1], 6, "Fig. 5a, read"),
    ("R = 5 µm: dip near 3000 cm-1 (%)", 42, -peak(WN, -T[5.0], 2700, 3300)[1], 6, "Fig. 5a, read"),
    ("R = 5 µm: at 4000 cm-1 (%)", 70, T[5.0][-1], 6, "Fig. 5a, read"),
    ("R = 10 µm: at 800 cm-1 (%)", 42, T[10.0][0], 8, "Fig. 5b, read"),
])
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for ax, R in zip(axes, (5.0, 10.0)):
        ax.plot(WN, T[R])
        ax.set_ylim(0, 100)
        ax.set_ylabel("transmission (%)")
        ax.set_title(f"R = {R:g} µm, n0 = {1.477 if R == 5 else 1.478}", fontsize=9)
    axes[1].set_xlabel("wavenumber (cm$^{-1}$)")
    axes[1].invert_xaxis()
    fig.suptitle("Davis, Carney & Bhargava 2011, Fig. 5 (constant n0), recomputed: NA 0.1-0.4", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
