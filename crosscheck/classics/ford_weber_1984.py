"""Ford & Weber, "Electromagnetic interactions of molecules with metal surfaces", Phys. Rep. 113, 195
(1984), doi:10.1016/0370-1573(84)90098-X: Fig. 7 -- where the energy of an excited molecule above a silver
mirror goes, against its distance d: into photons, surface plasmons or lossy surface waves.

Inputs: vacuum over a silver mirror (here 200 nm thick, opaque, since the calculator's exteriors must be
lossless) at ħω = 1.96 eV (632.8 nm), Ag ε = -19.04 + 0.67i from the paper's
local Drude fit (ħωp = 9.33 eV, ħΓ = 0.058 eV, ε_b = 3.6; footnote p. 227), quantum yield 1, perpendicular
and parallel dipoles.  The calculator splits power into escape (photons) and absorption in the metal, which
here is the surface plasmon and the lossy surface waves together (on lossy silver the plasmon is absorbed
before it leaves); so the checks are on the photon fraction.  The paper's Figs. 6-7 use a nonlocal
dielectric function, which matters below ~10 nm only.  Values read off Fig. 7 (±0.03).
"""

import numpy as np
from _common import report, save_csv

import pystratify as ps

LAM = 0.6328
AG = np.sqrt(3.6 - 9.33**2 / (1.96 * (1.96 + 0.058j)))  # = sqrt(-19.04 + 0.67i)
AG_THICKNESS = 0.2  # µm: the film exteriors must be lossless, so the mirror is an opaque 200 nm layer (T < 1e-14)
d = np.geomspace(0.001, 1.0, 120)  # µm
photon = np.empty((d.size, 2))
for i, z in enumerate(d):
    r = ps.solve_problem(ps.Problem("films", [np.inf, AG_THICKNESS, np.inf], [1.0, AG, 1.0], LAM, ps.PointDipole(-z, "electric", layer=0)))
    photon[i] = (r["escape"] / r["total"])[:2]
save_csv("ford_weber_1984_fig7", "d_um,photon_fraction_perp,photon_fraction_par", [d, *photon.T])
A = d * 1e4  # Ångström


def at(x, j):
    return np.interp(np.log(x), np.log(A), photon[:, j])


cross_par = np.exp(np.interp(0.5, photon[:, 1], np.log(A)))  # photon fraction 0.5, parallel (monotonic there)
ok = report("ford_weber_1984", [
    ("perp: photon fraction at 400 Å", 0.35, at(400, 0), 0.05, "Fig. 7, read"),
    ("perp: photon fraction dip near 1800 Å (OPEN: see README)", 0.22, photon[(A > 1000) & (A < 3000), 0].min(), None, "Fig. 7, read"),
    ("perp: photon fraction at 10^4 Å", 0.97, at(1e4 - 1, 0), 0.04, "Fig. 7, read"),
    ("par: distance where photons take half (Å)", 165, cross_par, 40, "Fig. 7, read"),
    ("par: photon fraction at 1000 Å", 0.97, at(1000, 1), 0.04, "Fig. 7, read"),
    ("perp: everything goes into the metal at 20 Å", 0.0, at(20, 0), 0.03, "Fig. 7"),
])
print(f"Ag n + ik = {AG:.4f}")
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(A, photon[:, 0], label="photons, perpendicular")
    ax.semilogx(A, photon[:, 1], "--", label="photons, parallel")
    ax.semilogx(A, 1 - photon[:, 0], label="plasmons + lossy waves, perpendicular")
    ax.set_xlabel("distance d (Å)")
    ax.set_ylabel("fraction of the decay rate")
    ax.legend(fontsize=8)
    ax.set_title("Ford & Weber 1984, Fig. 7, recomputed (Ag, 1.96 eV, local)", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
