"""Mohlenhoff, Romeo, Diem & Wood, "Mie-type scattering and non-Beer-Lambert absorption behavior of human
cells in infrared microspectroscopy", Biophys. J. 88, 3635 (2005), doi:10.1529/biophysj.104.057950:
Fig. 4 -- the scattering efficiency of non-absorbing spheres of index 1.33 and radius 4-10 µm across the
mid-infrared, the broad "Mie wave" under the spectra of cells and nuclei (the same story as Bassan et al.
2009, whose PMMA optical constants are not printed).

Inputs (caption and Eq. 8): homogeneous sphere, n = 1.33, k = 0, in air; radius a = 4.0 ... 10 µm (the
legend writes "d", the caption and Eq. 8 use the radius, and the first maximum's position confirms it);
plane wave.  Fig. 4 is drawn from van de Hulst's anomalous-diffraction formula (their Eq. 8,
Q = 2 - (4/ρ) sin ρ + (4/ρ²)(1 - cos ρ), ρ = 4πa(n - 1)/λ), whose maximum is 3.17 for every size: the
script evaluates it to confirm the inputs (maximum ~3.2, first maximum near 2470 cm-1 for a = 4 µm and
1000 cm-1 for a = 10 µm), and lays exact Mie beside it -- the calculator's curve, whose first maximum
is higher (~4.0) and sits ~5% further to the blue, since n = 1.33 is not close enough to 1 for the
approximation.
"""

import numpy as np
from _common import peak, report, save_csv

import pystratify as ps

WN = np.linspace(600, 7000, 641)  # cm-1
LAM = 1e4 / WN  # µm
RADII = [4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]
q = {a: ps.cross_sections(ps.solve([a], [1.33, 1.0], LAM)).q_sca for a in RADII}
save_csv("mohlenhoff_2005_fig4", "wavenumber_cm-1," + ",".join(f"Qsca a={a}um" for a in RADII), [WN, *q.values()])

def van_de_hulst(a):
    rho = 4 * np.pi * a * (1.33 - 1) / LAM
    return 2 - 4 / rho * np.sin(rho) + 4 / rho**2 * (1 - np.cos(rho))


ada = {a: van_de_hulst(a) for a in RADII}
save_csv("mohlenhoff_2005_fig4_vdh", "wavenumber_cm-1," + ",".join(f"Q_vdH a={a}um" for a in RADII), [WN, *ada.values()])
ok = report("mohlenhoff_2005", [
    ("their formula, a = 4 µm: first maximum (cm-1)", 2470, peak(WN, ada[4.0], 1500, 3200)[0], 150, "Fig. 4, read"),
    ("their formula, a = 10 µm: first maximum (cm-1)", 1000, peak(WN, ada[10.0], 600, 1400)[0], 100, "Fig. 4, read"),
    ("their formula: maximum Q", 3.2, peak(WN, ada[4.0], 1500, 3200)[1], 0.15, "Fig. 4, read"),
    ("exact Mie, a = 4 µm: first maximum (cm-1)", 2470, peak(WN, q[4.0], 1500, 3200)[0], None, "differs: exact vs Eq. 8"),
    ("exact Mie, a = 4 µm: maximum Q", 3.2, peak(WN, q[4.0], 1500, 3200)[1], None, "differs: exact vs Eq. 8"),
])
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for a in (4.0, 6.0, 10.0):
        line, = ax.plot(WN, q[a], lw=1.2, label=f"exact Mie, a = {a} µm")
        ax.plot(WN, ada[a], "--", lw=1, color=line.get_color(), label=f"van de Hulst (their Fig. 4), a = {a} µm")
    ax.set_xlabel("wavenumber (cm$^{-1}$)")
    ax.set_ylabel("scattering efficiency Q")
    ax.set_title("Mohlenhoff et al. 2005, Fig. 4, recomputed: n = 1.33 spheres", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
