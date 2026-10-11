"""Anger, Bharadwaj & Novotny, "Enhancement and quenching of single-molecule fluorescence", Phys. Rev.
Lett. 96, 113002 (2006), doi:10.1103/PhysRevLett.96.113002: Fig. 1 -- (a) the excitation rate and the
quantum yield of a molecule beside an 80 nm gold sphere, against its distance z from the surface; (b) the
fluorescence rate, their product, for gold spheres of 20, 60, 80 and 100 nm, with its maximum near
z = 10 nm: the enhancement-to-quenching transition.

Inputs, all printed (Fig. 1 caption): gold ε = -12.99 + 1.09i at λ = 650 nm (n + ik = 0.1510 + 3.6094i),
free space, plane wave polarized along the axis through the sphere and the molecule, molecular dipole
along that axis (radial), high intrinsic quantum yield (1).  Excitation rate: |E_radial|² at the molecule
over |E0|² (Eq. 5); quantum yield: radiative over total rate of the radial dipole; fluorescence rate:
their product (Eq. 1).  In the calculator these are two runs, a plane-wave near field and a dipole
distance sweep.  Values read off Fig. 1 (±5%); the paper's solid curves are exact (MMP, 2%).
"""

import numpy as np
from _common import peak, report, save_csv

import pystratify as ps

LAM = 0.650
GOLD = np.sqrt(-12.99 + 1.09j)
z = np.linspace(0.0005, 0.100, 200)  # µm, molecule-surface distance
out, curves = {}, {}
for d in (0.020, 0.060, 0.080, 0.100):
    R = d / 2
    sol = ps.solve([R], [GOLD, 1.0], [LAM])
    f = ps.near_field(sol, R + z, np.zeros_like(z), np.zeros_like(z))  # on the polarization (x) axis
    excitation = abs(f.e["x"]) ** 2
    rates = ps.decay_rates([R], [GOLD, 1.0], LAM, R + z)
    q = rates.radiative[:, 0] / rates.total[:, 0]  # radial dipole
    curves[d] = (excitation, q, excitation * q)
save_csv("anger_2006_fig1", "z_um," + ",".join(f"{name} d={d*1e3:.0f}nm" for d in curves for name in ("excitation", "quantum_yield", "fluorescence")),
         [z, *[c for d in curves for c in curves[d]]])

exc80, q80, fl80 = curves[0.080]
zz = z * 1e3
ok = report("anger_2006", [
    ("80 nm: excitation at z = 20 nm", 4.0, np.interp(20, zz, exc80), 0.3, "Fig. 1a, read"),
    ("80 nm: excitation at z = 30 nm", 2.8, np.interp(30, zz, exc80), 0.25, "Fig. 1a, read"),
    ("80 nm: excitation at z = 60 nm", 1.6, np.interp(60, zz, exc80), 0.2, "Fig. 1a, read"),
    ("80 nm: quantum yield at z = 20 nm", 0.80, np.interp(20, zz, q80), 0.05, "Fig. 1a, read"),
    ("80 nm: quantum yield at z = 30 nm", 0.88, np.interp(30, zz, q80), 0.04, "Fig. 1a, read"),
    ("80 nm: quantum yield at z = 60 nm", 0.95, np.interp(60, zz, q80), 0.05, "Fig. 1a, read"),
    ("80 nm: fluorescence maximum", 3.5, peak(zz, fl80)[1], 0.35, "Fig. 1b, read"),
    ("80 nm: z of the maximum (nm)", 10, peak(zz, fl80)[0], 4, "Fig. 1b, read"),
    ("100 nm: fluorescence maximum", 5.3, peak(zz, curves[0.100][2])[1], 0.5, "Fig. 1b, read"),
    ("60 nm: fluorescence maximum", 2.3, peak(zz, curves[0.060][2])[1], 0.25, "Fig. 1b, read"),
    ("20 nm: fluorescence maximum", 1.05, peak(zz, curves[0.020][2])[1], 0.15, "Fig. 1b, read"),
    ("80 nm: quenched at contact (z = 0.5 nm)", 0.0, fl80[0], 0.3, "Fig. 1b"),
])
try:
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(zz, exc80, color="tab:red", label="excitation rate")
    a2 = a.twinx()
    a2.plot(zz, q80, color="tab:blue", label="quantum yield")
    a2.set_ylim(0, 1)
    a.set_xlabel("z (nm)")
    a.set_title("(a) 80 nm Au sphere", fontsize=9)
    for d, (_, _, fl) in curves.items():
        b.plot(zz, fl, label=f"{d*1e3:.0f} nm")
    b.set_xlabel("z (nm)")
    b.set_ylabel("fluorescence rate / free space")
    b.legend(fontsize=8)
    fig.suptitle("Anger, Bharadwaj & Novotny 2006, Fig. 1, recomputed (λ = 650 nm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
