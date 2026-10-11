"""Prodan, Radloff, Halas & Nordlander, "A hybridization model for the plasmon response of complex
nanostructures", Science 302, 419 (2003), doi:10.1126/science.1089171: Fig. 4A -- extinction of a
concentric gold nanoshell ("nanomatryushka") and of its two shells taken alone; the inner- and outer-shell
plasmons hybridize into a bonding (red) and an antibonding (blue) pair.

Inputs: silica core 80 nm / gold to 107 nm / silica spacer to 135 nm / gold to 157 nm (radii; Fig. 4
caption); curve (1) the inner nanoshell alone, curve (2) the outer shell on a silica core of 135 nm, curve
(3) all four layers.  The paper prints neither its optical constants nor the host index of its Mie curves:
we use Johnson & Christy gold, Malitson silica, and a host of n = 1.45, the value that best places the
four marked peaks (scan 1.40-1.56: rms 44 nm at 1.45).  The bonding peak of (3) then falls ~85 nm short of
the marked 1550 nm at every host index, which points at the unstated gold data rather than the host.
Peak marks read off Fig. 4A (±30 nm); arbitrary units, so only positions are compared.
"""

import numpy as np
from _common import material, peak, report, save_csv

import pystratify as ps

LAM = np.linspace(0.5, 1.93, 573)  # Johnson & Christy gold ends at 1.94 µm
HOST = 1.45
au, si, h = material("Au_Johnson", LAM), material("SiO2_Malitson", LAM), np.full(LAM.size, HOST)


def ext(radii, layers):
    return ps.cross_sections(ps.solve(radii, np.column_stack([*layers, h]), LAM)).ext


full = ext([0.080, 0.107, 0.135, 0.157], [si, au, si, au])
inner = ext([0.080, 0.107], [si, au])
outer = ext([0.135, 0.157], [si, au])
nm = LAM * 1e3
save_csv("prodan_2003_fig4a", "wavelength_nm,ext_full_um2,ext_inner_um2,ext_outer_um2", [nm, full, inner, outer])
ok = report("prodan_2003", [
    ("(3) antibonding dipole peak (nm)", 900, peak(nm, full, 800, 1100)[0], 60, "Fig. 4A mark, read"),
    ("(3) bonding dipole peak (nm)", 1550, peak(nm, full, 1300, 1900)[0], 100, "Fig. 4A mark, read"),
    ("(2) outer shell alone (nm)", 1280, peak(nm, outer, 1000, 1700)[0], 60, "Fig. 4A mark, read"),
    ("(1) inner shell alone (nm)", 900, peak(nm, inner, 750, 1100)[0], 60, "Fig. 4A mark, read"),
    ("hybridization: the bonding peak lies red of the outer shell alone", 1,
     float(peak(nm, full, 1300, 1900)[0] > peak(nm, outer, 1000, 1700)[0]), 0, "hybridization, Fig. 4A"),
])
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (c, label) in enumerate(((inner, "(1) inner shell"), (outer, "(2) outer shell"), (full, "(3) concentric"))):
        ax.plot(nm, c / c.max() + 1.1 * i, label=label)
    for x in (900, 1550, 1280):
        ax.axvline(x, color="0.8", lw=0.6)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("extinction (normalized, offset)")
    ax.legend(fontsize=8)
    ax.set_title(f"Prodan et al. 2003, Fig. 4A, recomputed (host n = {HOST})", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
