"""Kuznetsov, Miroshnichenko, Fu, Zhang & Luk'yanchuk, "Magnetic light", Sci. Rep. 2, 492 (2012),
doi:10.1038/srep00492 (open access): Fig. 3, row (iv) -- Mie scattering and extinction of silicon
spheres of 100-270 nm in free space, with the magnetic-dipole (md), electric-dipole (ed) and
magnetic-quadrupole (mq) peaks labelled.

Inputs: homogeneous Si spheres, sizes as labelled in Fig. 3 (read as diameters: the SEM sizes, and
the only reading for which the 100 nm sphere's md falls at ~466 nm), vacuum host, plane wave.
The paper does not name its Si data; we use Green 2008 (refractiveindex.info), the calculator's
default crystalline-Si table.  Peak wavelengths are read off Fig. 3(iv) (±8 nm reading error), so
the checks allow ±15 nm for the unstated Si data.
"""

import numpy as np
from _common import curve, figure, mark, material, panel, peak, png, report, save_csv

import pystratify as ps

LAM = np.linspace(0.43, 0.80, 371)
DIAMETERS = [0.100, 0.140, 0.150, 0.182, 0.220, 0.270]
N = np.column_stack([material("Si_Green-2008", LAM), np.ones_like(LAM)])

curves, peaks, multipoles = {}, {}, {}
for d in DIAMETERS:
    cs = ps.cross_sections(ps.solve([d / 2], N, LAM))
    curves[d] = cs
    # each multipole's scattering efficiency, as the calculator plots it
    multipoles[d] = {k: cs.sca_by_order[:, pol, order] / cs.geometric
                     for k, pol, order in (("md_sca", ps.TE, 0), ("ed_sca", ps.TM, 0), ("mq_sca", ps.TE, 1), ("eq_sca", ps.TM, 1))}
    md, ed, mq = cs.sca_by_order[:, ps.TE, 0], cs.sca_by_order[:, ps.TM, 0], cs.sca_by_order[:, ps.TE, 1]
    peaks[d] = {name: peak(LAM, c)[0] * 1e3 for name, c in (("md", md), ("ed", ed), ("mq", mq))}

save_csv("kuznetsov_2012_fig3", "wavelength_um," + ",".join(f"Qsca d={d*1e3:.0f}nm,Qext d={d*1e3:.0f}nm" for d in DIAMETERS),
         [LAM, *[q for d in DIAMETERS for q in (curves[d].q_sca, curves[d].q_ext)]])

CHECKS = [
    ("100 nm: md peak (nm)", 466, peaks[0.100]["md"], 15, "Fig. 3a(iv), read"),
    ("140 nm: md peak (nm)", 581, peaks[0.140]["md"], 15, "Fig. 3b(iv), read"),
    ("140 nm: ed peak (nm)", 478, peaks[0.140]["ed"], 15, "Fig. 3b(iv), read"),
    ("150 nm: md peak (nm)", 607, peaks[0.150]["md"], 15, "Fig. 3c(iv), read"),
    ("150 nm: ed peak (nm)", 492, peaks[0.150]["ed"], 15, "Fig. 3c(iv), read"),
    ("182 nm: ed peak (nm)", 568, peaks[0.182]["ed"], 15, "Fig. 3d(iv), read"),
    ("182 nm: mq peak (nm)", 538, peaks[0.182]["mq"], 15, "Fig. 3d(iv), read"),
    ("220 nm: mq peak (nm)", 618, peaks[0.220]["mq"], 15, "Fig. 3e(iv), read"),
    ("220 nm: ed peak (nm)", 662, peaks[0.220]["ed"], 15, "Fig. 3e(iv), read"),
    ("270 nm: mq peak (nm)", 728, peaks[0.270]["mq"], 15, "Fig. 3f(iv), read"),
]
ok = report("kuznetsov_2012", CHECKS)

# The card's figure: each size's extinction and scattering with its dipoles and quadrupoles, as the calculator draws a
# sphere's multipoles; the paper's peak positions are rules. The 150 nm sphere is the calculator setup.
LABELS = {"md_sca": "magnetic dipole b_1", "ed_sca": "electric dipole a_1", "mq_sca": "magnetic quadrupole b_2",
          "eq_sca": "electric quadrupole a_2"}  # as the multipoles example labels them
figure("kuznetsov_2012", "Silicon spheres in free space, 100–270 nm in diameter", [
    panel(f"d{d*1e3:.0f}", f"{d*1e3:.0f} nm", LAM * 1e3, "λ [nm]", "Q",
          [curve("q_ext", "Q_ext", curves[d].q_ext, quantity="q_ext"), curve("q_sca", "Q_sca", curves[d].q_sca, quantity="q_sca")]
          + [curve(k, LABELS[k], v, quantity=k) for k, v in multipoles[d].items()],
          [mark(check, series=f"{kind}_sca", label=kind) for check, *_ in CHECKS
           if check.startswith(f"{d*1e3:.0f} nm:") for kind in [check.split(": ")[1].split()[0]]], points=200)
    for d in DIAMETERS], note="Peak positions read off Fig. 3(iv), ±8 nm.", default=DIAMETERS.index(0.150))

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, d in zip(axes.flat, DIAMETERS):
        ax.plot(LAM * 1e3, curves[d].q_ext, color="tab:blue", label="extinction")
        ax.plot(LAM * 1e3, curves[d].q_sca, color="tab:red", label="scattering")
        for name, x in peaks[d].items():
            ax.axvline(x, color="0.7", lw=0.6)
            ax.text(x, ax.get_ylim()[1] * 0.92, name, ha="center", fontsize=7)
        ax.set_title(f"Si sphere, {d*1e3:.0f} nm", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("wavelength (nm)")
    fig.suptitle("Kuznetsov et al. 2012, Fig. 3(iv), recomputed (efficiencies)")
    fig.tight_layout()
    fig.savefig(png("kuznetsov_2012"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
