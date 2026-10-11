"""Ruan & Fan, "Superscattering of light from subwavelength nanostructures", Phys. Rev. Lett. 105, 013901
(2010), doi:10.1103/PhysRevLett.105.013901: Fig. 1 -- a three-layer nanorod whose resonances in four angular
channels (m = ±1 ... ±4) are aligned at one frequency, so that it scatters ~8 times the single-channel limit
2λ/π; Fig. 4a -- the same rod with silver-like losses.

Inputs (scale-free in the paper): Drude core ε = 1 - ωp²/(ω² + iγω), radius 0.3485 λp; dielectric shell
ε = 12.96 to 0.5623 λp; Drude shell to 0.6370 λp; air; TM in the paper's sense (H along the axis), i.e.
the calculator's axial-magnetic (s) polarization at normal incidence.  Lossless: γ = 0; lossy (Fig. 4a):
γ = 0.004 ωp (core), 0.012 ωp (shell).  We fix the scale with silver's ħωp = 9.01 eV (λp = 137.6 nm), which
only relabels the axis: ω/ωp = 0.25-0.259 is λ = 531-550 nm.  Widths are shown in units of 2λ/π.
"""

import numpy as np
from _common import drude_nk, nk_rows, peak, report, save_csv

import pystratify as ps

LP = 0.1376
RADII = np.array([0.3485, 0.5623, 0.6370]) * LP
x = np.linspace(0.25, 0.259, 1801)  # ω / ωp
LAM = LP / x


def widths(g_core, g_shell):
    out = np.empty((x.size, 3))
    per_order = np.empty((x.size, 5))
    for i, lam in enumerate(LAM):
        n = [drude_nk(lam, LP, g_core), 3.6, drude_nk(lam, LP, g_shell), 1.0]
        s = ps.solve_cylinder(RADII, n, lam)
        w = ps.cross_widths(s, "axial-magnetic")
        unit = 2 * lam / np.pi
        out[i] = w["scattering"] / unit, w["absorption"] / unit, w["extinction"] / unit
        t = np.abs(s.t[:, 1, 1]) ** 2  # axial-magnetic channel, |t_m|² in units of 2λ/π
        per_order[i] = [t[s.orders == m].sum() for m in range(5)]  # one sign of m (m = 0 alone)
    return out, per_order


lossless, orders = widths(0.0, 0.0)
lossy, _ = widths(0.004, 0.012)
save_csv("ruan_fan_2010_fig1_fig4a", "omega_over_omega_p,wavelength_um,sca_lossless,sca_lossy,abs_lossy,m1,m2,m3,m4",
         [x, LAM, lossless[:, 0], lossy[:, 0], lossy[:, 1], *orders[:, 1:].T])
for name, g in (("ruan_fan_2010_drude_lossless_nk", 0.0), ("ruan_fan_2010_drude_core_lossy_nk", 0.004), ("ruan_fan_2010_drude_shell_lossy_nk", 0.012)):
    grid = np.linspace(0.53, 0.552, 221)
    (__import__("pathlib").Path(__file__).parent / "results" / f"{name}.json").write_text(__import__("json").dumps(nk_rows(grid, drude_nk(grid, LP, g))))

# The lossless m = 3, 4 resonances are needles (width ~1e-6 ωp): resolve them on a fine local grid.
fine = np.linspace(0.2535, 0.2550, 6001)
_x, _lam = x, LAM
x, LAM = fine, LP / fine
fine_total, fine_orders = widths(0.0, 0.0)
x, LAM = _x, _lam
xp, sp = peak(fine, fine_total[:, 0])
ok = report("ruan_fan_2010", [
    ("lossless: total scattering peak (2λ/π)", 7.94, sp, 0.15, "p. 3, printed"),
    ("lossless: frequency of the peak (ω/ωp)", 0.2542, xp, 0.0005, "p. 3, printed"),
    ("lossless: each channel m = 1..4 peaks at the single-channel limit (min of the four maxima)", 1.0, fine_orders[:, 1:].max(axis=0).min(), 0.03, "p. 3"),
    ("lossy: total scattering peak (2λ/π)", 1.92, peak(x, lossy[:, 0])[1], 0.1, "p. 3, printed"),
    ("lossy: absorption peak (2λ/π)", 0.73, peak(x, lossy[:, 1])[1], 0.05, "p. 4, printed"),
])
try:
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(x, lossless[:, 0], "k", label="total")
    for m in range(1, 5):
        a.plot(x, orders[:, m], lw=0.8, label=f"m = {m}")
    a.set_xlabel("ω / ωp")
    a.set_ylabel("scattering width (2λ/π)")
    a.legend(fontsize=7)
    a.set_title("Fig. 1: lossless", fontsize=9)
    b.plot(x, lossy[:, 0], label="scattering")
    b.plot(x, lossy[:, 1], label="absorption")
    b.set_xlabel("ω / ωp")
    b.legend(fontsize=7)
    b.set_title("Fig. 4a: silver-like loss", fontsize=9)
    fig.suptitle("Ruan & Fan 2010, recomputed", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
