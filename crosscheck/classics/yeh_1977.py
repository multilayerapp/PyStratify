"""Yeh, Yariv & Hong, "Electromagnetic propagation in periodic stratified media. I. General theory",
J. Opt. Soc. Am. 67, 423 (1977), doi:10.1364/JOSA.67.000423: Figs. 7-8 -- the reflectivity of a 15-period
Bragg reflector for TE and TM waves at incidence angles from 0 to 85 degrees: forbidden bands, the N - 1
zeros between them, and the TM gaps that close at the Brewster angle.

The figures are dimensionless and the paper prints no indices for them.  We use the GaAs / AlGaAs pair the
same paper uses for its Fig. 12 (n2 = 3.38, n1 = 2.89), quarter-wave layers at λ0 = 1 µm, and measure the
angle inside GaAs, which is how Yeh's angles reach 85 degrees (the n2 -> n1 critical angle is 58.8 degrees
and the TM Brewster angle, tan θ_B = n1 / n2, is 40.5 degrees there): incident GaAs / 15 x (AlGaAs, GaAs)
/ AlGaAs.  Checks are the paper's general statements (p. 426 and Fig. 4): N - 1 = 14 zeros of R between
the first band and the next (closed, for quarter-wave layers, at λ0/2), the TM gap closing at θ_B, near-total reflection beyond the critical angle.
"""

import numpy as np
from _common import report, save_csv

import pystratify.planar as pl

N1, N2, LAM0, PERIODS = 2.89, 3.38, 1.0, 15
d1, d2 = LAM0 / (4 * N1), LAM0 / (4 * N2)
n_list = [N2] + [N1, N2] * PERIODS + [N1]
d_list = [np.inf] + [d1, d2] * PERIODS + [np.inf]
LAM = np.linspace(0.6, 2.5, 1901)


def R(pol, theta_deg):
    return np.array([pl.coh_tmm(pol, n_list, d_list, np.radians(theta_deg), l)["R"] for l in LAM])


r0 = R("s", 0)
# zeros of R between the first stop band (around λ0) and the second (around λ0 / 3): count the minima
window = (LAM > 0.40) & (LAM < 0.95)
lam_w = np.linspace(0.45, 0.95, 6000)
r_w = np.array([pl.coh_tmm("s", n_list, d_list, 0.0, l)["R"] for l in lam_w])
# with quarter-wave layers the second band is closed at λ0/2: count from just past it to the first band
interior = (lam_w > LAM0 / 2 * 1.01) & (lam_w < LAM0 * 0.9)
minima = np.flatnonzero((r_w[1:-1] < r_w[:-2]) & (r_w[1:-1] < r_w[2:]) & interior[1:-1] & (r_w[1:-1] < 1e-3))
theta_b = np.degrees(np.arctan(N1 / N2))
tm_b = R("p", theta_b)
tm_0 = R("p", 0)
te_85 = R("s", 85)
angles = np.arange(0, 86, 5)
save_csv("yeh_1977_fig7_8", "wavelength_um," + ",".join(f"R_{p}_{a}deg" for p in "sp" for a in angles),
         [LAM, *[R(p, a) for p in "sp" for a in angles]])
ok = report("yeh_1977", [
    ("zeros of R between the first two bands (N - 1)", 14, len(minima), 0, "p. 426"),
    ("first band peak R at 0 deg (Yeh's unprinted pair has more contrast)", 1.0, r0[(LAM > 0.9) & (LAM < 1.1)].max(), None, "Fig. 7"),
    ("TM at the Brewster angle: largest R anywhere", 0.0, tm_b.max(), 1e-6, "Fig. 4 / Fig. 8: gaps close at θ_B"),
    ("TM at 0 deg: band peak equals TE (the gap exists off θ_B)", r0[(LAM > 0.9) & (LAM < 1.1)].max(), tm_0[(LAM > 0.9) & (LAM < 1.1)].max(), 1e-9, "Fig. 8"),
    ("TE at 85 deg: R across the long-wavelength range", 1.0, te_85[LAM > 1.2].min(), 0.01, "Fig. 7, top panel"),
])
print(f"Brewster angle in GaAs: {theta_b:.2f} deg; critical angle {np.degrees(np.arcsin(N1 / N2)):.2f} deg")
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, pol, title in zip(axes, "sp", ("TE (Fig. 7)", "TM (Fig. 8)")):
        omega = 2 * np.pi * (d1 + d2) / LAM / np.pi  # ωΛ/c in units of π
        for a in (0, 30, theta_b, 55, 75):
            ax.plot(omega, R(pol, a), lw=0.8, label=f"{a:.1f}°")
        ax.set_xlabel("ωΛ/c (units of π)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("R")
    axes[1].legend(fontsize=7)
    fig.suptitle("Yeh, Yariv & Hong 1977, Figs. 7-8, recomputed: 15 GaAs/AlGaAs periods", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
