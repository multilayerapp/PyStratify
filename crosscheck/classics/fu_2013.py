"""Fu, Kuznetsov, Miroshnichenko, Yu & Luk'yanchuk, "Directional visible light scattering by silicon
nanoparticles", Nat. Commun. 4, 1527 (2013), doi:10.1038/ncomms2538: Fig. 1a -- forward and backward
scattering and their ratio F/B for a silicon sphere of radius 75 nm in free space, the theory behind the
paper's "forward-to-backward ratio above six" (the first Kerker condition with real particles).

Inputs: Si sphere, R = 75 nm, vacuum host, unpolarized plane wave.  Forward (backward) scattering is the
scattered power in the forward (backward) hemisphere (Fig. 1a inset).  The paper does not name its Si
data; we use Green 2008.  Checks: F/B maximum ~8 at 660 nm, minimum below 0.5 at 564 nm, F = B at 500 nm
(electric dipole) and 603 nm (magnetic dipole), all printed on p. 3.
"""

import numpy as np
from _common import material, peak, report, save_csv

import pystratify as ps

LAM = np.linspace(0.38, 0.78, 401)
N = np.column_stack([material("Si_Green-2008", LAM), np.ones_like(LAM)])
sol = ps.solve([0.075], N, LAM)

theta = np.linspace(0, np.pi, 721)
phi = np.linspace(0, 2 * np.pi, 145)[:-1]
T, P = np.meshgrid(theta, phi, indexing="ij")
k = 2 * np.pi / LAM
dsdo = 0
for pol in ((1.0, 0.0), (0.0, 1.0)):  # unpolarized: the mean of two orthogonal polarizations
    pat = ps.scattering_pattern(sol, T, P, polarization=pol)
    dsdo = dsdo + 0.5 * (abs(pat.e_theta) ** 2 + abs(pat.e_phi) ** 2) / k[:, None, None] ** 2
per_theta = dsdo.mean(axis=2) * 2 * np.pi * np.sin(theta)  # integrate φ
fwd = np.trapezoid(np.where(theta <= np.pi / 2, per_theta, 0), theta, axis=1)
bwd = np.trapezoid(np.where(theta >= np.pi / 2, per_theta, 0), theta, axis=1)
ratio = fwd / bwd
nm = LAM * 1e3
crossings = nm[1:][np.diff(np.sign(ratio - 1)) != 0]
save_csv("fu_2013_fig1a", "wavelength_nm,forward_um2,backward_um2,F_over_B", [nm, fwd, bwd, ratio])

x_max, r_max = peak(nm, ratio, 620, 720)
x_min = nm[np.argmin(np.where((nm > 530) & (nm < 590), ratio, np.inf))]
ok = report("fu_2013", [
    ("F/B maximum", 8, r_max, 1.5, "p. 3: ~8 at 660 nm"),
    ("wavelength of the F/B maximum (nm)", 660, x_max, 15, "p. 3"),
    ("F/B minimum wavelength (nm)", 564, x_min, 15, "p. 3"),
    ("F/B minimum value (< 0.5)", 0.25, ratio[nm == x_min][0], 0.25, "p. 3: below 0.5"),
    ("F = B, electric dipole (nm)", 500, crossings[np.argmin(abs(crossings - 500))], 15, "p. 3"),
    ("F = B, magnetic dipole (nm)", 603, crossings[np.argmin(abs(crossings - 603))], 15, "p. 3"),
    ("hemispheres add up to the cross section (rel. error)", 0, np.max(abs(fwd + bwd - ps.cross_sections(sol).sca) / ps.cross_sections(sol).sca), 5e-3, "energy check (quadrature)"),
])
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(nm, fwd / fwd.max() * 9, color="tab:green", label="forward (scaled)")
    ax.plot(nm, bwd / fwd.max() * 9, color="tab:blue", label="backward (scaled)")
    ax2 = ax.twinx()
    ax2.plot(nm, ratio, color="tab:orange", label="F/B")
    ax2.axhline(1, ls="--", color="0.6")
    ax2.set_ylim(0, 10)
    ax.set_xlabel("wavelength (nm)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Fu et al. 2013, Fig. 1a, recomputed: Si sphere R = 75 nm", fontsize=9)
    fig.tight_layout()
    fig.savefig(__file__.replace("classics/", "classics/results/").replace(".py", ".png"), dpi=130)
except ImportError:
    pass
print("PASS" if ok else "CHECK")
