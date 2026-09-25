"""tst_near_fld.m: |E|^2 around SiO2/Au matryoshkas in water at 690 nm
(Meng et al., ACS Nano 11, 7915 (2017), Fig. 3)."""

import numpy as np
from _plot import figure

import pystratify as ps

lam, nh = 690.0, 1.33
nau, nsio2 = ps.refractive_index("Au_JC", lam), 1.45
particles = [np.array([23, 28.0]), np.array([10, 13, 36, 48.0]), np.array([4, 7, 10, 16, 34, 42.0])]
g = np.linspace(-70, 70, 281)
X, Y = np.meshgrid(g, g)
fig, path = figure("near_fld")
for k, rad in enumerate(particles):
    ref = [nsio2 if i % 2 == 0 else nau for i in range(rad.size)] + [nh]
    sol = ps.solve(rad, ref, np.ones(rad.size + 1), lam, ps.l_max(rad[-1], nh, lam, "near"))
    f = ps.near_field(sol, X, Y, 0 * X)
    print(f"{rad.size} shells: max |E|^2 = {f.intensity_E.max():.1f}")
    if fig:
        ax = fig.add_subplot(1, 3, k + 1)
        ax.imshow(np.log10(f.intensity_E), extent=[g[0], g[-1], g[0], g[-1]], origin="lower", cmap="inferno")
        ax.set_title(f"{rad.size} shells, log10|E|²")
if fig:
    fig.tight_layout(), fig.savefig(path)
