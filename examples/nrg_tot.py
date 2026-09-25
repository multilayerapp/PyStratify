"""tst_nrg_tot.m: electric energy in SiO2@Au (5-nm shell) vs core radius and wavelength."""

import numpy as np
from _plot import figure

import pystratify as ps

lam = np.linspace(500, 900, 201)
rc = np.linspace(10, 100, 46)
nau = ps.refractive_index("Au_JC", lam)
W = np.zeros((lam.size, rc.size, 2))
for j, r in enumerate(rc):
    rad = np.array([r, r + 5.0])
    ref = np.stack([np.full(lam.size, 1.45), ps.free_path_correction(lam, nau, rad, "Au_Ord"), np.ones(lam.size)], 1)
    sol = ps.solve(rad, ref, np.ones(3), lam, ps.l_max(rad[-1], 1.0, lam, "near"))  # all wavelengths at once
    for i, lm in enumerate(lam):
        W[i, j] = ps.total_energy(sol, ps.g_prefactors(ref[i], np.ones(3), lm, [None, "Au_Ord", None]), w=i).e
print("max core / shell electric energy enhancement: %.1f / %.1f" % (W[..., 0].max(), W[..., 1].max()))
fig, path = figure("nrg_tot")
if fig:
    for k, name in enumerate(("core", "shell")):
        ax = fig.add_subplot(1, 2, k + 1)
        im = ax.imshow(
            np.log10(np.abs(W[..., k])),
            aspect="auto",
            origin="lower",
            extent=[rc[0], rc[-1], lam[0], lam[-1]],
            cmap="hot",
        )
        fig.colorbar(im, ax=ax), ax.set_title(name), ax.set_xlabel("r_c, nm")
    fig.tight_layout(), fig.savefig(path)
