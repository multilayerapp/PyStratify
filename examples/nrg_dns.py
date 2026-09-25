"""tst_nrg_dns.m: radial electric energy density of Au@SiO2@Au {10, 40, 45} nm in vacuum.

Uses OSAC Eq. (23) for the Au shells (AUDIT.md M5: G_prefac.m overstates it)."""

import numpy as np
from _plot import figure

import pystratify as ps

lam = np.linspace(400, 900, 251)
rad = np.array([10.0, 40.0, 45.0])
rw = np.linspace(0, 60, 241)
nau = ps.refractive_index("Au_JC", lam)
ref = np.stack(
    [
        ps.free_path_correction(lam, nau, rad[:1], "Au_Ord"),
        np.full(lam.size, 1.45),
        ps.free_path_correction(lam, nau, rad[1:], "Au_Ord"),
        np.ones(lam.size),
    ],
    1,
)
sol = ps.solve(rad, ref, np.ones(4), lam, ps.l_max(rad[-1], 1.0, lam, "near"))
w = np.array(
    [
        ps.energy_density(sol, rw, ps.g_prefactors(ref[i], np.ones(4), lm, ["Au_Ord", None, "Au_Ord", None]), w=i).w_e
        for i, lm in enumerate(lam)
    ]
)
print("max normalised electric energy density: %.1f" % w.max())
fig, path = figure("nrg_dns")
if fig:
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(np.log10(w), aspect="auto", origin="lower", extent=[rw[0], rw[-1], lam[0], lam[-1]], cmap="hot")
    fig.colorbar(im), ax.set_xlabel("r, nm"), ax.set_ylabel("λ, nm")
    fig.savefig(path)
