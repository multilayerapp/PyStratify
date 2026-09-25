"""tst_sca_l.m: Au@SiO2 {110, 120} nm in water, multipole decomposition (l = 1..3)."""

import numpy as np
from _plot import figure

import pystratify as ps

lam = np.linspace(300, 1500, 1201)
rad = np.array([110.0, 120.0])
ref = np.stack([ps.refractive_index("Au_JC", lam), np.full(lam.size, 1.45), np.full(lam.size, 1.33)], 1)
cs = ps.cross_sections(ps.solve(rad, ref, [1, 1, 1], lam, 3))
ext_l = cs.ext_l / cs.geometric  # (W, e/m, l)
for p, name in enumerate(("electric", "magnetic")):
    for j in range(3):
        print(f"{name} l={j + 1}: max Q_ext = {ext_l[:, p, j].max():.3f} at {lam[ext_l[:, p, j].argmax()]:.0f} nm")
fig, path = figure("sca_l")
if fig:
    ax = fig.add_subplot(1, 1, 1)
    for p, ls in ((0, "-"), (1, "--")):
        for j in range(3):
            ax.plot(lam, ext_l[:, p, j], ls, label=f"{'EM'[p]}{j + 1}")
    ax.plot(lam, cs.q_ext, "k", label="total"), ax.legend(), ax.set_xlabel("λ, nm")
    fig.savefig(path)
