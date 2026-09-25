"""tst_dcy.m / OSAC Fig. 2(c): electric-dipole decay rates near Au@SiO2 {50, 70} nm in water."""

import numpy as np
from _plot import figure

import pystratify as ps

rad, nh = np.array([50.0, 70.0]), 1.33
lam0 = 614.0
ref = [ps.refractive_index("Au_JC", lam0), 1.45, nh]
rd = np.linspace(52, 150, 99)
r = ps.decay_rates(rad, ref, [1, 1, 1], lam0, rd, norm="host")
print("max energy-balance error: %.1e" % r.balance_error.max())
lam = np.linspace(400, 1000, 301)
gl = [
    ps.decay_rates(rad, [ps.refractive_index("Au_JC", lm), 1.45, nh], [1, 1, 1], lm, [71.0], norm="host") for lm in lam
]
fig, path = figure("dcy")
if fig:
    ax = fig.add_subplot(1, 2, 1)
    ax.semilogy(rd, r.radiative, rd, r.nonradiative, "--"), ax.axvline(71, ls=":", c="k")
    ax.legend(["rad ⊥", "rad ∥", "nrad ⊥", "nrad ∥"]), ax.set_xlabel("r, nm"), ax.set_title("λ = 614 nm")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(lam, [g.radiative[0] for g in gl], lam, [g.nonradiative[0] for g in gl], "--"), ax.set_xlabel("λ, nm")
    fig.tight_layout(), fig.savefig(path)
