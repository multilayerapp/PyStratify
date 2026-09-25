"""tst_mdcy.m: electric vs magnetic dipole 10 nm from an Ag sphere (r = 50 nm) in air
(Schmidt et al., Opt. Express 20, 13636 (2012), Fig. 4(c,d))."""

import numpy as np
from _plot import figure

import pystratify as ps

lam = np.linspace(300, 700, 201)
res = {
    d: np.array(
        [
            np.r_[g.radiative[0], g.nonradiative[0]]
            for g in (
                ps.decay_rates([50.0], [ps.refractive_index("Ag_JC", lm), 1.0], [1, 1], lm, [60.0], dipole=d)
                for lm in lam
            )
        ]
    )
    for d in ("electric", "magnetic")
}
for d, v in res.items():
    print(f"{d}: max total (perp) = {(v[:, 0] + v[:, 2]).max():.1f}")
fig, path = figure("mdcy")
if fig:
    for k, d in enumerate(res):
        ax = fig.add_subplot(1, 2, k + 1)
        ax.semilogy(lam, res[d]), ax.set_title(d), ax.set_xlabel("λ, nm")
        ax.legend(["rad ⊥", "rad ∥", "nrad ⊥", "nrad ∥"])
    fig.tight_layout(), fig.savefig(path)
