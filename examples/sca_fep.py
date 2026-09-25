"""tst_sca_fep.m: SiO2@Au {50, 55} nm in vacuum, bulk vs free-path-corrected Au shell."""

import numpy as np
from _plot import figure

import pystratify as ps

lam = np.linspace(600, 1000, 401)  # nm, solved as one vectorised batch
rad = np.array([50.0, 55.0])
nau = ps.refractive_index("Au_JC", lam)
L = ps.l_max(rad[-1], 1.0, lam, "far")
out = []
for shell in (nau, ps.free_path_correction(lam, nau, rad, "Au_Ord")):
    ref = np.stack([np.full(lam.size, 1.45), shell, np.ones(lam.size)], axis=1)
    cs = ps.cross_sections(ps.solve(rad, ref, [1, 1, 1], lam, L))
    out.append(np.stack([cs.q_abs, cs.q_sca, cs.q_ext], 1))
out = np.stack(out, 1)  # (W, bulk/corrected, abs/sca/ext)

print("peak Q_ext  bulk: %.3f at %.0f nm" % (out[:, 0, 2].max(), lam[out[:, 0, 2].argmax()]))
print("peak Q_ext  corrected: %.3f at %.0f nm" % (out[:, 1, 2].max(), lam[out[:, 1, 2].argmax()]))
fig, path = figure("sca_fep")
if fig:
    for k, name in enumerate(("absorption", "scattering", "extinction")):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.plot(lam, out[:, 0, k], "--", lam, out[:, 1, k])
        ax.set_title(name), ax.set_xlabel("λ, nm")
    fig.axes[0].legend(["bulk", "corrected"])
    fig.tight_layout(), fig.savefig(path)
