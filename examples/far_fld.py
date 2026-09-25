"""tst_far_fld.m: angular scattering of SiO2@Au {50, 70} nm in vacuum at 750 nm."""

import numpy as np
from _plot import figure

import pystratify as ps

lam, rad = 750.0, np.array([50.0, 70.0])
sol = ps.solve(rad, [1.45, ps.refractive_index("Au_JC", lam), 1.0], [1, 1, 1], lam, 5)
th = np.linspace(0, np.pi, 181)  # includes backscattering (far_fld.m returns NaN there)
par, per = (a[0] for a in ps.scattering_amplitudes(sol, th))
print("|S_par|^2 forward/backward: %.4g / %.4g" % (abs(par[0]) ** 2, abs(par[-1]) ** 2))
fig, path = figure("far_fld")
if fig:
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.plot(th, np.abs(par) ** 2, label="parallel"), ax.plot(th, np.abs(per) ** 2, label="perpendicular")
    ax.plot(-th, np.abs(par) ** 2, "C0"), ax.plot(-th, np.abs(per) ** 2, "C1"), ax.legend()
    fig.savefig(path)
