"""SiO2@Au {50, 70} nm in vacuum at 750 nm: polarised angular scattering, backscattering included."""

import numpy as np
from _common import figure, gold

import pystratify as ps

wavelength = 750.0
sol = ps.solve([50.0, 70.0], [1.45, gold(wavelength), 1.0], wavelength)
theta = np.linspace(0, np.pi, 181)
s_par, s_per = (s[0] for s in ps.scattering_amplitudes(sol, theta))
print(f"|S_par|^2 forward / backward: {abs(s_par[0]) ** 2:.4g} / {abs(s_par[-1]) ** 2:.4g}")

fig, path = figure("scattering_pattern", (5, 5))
if fig:
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    for s, label, colour in ((s_par, "parallel", "C0"), (s_per, "perpendicular", "C1")):
        ax.plot(theta, np.abs(s) ** 2, colour, label=label)
        ax.plot(-theta, np.abs(s) ** 2, colour)
    ax.legend()
    fig.savefig(path)
