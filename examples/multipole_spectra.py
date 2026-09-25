"""Au@SiO2 {110, 120} nm in water: extinction decomposed into electric and magnetic multipoles."""

import numpy as np
from _common import figure, gold

import pystratify as ps

wavelength = np.linspace(400, 1000, 601)
radii = [110.0, 120.0]
n = np.stack([gold(wavelength), np.full(wavelength.size, 1.45), np.full(wavelength.size, 1.33)], axis=1)
cs = ps.cross_sections(ps.solve(radii, n, wavelength))  # the whole spectrum in one call
q = cs.ext_by_order / cs.geometric
for pol, name in ((ps.TM, "electric"), (ps.TE, "magnetic")):
    for order in (1, 2, 3):
        peak = q[:, pol, order - 1]
        print(f"{name} l={order}: max Q_ext = {peak.max():.3f} at {wavelength[peak.argmax()]:.0f} nm")

fig, path = figure("multipole_spectra")
if fig:
    ax = fig.add_subplot(1, 1, 1)
    for pol, style in ((ps.TM, "-"), (ps.TE, "--")):
        for order in (1, 2, 3):
            ax.plot(wavelength, q[:, pol, order - 1], style, label=f"{'EM'[pol]}{order}")
    ax.plot(wavelength, cs.q_ext, "k", label="total")
    ax.set_xlabel("wavelength, nm"), ax.set_ylabel("Q_ext"), ax.legend()
    fig.savefig(path)
