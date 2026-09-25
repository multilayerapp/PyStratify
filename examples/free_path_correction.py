"""SiO2@Au {50, 55} nm in vacuum: a 5-nm gold shell with bulk and with surface-limited damping."""

import numpy as np
from _common import figure, gold

import pystratify as ps

wavelength = np.linspace(600, 1000, 401)
radii = [50.0, 55.0]
bulk = gold(wavelength)
spectra = {}
for name, shell in (("bulk", bulk), ("corrected", ps.free_path_correction(wavelength, bulk, radii, "Au_Ord"))):
    n = np.stack([np.full(wavelength.size, 1.45), shell, np.ones(wavelength.size)], axis=1)
    cs = ps.cross_sections(ps.solve(radii, n, wavelength))
    spectra[name] = (cs.q_abs, cs.q_sca, cs.q_ext)
    print(f"{name:9s} peak Q_ext {cs.q_ext.max():.3f} at {wavelength[cs.q_ext.argmax()]:.0f} nm")

fig, path = figure("free_path_correction")
if fig:
    for k, title in enumerate(("absorption", "scattering", "extinction")):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.plot(wavelength, spectra["bulk"][k], "--", wavelength, spectra["corrected"][k])
        ax.set_title(title), ax.set_xlabel("wavelength, nm")
    fig.axes[0].legend(["bulk", "corrected"])
    fig.tight_layout(), fig.savefig(path)
