"""A 5-nm chiral molecular shell on a 20-nm core in water: gold versus silica core.

The shell's chirality is a Condon-Lorentz band in the UV (300 nm), so the
circular dichroism falls towards the red; the gold core roughly doubles it
around the plasmon resonance.  An emitter beside the particle radiates the two
circular polarisations slightly unequally."""

import numpy as np
from _common import figure, gold

import pystratify as ps

wavelength = np.linspace(400, 800, 401)
radii = [20.0, 25.0]
w, w0, width = 1 / wavelength, 1 / 300.0, 1 / 3000.0  # frequencies in 1/nm
kappa = np.zeros((wavelength.size, 3), dtype=complex)
kappa[:, 1] = 2e-3 * w * w0 / (w0**2 - w**2 - 1j * w * width)
spectra = {}
for name, core in (("gold", gold(wavelength)), ("silica", np.full(wavelength.size, 1.45 + 0j))):
    n = np.stack([core, np.full(wavelength.size, 1.45), np.full(wavelength.size, 1.33)], axis=1)
    spectra[name] = ps.helicity_cross_sections(ps.solve_chiral(radii, n, kappa, wavelength))
plasmon = np.argmax(spectra["gold"].q_ext[:, 0])
ratio = spectra["gold"].cd_ext[plasmon] / spectra["silica"].cd_ext[plasmon]
print(f"plasmon at {wavelength[plasmon]:.0f} nm: CD_ext gold / silica core = {ratio:.2f}")
n_i = [gold(wavelength[plasmon]), 1.45, 1.33]
powers = [
    ps.dipole_far_field(radii, n_i, wavelength[plasmon], [0, 0, 30.0], [1, s * 1j, 0], 0.0, kappa=kappa[plasmon]).power
    for s in (1, -1)
]
print(f"circular dipoles 5 nm outside: P_rad/P_0 = {powers[0]:.5f} (+) and {powers[1]:.5f} (-)")

fig, path = figure("chiral_cd")
if fig:
    ax = fig.add_subplot(1, 2, 1)
    for name, style in (("gold", "-"), ("silica", "--")):
        ax.plot(wavelength, spectra[name].cd_ext, style, label=f"{name} core")
    ax.set_xlabel("wavelength, nm"), ax.set_ylabel("C_ext(+) - C_ext(-), nm²"), ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(wavelength, spectra["gold"].q_ext[:, 0]), ax.set_xlabel("wavelength, nm"), ax.set_title("Q_ext, gold core")
    fig.tight_layout(), fig.savefig(path)
