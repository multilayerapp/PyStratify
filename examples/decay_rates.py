"""Electric dipole near Au@SiO2 {50, 70} nm in water: decay rates versus position and wavelength."""

import numpy as np
from _common import figure, gold

import pystratify as ps

radii, n_water = [50.0, 70.0], 1.33
wavelength = 614.0
r = np.linspace(52, 150, 99)
r = r[np.abs(r - radii[1]) > 1]  # an emitter on an interface needs unboundedly many orders
rates = ps.decay_rates(radii, [gold(wavelength), 1.45, n_water], wavelength, r)
print(f"max energy-balance error: {rates.balance_error.max():.1e}")
spectrum = np.linspace(450, 900, 226)
at_71 = [ps.decay_rates(radii, [gold(lam), 1.45, n_water], lam, [71.0]) for lam in spectrum]

fig, path = figure("decay_rates")
if fig:
    ax = fig.add_subplot(1, 2, 1)
    ax.semilogy(r, rates.radiative, r, rates.nonradiative, "--"), ax.axvline(71, ls=":", c="k")
    ax.legend(["rad ⊥", "rad ∥", "nonrad ⊥", "nonrad ∥"]), ax.set_xlabel("r, nm"), ax.set_title("614 nm")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(spectrum, [d.radiative[0] for d in at_71], spectrum, [d.nonradiative[0] for d in at_71], "--")
    ax.set_xlabel("wavelength, nm"), ax.set_title("r = 71 nm, radial dipole")
    fig.tight_layout(), fig.savefig(path)
