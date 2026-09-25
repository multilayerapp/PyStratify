"""Electric versus magnetic dipole 10 nm from a high-index (n = 3.5) sphere of radius 90 nm in air.

The sphere's magnetic dipole resonance enhances the magnetic emitter far more
than the electric one."""

import numpy as np
from _common import figure

import pystratify as ps

wavelength = np.linspace(500, 1000, 251)
rates = {
    dipole: np.array(
        [ps.decay_rates([90.0], [3.5, 1.0], lam, [100.0], dipole=dipole).radiative[0] for lam in wavelength]
    )
    for dipole in ("electric", "magnetic")
}
for dipole, v in rates.items():
    print(f"{dipole}: max radiative enhancement {v.max():.1f} at {wavelength[v.max(axis=1).argmax()]:.0f} nm")

fig, path = figure("magnetic_dipole_decay")
if fig:
    for k, dipole in enumerate(rates):
        ax = fig.add_subplot(1, 2, k + 1)
        ax.plot(wavelength, rates[dipole]), ax.set_title(dipole), ax.set_xlabel("wavelength, nm")
        ax.legend(["⊥", "∥"])
    fig.tight_layout(), fig.savefig(path)
