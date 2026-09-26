"""High-index sphere (n = 3.5, R = 75 nm) in air: scattering and emission directivity.

At 603 nm the electric and magnetic dipoles balance (first Kerker condition) and
the sphere scatters almost nothing backwards; an emitter 10 nm below it then
radiates preferentially through the sphere."""

import numpy as np
from _common import figure

import pystratify as ps

radii, n = [75.0], [3.5, 1.0]
theta = np.linspace(0, np.pi, 361)
for wavelength in (548.0, 603.0):  # magnetic-dipole resonance, Kerker condition
    sol = ps.solve(radii, n, wavelength)
    pattern = ps.scattering_pattern(sol, np.array([0.0, np.pi]), 0.0)
    forward, backward = pattern.directivity[0]
    print(f"{wavelength:.0f} nm plane wave: D(0) = {forward:.2f}, D(pi) = {backward:.2g}")
emitter = ps.dipole_far_field(radii, n, 603.0, [0, 0, -85.0], [1, 0, 0], theta, 0.0)
rates = ps.decay_rates(radii, n, 603.0, [85.0])
print(f"emitter: P_rad/P_0 = {emitter.power:.4f} (decay_rates: {rates.radiative[0, 1]:.4f})")
print(f"emitter: D(+z) = {emitter.directivity[0]:.2f}, D(-z) = {emitter.directivity[-1]:.2f}")

fig, path = figure("directivity", (10, 5))
if fig:
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    for phi, label in ((0.0, "E-plane"), (np.pi / 2, "H-plane")):
        d = ps.scattering_pattern(ps.solve(radii, n, 603.0), theta, phi).directivity[0]
        ax.plot(theta, d, label=label), ax.plot(-theta, d, "C0" if phi == 0 else "C1")
    ax.set_title("plane wave along +z, 603 nm"), ax.legend()
    ax = fig.add_subplot(1, 2, 2, projection="polar")
    for phi, label in ((0.0, "xz-plane"), (np.pi / 2, "yz-plane")):
        f = ps.dipole_far_field(radii, n, 603.0, [0, 0, -85.0], [1, 0, 0], theta, phi)
        ax.plot(theta, f.directivity, label=label), ax.plot(-theta, f.directivity, "C0" if phi == 0 else "C1")
    ax.set_title("x dipole at z = -85 nm"), ax.legend()
    fig.tight_layout(), fig.savefig(path)
