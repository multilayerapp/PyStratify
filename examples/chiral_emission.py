"""Circularly polarised emission of chiral emitters beside achiral spheres.

1. A randomly oriented chiral molecule (m = 0.01 i p, free g_lum = -0.04) 5 nm from a
   silicon-like (n = 3.5, R = 75 nm) or a gold (R = 50 nm) sphere in air: gold
   brightens the emission but dilutes its dissymmetry; near its first Kerker
   wavelength the silicon sphere is nearly dual (helicity-preserving) and keeps it.
2. A valley exciton of a 2D semiconductor, a circular in-plane dipole (x + i y),
   beside the same spheres: alone it emits no net helicity, beside a sphere it does."""

import numpy as np
from _common import figure, gold

import pystratify as ps

p, m = np.array([0, 0, 1.0]), np.array([0, 0, 0.01j])
g_free = 4 * np.imag(np.vdot(m, p)) / (np.vdot(p, p).real + np.vdot(m, m).real)
wavelength = np.linspace(500, 1000, 101)
spheres = {"silicon-like, R = 75 nm": (75.0, lambda lam: 3.5), "gold, R = 50 nm": (50.0, gold)}
molecule = {}
for name, (radius, index) in spheres.items():
    runs = [
        ps.dipole_far_field(
            [radius], [index(lam), 1.0], lam, [0, 0, radius + 5], p, 0.0, magnetic_moment=m, orientation="isotropic"
        )  # fmt: skip
        for lam in wavelength
    ]
    molecule[name] = np.array([[f.dissymmetry / g_free, f.power] for f in runs])
    ratio, i = molecule[name][:, 0], np.argmin(np.abs(molecule[name][:, 0] - 1))
    print(f"{name}: g/g_free from {ratio.min():.2f} to {ratio.max():.2f}, closest to 1 at {wavelength[i]:.0f} nm")
gaps = np.geomspace(1, 100, 41)
valley = {
    name: [ps.dipole_far_field([r], [idx(614.0), 1.0], 614.0, [0, 0, r + d], [1, 1j, 0], 0.0).dissymmetry for d in gaps]
    for name, (r, idx) in spheres.items()
}
for name, g in valley.items():
    print(f"valley exciton beside the {name} sphere at 614 nm: g_lum from {min(g):.3f} to {max(g):.3f}")

fig, path = figure("chiral_emission", (12, 4))
if fig:
    ax = fig.add_subplot(1, 3, 1)
    for name, v in molecule.items():
        ax.plot(wavelength, v[:, 0], label=name)
    ax.axhline(1, c="k", ls=":"), ax.set_xlabel("wavelength, nm"), ax.set_ylabel("g_lum / g_lum(free)"), ax.legend()
    ax = fig.add_subplot(1, 3, 2)
    for name, v in molecule.items():
        ax.plot(wavelength, v[:, 1], label=name)
    ax.set_xlabel("wavelength, nm"), ax.set_ylabel("P_rad / P_0 (orientation average)")
    ax = fig.add_subplot(1, 3, 3)
    for name, g in valley.items():
        ax.semilogx(gaps, g, label=name)
    ax.axhline(0, c="k", ls=":"), ax.set_xlabel("gap, nm"), ax.set_ylabel("g_lum of an x + iy dipole")
    fig.tight_layout(), fig.savefig(path)
