"""Au@SiO2@Au {10, 40, 45} nm in vacuum: radial electric energy density versus wavelength.

The gold carries the free-path correction, and its energy is weighted by the
Drude prefactor Re eps + 2 (omega/gamma_s) Im eps with the surface-limited damping."""

import numpy as np
from _common import figure, gold

import pystratify as ps

wavelength = np.linspace(400, 900, 251)
radii = [10.0, 40.0, 45.0]
r = np.linspace(0, 60, 241)
model = ps.DRUDE["Au_Ord"]
core, shell = [ps.free_path_correction(wavelength, gold(wavelength), rr, model) for rr in (radii[:1], radii[1:])]
n = np.stack([core, np.full(wavelength.size, 1.45), shell, np.ones(wavelength.size)], axis=1)
sol = ps.solve(radii, n, wavelength, l_max=ps.truncation_order(radii[-1], 1.0, wavelength, "near"))
damping = [ps.surface_damping_wavelength(model, radii[:1]), None, ps.surface_damping_wavelength(model, radii[1:]), None]
density = np.empty((wavelength.size, r.size))
for i, lam in enumerate(wavelength):
    eps = n[i] ** 2
    g_e = np.array([ps.electric_prefactor(e, lam, d) for e, d in zip(eps, damping)])
    density[i] = ps.energy_density(sol, r, (g_e, np.ones(4)), wavelength_index=i).density_e
print(f"max normalised electric energy density: {density.max():.1f}")

fig, path = figure("energy_density")
if fig:
    ax = fig.add_subplot(1, 1, 1)
    extent = [r[0], r[-1], wavelength[0], wavelength[-1]]
    im = ax.imshow(np.log10(density), aspect="auto", origin="lower", extent=extent, cmap="hot")
    fig.colorbar(im), ax.set_xlabel("r, nm"), ax.set_ylabel("wavelength, nm")
    fig.savefig(path)
