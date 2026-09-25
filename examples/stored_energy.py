"""SiO2@Au with a 5-nm shell: electric energy stored in core and shell versus core radius and wavelength."""

import numpy as np
from _common import figure, gold

import pystratify as ps

wavelength = np.linspace(500, 900, 201)
cores = np.linspace(10, 100, 46)
model = ps.DRUDE["Au_Ord"]
energy = np.zeros((wavelength.size, cores.size, 2))
for j, core in enumerate(cores):
    radii = [core, core + 5.0]
    shell = ps.free_path_correction(wavelength, gold(wavelength), radii, model)
    n = np.stack([np.full(wavelength.size, 1.45), shell, np.ones(wavelength.size)], axis=1)
    sol = ps.solve(radii, n, wavelength, l_max=ps.truncation_order(radii[-1], 1.0, wavelength, "near"))
    damping = ps.surface_damping_wavelength(model, radii)
    for i, lam in enumerate(wavelength):
        g_e = np.array([1.45**2, ps.electric_prefactor(shell[i] ** 2, lam, damping), 1.0])
        energy[i, j] = ps.shell_energy(sol, (g_e, np.ones(3)), wavelength_index=i).electric
print(f"max core / shell electric energy enhancement: {energy[..., 0].max():.1f} / {energy[..., 1].max():.1f}")

fig, path = figure("stored_energy")
if fig:
    for k, name in enumerate(("core", "shell")):
        ax = fig.add_subplot(1, 2, k + 1)
        extent = [cores[0], cores[-1], wavelength[0], wavelength[-1]]
        im = ax.imshow(np.log10(np.abs(energy[..., k])), aspect="auto", origin="lower", extent=extent, cmap="hot")
        fig.colorbar(im, ax=ax), ax.set_title(name), ax.set_xlabel("core radius, nm")
    fig.tight_layout(), fig.savefig(path)
