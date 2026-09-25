"""|E|^2 around SiO2/Au matryoshkas in water at 690 nm (Meng et al., ACS Nano 11, 7915 (2017))."""

import numpy as np
from _common import figure, gold

import pystratify as ps

wavelength, n_water = 690.0, 1.33
particles = [[23.0, 28.0], [10.0, 13.0, 36.0, 48.0], [4.0, 7.0, 10.0, 16.0, 34.0, 42.0]]
grid = np.linspace(-70, 70, 281)
x, y = np.meshgrid(grid, grid)
fig, path = figure("near_field_map", (12, 4))
for k, radii in enumerate(particles):
    n = [1.45 if i % 2 == 0 else gold(wavelength) for i in range(len(radii))] + [n_water]
    l_max = ps.truncation_order(radii[-1], n_water, wavelength, "near")
    field = ps.near_field(ps.solve(radii, n, wavelength, l_max=l_max), x, y, 0 * x)
    print(f"{len(radii)} interfaces: max |E|^2 = {field.intensity_e.max():.1f}")
    if fig:
        ax = fig.add_subplot(1, 3, k + 1)
        extent = [grid[0], grid[-1], grid[0], grid[-1]]
        ax.imshow(np.log10(field.intensity_e), extent=extent, origin="lower", cmap="inferno")
        ax.set_title(f"{len(radii)} interfaces, log10 |E|²")
if fig:
    fig.tight_layout(), fig.savefig(path)
