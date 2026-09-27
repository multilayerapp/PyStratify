"""Two-dimensional materials on spheres: a TMD monolayer on a silicon sphere, graphene on silica.

1. A WS2-like monolayer (A exciton at 2.01 eV as one Lorentzian, eps_b = 16,
   out-of-plane eps = 6.5; an illustrative model - use measured data for real
   work) on a silicon-like sphere (n = 3.9 + 0.02i, R = 78 nm) in air, whose
   magnetic-dipole Mie resonance sits on the exciton: extinction without the
   monolayer, with it as an explicit isotropic 0.618 nm film, and as a sheet -
   in-plane conductivity only, with the film's isotropic out-of-plane response,
   and with a TMD-like out-of-plane permittivity.  The isotropic sheet is the
   film's first-order limit (here within 0.5% of the monolayer's effect, the
   rest O(d^2)); the in-plane term alone misses ~7% of it.  Cross sections are
   compared on the sphere's own area: efficiencies q_ext would be normalised to
   the film's larger outer radius, an O(d/R) difference of their own.
2. Graphene (local RPA conductivity, T = 300 K, hbar gamma = 5 meV) on a silica
   sphere (R = 25 nm) in air: the l = 1 plasmon moves with the Fermi energy as
   the quasi-static condition eps_1 + 2 eps_2 + 2 i sigma / (k0 R) = 0 predicts.
"""

import numpy as np
from _common import figure

import pystratify as ps

# 1. TMD monolayer on a Mie resonator
wavelength = np.linspace(560, 680, 481)
energy = 1239.841984 / wavelength
eps_ws2 = 16.0 + 1.0 / (2.01**2 - energy**2 - 1j * 0.033 * energy)
thickness, radius, n_si = 0.618, 78.0, 3.9 + 0.02j
sheet = ps.Sheet.from_film(eps_ws2, thickness, wavelength, 1.0)
cases = {
    "bare Si sphere": ps.solve([radius], [n_si, 1.0], wavelength),
    "isotropic 0.618 nm film": ps.solve(
        [radius, radius + thickness],
        np.stack([np.full_like(eps_ws2, n_si), np.sqrt(eps_ws2), np.ones_like(eps_ws2)], 1),
        wavelength,
    ),
    "sheet, in-plane only": ps.solve([radius], [n_si, 1.0], wavelength, sheets={0: sheet.conductivity}),
    "sheet, isotropic": ps.solve([radius], [n_si, 1.0], wavelength, sheets={0: sheet}),
    "sheet, out-of-plane eps 6.5": ps.solve(
        [radius], [n_si, 1.0], wavelength, sheets={0: ps.Sheet.from_film(eps_ws2, thickness, wavelength, 1.0, 6.5)}
    ),
}
area = np.pi * radius**2
extinction = {name: ps.cross_sections(sol).ext / area for name, sol in cases.items()}
effect = np.abs(extinction["isotropic 0.618 nm film"] - extinction["bare Si sphere"]).max()
for name, q in extinction.items():
    peaks = wavelength[1:-1][(q[1:-1] > q[:-2]) & (q[1:-1] > q[2:])]
    gap = np.abs(q - extinction["isotropic 0.618 nm film"]).max() / effect
    print(f"{name:28s} extinction maxima at {np.round(peaks, 1)} nm; differs from the film by {gap:.1%} of its effect")

# 2. graphene-coated sphere
wavelength_ir = np.linspace(2000, 12000, 1001)
radius_g, eps_1 = 25.0, 2.1
graphene = {}
for fermi in (0.3, 0.5, 0.7):
    sigma = ps.graphene_conductivity(wavelength_ir, fermi, 0.005, 300.0)
    sol = ps.solve([radius_g], [np.sqrt(eps_1), 1.0], wavelength_ir, sheets={0: sigma})
    graphene[fermi] = ps.cross_sections(sol).q_ext
    quasi_static = eps_1 + 2 + 2j * sigma * wavelength_ir / (2 * np.pi * radius_g)
    peak = wavelength_ir[np.argmax(graphene[fermi])]
    print(
        f"graphene E_F = {fermi} eV: extinction peak at {peak:.0f} nm, "
        f"quasi-static resonance at {wavelength_ir[np.argmin(np.abs(quasi_static.real))]:.0f} nm"
    )

fig, path = figure("monolayer_sheet", (11, 4))
if fig:
    ax = fig.add_subplot(1, 2, 1)
    for (name, q), style in zip(extinction.items(), ("k:", "C0-", "C1--", "C2:", "C3-")):
        ax.plot(wavelength, q, style, label=name)
    ax.axvline(1239.841984 / 2.01, c="0.7", lw=0.8)
    ax.set_xlabel("wavelength, nm"), ax.set_ylabel("extinction / (pi R^2)"), ax.legend(fontsize=8)
    ax.set_title("WS2-like monolayer on a Si sphere")
    ax = fig.add_subplot(1, 2, 2)
    for fermi, q in graphene.items():
        ax.plot(wavelength_ir / 1000, q, label=f"E_F = {fermi} eV")
    ax.set_xlabel("wavelength, um"), ax.set_ylabel("extinction efficiency"), ax.legend()
    ax.set_title("graphene on a silica sphere, R = 25 nm")
    fig.tight_layout(), fig.savefig(path)
