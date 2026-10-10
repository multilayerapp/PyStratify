"""Spheres against pyGDM2, a volume Green dyadic method (benchmarks/MATRIX.md, spheres).

    pip install -e ".[benchmarks]"
    python benchmarks/gdm_spheres.py        # prints a Markdown report (GDM_SPHERES.md)

pyGDM2 (Wiecha, Comput. Phys. Commun. 233, 167 (2018); Wiecha et al., CPC 270, 108142 (2022))
discretizes the particle's volume into point dipoles and solves the coupled system: a method that
shares nothing with Mie theory or with MNPBEM's surface elements. Its answer depends on the mesh, so
this script refines it and reports the difference to PyStratify at each step. How it is compared:

* **Volume-equivalent radius.** A cube or hexagonal-compact lattice does not hold a sphere's volume
  exactly (it swings by +-5% with the step), so PyStratify is evaluated for the sphere of the same
  volume, r_eff = (3 V / 4 pi)^(1/3), V = N step^3 (cube) or N step^3 / sqrt(2) (hexagonal), the usual
  practice for volume-discretization methods.
* **Centroid.** ``pyGDM2.structures.sphere`` builds the sphere resting on the XY plane (centroid at
  z ~ R + step/2), so emitters are placed relative to the centroid of the dipoles.
* **Double precision** (``dtype='d'``; pyGDM2's default is single).
* Decay rates are relative to the dipole in vacuum on both sides (the host is vacuum).

Cases at 600 nm, hexagonal mesh, 425 to 2235 dipoles (the finest peaks at ~1.7 GB):

* asserted: a 40 nm-radius sphere with n = 2 and n = 2 + 0.1i (cross sections; total decay rates at
  20 and 40 nm from the surface, radial and tangential);
* shown, not asserted: 20 nm metal spheres, eps = -2.5 + i and -10 + i. A volume lattice converges
  slowly for metals: scattering of the -2.5 + i sphere is still -17% at 2235 dipoles, absorption of
  the -10 + i sphere +173% (+278% at 425); finer meshes need more memory than this 8 GB machine has.

The dielectric cross sections settle near +2.8% instead of shrinking further: that is the method's
error at these meshes (lattice and surface), and the reason the guard test asserts a band, not a limit.
C_abs is compared relative to C_ext (it vanishes for the lossless sphere).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402

WAVELENGTH = 600.0
STEPS = (4.0, 5.0, 6.0, 7.0)  # dipoles per radius; 425, 853, 1405, 2235 dipoles on the hexagonal mesh
GAPS = (20.0, 40.0)

# name -> (radius in nm, index, emitters?, asserted?)
CASES = {
    "dielectric, n = 2, R = 40 nm": (40.0, 2.0, True, True),
    "absorbing dielectric, n = 2 + 0.1i, R = 40 nm": (40.0, 2.0 + 0.1j, True, True),
    "metal, eps = -2.5 + i, R = 20 nm": (20.0, np.sqrt(-2.5 + 1j + 0j), False, False),
    "metal, eps = -10 + i, R = 20 nm": (20.0, np.sqrt(-10 + 1j + 0j), False, False),
}


def gdm(radius, n, per_radius, emitters):
    """pyGDM2's values and the volume-equivalent radius for a mesh of ``per_radius`` dipoles per radius."""
    from pyGDM2 import core, fields, linear, materials, propagators, structures

    step = radius / per_radius
    structure = structures.struct(step, structures.sphere(step, R=per_radius, mesh="hex"), materials.dummy(n),
                                  verbose=False)
    dipoles = np.asarray(structure.geometry)
    r_eff = (3 * len(dipoles) * step ** 3 / np.sqrt(2) / (4 * np.pi)) ** (1 / 3)
    field = fields.efield(fields.plane_wave, wavelengths=[WAVELENGTH], kwargs=dict(theta=0, inc_angle=180))
    simulation = core.simulation(structure, field, propagators.DyadsQuasistatic123(n1=1, n2=1), dtype="d",
                                 verbose=False)
    simulation.scatter(verbose=False)
    ext, sca, absorbed = (float(v) for v in linear.extinct(simulation, 0))
    out = {"dipoles": len(dipoles), "r_eff": r_eff, "C_ext": ext, "C_sca": sca, "C_abs": absorbed}
    if emitters:
        center = dipoles.mean(axis=0)
        probes = [(center[0], center[1], center[2] + radius + gap) for gap in GAPS]
        rates = core.decay_rate(simulation, wavelength=WAVELENGTH, r_probe=probes, verbose=False)
        for i, gap in enumerate(GAPS):
            out[f"radial, {gap:g} nm"] = float(np.asarray(rates[2])[i, 3])
            out[f"tangential, {gap:g} nm"] = float(np.asarray(rates[0])[i, 3])
    return out


def stratify(radius, n, r_eff, emitters):
    sections = ps.cross_sections(ps.solve([r_eff], np.array([[n, 1.0]]), np.array([WAVELENGTH]), l_max=20))
    area = np.pi * r_eff ** 2
    out = {"C_ext": sections.q_ext[0] * area, "C_sca": sections.q_sca[0] * area, "C_abs": sections.q_abs[0] * area}
    if emitters:
        rates = ps.decay_rates([r_eff], [n, 1.0], WAVELENGTH, r=[radius + gap for gap in GAPS])
        for i, gap in enumerate(GAPS):
            out[f"radial, {gap:g} nm"] = rates.total[i, 0]
            out[f"tangential, {gap:g} nm"] = rates.total[i, 1]
    return out


def difference(quantity, values):
    """pyGDM2 relative to PyStratify; absorption relative to extinction (it vanishes for lossless spheres)."""
    theirs, ours = values[quantity]
    if quantity == "C_abs":
        return (theirs - ours) / values["C_ext"][1]
    return theirs / ours - 1


def compare(cases=CASES, steps=STEPS):
    """{case: [(dipoles, {quantity: (pyGDM2, PyStratify)}) per mesh]}."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (radius, n, emitters, _) in cases.items():
            rows = []
            for per_radius in steps:
                theirs = gdm(radius, n, per_radius, emitters)
                ours = stratify(radius, n, theirs["r_eff"], emitters)
                rows.append((theirs["dipoles"], {q: (theirs[q], ours[q]) for q in ours}))
            out[name] = rows
    return out


def report(results):
    print("# PyStratify spheres against pyGDM2 (volume Green dyadic method)\n")
    print("Generated by `python benchmarks/gdm_spheres.py`. Relative difference of pyGDM2 to PyStratify for the")
    print(f"volume-equivalent sphere, {WAVELENGTH:g} nm, vacuum host, as the hexagonal mesh is refined; decay rates")
    print("are total rates over the dipole in vacuum, at the given distance from the surface.\n")
    for name, rows in results.items():
        quantities = list(rows[0][1])
        asserted = CASES[name][3]
        print(f"## {name}{'' if asserted else ' (shown, not asserted)'}\n")
        print("| dipoles | " + " | ".join(quantities) + " |")
        print("|---|" + "---|" * len(quantities))
        for dipoles, values in rows:
            print(f"| {dipoles} | " + " | ".join(f"{difference(q, values):+.2%}" for q in quantities) + " |")
        print()
        if not asserted:
            print("A volume lattice converges slowly for metals; finer meshes need more memory than 8 GB.\n")


if __name__ == "__main__":
    report(compare())
