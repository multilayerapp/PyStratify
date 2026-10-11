"""Point dipoles beside a cylinder against MIT Meep's 3D FDTD (benchmarks/MATRIX.md, cylinders).

    conda create -p <env> -c conda-forge python=3.12 "pymeep=1.35.0=nompi*" numpy scipy
    <env>/bin/pip install -e .
    <env>/bin/python benchmarks/fdtd_meep.py              # prints FDTD_MEEP.md (~30 min, one core, 0.7 GB)
    <env>/bin/python benchmarks/fdtd_meep.py 20 30        # chosen 3D resolutions only (pixels per um)
    <env>/bin/python benchmarks/fdtd_meep.py axial        # the sphere and film on an axis only (~3 min)

No closed form exists for a dipole beside a cylinder, so a volume method is the outside check:
it shares nothing with PyStratify's cylindrical-wave expansion and axial (beta) integral. Meep
runs in 3D with the infinite cylinder passing through the PML, a Gaussian pulse and Meep's LDOS
(``dft_ldos``) at the dipole; the total decay rate relative to free space is the LDOS with the
cylinder over the LDOS of the same grid without it, so the grid's own dispersion largely cancels.
Two mirror planes through the dipole (y = 0, z = 0; the cylinder's axis is z, the dipole sits on
x) cut each run to a quarter of the cell. Meep's MPI-free conda build (``pymeep`` from
conda-forge; the PyPI ``meep`` is another project) runs on one core; runs go one at a time.

Agreement is at the percent level by design: what is checked is that the FDTD answer approaches
PyStratify's as the grid is refined, at every wavelength and for every orientation, through the
cutoff of the TE01 and TM01 guided modes. This benchmark found the bug fixed with it: PyStratify's
guided-mode search dropped one of two nearby modes (the cutoff band here), while reporting
convergence, and the radial rate jumped by up to 12% from one wavelength to the next (0.10.3 gave
1.2568 at 645 nm, radial, where 0.10.4 gives 1.40628 and Meep at 60 px/um 0.43% more).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402

# name -> (radii in um, indices from the core out, host last, source radius in um)
CASES = {
    "fibre n = 2, R = 150 nm, dipole 100 nm outside": ([0.15], [2.0, 1.0], 0.25),
}
WAVELENGTHS = 1 / np.linspace(0.85, 1.65, 9)  # um; 1.18 ... 0.61, across the TE01/TM01 cutoff (0.678)
ORIENTATIONS = ("radial", "azimuthal", "axial")  # dipole along x, y, z: radial, azimuthal, axial at (r, 0, 0)
RESOLUTIONS = (20, 30, 40, 60)  # pixels per um
PML, HALF_WIDTH = 1.0, 0.9  # um: PML thickness and the half-width of the cell inside it


def stratify(case):
    """{orientation: total rate relative to the free-space rate, per wavelength}."""
    radii, n, source = CASES[case]
    totals = np.array([ps.solve_problem(ps.Problem("cylinders", radii, n, wavelength, ps.PointDipole(source, "electric"),
                                                   1e-6))["total"][:3] for wavelength in WAVELENGTHS])
    return dict(zip(ORIENTATIONS, totals.T))


def meep_ldos(case, orientation, resolution, cylinder=True):
    """Meep's LDOS at the dipole per wavelength, with or without the cylinder."""
    import meep as mp

    radii, n, source = CASES[case]
    frequencies = 1 / WAVELENGTHS
    centre, width = (frequencies[0] + frequencies[-1]) / 2, frequencies[-1] - frequencies[0]
    component = {"radial": mp.Ex, "azimuthal": mp.Ey, "axial": mp.Ez}[orientation]
    # a mirror plane containing the dipole is even (+1), one normal to it odd (-1)
    symmetries = [mp.Mirror(mp.Y, phase=-1 if orientation == "azimuthal" else 1),
                  mp.Mirror(mp.Z, phase=-1 if orientation == "axial" else 1)]
    geometry = [mp.Cylinder(radius=r, height=mp.inf, material=mp.Medium(index=float(np.real(v))))
                for r, v in reversed(list(zip(radii, n[:-1])))] if cylinder else []
    position = mp.Vector3(source, 0, 0)
    side = 2 * (HALF_WIDTH + PML)
    simulation = mp.Simulation(cell_size=mp.Vector3(side, side, side), boundary_layers=[mp.PML(PML)],
                               geometry=geometry, default_material=mp.Medium(index=float(np.real(n[-1]))),
                               sources=[mp.Source(mp.GaussianSource(centre, fwidth=1.2 * width), component, position)],
                               resolution=resolution, symmetries=symmetries, Courant=0.5)
    ldos = mp.Ldos(centre, width, len(frequencies))
    simulation.run(mp.dft_ldos(ldos=ldos),
                   until_after_sources=mp.stop_when_fields_decayed(20, component, position, 1e-6))
    assert np.allclose(mp.get_ldos_freqs(ldos), frequencies)
    return np.array(simulation.ldos_data)


def compare(cases=tuple(CASES), resolutions=RESOLUTIONS):
    """{case: (exact {orientation: totals}, {resolution: {orientation: Meep totals}}, seconds per resolution)}."""
    import meep as mp

    mp.verbosity(0)
    out = {}
    for case in cases:
        exact, runs, seconds = stratify(case), {}, {}
        for resolution in resolutions:
            start = time.time()
            runs[resolution] = {o: meep_ldos(case, o, resolution) / meep_ldos(case, o, resolution, cylinder=False)
                                for o in ORIENTATIONS}
            seconds[resolution] = time.time() - start
        out[case] = exact, runs, seconds
    return out


def report(results):
    print("# Dipoles beside a cylinder: PyStratify against Meep (3D FDTD)\n")
    print("Generated by `python benchmarks/fdtd_meep.py` (Meep 1.35, conda-forge `pymeep`, one core). Total decay")
    print("rate relative to free space; Meep's is its LDOS with the cylinder over its LDOS without, on the same")
    print(f"grid; cell {2 * HALF_WIDTH:g} um plus {PML:g} um of PML per side, two mirror planes. Relative difference")
    print("Meep / PyStratify - 1 per resolution (pixels per um); the FDTD error should shrink as the grid is refined.")
    for case, (exact, runs, seconds) in results.items():
        resolutions = sorted(runs)
        print(f"\n## {case}\n")
        print("Meep run time per resolution (6 runs: 3 orientations, with and without the cylinder): "
              + ", ".join(f"{r}: {seconds[r]:.0f} s" for r in resolutions) + ".\n")
        print("| wavelength (um) | orientation | PyStratify | " + " | ".join(f"Meep {r}" for r in resolutions) + " |")
        print("|---|---|---|" + "---|" * len(resolutions))
        for orientation in ORIENTATIONS:
            for i, wavelength in enumerate(WAVELENGTHS):
                cells = [f"{runs[r][orientation][i] / exact[orientation][i] - 1:+.2%}" for r in resolutions]
                print(f"| {wavelength:.3f} | {orientation} | {exact[orientation][i]:.5f} | " + " | ".join(cells) + " |")
        print("\nWorst |Meep / PyStratify - 1| over wavelengths and orientations: "
              + ", ".join(f"{r} px/um: {worst(exact, runs[r]):.2%}" for r in resolutions) + ".")


def worst(exact, run, orientations=ORIENTATIONS):
    return max(float(np.max(np.abs(run[o] / exact[o] - 1))) for o in orientations)


# Dipoles on an axis of symmetry: a sphere (on its axis) and a film (on its normal), in Meep's
# cylindrical coordinates (2D, so a grid far finer than the cylinder's). A dipole along the axis is
# E_z at r = 0 with m = 0; one across it is E_r at r = 0 with m = 1 (m = -1 gives the same LDOS).
# name -> (geometry, PyStratify dimensions, indices, source position, orientation names)
AXIAL_CASES = {
    "sphere n = 2, R = 150 nm, dipole 100 nm outside on its axis":
        ("spheres", [0.15], [2.0, 1.0], 0.25, ("radial", "tangential")),
    "film: dipole 100 nm above 200 nm of n = 2.3 on glass":
        ("films", [np.inf, 0.2, np.inf], [1.0, 2.3, 1.52], 0.1, ("perpendicular", "parallel")),
}
AXIAL_RESOLUTIONS = (25, 50, 100)  # pixels per um
AXIAL_PML, AXIAL_HALF_WIDTH, FILM_RADIUS = 1.0, 1.0, 3.0  # um


def stratify_axial(case):
    geometry, dimensions, n, source, names = AXIAL_CASES[case]
    position, layer = (-source, 0) if geometry == "films" else (source, 1)
    totals = np.array([ps.solve_problem(ps.Problem(geometry, dimensions, n, wavelength,
                                                   ps.PointDipole(position, "electric", layer), 1e-8))["total"][:2]
                       for wavelength in WAVELENGTHS])
    return dict(zip(names, totals.T))


def meep_ldos_axial(case, orientation, resolution, structure=True):
    """Meep's LDOS per wavelength for a dipole on the axis, with or without the sphere or film."""
    import meep as mp

    geometry, dimensions, n, source, names = AXIAL_CASES[case]
    frequencies = 1 / WAVELENGTHS
    centre, width = (frequencies[0] + frequencies[-1]) / 2, frequencies[-1] - frequencies[0]
    along = orientation == names[0]
    component = mp.Ez if along else mp.Er
    if geometry == "spheres":
        cell, middle = mp.Vector3(AXIAL_HALF_WIDTH + AXIAL_PML, 0, 2 * (AXIAL_HALF_WIDTH + AXIAL_PML)), 0.0
        shapes = [mp.Sphere(radius=r, material=mp.Medium(index=float(v)))
                  for r, v in reversed(list(zip(dimensions, n[:-1])))]
    else:  # films: layers below z = 0, the substrate through the lower PML
        thickness = float(sum(dimensions[1:-1]))
        cell = mp.Vector3(FILM_RADIUS + AXIAL_PML, 0, 2 * (AXIAL_HALF_WIDTH + AXIAL_PML) + thickness)
        middle, top, shapes = -thickness / 2, 0.0, []
        for d, v in zip(dimensions[1:-1], n[1:-1]):
            shapes.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, d), center=mp.Vector3(0, 0, top - d / 2),
                                   material=mp.Medium(index=float(v))))
            top -= d
        depth = AXIAL_HALF_WIDTH + AXIAL_PML
        shapes.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, depth), center=mp.Vector3(0, 0, top - depth / 2),
                               material=mp.Medium(index=float(n[-1]))))
    position = mp.Vector3(0, 0, source)
    simulation = mp.Simulation(cell_size=cell, dimensions=mp.CYLINDRICAL, m=0 if along else 1,
                               boundary_layers=[mp.PML(AXIAL_PML)], resolution=resolution, Courant=0.5,
                               geometry=shapes if structure else [], geometry_center=mp.Vector3(0, 0, middle),
                               default_material=mp.Medium(index=float(np.real(n[0] if geometry == "films" else n[-1]))),
                               sources=[mp.Source(mp.GaussianSource(centre, fwidth=2 * width), component, position)])
    ldos = mp.Ldos(centre, width, len(frequencies))
    simulation.run(mp.dft_ldos(ldos=ldos),
                   until_after_sources=mp.stop_when_fields_decayed(20, component, position, 1e-9))
    return np.array(simulation.ldos_data)


def compare_axial(cases=tuple(AXIAL_CASES), resolutions=AXIAL_RESOLUTIONS):
    """{case: (exact {orientation: totals}, {resolution: {orientation: Meep totals}}, seconds per resolution)}."""
    import meep as mp

    mp.verbosity(0)
    out = {}
    for case in cases:
        names = AXIAL_CASES[case][-1]
        exact, runs, seconds = stratify_axial(case), {}, {}
        for resolution in resolutions:
            start = time.time()
            runs[resolution] = {o: meep_ldos_axial(case, o, resolution) / meep_ldos_axial(case, o, resolution, False)
                                for o in names}
            seconds[resolution] = time.time() - start
        out[case] = exact, runs, seconds
    return out


def report_axial(results):
    print("\n# Dipoles on an axis: a sphere and a film against Meep in cylindrical coordinates\n")
    print("Same script and method, 2D: a dipole along the axis is E_z at r = 0 (m = 0), one across it E_r at r = 0")
    print(f"(m = 1); {AXIAL_PML:g} um of PML; the film's layers and substrate are infinite in r (the cell reaches")
    print(f"{FILM_RADIUS:g} um). PyStratify's sphere rates are pinned to a 60-digit Mie reference and its film rates to")
    print("written-out Sommerfeld integrals and smuthi, so the differences here are Meep's discretisation: a curved")
    print("surface converges in an oscillating way, and the film (5 to 20 pixels thick) at about first order. Not")
    print("shown: at 200 px/um the lowest frequency of the E_z (m = 0) runs jumps, +9% for the sphere and +12% for")
    print("the film, while every other value keeps converging (within 1.8%); unchanged by run length and cell size.")
    for case, (exact, runs, seconds) in results.items():
        resolutions, names = sorted(runs), AXIAL_CASES[case][-1]
        print(f"\n## {case}\n")
        print("Meep run time per resolution (4 runs): " + ", ".join(f"{r}: {seconds[r]:.0f} s" for r in resolutions) + ".\n")
        print("| wavelength (um) | orientation | PyStratify | " + " | ".join(f"Meep {r}" for r in resolutions) + " |")
        print("|---|---|---|" + "---|" * len(resolutions))
        for orientation in names:
            for i, wavelength in enumerate(WAVELENGTHS):
                cells = [f"{runs[r][orientation][i] / exact[orientation][i] - 1:+.2%}" for r in resolutions]
                print(f"| {wavelength:.3f} | {orientation} | {exact[orientation][i]:.5f} | " + " | ".join(cells) + " |")
        print("\nWorst |Meep / PyStratify - 1|: "
              + ", ".join(f"{r} px/um: {worst(exact, runs[r], names):.2%}" for r in resolutions) + ".")


if __name__ == "__main__":
    import atexit
    import os

    import meep  # noqa: F401  (imported first: exit handlers run last-registered first)

    atexit.register(lambda: (sys.stdout.flush(), os.dup2(os.open(os.devnull, os.O_WRONLY), 1)))  # Meep's exit line
    if sys.argv[1:] == ["axial"]:
        report_axial(compare_axial())
    else:
        report(compare(resolutions=tuple(int(r) for r in sys.argv[1:]) or RESOLUTIONS))
        report_axial(compare_axial())
