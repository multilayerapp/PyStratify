"""Point dipoles beside a cylinder against MIT Meep's 3D FDTD (benchmarks/MATRIX.md, cylinders).

    conda create -p <env> -c conda-forge python=3.12 "pymeep=1.35.0=nompi*" numpy scipy
    <env>/bin/pip install -e .
    <env>/bin/python benchmarks/fdtd_meep.py              # prints FDTD_MEEP.md (~30 min, one core, 0.7 GB)
    <env>/bin/python benchmarks/fdtd_meep.py 20 30        # chosen resolutions only (pixels per um)

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
        print(f"Meep run time per resolution (6 runs: 3 orientations, with and without the cylinder): "
              + ", ".join(f"{r}: {seconds[r]:.0f} s" for r in resolutions) + ".\n")
        print("| wavelength (um) | orientation | PyStratify | " + " | ".join(f"Meep {r}" for r in resolutions) + " |")
        print("|---|---|---|" + "---|" * len(resolutions))
        for orientation in ORIENTATIONS:
            for i, wavelength in enumerate(WAVELENGTHS):
                cells = [f"{runs[r][orientation][i] / exact[orientation][i] - 1:+.2%}" for r in resolutions]
                print(f"| {wavelength:.3f} | {orientation} | {exact[orientation][i]:.5f} | " + " | ".join(cells) + " |")
        print("\nWorst |Meep / PyStratify - 1| over wavelengths and orientations: "
              + ", ".join(f"{r} px/um: {worst(exact, runs[r]):.2%}" for r in resolutions) + ".")


def worst(exact, run):
    return max(float(np.max(np.abs(run[o] / exact[o] - 1))) for o in ORIENTATIONS)


if __name__ == "__main__":
    report(compare(resolutions=tuple(int(r) for r in sys.argv[1:]) or RESOLUTIONS))
