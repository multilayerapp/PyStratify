"""Point dipoles beside a cylinder against NGSolve's finite elements (benchmarks/MATRIX.md, cylinders).

    pip install ngsolve                          # 6.2; its own environment with `pip install -e .` is simplest
    python benchmarks/fem_ngsolve.py             # prints FEM_NGSOLVE.md (two meshes: ~3 min, ~3.2 GB)
    python benchmarks/fem_ngsolve.py 3           # three meshes (adds 196k unknowns: ~4.1 GB, keep it free)

A second volume method beside Meep's FDTD (``fdtd_meep.py``): 3D Maxwell finite elements, sharing nothing
with PyStratify's cylindrical-wave expansion or with FDTD. The cell is a sphere around the dipole and the
cylinder's axis with a radial PML shell; the cylinder runs through the PML, so it stays infinite. The total
field of a point dipole p at r0 is solved with second-order Nedelec elements, the source assembled from the
basis functions of the element holding r0; the radiated power is Im(p . E(r0)), finite although Re E
diverges there, and the decay rate is its ratio to the same mesh with the cylinder's permittivity set to
the host's (so most discretisation error cancels, as with Meep's LDOS ratio).

A scattered-field formulation (the dipole's analytic field as the source inside the cylinder) was tried
first and rejected: it must stop the source where the cylinder enters the PML, and that cut radiates back
(the result swung by +-6% with the PML radius and grew with the PML's strength). The total field moves by
~2% over PML radii 0.6-1.0 um; that, not the mesh, is the floor at the meshes 8 GB allows, so agreement is
at the percent level by design and the check is that it approaches PyStratify as the mesh is refined.

At 0.645 um the fibre's TE01 and TM01 modes sit just above cutoff (n_eff 1.0098 and 1.043): their fields
reach ~0.7 um into the vacuum, into a PML that starts at 0.8 um, so this cell cannot resolve them (-8%);
it is shown, not asserted. Even so the radial rate sides with 0.10.4 (NGSolve 1.443, 0.10.4 1.406, 0.10.3
1.257). The CI-sized test runs 0.8 um on the coarsest mesh. NGSolve threads by itself (its TaskManager);
peak memory is set by the mesh, ~3.2 GB at 126k unknowns.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402

RADIUS, INDEX, SOURCE = 0.15, 2.0, 0.25  # um: an n = 2, 150 nm fibre in vacuum, the dipole 100 nm outside
WAVELENGTHS = (0.8, 0.645)  # um; 0.645: just above the TE01/TM01 cutoffs, where 0.10.3 was 12% low (shown only)
ORIENTATIONS = ("radial", "azimuthal", "axial")  # dipole along x, y, z at (r0, 0, 0); the axis is z
MESHES = ((0.32, 0.08), (0.25, 0.06), (0.22, 0.05))  # (maximum element size in vacuum, in the cylinder), um
PML_RADIUS, PML_THICKNESS, ORDER = 0.8, 0.5, 2


def stratify(wavelength):
    rates = ps.solve_problem(ps.Problem("cylinders", [RADIUS], [INDEX, 1.0], wavelength,
                                        ps.PointDipole(SOURCE, "electric"), 1e-6))
    return np.asarray(rates["total"][:3])


def ngsolve_rates(wavelength, h_vacuum, h_cylinder):
    """(decay rates radial/azimuthal/axial, degrees of freedom) from one mesh."""
    from netgen.occ import Cylinder, Glue, OCCGeometry, Pnt, Sphere, Z
    from ngsolve import BilinearForm, ElementId, GridFunction, HCurl, Mesh, VOL, curl, dx, pml

    k = 2 * np.pi / wavelength
    outer, inner = Sphere(Pnt(0, 0, 0), PML_RADIUS + PML_THICKNESS), Sphere(Pnt(0, 0, 0), PML_RADIUS)
    rod = Cylinder(Pnt(0, 0, -(PML_RADIUS + PML_THICKNESS) - 0.1), Z, r=RADIUS, h=2 * (PML_RADIUS + PML_THICKNESS) + 0.2)
    near = Sphere(Pnt(SOURCE, 0, 0), 0.06)
    core = rod * inner
    core.mat("cylinder")
    core.maxh = h_cylinder
    core_pml = rod * outer - inner
    core_pml.mat("cylinder pml")
    core_pml.maxh = 2 * h_cylinder
    source = near * inner - rod
    source.mat("vacuum")
    vacuum = inner - rod - near
    vacuum.mat("vacuum")
    shell = outer - inner - rod
    shell.mat("pml")
    mesh = Mesh(OCCGeometry(Glue([core, core_pml, source, vacuum, shell])).GenerateMesh(maxh=h_vacuum))
    mesh.Curve(ORDER)
    mesh.SetPML(pml.Radial(rad=PML_RADIUS, alpha=1j, origin=(0, 0, 0)), "pml|cylinder pml")
    space = HCurl(mesh, order=ORDER, complex=True)
    u, v = space.TnT()
    point = mesh(SOURCE, 0, 0)
    dofs = space.GetDofNrs(ElementId(VOL, point.nr))
    basis = GridFunction(space)
    values = []
    for d in dofs:  # the point source: every basis function of r0's element, evaluated at r0
        basis.vec[:] = 0
        basis.vec[d] = 1
        values.append(np.real(basis(point)))
    powers = {}
    for case, permittivity in (("cylinder", INDEX ** 2), ("reference", 1.0)):
        eps = mesh.MaterialCF({"cylinder": permittivity, "cylinder pml": permittivity}, default=1)
        form = BilinearForm(space, symmetric=True)
        form += (curl(u) * curl(v) - k ** 2 * eps * u * v) * dx
        form.Assemble()
        inverse = form.mat.Inverse(space.FreeDofs(), inverse="sparsecholesky")
        powers[case] = []
        for p in np.eye(3):
            f = basis.vec.CreateVector()
            f[:] = 0
            for d, value in zip(dofs, values):
                f[d] += float(value @ p)
            field = basis.vec.CreateVector()
            field.data = inverse * f
            powers[case].append(np.imag(np.dot(f.FV().NumPy(), field.FV().NumPy())))
        del inverse
    return np.array(powers["cylinder"]) / np.array(powers["reference"]), space.ndof


def compare(wavelengths=WAVELENGTHS, meshes=MESHES):
    """{wavelength: (PyStratify rates, [(mesh, dofs, NGSolve rates, seconds)])}."""
    out = {}
    for wavelength in wavelengths:
        runs = []
        for mesh in meshes:
            start = time.time()
            rates, dofs = ngsolve_rates(wavelength, *mesh)
            runs.append((mesh, dofs, rates, time.time() - start))
        out[wavelength] = stratify(wavelength), runs
    return out


def worst(exact, rates):
    return float(np.max(np.abs(np.asarray(rates) / exact - 1)))


def report(results):
    print("# Dipoles beside a cylinder: PyStratify against NGSolve (3D finite elements)\n")
    print("Generated by `python benchmarks/fem_ngsolve.py` (NGSolve 6.2, its own threads). Total decay rate relative to")
    print(f"free space of a dipole {SOURCE - RADIUS:g} um from an n = {INDEX:g}, {RADIUS * 1000:g} nm fibre in vacuum;")
    print("NGSolve's is Im(p . E(r0)) with the fibre over the same with the fibre's permittivity set to 1, on the")
    print(f"same mesh; order-{ORDER} Nedelec elements, PML from r = {PML_RADIUS:g} um, {PML_THICKNESS:g} um thick.")
    print("Relative difference NGSolve / PyStratify - 1; the floor (~1-2%) is the cell's size, see the script.")
    print("At 0.645 um the TE01/TM01 modes, just above cutoff, reach the PML: shown, not asserted (the radial")
    print("rate still sides with 0.10.4: 0.10.3 gave 1.257).")
    for wavelength, (exact, runs) in results.items():
        print(f"\n## {wavelength:g} um\n")
        print("| mesh (vacuum / fibre, um) | unknowns | time | " + " | ".join(ORIENTATIONS) + " |")
        print("|---|---|---|" + "---|" * len(ORIENTATIONS))
        print("| PyStratify | | | " + " | ".join(f"{v:.5f}" for v in exact) + " |")
        for (h_vacuum, h_cylinder), dofs, rates, seconds in runs:
            cells = [f"{r:.5f} ({r / e - 1:+.2%})" for r, e in zip(rates, exact)]
            print(f"| {h_vacuum:g} / {h_cylinder:g} | {dofs} | {seconds:.0f} s | " + " | ".join(cells) + " |")
        print("\nWorst |NGSolve / PyStratify - 1| per mesh: "
              + ", ".join(f"{worst(exact, rates):.2%}" for _, _, rates, _ in runs) + ".")


if __name__ == "__main__":
    report(compare(meshes=MESHES[:int(sys.argv[1]) if sys.argv[1:] else 2]))
