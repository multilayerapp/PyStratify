"""Point dipoles beside a cylinder against NGSolve's finite elements (benchmarks/MATRIX.md, cylinders).

    pip install ngsolve                          # 6.2; its own environment (benchmarks/envs/README.md)
    python benchmarks/fem_ngsolve.py             # prints FEM_NGSOLVE.md (three meshes, two wavelengths: ~9 min, ~2.5 GB)

A second volume method beside Meep's FDTD (``fdtd_meep.py``): 3D Maxwell finite elements, sharing nothing
with PyStratify's cylindrical-wave expansion or with FDTD. The total field of a point dipole p at r0 is solved with
second-order Nedelec elements, the source assembled from the basis functions of the element holding r0; the radiated
power is Im(p . E(r0)), finite although Re E diverges there, and the decay rate is its ratio to the same mesh with the
fibre's permittivity set to the host's (so most discretisation error cancels, as with Meep's LDOS ratio).

**The cell is a cylinder around the fibre** (radius and half-height CELL = 1.2 um) inside a PML 1 um thick that is the
tensor product of a radial PML in rho and a Cartesian one in z (``pml.Compound``): the fibre runs through the z-PML,
which meets its guided modes head on, and stays infinite. **Mirror symmetry** cuts it to a quarter: the planes y = 0
and z = 0 through the dipole are a magnetic wall (natural boundary) where the dipole lies in the plane and an electric
wall (tangential E = 0) where it is normal to it, so each orientation is its own quarter problem; a source on the
walls carries a quarter of its strength, which the rate's ratio cancels. A quarter of a larger cell costs a quarter of
the unknowns: ~100k at the middle mesh, ~1 min and ~1.6 GB per wavelength.

The first cell (until 2026-10-13) was a sphere of 0.8 um with a 0.5 um radial PML. It agreed to 1.5% at 0.8 um but,
swept over 0.75-1.2 um, differed by up to 7% with alternating sign, set by where the PML began (0.8 -> 1.0 um turned
+8% into -6% at 0.75 um) and not by the mesh or the PML's strength; just above the TE01/TM01 cutoff (0.645 um) it was
8-11% low. The cylindrical cell converges toward PyStratify as the mesh is refined, worst 1.4% -> 0.8% -> 0.5% at
0.8 um and 4.4% -> 1.3% -> 0.8% at 0.645 um (FEM_NGSOLVE.md), and hardly moves when it grows (1.2 -> 1.6 um); the
coarsest mesh is too coarse for the far-reaching modes just above cutoff. NGSolve threads by itself (its TaskManager).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402

RADIUS, INDEX, SOURCE = 0.15, 2.0, 0.25  # um: an n = 2, 150 nm fibre in vacuum, the dipole 100 nm outside
WAVELENGTHS = (0.8, 0.645)  # um; 0.645: just above the TE01/TM01 cutoff (0.678 um), modes reaching far out
ORIENTATIONS = ("radial", "azimuthal", "axial")  # dipole along x, y, z at (r0, 0, 0); the axis is z
MESHES = ((0.32, 0.08), (0.25, 0.06), (0.2, 0.05))  # (maximum element size in vacuum and the PML, in the fibre), um
CELL, PML_THICKNESS, ORDER = 1.2, 1.0, 2  # um: the cell's radius and half-height, the PML around it
# each orientation's electric wall among the two mirror planes (the other is a magnetic wall)
WALLS = {"radial": "", "azimuthal": "ysym", "axial": "zsym"}


def stratify(wavelength):
    rates = ps.solve_problem(ps.Problem("cylinders", [RADIUS], [INDEX, 1.0], wavelength,
                                        ps.PointDipole(SOURCE, "electric"), 1e-6))
    return np.asarray(rates["total"][:3])


def ngsolve_rates(wavelength, h_vacuum, h_cylinder):
    """(decay rates radial/azimuthal/axial, the largest number of free unknowns) from one mesh."""
    from netgen.occ import Box, Cylinder, Glue, OCCGeometry, Pnt, Sphere, Y, Z
    from ngsolve import BilinearForm, ElementId, GridFunction, HCurl, Mesh, VOL, curl, dx, pml

    k = 2 * np.pi / wavelength
    big = CELL + PML_THICKNESS + 1
    quarter = Box(Pnt(-big, 0, 0), Pnt(big, big, big))  # y >= 0, z >= 0
    quarter.faces.Min(Y).name = "ysym"
    quarter.faces.Min(Z).name = "zsym"
    whole = Cylinder(Pnt(0, 0, -CELL - PML_THICKNESS), Z, r=CELL + PML_THICKNESS, h=2 * (CELL + PML_THICKNESS))
    whole.faces.name = "outer"
    outer = whole * quarter
    inner = Cylinder(Pnt(0, 0, -CELL), Z, r=CELL, h=2 * CELL) * quarter
    rod = Cylinder(Pnt(0, 0, -big), Z, r=RADIUS, h=2 * big) * quarter
    near = Sphere(Pnt(SOURCE, 0, 0), 0.06) * quarter
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
    mesh.SetPML(pml.Compound(pml.Radial(rad=CELL, alpha=1j, origin=(0, 0)),
                             pml.Cartesian(mins=[-CELL], maxs=[CELL], alpha=1j), dims1=(1, 2), dims2=(3,)),
                "pml|cylinder pml")
    point = mesh(SOURCE, 0, 0)
    out, unknowns = [], 0
    for i, orientation in enumerate(ORIENTATIONS):
        space = HCurl(mesh, order=ORDER, complex=True, dirichlet="|".join(w for w in ("outer", WALLS[orientation]) if w))
        unknowns = max(unknowns, sum(space.FreeDofs()))
        u, v = space.TnT()
        dofs = space.GetDofNrs(ElementId(VOL, point.nr))
        basis = GridFunction(space)
        values = []
        for d in dofs:  # the point source: every basis function of r0's element, evaluated at r0
            basis.vec[:] = 0
            basis.vec[d] = 1
            values.append(float(np.real(basis(point)) @ np.eye(3)[i]))
        powers = []
        for permittivity in (INDEX ** 2, 1.0):  # with the fibre, then without it on the same mesh
            eps = mesh.MaterialCF({"cylinder": permittivity, "cylinder pml": permittivity}, default=1)
            form = BilinearForm(space, symmetric=True)
            form += (curl(u) * curl(v) - k ** 2 * eps * u * v) * dx
            form.Assemble()
            inverse = form.mat.Inverse(space.FreeDofs(), inverse="sparsecholesky")
            f = basis.vec.CreateVector()
            f[:] = 0
            for d, value in zip(dofs, values):
                f[d] += value
            field = basis.vec.CreateVector()
            field.data = inverse * f
            powers.append(np.imag(np.dot(f.FV().NumPy(), field.FV().NumPy())))
            del inverse
        out.append(powers[0] / powers[1])
    return np.array(out), unknowns


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
    print(f"same mesh; order-{ORDER} Nedelec elements in a cylindrical cell of radius and half-height {CELL:g} um inside a")
    print(f"{PML_THICKNESS:g} um PML (radial in rho, Cartesian in z), a quarter of it by mirror symmetry (unknowns: the")
    print("largest of the three orientations' quarter problems). Relative difference NGSolve / PyStratify - 1; it")
    print("shrinks as the mesh is refined. At 0.645 um TE01 and TM01 sit just above their cutoff, their fields far out in")
    print("the vacuum: the coarsest mesh is too coarse there (the old spherical cell was 8-11% low at any mesh).")
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
    report(compare(meshes=MESHES[:int(sys.argv[1]) if sys.argv[1:] else 3]))
