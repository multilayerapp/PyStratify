"""Spheres against MNPBEM, a boundary-element method (benchmarks/MATRIX.md, spheres).

    python benchmarks/mnpbem_spheres.py        # needs GNU Octave; prints MNPBEM_SPHERES.md

Every Mie code shares the multipole formulation; a boundary-element method discretizes the surface
and solves the integral equations instead, so it checks the formulation, not only the numerics.
MNPBEM's answer depends on its mesh: this script refines the mesh and reports how the difference to
PyStratify's exact multilayer Mie result shrinks. Agreement is at the percent level by design; the
check is that the difference falls as the mesh is refined. MNPBEM runs under GNU Octave through
``mnpbem_octave.py`` (full matrices; the finest single-sphere mesh, 2044 faces, peaks at 1.2 GB).

Cases, at 600 nm with constant permittivities (no tabulated data, so both codes see the same eps):

* a 40 nm sphere, eps = -10 + i, in vacuum: extinction and scattering cross sections, and the total
  and radiative decay rates of a dipole 5 nm from the surface, radial and tangential (MNPBEM
  normalizes to the dipole in vacuum, PyStratify to the dipole in the host; the host is vacuum);
* the same sphere in water (n = 1.33): cross sections;
* a nanoshell, SiO2 core (eps = 2.13) of radius 10 nm in a 10 nm shell of eps = -10 + i, in vacuum:
  cross sections (two surfaces, so coarser meshes);
* a thin nanoshell, core radius 15 nm, shell 5 nm: shown, not asserted. With elements of 2-4 nm on
  surfaces 5 nm apart MNPBEM's default integration stays ~12% low and barely moves with the mesh,
  while three Mie codes (PyStratify, treams, scattnlay) agree to all printed digits (C_ext 995.384013
  nm^2): a resolution limit of the boundary elements at the meshes this machine affords, not a
  disagreement about the physics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mnpbem_octave  # noqa: E402
import pystratify as ps  # noqa: E402

WAVELENGTH = 600.0
METAL = -10 + 1j
SILICA = 2.13
SPHERE_MESHES = (144, 256, 484, 1024)
SHELL_MESHES = (144, 256, 484)
SHELLS = {"nanoshell": 10.0, "thin nanoshell": 15.0}  # core radius in nm; the shell ends at 20 nm


def mnpbem(case):
    """MNPBEM's results per mesh: {vertices: {quantity: value}}."""
    if case in SHELLS:
        core = 2 * SHELLS[case]
        particle = ("p = comparticle( { epsconst( 1 ), epsconst( %s ), epsconst( %s ) }, "
                    "{ trisphere( nv, %g ), trisphere( nv, 40 ) }, [ 3, 2; 2, 1 ], 1, 2, op );"
                    % (_matlab(METAL), _matlab(SILICA), core))
        meshes, dipole = SHELL_MESHES, False
    else:
        host = 1.0 if case == "sphere in vacuum" else 1.33 ** 2
        particle = ("p = comparticle( { epsconst( %s ), epsconst( %s ) }, { trisphere( nv, 40 ) }, [ 2, 1 ], 1, op );"
                    % (_matlab(host), _matlab(METAL)))
        meshes, dipole = SPHERE_MESHES, case == "sphere in vacuum"
    script = f"""
op = bemoptions( 'sim', 'ret', 'interp', 'curv', 'waitbar', 0 );
for nv = [ {' '.join(str(m) for m in meshes)} ]
  {particle}
  bem = bemsolver( p, op );
  exc = planewave( [ 1, 0, 0 ], [ 0, 0, 1 ], op );
  sig = bem \\ exc( p, {WAVELENGTH} );
  printf( 'mesh %d faces %d ext %.12g sca %.12g\\n', nv, p.n, exc.ext( sig ), exc.sca( sig ) );
  if {int(dipole)}
    dip = dipoleret( compoint( p, [ 0, 0, 25 ] ), [ 0, 0, 1; 1, 0, 0 ], op );
    [ tot, rad ] = dip.decayrate( bem \\ dip( p, {WAVELENGTH} ) );
    printf( 'mesh %d rates %.12g %.12g %.12g %.12g\\n', nv, tot( 1 ), tot( 2 ), rad( 1 ), rad( 2 ) );
  end
  fflush( stdout );
end
"""
    results = {}
    for line in mnpbem_octave.run(script).splitlines():
        words = line.split()
        if not words or words[0] != "mesh":
            continue
        entry = results.setdefault(int(words[1]), {})
        if words[2] == "faces":
            entry.update(faces=int(words[3]), ext=float(words[5]), sca=float(words[7]))
        elif words[2] == "rates":
            entry.update(dict(zip(("radial total", "tangential total", "radial radiative", "tangential radiative"),
                                  map(float, words[3:7]))))
    return results


def _matlab(value):
    value = complex(value)
    return f"{value.real:.17g}" if value.imag == 0 else f"{value.real:.17g} + {value.imag:.17g}i"


def stratify(case):
    """PyStratify's exact values for the same quantities."""
    if case in SHELLS:
        radii, n = [SHELLS[case], 20.0], [np.sqrt(SILICA + 0j), np.sqrt(METAL + 0j), 1.0]
    else:
        radii, n = [20.0], [np.sqrt(METAL + 0j), 1.0 if case == "sphere in vacuum" else 1.33]
    sections = ps.cross_sections(ps.solve(radii, np.array([n]), np.array([WAVELENGTH])))
    area = np.pi * radii[-1] ** 2
    out = {"ext": sections.q_ext[0] * area, "sca": sections.q_sca[0] * area}
    if case == "sphere in vacuum":
        rates = ps.decay_rates(radii, n, WAVELENGTH, r=[25.0])
        out.update({"radial total": rates.total[0, 0], "tangential total": rates.total[0, 1],
                    "radial radiative": rates.radiative[0, 0], "tangential radiative": rates.radiative[0, 1]})
    return out


CASES = ("sphere in vacuum", "sphere in water", "nanoshell", "thin nanoshell")


def compare(cases=CASES):
    """{case: (exact values, {vertices: {quantity: MNPBEM value}})}."""
    return {case: (stratify(case), mnpbem(case)) for case in cases}


def report(results):
    print("# PyStratify spheres against MNPBEM (boundary elements)\n")
    print("Generated by `python benchmarks/mnpbem_spheres.py` (MNPBEM17 from the CPC program library, run")
    print(f"under GNU Octave by `mnpbem_octave.py`). {WAVELENGTH:g} nm, eps = {METAL} for the metal. Each cell is")
    print("MNPBEM's value and, in parentheses, its relative difference to PyStratify; the last row is PyStratify's")
    print("exact multilayer Mie value. Cross sections in nm^2; decay rates over the dipole in vacuum, 5 nm from")
    print("the 20 nm sphere.\n")
    for case, (exact, meshes) in results.items():
        quantities = list(exact)
        print(f"## {case}\n")
        print("| mesh (faces) | " + " | ".join(quantities) + " |")
        print("|---|" + "---|" * len(quantities))
        for vertices, values in sorted(meshes.items()):
            cells = [f"{values[q]:.6g} ({values[q] / exact[q] - 1:+.2%})" for q in quantities]
            print(f"| {values['faces']} | " + " | ".join(cells) + " |")
        print("| **PyStratify** | " + " | ".join(f"**{exact[q]:.6g}**" for q in quantities) + " |\n")
        if case == "thin nanoshell":
            print("Not a disagreement about the physics: treams and scattnlay give PyStratify's values to all printed")
            print("digits. With elements of 2-4 nm on two surfaces 5 nm apart, MNPBEM's default integration is")
            print("under-resolved at the meshes this machine affords (a finer mesh needs more than its 8 GB).\n")


if __name__ == "__main__":
    report(compare())
