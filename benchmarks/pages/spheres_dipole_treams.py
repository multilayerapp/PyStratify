"""Spheres, point dipole: PyStratify against treams.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_dipole_treams.py                                                                #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-treams.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, spheres_dipole_treams.json (the curves the page draws, value for value) and
spheres_dipole_treams.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

import pystratify as ps


def _treams_rates(tmatrix, k0, host, z0):
    """treams' radial and tangential rates of an electric dipole on the z axis at z0: its ℓ = 1 singular wave
    translated to the sphere's centre, scattered by the T-matrix and read back at the dipole."""
    import treams

    source = treams.SphericalWaveBasis.default(1, positions=[[0, 0, z0]])
    common = dict(basis=source, k0=k0, material=host, poltype="parity")
    rates = []
    for m, axis in ((0, 2), (1, 0)):
        c = np.zeros(len(source), complex)
        if m == 0:
            c[(source.m == 0) & (source.pol == 1)] = 1
        else:
            c[(source.m == -1) & (source.pol == 1)], c[(source.m == 1) & (source.pol == 1)] = 1, -1
        own, regular = (treams.PhysicsArray(c, modetype=t, **common) for t in ("singular", "regular"))
        scattered = tmatrix @ own.expand(tmatrix.basis, "regular")
        at = np.array([[0, 0, z0]])
        e_s, e_j = np.asarray(scattered.efield(at))[0, axis], np.asarray(regular.efield(at))[0, axis]
        rates.append(1 + np.real(e_s / e_j))
    return rates


@figure("treams-dipole-nanoshell", "Nanoshell", "gap [nm]", "Γ / Γ₀", log=True)
def plot_nanoshell():
    import treams

    gaps = np.geomspace(5, 60, 24)
    radii, n = [0.010, 0.020], [np.sqrt(2.13), np.sqrt(-10 + 1j + 0j), 1.0]
    lam, l_max = 0.6, 50
    k0 = 2 * np.pi / lam
    materials = [treams.Material(complex(v) ** 2) for v in n]
    tmatrix = treams.TMatrix.sphere(l_max, k0, radii, materials, poltype="parity")
    z0 = radii[-1] + gaps * 1e-3
    ours = np.array([ps.solve_problem(ps.Problem("spheres", radii, n, lam, ps.PointDipole(z, "electric", len(radii)),
                                                 1e-10))["total"][:2] for z in z0])
    ref = np.array([_treams_rates(tmatrix, k0, materials[-1], z) for z in z0])
    return result(
        gaps,
        [line("perp", "radial, PyStratify", ours[:, 0]), line("par", "tangential, PyStratify", ours[:, 1]),
         dots("perp_ref", "radial, treams", ref[:, 0]), dots("par_ref", "tangential, treams", ref[:, 1])],
        worst=largest(ours, ref, relative=True),
        worst_label="largest relative difference",
        note="treams translates the dipole's ℓ = 1 singular wave to the sphere's centre, scatters it with the "
             "T-matrix (50 orders; closer than 5 nm its truncation no longer converges, PyStratify's adaptive order "
             "does) and reads the scattered field back at the dipole; PyStratify places the jump in its "
             "transfer-matrix recursion.")


if __name__ == "__main__":
    run(__file__)
