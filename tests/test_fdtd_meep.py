"""Dipoles beside a cylinder against Meep's 3D FDTD through benchmarks/fdtd_meep.py.

Skipped unless MIT Meep is importable (conda-forge ``pymeep``; see the script). Agreement is at the
percent level by design: the check is that the FDTD answer approaches PyStratify's as the grid is
refined. Coarse grids only (~4 min); the script's full ladders are in FDTD_MEEP.md.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("meep")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import fdtd_meep  # noqa: E402


def test_meep_converges_to_pystratify_beside_a_fibre():
    (case,) = fdtd_meep.CASES
    exact, runs, _ = fdtd_meep.compare([case], resolutions=(20, 30))[case]
    coarse, fine = fdtd_meep.worst(exact, runs[20]), fdtd_meep.worst(exact, runs[30])
    assert fine < COARSE_LIMIT and fine < coarse, (coarse, fine)


COARSE_LIMIT = 0.05


def test_meep_on_an_axis_converges_to_pystratify_for_a_sphere_and_a_film():
    """Cylindrical coordinates (~2 min): within 3% for the sphere and 5% for the film at 100 px/um, and
    closer than at 25 px/um."""
    for case, limit in zip(fdtd_meep.AXIAL_CASES, (0.03, 0.05)):
        exact, runs, _ = fdtd_meep.compare_axial([case], resolutions=(25, 100))[case]
        names = fdtd_meep.AXIAL_CASES[case][-1]
        coarse, fine = fdtd_meep.worst(exact, runs[25], names), fdtd_meep.worst(exact, runs[100], names)
        assert fine < limit and fine < coarse, (case, coarse, fine)
