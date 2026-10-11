"""Dipoles beside a cylinder against Meep's 3D FDTD through benchmarks/fdtd_meep.py.

Skipped unless MIT Meep is importable (conda-forge ``pymeep``; see the script). Agreement is at the
percent level by design: the check is that the FDTD answer approaches PyStratify's as the grid is
refined. Two coarse grids only (~2 min); the script's full ladder is in FDTD_MEEP.md.
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
