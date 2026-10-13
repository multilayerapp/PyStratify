"""Dipoles beside a cylinder against NGSolve's finite elements through benchmarks/fem_ngsolve.py.

Skipped unless NGSolve is importable (``pip install ngsolve``). The coarsest mesh at 0.8 um, and the middle one at
0.645 um, just above the TE01/TM01 cutoff, where the coarsest is 4% off (~1.5 min, ~1.6 GB); the mesh ladder is in
FEM_NGSOLVE.md.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("ngsolve")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import fem_ngsolve  # noqa: E402


@pytest.mark.parametrize("wavelength, mesh", [(0.8, 0), (0.645, 1)])
def test_finite_elements_agree_with_pystratify_beside_a_fibre(wavelength, mesh):
    rates, _ = fem_ngsolve.ngsolve_rates(wavelength, *fem_ngsolve.MESHES[mesh])
    assert fem_ngsolve.worst(fem_ngsolve.stratify(wavelength), rates) < 0.03
