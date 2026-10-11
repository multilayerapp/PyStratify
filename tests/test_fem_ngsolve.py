"""Dipoles beside a cylinder against NGSolve's finite elements through benchmarks/fem_ngsolve.py.

Skipped unless NGSolve is importable (``pip install ngsolve``). The coarsest mesh at 0.8 um (~30 s, ~3 GB);
the script's mesh ladder, and why 0.645 um is shown only, are in FEM_NGSOLVE.md.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("ngsolve")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import fem_ngsolve  # noqa: E402


def test_finite_elements_agree_with_pystratify_beside_a_fibre():
    wavelength = 0.8
    rates, _ = fem_ngsolve.ngsolve_rates(wavelength, *fem_ngsolve.MESHES[0])
    assert fem_ngsolve.worst(fem_ngsolve.stratify(wavelength), rates) < 0.05
