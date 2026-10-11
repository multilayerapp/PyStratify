"""Dipoles beside a sphere against smuthi through benchmarks/smuthi_spheres.py.

Skipped unless smuthi is importable (it builds only on Python <= 3.10 with NumPy < 2; see
benchmarks/emitter_references.py). Three cases at moderate truncation (~35 s).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("smuthi")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import smuthi_spheres  # noqa: E402

LADDERS = {"dielectric, n = 2, R = 150 nm, 100 nm away": ((10,), 1e-12),
           "in water, n = 1.5, R = 100 nm, 20 nm away": ((20,), 1e-12),
           "metal-like, eps = -10 + i, R = 20 nm, 5 nm away": ((40,), 1e-5)}


def test_sphere_emitters_agree_with_smuthi():
    results = smuthi_spheres.compare(tuple(LADDERS), {case: ladder for case, (ladder, _) in LADDERS.items()})
    for case, (exact, runs) in results.items():
        (_, rates, _), = runs
        assert np.max(np.abs(rates / exact - 1)) < LADDERS[case][1], (case, rates, exact)
