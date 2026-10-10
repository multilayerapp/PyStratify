"""Planar films against PyMoosh and pyElli, codes outside the ``tmm`` lineage (benchmarks/planar_codes.py).

Skipped unless the ``benchmarks`` extra is installed: ``pip install -e ".[test,benchmarks]"``.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("PyMoosh")
pytest.importorskip("elli")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from planar_codes import compare  # noqa: E402

TOLERANCE = {"Psi (deg)": 1e-10, "Delta (deg)": 1e-10}


def test_planar_films_agree_with_pymoosh_and_pyelli():
    worst = compare()
    assert {reference for _, reference, _ in worst} == {"PyMoosh", "pyElli 2x2", "pyElli 4x4"}
    failures = {key: value for key, value in worst.items() if not value <= TOLERANCE.get(key[2], 1e-12)}
    assert not failures, failures
