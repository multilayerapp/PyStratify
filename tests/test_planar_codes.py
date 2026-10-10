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

# PyMoosh's mode search stops at 1e-10; PyStratify's pole search at 2e-13 in n_eff / n_source.
TOLERANCE = {"Psi (deg)": 1e-10, "Delta (deg)": 1e-10, ("PyMoosh", "guided n_eff"): 1e-9}


def test_planar_films_agree_with_pymoosh_pyelli_and_the_slab_dispersion():
    worst = compare()
    assert {reference for _, reference, _ in worst} == {"PyMoosh", "pyElli 2x2", "pyElli 4x4", "slab dispersion"}
    assert any(quantity == "E_y (s), H_y (p)" for _, _, quantity in worst)

    def tolerance(reference, quantity):
        return TOLERANCE.get((reference, quantity), TOLERANCE.get(quantity, 1e-12))

    failures = {key: value for key, value in worst.items() if not value <= tolerance(*key[1:])}
    assert not failures, failures
