"""Spheres against scattnlay and treams through benchmarks/mie_codes.py.

Skipped unless both codes are installed (the ``benchmarks`` extra installs them).
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("scattnlay")
pytest.importorskip("treams")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from mie_codes import EXTRA_ORDERS, compare  # noqa: E402


def test_spheres_agree_with_scattnlay_and_treams():
    results = compare(chiral={})
    for (name, reference), values in results.items():
        # converged: the formulation at machine precision; default: Wiscombe's truncation, ~1e-8
        limit = 1e-7 if "default truncation" in reference else 1e-10
        assert all(value <= limit for value in values.values()), (name, reference, values)
    assert any(f"+ {EXTRA_ORDERS} orders" in reference for _, reference in results)
