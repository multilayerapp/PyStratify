"""Films against meent's RCWA with zero patterning (benchmarks/rcwa_meent.py); skipped unless meent is installed."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("meent")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import rcwa_meent  # noqa: E402


def test_rcwa_reduces_to_the_transfer_matrix_on_an_unpatterned_stack():
    assert all(rcwa_meent.worst() < 1e-9)
