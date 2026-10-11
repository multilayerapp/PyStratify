"""Fresnel at oblique incidence and incoherent stacks against an explicit phase average (benchmarks/planar_checks.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import planar_checks  # noqa: E402
from checks import assert_rows  # noqa: E402


def test_fresnel_oblique_brewster_and_normal_incidence():
    assert_rows(planar_checks.fresnel_oblique_brewster_normal())


def test_incoherent_stack_against_explicit_phase_average():
    assert_rows(planar_checks.incoherent_stack_against_phase_average())
