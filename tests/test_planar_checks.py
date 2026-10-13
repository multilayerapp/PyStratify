"""Fresnel at oblique incidence, the bare-substrate ratio, the S-matrix cross-check and incoherent stacks against an explicit phase average (benchmarks/planar_checks.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import planar_checks  # noqa: E402
from checks import assert_rows  # noqa: E402


def test_fresnel_oblique_brewster_and_normal_incidence():
    assert_rows(planar_checks.fresnel_oblique_brewster_normal())


def test_incoherent_stack_against_explicit_phase_average():
    assert_rows(planar_checks.incoherent_stack_against_phase_average())


def test_bare_substrate_matches_the_fresnel_ratio():
    assert_rows(planar_checks.bare_substrate_ratio())


def test_scattering_matrix_cross_check():
    assert_rows(planar_checks.scattering_matrix_cross_check())


def test_position_resolved_absorption_integrates_to_each_layers_absorption():
    assert_rows(planar_checks.position_resolved_absorption())
