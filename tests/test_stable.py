"""`pystratify.stable`, the stable coherent planar kernel (benchmarks/stable_checks.py, benchmarks/byrnes_tmm.py).

The rows of ``benchmarks/stable_checks.py`` (against ``planar.coh_tmm``, repeated stacks, uniaxial, PEC and Bloch
physics, the harvester's numbers in ``tests/stable_fixtures.json``) and ``benchmarks/byrnes_tmm.py`` (lineage:
``planar`` and ``stable`` against Byrnes's ``tmm`` 0.2.0, skipped without it) are asserted here, with the refusals
and the two call forms that steps (b) and (c) of WP07 map onto.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import stable_checks  # noqa: E402
from checks import assert_rows  # noqa: E402

from pystratify import stable  # noqa: E402
from pystratify.stable import BlochExit, Layer, PEC_INDEX, Repeat, stable_tmm  # noqa: E402


def test_isotropic_stacks_against_coh_tmm():
    assert_rows(stable_checks.isotropic_against_coh_tmm())


def test_long_stacks_stay_finite_and_match_coh_tmm():
    assert_rows(stable_checks.long_stacks())


def test_repeat_groups_and_detection_equal_the_literal_stack():
    assert_rows(stable_checks.repeats())


def test_uniaxial_layers():
    assert_rows(stable_checks.uniaxial_physics())


def test_perfect_conductor_exit():
    assert_rows(stable_checks.pec_physics())


def test_bloch_exit():
    assert_rows(stable_checks.bloch_physics())


def test_the_harvesters_numbers():
    assert_rows(stable_checks.harvester_fixture())


def test_against_byrnes_tmm_lineage():
    pytest.importorskip("tmm")
    import byrnes_tmm

    assert_rows(byrnes_tmm.rows(byrnes_tmm.compare()))


# --- the two call forms ------------------------------------------------------------------------------------------

def test_harvester_list_form_and_its_markers():
    lam = np.linspace(500.0, 700.0, 5)
    layers_n = [np.full(5, 2.1 + 0j), np.full(5, 1.46 + 0.01j)]
    d = [80.0, 120.0]
    stack = stable.from_lists(layers_n, d, [1.9 + 0j, None])
    assert stack == [Layer(layers_n[0], 80.0, 1.9 + 0j), Layer(layers_n[1], 120.0, None)]
    res = stable_tmm("p", stack, np.ones(5), np.full(5, 1.52 + 0j), 0.4, lam)
    assert all(res[key].shape == (5,) for key in ("r", "t", "R", "T"))
    # The harvester's PEC_INDEX is this value; an array of it everywhere is a PEC exit too.
    assert PEC_INDEX == complex(0.0, np.inf) and stable.is_pec(np.full(5, PEC_INDEX)) and not stable.is_pec(1.5)
    pec = stable_tmm("s", stack, 1.0, np.full(5, PEC_INDEX), 0.4, lam)
    assert np.all(pec["T"] == 0)

    class HarvesterBlochExit(np.ndarray):  # verify.BlochExit: NaN values plus period_layers
        pass

    marker = np.full(5, complex(np.nan)).view(HarvesterBlochExit)
    marker.period_layers = 2
    crystal = stable.from_lists([np.full(5, 1.755 + 0j), np.full(5, 2.486 + 0j)], [38.0, 57.0])
    via_marker = stable_tmm("s", crystal, np.full(5, 2.478 + 0j), marker, 0.3, lam)
    explicit = stable_tmm("s", crystal, np.full(5, 2.478 + 0j), BlochExit(2), 0.3, lam)
    np.testing.assert_array_equal(via_marker["R"], explicit["R"])
    assert np.all(np.isfinite(explicit["R"]))


def test_api_segment_form():
    segments = [(1.9 + 0.02j, 110.0), ("group", [(2.3 + 0j, 60.0), (1.46 + 0j, 95.0)], 12)]
    as_named = [Layer(1.9 + 0.02j, 110.0), Repeat([Layer(2.3 + 0j, 60.0), Layer(1.46 + 0j, 95.0)], 12)]
    th = np.deg2rad(np.array([0.0, 30.0]))[:, None]
    lam = np.linspace(400.0, 900.0, 4)
    for pol in ("s", "p"):
        a = stable_tmm(pol, segments, 1.0, 1.52, th, lam)
        b = stable_tmm(pol, as_named, 1.0, 1.52, th, lam)
        assert set(a) == {"r", "t", "R", "T"} and a["R"].shape == (2, 4)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert stable.expanded_layer_count(segments) == 25


def test_unpolarized_is_the_mean():
    res = stable_tmm("unpolarized", [(2.0, 100.0)], 1.0, 1.5, 0.8, 600.0)
    assert set(res) == {"R", "T", "s", "p"}
    assert res["R"] == pytest.approx((res["s"]["R"] + res["p"]["R"]) / 2, abs=0)


# --- refusals ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"n_in": 1.5 + 0.01j, "th_0": 0.3}, "lossless at oblique incidence"),
    ({"n_in": 1.5 - 0.01j, "th_0": 0.0}, "gain in the incident medium"),
    ({"n_out": 1.5 - 0.1j}, "gain in the exit medium"),
    ({"th_0": 2.0}, "not the incoming"),
    ({"n_in": PEC_INDEX}, "incident medium cannot be a perfect conductor"),
    ({"n_out": np.array([PEC_INDEX, 1.5])}, "infinite at some points only"),
    ({"pol": "x"}, "polarization must be"),
    ({"lam_vac": -500.0}, "lam_vac must be positive"),
    ({"stack": [(2.0, -5.0)]}, "finite and >= 0"),
    ({"stack": [(2.0, np.inf)]}, "finite and >= 0"),
    ({"stack": [(2.0, np.array([1.0, 2.0]))]}, "real scalar"),
    ({"stack": [("group", [(2.0, 10.0)], 0)]}, "repeat count"),
    ({"stack": [("group", [(2.0, 10.0)], 2.5)]}, "repeat count"),
    ({"stack": [("group", [], 2)]}, "at least one member"),
    ({"stack": [("grup", [(2.0, 10.0)], 2)]}, "('group', members, count)"),
    ({"stack": [object()]}, "a stack entry must be"),
    ({"stack": [(2.0, 10.0)], "n_out": BlochExit(2)}, "Bloch period of 2 layers"),
    ({"stack": [Repeat([(2.0, 10.0)], 3)], "n_out": BlochExit(1)}, "plain layers"),
    ({"stack": [(2.0, 10.0)], "n_in": 1.5 + 1e-6j, "th_0": 0.0, "n_out": BlochExit(1)}, "lossless for a Bloch exit"),
])
def test_refusals_are_value_errors(kwargs, message):
    args = {"pol": "s", "stack": [(2.0, 100.0)], "n_in": 1.0, "n_out": 1.5, "th_0": 0.0, "lam_vac": 600.0, **kwargs}
    with pytest.raises(ValueError, match=__import__("re").escape(message)):
        stable_tmm(args["pol"], args["stack"], args["n_in"], args["n_out"], args["th_0"], args["lam_vac"])


def test_list_form_mismatches_are_refused():
    with pytest.raises(ValueError, match="one thickness per finite layer"):
        stable.from_lists([1.5, 2.0], [10.0])
    with pytest.raises(ValueError, match="layers_nz needs one entry"):
        stable.from_lists([1.5, 2.0], [10.0, 20.0], [None])


def test_bloch_drops_a_negligible_incident_loss_as_the_harvester_does():
    crystal = [(1.755, 38.0), (2.486, 57.0)]
    tiny = stable_tmm("p", crystal, 2.478 + 1e-12j, BlochExit(2), 0.0, 1000.0)
    clean = stable_tmm("p", crystal, 2.478, BlochExit(2), 0.0, 1000.0)
    np.testing.assert_array_equal(tiny["R"], clean["R"])
