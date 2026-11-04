"""H in `position_resolved` and the field of stacks with incoherent layers (`inc_field`), assurance WP07-D2 / O-W7-12.

The physics rows (Maxwell's equations for H, the incoherent slab in closed form, the band-averaged coherent field,
a coherent stack routed through `inc_tmm`) are `benchmarks/planar_field_checks.py`; the rest pin `inc_field`'s
interface: vectorised depths, its refusals, and `inc_tmm`'s new keys.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import planar_field_checks  # noqa: E402
from checks import assert_rows  # noqa: E402

from pystratify import planar  # noqa: E402

inf = np.inf


def test_h_satisfies_maxwell_on_random_stacks():
    assert_rows(planar_field_checks.maxwell_rows())


def test_incoherent_slab_against_its_closed_form():
    assert_rows(planar_field_checks.analytic_rows())


def test_inc_field_against_the_band_averaged_coherent_field():
    assert_rows(planar_field_checks.band_rows())


def test_a_coherent_stack_routed_through_inc_tmm_is_position_resolved():
    assert_rows(planar_field_checks.routing_rows())


MIXED = ([1.0, 2.0, 1.5 + 1e-4j, 1.38, 1.0], [inf, 0.1, 50.0, 0.12, inf], ["i", "c", "i", "c", "i"])


@pytest.mark.parametrize("pol", ["s", "p"])
def test_inc_field_takes_an_array_of_depths(pol):
    data = planar.inc_tmm(pol, *MIXED, np.radians(30), 0.6)
    for layer, depths in ((0, np.array([-1.0, -0.2, 0.0])), (1, np.linspace(0, 0.1, 7)), (2, np.linspace(0, 50, 9)),
                          (3, np.linspace(0, 0.12, 5)), (4, np.array([0.0, 3.0]))):
        vector = planar.inc_field(layer, depths, data)
        assert vector["E2"].shape == vector["H2"].shape == depths.shape
        for i, z in enumerate(depths):
            scalar = planar.inc_field(layer, float(z), data)
            assert isinstance(scalar["E2"], float) and isinstance(scalar["H2"], float)
            assert scalar["E2"] == pytest.approx(vector["E2"][i], rel=1e-14)
            assert scalar["H2"] == pytest.approx(vector["H2"][i], rel=1e-14)
            assert scalar["coherent"] is vector["coherent"] is (layer in (1, 3))


def test_inc_tmm_carries_what_inc_field_reads():
    data = planar.inc_tmm("p", *MIXED, 0.3, 0.6)
    assert data["pol"] == "p" and data["th_0"] == 0.3 and data["lam_vac"] == 0.6 and data["c_list"] == MIXED[2]
    np.testing.assert_array_equal(data["n_list"], np.array(MIXED[0]))
    np.testing.assert_array_equal(data["th_list"], planar.list_snell(np.array(MIXED[0]), 0.3))
    pim = planar.pim_tmm("p", *MIXED, 0.3, 0.6)
    assert planar.inc_field(2, 10.0, pim) == planar.inc_field(2, 10.0, data)


def test_inc_field_refuses_what_it_cannot_define():
    data = planar.inc_tmm("s", *MIXED, 0.0, 0.6)
    for layer, z in ((0, 0.1), (1, -0.01), (1, 0.2), (2, 51.0), (4, -1.0), (5, 0.0), (-1, 0.0)):
        with pytest.raises(ValueError):
            planar.inc_field(layer, z, data)
    with pytest.raises(ValueError, match="inc_tmm's output"):
        planar.inc_field(1, 0.0, planar.coh_tmm("s", [1.0, 1.5, 1.0], [inf, 0.1, inf], 0.0, 0.6))


def test_inc_field_refuses_an_evanescent_incoherent_layer():
    """Glass into air beyond the critical angle: the air exit carries no flux, so its power fixes no field."""
    data = planar.inc_tmm("s", [1.5, 1.38, 1.0], [inf, 0.1, inf], ["i", "c", "i"], np.radians(60), 0.6)
    assert planar.inc_field(1, 0.05, data)["E2"] > 0
    with pytest.raises(ValueError, match="no propagating wave"):
        planar.inc_field(2, 0.0, data)


def test_inc_field_refuses_a_coherent_layer_at_its_own_critical_angle():
    """O-W7-10: an air gap between glass prisms at its critical angle has no forward/backward split."""
    theta = float(np.arcsin(1.0 / 1.5))
    data = planar.inc_tmm("s", [1.5, 1.0, 1.5, 1.5], [inf, 0.12, 100.0, inf], ["i", "c", "i", "i"], theta, 0.6)
    with pytest.raises(ValueError, match="critical angle"):
        planar.inc_field(1, 0.05, data)
