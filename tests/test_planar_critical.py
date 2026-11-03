"""`coh_tmm` at a layer's own critical angle (assurance WP07, O-W7-10).

Where cos(theta) = 0 inside an interior layer, the reflection recursion meets an interface that reflects
perfectly while the layer's phase vanishes, and loses the 0 * inf limit between them: an air gap between two
glass prisms at its critical angle came back as R_s = 1. The characteristic matrix of such a layer tends to
[[1, i k0 d], [0, 1]] (s), so for a gap between equal half-spaces R_s = x^2 / (4 + x^2) with
x = n0 cos(theta0) k0 d. These pin that `coh_tmm` and the wavelength sweep now return that limit, that the
per-layer outputs refuse rather than return amplitudes that cannot exist there, and that every other point is
untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from pystratify import planar
from pystratify.stable import stable_tmm

N0, NG, LAM, D = 1.5, 1.0, 0.6, 0.12
THC = float(np.arcsin(NG / N0))          # the gap's critical angle, inside the prism
STACK = ([N0, NG, N0], [np.inf, D, np.inf])


def closed_form_rs(theta=THC):
    x = N0 * np.cos(theta) * 2 * np.pi / LAM * D
    return x ** 2 / (4 + x ** 2)


def test_r_s_at_the_critical_angle_is_the_characteristic_matrix_limit():
    data = planar.coh_tmm("s", *STACK, THC, LAM)
    assert data["R"] == pytest.approx(closed_form_rs(), abs=1e-12)
    assert data["R"] + data["T"] == pytest.approx(1.0, abs=1e-12)        # a lossless stack
    assert data["critical_layers"] == [1]
    assert isinstance(data["r"], complex) and isinstance(data["t"], complex)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_both_polarizations_equal_the_stable_kernel(pol):
    data = planar.coh_tmm(pol, *STACK, THC, LAM)
    ref = stable_tmm(pol, [(NG, D)], N0, N0, THC, LAM)
    assert abs(data["r"] - complex(np.squeeze(ref["r"]))) < 1e-12
    assert abs(data["t"] - complex(np.squeeze(ref["t"]))) < 1e-12


def test_the_wavelength_sweep_takes_the_same_limit():
    lams = np.array([0.5, 0.6, 0.7])
    out = planar.coh_tmm_wavelength_sweep("s", *STACK, THC, lams, include_layer_absorption=True)
    for lam, R in zip(lams, out["R"]):
        assert R == pytest.approx(planar.coh_tmm("s", *STACK, THC, lam)["R"], abs=1e-12)
    assert out["critical_layers"] == [1]
    assert np.isnan(out["layer_absorption"][:, 1]).all()                # undefined, never a number


def test_per_layer_outputs_refuse_at_the_critical_angle():
    data = planar.coh_tmm("s", *STACK, THC, LAM)
    for call in (lambda: planar.absorp_in_each_layer(data),
                 lambda: planar.position_resolved(1, 0.05, data),
                 lambda: planar.absorp_analytic_fn().fill_in(data, 1)):
        with pytest.raises(ValueError, match="critical angle"):
            call()


@pytest.mark.parametrize("offset", [1e-3, 1e-6])
def test_off_the_critical_angle_nothing_changes(offset):
    """Off the critical angle the recursion's own answer stands (no stable path), and it agrees."""
    data = planar.coh_tmm("s", *STACK, THC + offset, LAM)
    assert data["critical_layers"] == []
    assert np.isfinite(data["vw_list"]).all()
    ref = stable_tmm("s", [(NG, D)], N0, N0, THC + offset, LAM)
    assert data["R"] == pytest.approx(float(np.squeeze(ref["R"])), abs=1e-9)
    assert sum(planar.absorp_in_each_layer(data)) == pytest.approx(1.0, abs=1e-12)


def test_a_magnetic_stack_at_the_critical_angle_refuses():
    with pytest.raises(ValueError, match="magnetic"):
        planar.coh_tmm("s", *STACK, THC, LAM, mu_list=[1, 1.2, 1])
