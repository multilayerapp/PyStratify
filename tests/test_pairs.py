"""Two-point Green's dyadic: free-field series, reciprocity, the coinciding limit and extended precision."""

import sys
from pathlib import Path
import numpy as np
import pytest

import pystratify as ps
from pystratify import TE, TM
from pystratify.emission import _local_frame
from pystratify.normalized import _Sweep
from pystratify.pairs import _free_dyadic, _pair_terms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import sphere_emitter_checks as emitters  # noqa: E402
from checks import assert_rows  # noqa: E402
from sphere_emitter_checks import _mp_two_point  # noqa: E402

AU = 0.1412 + 3.1518j  # Etchegoin, Le Ru & Meyer gold at 614 nm
SHELL = ([50.0, 55.0], [1.45, AU, 1.33], 614.0)


def _free_series(radii, n, lam, r1, r2, L):
    radii, n = np.asarray(radii), np.asarray(n, complex)
    k = 2 * np.pi * n / lam
    a, b = np.linalg.norm(r1), np.linalg.norm(r2)
    frame = _local_frame(r1)
    v = frame @ r2
    phi = np.arctan2(v[1], v[0])
    frame = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1.0]]) @ frame
    gamma = float(np.arccos(np.clip((frame @ r2)[2] / b, -1, 1)))
    terms, _ = _pair_terms(radii, n, np.ones_like(n), k, L, a, b, gamma, series_free=True)
    ct, st = np.cos(gamma), np.sin(gamma)
    basis = np.array([[st, ct, 0.0], [0.0, 0.0, 1.0], [ct, -st, 0.0]])
    return frame.T @ ((1j * k[-1].real / (6 * np.pi)) * basis @ terms.sum(axis=0)) @ frame


@pytest.mark.parametrize(
    "r1, r2", [([0, 10, 85.0], [30, -20, 64.0]), ([50, 40, 30.0], [10, 5, 63.0]), ([20, -40, 50.0], [-35, 50, 55.0])]
)
def test_free_field_series_is_the_closed_form(r1, r2):
    """The multipole series of the free part, with the angular functions and constants of the scattered
    part, reproduces the closed-form free dyadic: every family, polarization and frame convention."""
    r1, r2 = np.array(r1), np.array(r2)
    g = _free_series([30.0, 60.0], [1.4, 1.4, 1.4], 600.0, r1, r2, 1500)
    g0 = _free_dyadic(2 * np.pi * 1.4 / 600.0, r1, r2)
    assert np.max(np.abs(g - g0)) <= 1e-13 * np.max(np.abs(g0))


def test_index_matched_and_inputs():
    out = ps.green_dyadic([30.0, 60.0], [1.4, 1.4, 1.4], 600.0, [0, 0, 70.0], [10, 20, 80.0])
    assert np.all(out.scattered == 0) and np.allclose(out.total, out.free)
    # points in different shells: the whole field, carried through the (fictitious) interfaces
    for r1, r2 in (([0, 0, 20.0], [10, 20, 80.0]), ([10, 20, 80.0], [3, -4, 40.0])):
        out = ps.green_dyadic([30.0, 60.0], [1.4, 1.4, 1.4], 600.0, r1, r2)
        g0 = _free_dyadic(2 * np.pi * 1.4 / 600.0, np.array(r1), np.array(r2))
        assert np.all(out.free == 0) and np.max(np.abs(out.total - g0)) <= 1e-13 * np.max(np.abs(g0))
    with pytest.raises(ValueError):
        ps.green_dyadic(*SHELL, [0, 0, 52.0], [0, 0, 60.0])  # in the gold shell
    with pytest.raises(ValueError):
        ps.green_dyadic(*SHELL, [0, 0, 60.0], [0, 0, 60.0])


@pytest.mark.parametrize("r1, r2", [([3.0, 1.0, 57.0], [-10.0, 6.0, 58.0]), ([5.0, 0, 40.0], [0, -8.0, 45.0])])
def test_reciprocity(r1, r2):
    g12, g21 = ps.green_dyadic(*SHELL, r1, r2), ps.green_dyadic(*SHELL, r2, r1)
    assert g12.converged and np.max(np.abs(g12.scattered - g21.scattered.T)) <= 1e-13 * np.max(np.abs(g12.scattered))


def test_coinciding_limit_gives_the_rates_and_the_shift():
    r = 57.0
    rates = ps.normalized_decay_rates(*SHELL, r, None, None, "electric")
    g = ps.green_dyadic(*SHELL, [0, 0, r], [0, 1e-5, r])
    s = g.scattered * 6 * np.pi / (1j * g.k)  # = g of the decay rates on the diagonal (local frame = global here)
    assert np.isclose(1 + s[2, 2].real, rates.total[0], rtol=1e-9) and np.isclose(
        1 + s[0, 0].real, rates.total[1], rtol=1e-9
    )
    assert np.isclose(s[2, 2].imag / 2, rates.shift[0], rtol=1e-9)



@pytest.mark.parametrize("a, b", emitters.TWO_POINT_CASES)
def test_two_point_radial_form_against_extended_precision(a, b):
    assert_rows(emitters.two_point_radial_form(a, b))


@pytest.mark.parametrize(
    "radii, n, lam, a, b",
    [
        (*SHELL, 40.0, 60.0),  # core -> host through a gold shell
        ([30.0, 40.0, 50.0], [1.45, AU, 2.0, 1.33], 614.0, 20.0, 45.0),
        ([30.0, 40.0, 50.0], [3.5, 1.2, 2.5, 1.0], 400.0, 25.0, 70.0),  # three dielectric interfaces
    ],
)
def test_cross_shell_radial_form_against_extended_precision(radii, n, lam, a, b):
    """psi_1 C xi_2 (1 + rho_1)(1 + sigma_2)/(1 - R S) with the transmission chain, per order."""
    L = 80
    n = np.asarray(n, complex)
    sweep = _Sweep(np.asarray(radii), n, np.ones(n.size, complex), 2 * np.pi * n / lam, L)
    t = sweep.at(np.array([a, b]))
    d1, d2 = t["shell"]
    l = np.arange(1, L + 1)
    ref = _mp_two_point(radii, n, lam, a, b, L)
    for p in (TM, TE):
        log_c, chain = sweep.transmission(p, d1, d2)
        rho1, sig1, sig2 = t["rho"][p, 0], t["sigma"][p, 0], t["sigma"][p, 1]
        pref = t["P"][p, 1] * np.exp((l + 1) * np.log(a / b) + t["lj"][p, 0] - t["lj"][p, 1] + log_c) * chain
        form = pref * (1 + rho1) * (1 + sig2) / (1 - rho1 * sig1)
        assert np.max(np.abs(form - ref[p]) / np.abs(ref[p])) <= 5e-14, p


@pytest.mark.parametrize("mu", [None, [1, 1, 1.5, 1.0]])
@pytest.mark.parametrize("source", [[5.0, -3.0, 20.0], [10.0, 20.0, 64.0], [-20.0, 30.0, 26.0]])
def test_interface_conditions(source, mu):
    """Tangential E and normal D continuous across a lossless interface, approached from both sides:
    source in the core (outward chain on both sides), in the host (same shell outside, reciprocity
    inside) and in the shell itself (same shell inside, outward chain outside)."""
    radii, n = [30.0, 40.0, 50.0], np.array([1.45, AU, 2.0, 1.33])
    eps = n**2 / (np.ones(4) if mu is None else np.asarray(mu))
    u = np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])

    def limit(side):  # cubic extrapolation to the interface from three points 1e-4 apart
        g = [ps.green_dyadic(radii, n, 614.0, source, (50 + side * j * 1e-4) * u, mu=mu) for j in (1, 2, 3)]
        assert all(v.converged for v in g)
        return 3 * g[0].total - 3 * g[1].total + g[2].total

    inside, outside = limit(-1), limit(+1)
    tangential = (np.eye(3) - np.outer(u, u)) @ (inside - outside)
    assert np.max(np.abs(tangential)) <= 1e-13 * np.max(np.abs(inside))
    assert np.max(np.abs(u @ (eps[2] * inside - eps[3] * outside))) <= 1e-13 * np.max(np.abs(eps[2] * u @ inside))
