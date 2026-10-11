"""Casimir-Polder potential: the normalized series at imaginary frequency, and the non-retarded limit."""

import sys
from pathlib import Path
import numpy as np
import pytest

import pystratify as ps
from pystratify.normalized import _dipole_series, _Sweep
from pystratify.references import layered_green_sums

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import sphere_emitter_checks as emitters  # noqa: E402
from checks import assert_rows  # noqa: E402
from sphere_emitter_checks import _alpha as alpha, _eps as eps  # noqa: E402

WP, GAMMA = 1.37e16, 1e14  # a Drude metal (gold-like), rad/s
A0, W0 = 0.05, 2.4e15  # alpha/(4 pi eps_0) = 0.05 nm^3 (an alkali atom), resonance 785 nm



@pytest.mark.parametrize("kappa0", [1e-4, 0.01, 0.2])
@pytest.mark.parametrize(
    "radii, n, r",
    [([50.0], [2.0, 1.0], 51.0), ([40.0, 50.0], [1.5, 3.0, 1.2], 49.0), ([40.0, 50.0], [1.5, 3.0, 1.2], 52.0)],
)
def test_series_at_imaginary_frequency(radii, n, r, kappa0):
    """g_perp, g_par at k = i kappa0 n against mpmath transfer matrices: real, and equal to <= 1e-13."""
    n = np.asarray(n, complex)
    k = 1j * kappa0 * n
    terms = _Sweep(np.asarray(radii), n, np.ones_like(n), k, 300).at(np.array([r]))
    for dipole in ("electric", "magnetic"):
        g = _dipole_series(terms, dipole)[0][0].sum(axis=0)
        ref = np.array(layered_green_sums(radii, 1j * n, 2 * np.pi / kappa0, r, 300, 80, dipole=dipole))
        assert np.all(np.abs(g.imag) <= 1e-14 * np.abs(g)) and np.allclose(g, ref, rtol=1e-13, atol=0)


def test_non_retarded_limit_near_a_large_sphere():
    assert_rows(emitters.casimir_non_retarded_limit())


def test_quadrature_converges():
    coarse = ps.casimir_polder([200.0], eps, [202.0], alpha, nodes=48)
    fine = ps.casimir_polder([200.0], eps, [202.0], alpha, nodes=96)
    assert np.isclose(coarse.energy[0], fine.energy[0], rtol=1e-6, atol=0)
