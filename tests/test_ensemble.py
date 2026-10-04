"""Shell averages and the spectral density J(omega)."""

import numpy as np
import pytest

import pystratify as ps

AU = 0.1412 + 3.1518j  # Etchegoin, Le Ru & Meyer gold at 614 nm
COATED = ([50.0, 60.0], [AU, 1.45, 1.33], 614.0)  # dye-doped silica shell on a gold core, in water


def test_shell_average_free_space():
    s = ps.shell_average([40.0, 60.0], [1.4, 1.4, 1.4], 600.0, 1, intrinsic=0.5)
    assert np.isclose(s.total, 1, atol=1e-13) and np.isclose(s.radiative, 1, atol=1e-13)
    assert abs(s.nonradiative) < 1e-13 and np.isclose(s.quantum_yield, 0.5, atol=1e-13) and s.converged


def test_shell_average_converges_with_graded_nodes():
    """Nodes graded geometrically towards the gold: 12 per half already give the average to ~1e-13."""
    coarse = ps.shell_average(*COATED, 1, d_min=0.5, nodes=12, intrinsic=0.8)
    fine = ps.shell_average(*COATED, 1, d_min=0.5, nodes=48, intrinsic=0.8)
    for key in ("total", "radiative", "nonradiative", "quantum_yield"):
        assert np.isclose(getattr(coarse, key), getattr(fine, key), rtol=1e-12, atol=0), key
    assert fine.span == (50.5, 59.5) and np.isclose(fine.weights.sum(), 1)


def test_shell_average_cutoff():
    """Next to the gold the averaged nonradiative rate diverges as d_min^-2."""
    a = ps.shell_average(*COATED, 1, d_min=0.2).nonradiative
    b = ps.shell_average(*COATED, 1, d_min=0.1).nonradiative
    assert 3.5 < b / a < 4.0
    with pytest.raises(ValueError):
        ps.shell_average(*COATED, 1)  # d_min needed next to an absorbing layer
    with pytest.raises(ValueError):
        ps.shell_average(*COATED, 0, d_min=0.5)  # the gold core itself


def test_shell_average_thin_shell_is_the_midpoint():
    radii, n, lam = [50.0, 50.4, 60.0], [1.45, 1.6, 1.45, 1.33], 614.0
    s = ps.shell_average(radii, n, lam, 1)
    mid = ps.decay_rates(radii, n, lam, [50.2], normalization="host", tol=1e-12)
    assert np.isclose(s.total, (mid.total[0, 0] + 2 * mid.total[0, 1]) / 3, rtol=1e-4)


def test_spectral_density():
    radii, n = [50.0, 55.0], [1.45, AU, 1.33]
    sd = ps.spectral_density(radii, n, [600.0, 614.0, 630.0], 56.0, reference=614.0)
    ref = ps.decay_rates(radii, n, 614.0, [56.0], tol=1e-10)
    assert np.isclose(2 * np.pi * sd.J[1], (ref.total[0, 0] + 2 * ref.total[0, 1]) / 3, rtol=1e-12)
    assert np.isclose(sd.shift[1], (ref.shift[0, 0] + 2 * ref.shift[0, 1]) / 3, rtol=1e-12)
    free = ps.spectral_density(radii, [1.33] * 3, [500.0, 614.0, 700.0], 56.0, reference=614.0)
    assert np.allclose(2 * np.pi * free.J, (614.0 / np.array([500.0, 614.0, 700.0])) ** 3, rtol=1e-12)
    assert np.allclose(free.shift, 0, atol=1e-12)
