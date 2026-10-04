"""Chiral (Pasteur) layers in the normalized formulation: 2x2 maps in the (TM, TE) basis.

Checked against: the scalar sweep for kappa = 0 (TE included, whose mismatch vanishes exactly
between achiral nonmagnetic media); 80-digit 4x4 transfer matrices for the reflection matrices
seen from the emitter's shell; reciprocity (symmetric matrices); the logarithmic route where that
conserves energy; energy conservation; the mirror image (kappa -> -kappa with the enantiomer).
"""

import numpy as np
import pytest

import pystratify as ps
from pystratify import TE, TM
from pystratify.normalized import _HEL_TO_TT, _TT_TO_HEL, _ChiralSweep, _Sweep

from .mp_reference import chiral_t_matrix, log_psi_xi

AU = 0.1412 + 3.1518j
LAM = 614.0
SHELL = ([40.0, 50.0], [1.45, 1.6, 1.33], [0, 0.05, 0])
LOSSY_SHELL = ([40.0, 50.0], [1.45, 1.6 + 0.02j, 1.33], [0, 0.03 + 0.002j, 0])
CORE_AU = ([30.0, 40.0, 50.0], [1.6, AU, 1.45, 1.33], [0.05, 0, 0, 0])


@pytest.mark.parametrize(
    "radii, n, r", [([40.0, 50.0], [1.5, AU, 1.33], [30.0, 52.0]), ([30.0, 40.0, 50.0], [1.45, AU, 2.0, 1.33], [45.0])]
)
def test_achiral_limit_is_the_scalar_sweep(radii, n, r):
    radii, n = np.array(radii), np.array(n, complex)
    L = 60
    t = _Sweep(radii, n, np.ones_like(n), 2 * np.pi * n / LAM, L).at(np.array(r))
    c = _ChiralSweep(radii, n, np.zeros(n.size), np.ones_like(n), 2 * np.pi / LAM, L).at(np.array(r))
    for p, i in ((TM, 0), (TE, 1)):
        assert np.allclose(c["rho"][..., i, i], t["rho"][p], rtol=1e-13, atol=0)
        assert np.allclose(c["sigma"][..., i, i], t["sigma"][p], rtol=1e-13, atol=0)
        f = c["Fm"][..., i, i] * (1 + c["rho"][..., i, i])
        assert np.sum(np.abs(f - t["F"][p]) ** 2) <= 1e-26 * np.sum(np.abs(t["F"][p]) ** 2)
    assert np.all(c["rho"][..., 0, 1] == 0) and np.all(c["Fm"][..., 1, 0] == 0)


@pytest.mark.parametrize("radii, n, kappa, r", [(*SHELL, 55.0), (*LOSSY_SHELL, 30.0), (*CORE_AU, 45.0)])
def test_reflection_matrices_against_80_digits(radii, n, kappa, r):
    radii, n, kappa = np.array(radii), np.array(n, complex), np.array(kappa, complex)
    mu, L = np.ones_like(n), 20
    c = _ChiralSweep(radii, n, kappa, mu, 2 * np.pi / LAM, L).at(np.array([r]))
    d = int(np.searchsorted(radii, r, side="right"))
    x0 = 2 * np.pi * n[d].real / LAM * r
    for l in range(1, L + 1):
        args = (list(radii), list(n), list(kappa), list(mu), LAM, l)
        lp, lx = log_psi_xi(l, x0)
        refs = []
        if d:
            la, lb = chiral_t_matrix(*args, amplitudes_only=True)[d]
            refs.append((c["rho"][0, l - 1], np.exp(lb + lx - lp) @ np.linalg.inv(np.exp(la))))
        if d < radii.size:
            la, lb = chiral_t_matrix(*args, outgoing=True)[d]
            refs.append((c["sigma"][0, l - 1], np.exp(la + lp - lx) @ np.linalg.inv(np.exp(lb))))
        for mine, ref in refs:
            ref = _HEL_TO_TT @ ref @ _TT_TO_HEL
            assert np.max(np.abs(mine - ref)) <= 1e-13 * np.max(np.abs(ref))
            assert abs(mine[0, 1] - mine[1, 0]) <= 1e-14 * np.max(np.abs(mine))  # reciprocity


@pytest.mark.parametrize("radii, n, kappa", [SHELL, LOSSY_SHELL, CORE_AU])
def test_chiral_emitters_balance_log_route_and_mirror(radii, n, kappa):
    """A chiral emitter (p || m) near and inside chiral shells: energy conservation, agreement with
    the logarithmic route where that conserves energy, and the mirror image."""
    for r in (radii[-1] + 1.0, radii[-1] + 5.0):
        for p, m in (([0, 0, 1.0], [0, 0, 0.6j]), ([1.0, 0, 0], [0.6j, 0, 0])):
            kw = dict(kappa=kappa, tol=1e-12, warn=False)
            out = ps.emission_rates(radii, n, LAM, [0, 0, r], p, m, **kw)
            log = ps.emission_rates(radii, n, LAM, [0, 0, r], p, m, route="log", **kw)
            mirror = ps.emission_rates(
                radii, n, LAM, [0, 0, r], p, -np.array(m), kappa=-np.array(kappa), tol=1e-12, warn=False
            )
            absorbed = out.absorption.sum() + out.sheet_absorption.sum()
            assert out.route == "normalized" and out.converged
            assert abs(out.total - out.radiative - absorbed) <= 3e-13 * out.total
            if log.converged:
                assert abs(out.total / log.total - 1) <= 5e-13
            assert np.isclose(mirror.total, out.total, rtol=1e-14, atol=0)
            assert np.allclose(mirror.radiative_helicity[::-1], out.radiative_helicity, rtol=1e-13, atol=0)


def test_emitter_in_a_chiral_layer_takes_the_log_route():
    radii, n, kappa = SHELL
    out = ps.emission_rates(radii, n, LAM, [0, 0, 45.0], [0, 0, 1.0], kappa=kappa, warn=False)
    assert out.route == "log"
    with pytest.raises(ValueError):
        ps.emission_rates(radii, n, LAM, [0, 0, 45.0], [0, 0, 1.0], kappa=kappa, route="normalized")
