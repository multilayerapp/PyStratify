"""Feibelman d-parameters (mesoscopic boundary conditions) in the normalized formulation.

Checked against: the mesoscopic Mie theory of Goncalves et al., Nat. Commun. 11, 366 (2020), Eqs. (4a)
and (13a), for a metal sphere; mpmath transfer matrices built from the boundary conditions on the M
and N fields, with the metal inside or outside an interface; energy conservation, the interfaces
absorbing the jump of the radial flux; TE as a sheet of conductivity i k0 d_par (eps_other - eps_metal).
"""

import mpmath as mp
import numpy as np
import pytest

import pystratify as ps
from pystratify.references import layered_green_forms

AU = 0.1412 + 3.1518j  # Etchegoin, Le Ru & Meyer gold at 614 nm
F = ps.Feibelman
SHELL = ([8.0, 10.0], [1.45, AU, 1.33], 614.0)
SHELL_D = {0: F(0.25 + 0.08j, 0.1 + 0.02j, "outer"), 1: F(0.3 + 0.1j, -0.1 + 0.05j)}


def _goncalves_perp(R, n_m, n_d, lam, r, d_perp, d_par, orders=120):
    """Total rate of a radial dipole in host units from Eqs. (4a) and (13a) of Goncalves et al. (2020)."""
    with mp.workdps(40):
        k0 = 2 * mp.pi / lam
        xm, xd, y = k0 * mp.mpc(n_m) * R, k0 * n_d * R, k0 * n_d * r
        em, ed = mp.mpc(n_m) ** 2, mp.mpf(n_d) ** 2

        def j(l, z):
            return mp.sqrt(mp.pi / (2 * z)) * mp.besselj(l + 0.5, z)

        def h(l, z):
            return j(l, z) + 1j * mp.sqrt(mp.pi / (2 * z)) * mp.bessely(l + 0.5, z)

        def d_psi(l, z):  # (z j_l)'
            return z * j(l - 1, z) - l * j(l, z)

        def d_xi(l, z):
            return z * h(l - 1, z) - l * h(l, z)

        total = mp.mpf(0)
        for l in range(1, orders + 1):
            ll = l * (l + 1)
            num = em * j(l, xm) * d_psi(l, xd) - ed * j(l, xd) * d_psi(l, xm)
            num += (em - ed) * (j(l, xd) * j(l, xm) * ll * d_perp + d_psi(l, xd) * d_psi(l, xm) * d_par) / R
            den = em * j(l, xm) * d_xi(l, xd) - ed * h(l, xd) * d_psi(l, xm)
            den += (em - ed) * (h(l, xd) * j(l, xm) * ll * d_perp + d_xi(l, xd) * d_psi(l, xm) * d_par) / R
            total += (2 * l + 1) * ll * mp.re(-(num / den) * h(l, y) ** 2)
        return float(1 + 1.5 * total / y**2)


@pytest.mark.parametrize("d_perp, d_par", [(0.3 + 0.1j, 0.0), (0.3 + 0.1j, -0.1 + 0.05j), (-0.2 + 0.05j, 0.15 + 0.02j)])
def test_sphere_against_goncalves(d_perp, d_par):
    for r in (6.0, 7.0):
        ref = _goncalves_perp(5.0, AU, 1.0, 614.0, r, d_perp, d_par)
        rates = ps.decay_rates([5.0], [AU, 1.0], 614.0, r, sheets={0: F(d_perp, d_par)}, tol=1e-14)
        assert rates.route == "normalized" and rates.converged.all()
        assert abs(rates.total[0, 0] / ref - 1) <= 1e-13


@pytest.mark.parametrize(
    "radii, n, lam, r, sheets",
    [
        ([5.0], [AU, 1.0], 614.0, 6.0, {0: F(0.3 + 0.1j, -0.1 + 0.05j)}),
        ([5.0], [AU, 1.33], 614.0, 7.5, {0: F(0.3 + 0.1j)}),
        (*SHELL, 7.0, SHELL_D),
        (*SHELL, 11.0, SHELL_D),
    ],
)
def test_forms_against_boundary_conditions(radii, n, lam, r, sheets):
    """S, S^m, S^d per order against mpmath matrices from the boundary conditions on the fields."""
    L = 80
    t = ps.normalized_terms(radii, n, lam, r, l_max=L, sheets=sheets)
    for ref, form in zip(layered_green_forms(radii, n, lam, r, L, dps=80, sheets=sheets), (t.S, t.Sm, t.Sd)):
        assert np.max(np.abs((t.P * form)[:, 0] - ref) / np.abs(ref)) <= 1e-13


@pytest.mark.parametrize("r", [6.0, 7.0, 11.0, 12.0])
def test_energy_balance_with_surface_absorption(r):
    for moment in ([0, 0, 1.0], [1.0, 0, 0]):
        out = ps.emission_rates(*SHELL, [0, 0, r], moment, sheets=SHELL_D, tol=1e-13, warn=False)
        absorbed = out.absorption.sum() + out.sheet_absorption.sum()
        assert out.converged and np.all(out.sheet_absorption != 0)
        assert abs(out.total - out.radiative - absorbed) <= 1e-13 * out.total


def test_te_is_a_sheet_and_routes():
    """With d_perp = 0 TE sees the sheet sigma = i k0 d_par (eps_other - eps_metal), on either side."""
    radii, n, lam = [40.0, 50.0], [1.45, AU, 1.0], 614.0
    k0, eps = 2 * np.pi / lam, np.array(n) ** 2
    d_par = 0.12 + 0.03j
    for j, side in ((1, "inner"), (0, "outer")):
        sigma = 1j * k0 * d_par * ((eps[j + 1] - eps[j]) if side == "inner" else (eps[j] - eps[j + 1]))
        a = ps.normalized_terms(radii, n, lam, 52.0, l_max=60, sheets={j: F(0.0, d_par, side)})
        b = ps.normalized_terms(radii, n, lam, 52.0, l_max=60, sheets={j: ps.Sheet(sigma)})
        assert np.allclose(a.S[1], b.S[1], rtol=1e-13, atol=0)
    with pytest.raises(ValueError):
        ps.emission_rates(radii, n, lam, [0, 0, 52.0], [0, 0, 1.0], sheets={1: F(0.1)}, route="log")
    with pytest.raises(ValueError):
        F(0.1, metal="middle")
