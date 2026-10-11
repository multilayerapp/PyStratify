"""Feibelman d-parameters (mesoscopic boundary conditions) in the normalized formulation.

Checked against: the mesoscopic Mie theory of Goncalves et al., Nat. Commun. 11, 366 (2020), Eqs. (4a)
and (13a), for a metal sphere; mpmath transfer matrices built from the boundary conditions on the M
and N fields, with the metal inside or outside an interface; energy conservation, the interfaces
absorbing the jump of the radial flux; TE as a sheet of conductivity i k0 d_par (eps_other - eps_metal).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

import pystratify as ps
from pystratify.references import layered_green_forms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import sphere_emitter_checks as emitters  # noqa: E402
from checks import assert_rows  # noqa: E402

AU = 0.1412 + 3.1518j  # Etchegoin, Le Ru & Meyer gold at 614 nm
F = ps.Feibelman
SHELL = ([8.0, 10.0], [1.45, AU, 1.33], 614.0)
SHELL_D = {0: F(0.25 + 0.08j, 0.1 + 0.02j, "outer"), 1: F(0.3 + 0.1j, -0.1 + 0.05j)}




@pytest.mark.parametrize("d_perp, d_par", emitters.GONCALVES_CASES)
def test_sphere_against_goncalves(d_perp, d_par):
    assert_rows(emitters.sphere_against_goncalves(d_perp, d_par))


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
