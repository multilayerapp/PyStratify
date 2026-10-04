"""2D sheets (graphene, TMD monolayers, thin films) on the interfaces of a multilayered sphere.

Checked against: explicit thin films (the sheet of a film is its first-order
limit: the error falls as d^2 with the out-of-plane term, as d without it);
the quasi-static plasmon condition of graphene-coated spheres; the chiral and
achiral solvers against each other; energy conservation - Poynting-flux jump,
the sheet-loss formula and extinction minus scattering for plane waves, total
= radiative + absorbed for emitters; the far field by reciprocity.
"""

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

import pystratify as ps
from pystratify.chiral import _log_add

LAM = 620.0
AU = 0.27 + 2.93j
EPS_FILM = 15.0 + 5.0j  # a TMD-like monolayer near its exciton


def _t_error(sheet_solution, film_solution, orders=8):
    """Largest error of each T-matrix block element over the first orders, relative to its size."""
    t, t0 = sheet_solution.t_matrix[0, :orders], film_solution.t_matrix[0, :orders]
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.abs(t - t0).max(axis=0) / np.abs(t0).max(axis=0))


@pytest.mark.parametrize("radius, core", [(50.0, AU), (200.0, 3.5)])
def test_thin_film_limit_is_second_order(radius, core):
    """Sheet.from_film vs the explicit film: O(d^2) with the normal term, O(d) (TM) without it."""
    errors, in_plane = [], []
    for d in (1.0, 0.5, 0.25):
        film = ps.solve([radius, radius + d], [core, np.sqrt(EPS_FILM), 1.33], LAM, l_max=30)
        sheet = ps.Sheet.from_film(EPS_FILM, d, LAM, 1.33**2)
        full = ps.solve([radius], [core, 1.33], LAM, l_max=30, sheets={0: sheet})
        planar = ps.solve([radius], [core, 1.33], LAM, l_max=30, sheets={0: sheet.conductivity})
        errors.append(np.diag(_t_error(full, film)))
        in_plane.append(np.diag(_t_error(planar, film)))
    errors, in_plane = np.array(errors), np.array(in_plane)  # (d, [TM, TE])
    assert np.all(errors[1:] / errors[:-1] < 0.3)  # ~1/4 per halving
    assert np.all(in_plane[1:, 0] / in_plane[:-1, 0] > 0.45)  # TM: ~1/2 per halving
    assert np.allclose(in_plane[:, 1], errors[:, 1])  # TE does not see the normal term
    assert errors[-1].max() < 2e-3


def test_film_between_chiral_layers():
    """An achiral film on a chiral core under an achiral spacer and a lossy chiral shell."""
    errors = []
    for d in (1.0, 0.5, 0.25):
        film = ps.solve_chiral(
            [50.0, 50.0 + d, 70.0, 80.0], [1.6, np.sqrt(EPS_FILM), 1.5 + 0.01j, 1.7, 1.33],
            [0.08, 0.0, 0.0, -0.1 + 0.02j, 0.0], LAM, l_max=30,
        )  # fmt: skip
        sheet = ps.solve_chiral(
            [50.0, 70.0, 80.0], [1.6, 1.5 + 0.01j, 1.7, 1.33], [0.08, 0.0, -0.1 + 0.02j, 0.0], LAM, l_max=30,
            sheets={0: ps.Sheet.from_film(EPS_FILM, d, LAM, (1.5 + 0.01j) ** 2)},
        )  # fmt: skip
        errors.append(_t_error(sheet, film).max())
    assert errors[1] / errors[0] < 0.3 and errors[2] / errors[1] < 0.3 and errors[-1] < 1e-3


def test_chiral_and_achiral_solvers_agree_with_sheets():
    radii, n, mu = [40.0, 55.0, 70.0], [1.45, AU, 1.6 + 0.02j, 1.33], [1.0, 1.0, 1.2, 1.0]
    sheets = {0: ps.Sheet(0.05 + 0.3j, 0.4 + 0.1j), 1: 0.2 - 0.05j, 2: ps.Sheet.from_film(EPS_FILM, 0.6, LAM, 1.33**2)}
    a = ps.solve(radii, n, LAM, mu, l_max=400, sheets=sheets)
    c = ps.solve_chiral(radii, n, [0, 0, 0, 0], LAM, mu, l_max=400, sheets=sheets)
    # helicity +1 = M + N: T_++ = (T_TM + T_TE) / 2, alpha_++ likewise; compared as logarithms
    ref = _log_add(a.log_t[0, 0], a.log_t[1, 0]) - np.log(2)
    assert np.abs(np.exp(c.log_t_helicity[0, :, 0, 0] - ref) - 1).max() < 1e-11
    for j in range(4):
        ref = _log_add(a.log_a[0, j, 0], a.log_a[1, j, 0]) - np.log(2)
        assert np.abs(np.exp(c.log_alpha[0, j, :, 0, 0] - ref) - 1).max() < 1e-10


def test_zero_sheet_changes_nothing():
    radii, n = [40.0, 55.0], [1.45, AU, 1.33]
    bare = ps.solve(radii, n, LAM, l_max=60)
    empty = ps.solve(radii, n, LAM, l_max=60, sheets={1: ps.Sheet(0.0, 0.0)})
    assert np.array_equal(bare.log_t, empty.log_t) and np.array_equal(bare.log_a, empty.log_a)
    assert not empty.has_sheets and ps.solve(radii, n, LAM, sheets={0: 0.1}).has_sheets


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_graphene_sphere_plasmons(order):
    """Resonances of a graphene-coated sphere vs eps_1 l + eps_2 (l + 1) + i sigma l (l + 1)/(k0 R) = 0."""
    radius, e1, e2 = 10.0, 2.1, 1.77

    def quasi_static(lam):
        s = ps.graphene_conductivity(lam, 0.5, 1e-5, 1.0)
        return (e1 * order + e2 * (order + 1) + 1j * s * order * (order + 1) * lam / (2 * np.pi * radius)).real

    lam_qs = brentq(quasi_static, 800, 200000)

    def minus_t(lam):
        sheet = {0: ps.graphene_conductivity(lam, 0.5, 1e-5, 1.0)}
        sol = ps.solve([radius], [np.sqrt(e1), np.sqrt(e2)], lam, l_max=order, sheets=sheet)
        return -abs(np.exp(sol.log_t[0, 0, order - 1]))

    peak = minimize_scalar(minus_t, bracket=(0.98 * lam_qs, lam_qs, 1.02 * lam_qs), tol=1e-12).x
    retardation = (2 * np.pi * radius * np.sqrt(e2) / lam_qs) ** 2
    assert abs(peak / lam_qs - 1) < retardation


def test_graphene_conductivity_limits():
    alpha = 7.2973525693e-3
    # far above 2 E_F: the universal pi alpha; far below: Drude, i 4 alpha E_F / hbar omega at T = 0
    assert ps.graphene_conductivity(50.0, 0.2, 0.0, 0.0) == pytest.approx(np.pi * alpha, rel=2e-2)
    hw = 1239.841984 / 50000.0
    low = ps.graphene_conductivity(50000.0, 0.4, 0.0, 0.0)
    assert low.imag == pytest.approx(4 * alpha * 0.4 / hw, rel=5e-2) and abs(low.real) < 1e-6
    # Drude weight grows with temperature at small E_F (2 kT ln(2 cosh(E_F / 2 kT)) > E_F)
    assert (
        ps.graphene_conductivity(20000.0, 0.05, 0.001, 300).imag
        > ps.graphene_conductivity(20000.0, 0.05, 0.001, 0).imag
    )
    assert ps.graphene_conductivity(20000.0, 0.3, 0.01, 300).real > 0  # passive


def test_plane_wave_absorption_by_sheets():
    """Extinction - scattering = Poynting-flux jump = (Z_h) int [Re sigma |<E_t>|^2 + k0 Im zeta |<D_n>|^2] dA."""
    radii, n, mu = [50.0, 70.0], [1.45, 1.8, 1.33], [1.0, 1.2, 1.0]
    sheets = {
        0: ps.Sheet(0.05 + 0.3j, 0.4 + 0.1j),
        1: ps.Sheet.from_film(EPS_FILM, 0.62, LAM, 1.33**2, eps_normal=6 + 1j),
    }
    sol = ps.solve(radii, n, LAM, mu, sheets=sheets, l_max=40)
    cs = ps.cross_sections(sol)
    ct, wt = np.polynomial.legendre.leggauss(64)
    phi = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    theta, phi = np.meshgrid(np.arccos(ct), phi, indexing="ij")
    weights = wt[:, None] * (2 * np.pi / 128)
    eps = np.array(n) ** 2 / np.array(mu)
    z_h = mu[-1] / n[-1]
    formula = flux = 0.0
    for j, radius in enumerate(radii):
        inner, outer = (
            ps.near_field(
                sol, *(rr * v for v in (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))
            )
            for rr in (radius * (1 - 1e-9), radius * (1 + 1e-9))
        )
        e_t = [(inner.e[c] + outer.e[c]) / 2 for c in ("theta", "phi")]
        d_n = (eps[j] * inner.e["r"] + eps[j + 1] * outer.e["r"]) / 2
        s, zeta = sol.sheet_sigma[0, j], sol.sheet_zeta[0, j]
        k0 = 2 * np.pi / LAM
        density = s.real * (np.abs(e_t[0]) ** 2 + np.abs(e_t[1]) ** 2) + k0 * zeta.imag * np.abs(d_n) ** 2
        formula += z_h * radius**2 * np.sum(weights * density)

        def poynting(f):
            return np.real(f.e["theta"] * np.conj(f.h["phi"]) - f.e["phi"] * np.conj(f.h["theta"]))

        flux += z_h * radius**2 * np.sum(weights * (poynting(inner) - poynting(outer)))
    assert formula == pytest.approx(cs.abs[0], rel=1e-8)
    assert flux == pytest.approx(cs.abs[0], rel=1e-8)


@pytest.mark.parametrize("r0", [30.0, 49.0, 51.0, 60.0, 80.0, 89.0, 91.0, 120.0])
def test_emitter_energy_balance_with_sheets(r0):
    """Sheets with in- and out-of-plane loss beside lossless, chiral and gold layers."""
    radii = [50.0, 70.0, 90.0]
    configs = [
        ([1.45, 1.8, 1.6, 1.33], [0, 0, 0, 0], [1, 1, 1, 1]),
        ([1.45, AU, 1.6, 1.33], [0.05, 0, -0.08, 0], [1.3, 1, 1.7, 1.1]),
    ]
    sheets = {
        0: ps.Sheet(0.05 + 0.3j, 0.4 + 0.1j),
        1: 0.1 + 0.02j,
        2: ps.Sheet.from_film(18 + 8j, 0.62, 614.0, 1.33**2, eps_normal=6 + 0.1j),
    }
    for n, kappa, mu in configs:
        if np.imag(n[ps.locate_shell(radii, r0)]) != 0:
            continue
        for p, m in (([0, 0, 1.0], None), ([0.3, -0.5, 0.8], [0.2j, 0.1, -0.4])):
            out = ps.emission_rates(
                radii, n, 614.0, [0, 0, r0], p, magnetic_moment=m, mu=mu, kappa=kappa, sheets=sheets, tol=1e-9,
                l_cap=600, warn=False,
            )  # fmt: skip
            assert out.balance_error < 5e-11
            assert np.all(out.sheet_absorption > 0)


def test_far_field_and_decay_rates_with_sheets():
    radii, n = [50.0, 70.0], [1.45, 1.8, 1.33]
    sheets = {0: ps.Sheet(0.05 + 0.3j, 0.4 + 0.1j), 1: ps.Sheet.from_film(EPS_FILM, 0.62, LAM, 1.33**2)}
    for position in ([0, 0, 30.0], [0, 40.0, 40.0], [0, 0, 100.0]):
        ff = ps.dipole_far_field(radii, n, LAM, position, [1.0, 0.3j, 0.2], 0.0, sheets=sheets, tol=1e-12)
        er = ps.emission_rates(radii, n, LAM, position, [1.0, 0.3j, 0.2], sheets=sheets, tol=1e-12)
        assert er.radiative == pytest.approx(ff.power, rel=1e-11)
        assert er.balance_error < 1e-12
    r = np.array([30.0, 60.0, 100.0])
    dr = ps.decay_rates(radii, n, LAM, r, dipole="magnetic", sheets=sheets, tol=1e-10, normalization="shell")
    assert np.all(dr.balance_error < 1e-11) and dr.converged.all()
    for i, r0 in enumerate(r):
        er = ps.emission_rates(radii, n, LAM, [0, 0, r0], [1.0, 0, 0], dipole="magnetic", sheets=sheets, tol=1e-10)
        assert dr.total[i, 1] == pytest.approx(er.total / er.free_in_layer, rel=1e-12)


def test_sheet_input_validation():
    with pytest.raises(ValueError, match="interface index"):
        ps.solve([50.0], [1.5, 1.0], LAM, sheets={1: 0.1})
    with pytest.raises(ValueError, match="map"):
        ps.solve([50.0], [1.5, 1.0], LAM, sheets=[0.1])
    with pytest.raises(ValueError, match="per wavelength"):
        ps.solve([50.0], [1.5, 1.0], [500.0, 600.0], sheets={0: [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="finite"):
        ps.solve_chiral([50.0], [1.5, 1.0], [0.1, 0], LAM, sheets={0: np.nan})
    batch = ps.solve([50.0], [1.5, 1.0], [500.0, 600.0], sheets={0: [0.1, 0.2j]})
    assert np.allclose(batch.sheet_sigma[:, 0], [0.1, 0.2j])


# ---- the normalized formulation with sheets -------------------------------------------------------

GRAPHENE = ps.graphene_conductivity(1500.0, 0.4, 0.0066)
TWO_SHEETS = {0: ps.Sheet(0.02 + 0.3j, 0.4 + 0.05j), 1: ps.Sheet(0.05 + 0.01j, 0.2 + 0.1j)}


@pytest.mark.parametrize(
    "radii, n, lam, r, sheets",
    [
        ([50.0], [1.5, 1.0], 1500.0, 51.0, {0: ps.Sheet(GRAPHENE)}),
        ([50.0], [1.5, 1.33], 1500.0, 49.0, {0: ps.Sheet(GRAPHENE)}),
        ([40.0, 50.0], [2.0, 1.4, 1.0], 900.0, 52.0, TWO_SHEETS),
        ([40.0, 50.0], [2.0, 1.4, 1.0], 900.0, 45.0, TWO_SHEETS),
    ],
)
def test_normalized_forms_with_sheets_against_extended_precision(radii, n, lam, r, sheets):
    """S, S^m, S^d per order against mpmath transfer matrices built from the transition conditions on
    the M and N fields (independent of the Moebius-map coefficients of the solvers)."""
    from pystratify.references import layered_green_forms

    L = 100
    t = ps.normalized_terms(radii, n, lam, r, l_max=L, sheets=sheets)
    for ref, form in zip(layered_green_forms(radii, n, lam, r, L, dps=80, sheets=sheets), (t.S, t.Sm, t.Sd)):
        mine = (t.P * form)[:, 0]
        assert np.max(np.abs(mine - ref) / np.abs(ref)) <= 1e-13


def test_shift_with_sheets_against_extended_precision():
    from pystratify.references import layered_green_sums

    radii, n, lam, r = [40.0, 50.0], [2.0, 1.4, 1.0], 900.0, 53.0
    rates = ps.decay_rates(radii, n, lam, r, sheets=TWO_SHEETS, normalization="shell", tol=1e-14)
    g = np.array(layered_green_sums(radii, n, lam, r, 600, dps=60, sheets=TWO_SHEETS))
    assert rates.route == "normalized" and rates.converged.all()
    assert np.allclose(rates.total[0], 1 + g.real, rtol=1e-12, atol=0)
    assert np.allclose(rates.shift[0], g.imag / 2, rtol=1e-12, atol=0)


@pytest.mark.parametrize("gap", [1.0, 5.0, -1.0])
@pytest.mark.parametrize("setup", ["graphene", "two sheets on gold"])
def test_normalized_route_with_sheets_balance_and_log_route(setup, gap):
    """total = radiative + absorbed in layers + absorbed in sheets (absorption from the logarithmic
    route, so this is a test), and agreement with the logarithmic route where that converges."""
    if setup == "graphene":
        radii, n, lam, sheets = [50.0], [1.5, 1.0], 1500.0, {0: ps.Sheet(GRAPHENE)}
    else:
        radii, n, lam, sheets = [40.0, 50.0], [0.2 + 4j, 1.4, 1.0], 900.0, TWO_SHEETS
    for moment in ([0, 0, 1.0], [1.0, 0, 0]):
        kw = dict(sheets=sheets, warn=False)
        out = ps.emission_rates(radii, n, lam, [0, 0, 50.0 + gap], moment, tol=1e-13, **kw)
        log = ps.emission_rates(radii, n, lam, [0, 0, 50.0 + gap], moment, tol=1e-12, route="log", **kw)
        absorbed = out.absorption.sum() + out.sheet_absorption.sum()
        assert out.route == "normalized" and out.converged and log.converged
        assert abs(out.total - out.radiative - absorbed) <= 3e-13 * out.total
        assert abs(out.total - log.total) <= 3e-13 * out.total
        assert np.max(np.abs(out.sheet_absorption - log.sheet_absorption)) <= 1e-13 * out.total
