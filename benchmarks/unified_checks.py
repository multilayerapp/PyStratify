"""Closed forms, limits and channel closure for films, cylinders and point sources (benchmarks/MATRIX.md).

    python benchmarks/unified_checks.py        # prints UNIFIED_CHECKS.md (~1 min)

Every check here is independent of the code it tests: a Fresnel coefficient, Bessel boundary equations
solved directly, a plane wave through an index-matched cylinder, the free dipole, an energy budget whose
channels are computed separately, a quadrature of the emission pattern, the closed-form dispersion
relations of a step-index fibre, treams' cylinder T-matrix. ``tests/test_unified.py`` asserts every row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pystratify as ps  # noqa: E402
from checks import close, flag, report, row  # noqa: E402
from pystratify.cylindrical import cross_widths, cylinder_pattern, solve_cylinder  # noqa: E402
from pystratify.planar import coh_tmm, position_resolved  # noqa: E402

GEOMETRIES = ("films", "cylinders", "spheres")
DIPOLES = ("electric", "magnetic")


def fresnel_and_opaque():
    rows = []
    for pol in ("s", "p"):
        name = f"Fresnel, n = 1 | 1.5, normal incidence, {pol}"
        s = coh_tmm(pol, [1, 1.5], [np.inf, np.inf], 0, .6)
        rows += [row(name, "|R - 0.04|", abs(s["R"] - .04), 1e-14), row(name, "|T - 0.96|", abs(s["T"] - .96), 1e-14)]
        name = f"opaque 20 um, n = 0.2 + 3i, {pol}"
        opaque = coh_tmm(pol, [1, .2 + 3j, 1], [np.inf, 20, np.inf], 0, .6)
        rows += [flag(name, "R finite, T = 0", np.isfinite(opaque["R"]) and opaque["T"] == 0),
                 flag(name, "Poynting finite at 0, 10, 20 um",
                      all(np.isfinite(position_resolved(1, z, opaque)["poyn"]) for z in (0, 10, 20))),
                 row(name, "|R - |(1 - n)/(1 + n)|^2|", abs(opaque["R"] - abs((1 - (.2 + 3j)) / (1 + (.2 + 3j))) ** 2), 1e-14)]
    return rows


def cylinder_bessel_boundary():
    name = "cylinder R = 0.1, n = 1.5, 0.6 um: Bessel boundary equations"
    radius, wavelength, n = .1, .6, 1.5
    s = solve_cylinder([radius], [n, 1], wavelength, m_max=12)
    x = 2 * np.pi * radius / wavelength
    worst = 0.0
    for index, m in enumerate(s.orders):
        for pol, eta in [(0, n), (1, 1 / n)]:
            want = -(jvp(m, x) * jv(m, n * x) - eta * jv(m, x) * jvp(m, n * x)) / (
                h1vp(m, x) * jv(m, n * x) - eta * hankel1(m, x) * jvp(m, n * x))
            worst = max(worst, abs(s.t[index, pol, pol] - want))
    result = cross_widths(s)
    phi = np.linspace(0, 2 * np.pi, 2001)
    return [row(name, "max |t - t_Bessel|", worst, 2e-14),
            row(name, "|extinction - 0.129405795136107|", abs(result["extinction"] - .129405795136107), 2e-14),
            row(name, "|extinction - scattering| (lossless)", abs(result["extinction"] - result["scattering"]), 2e-14),
            row(name, "|integral of the pattern - scattering|",
                abs(np.trapezoid(cylinder_pattern(s, phi), phi) - result["scattering"]), 1e-12)]


def homogeneous_point_source(geometry, dipole):
    name = f"{geometry}, {dipole} dipole in a homogeneous n = 1.5 (quantum yield 0.4)"
    dimensions = [np.inf, .2, np.inf] if geometry == "films" else [.1]
    n = [1.5] * len(dimensions) if geometry == "films" else [1.5, 1.5]
    source = ps.PointDipole(.1 if geometry == "films" else 0, dipole, 1 if geometry == "films" else 0, .4)
    r = ps.solve_problem(ps.Problem(geometry, dimensions, n, .6, source, 1e-6, 15 if geometry == "cylinders" else None))
    return [flag(name, "converged", r["diagnostics"]["converged"]),
            row(name, "max |total - 1|", np.abs(r["total"] - 1), 1e-8),
            row(name, "max |escape - 1|", np.abs(r["escape"] - 1), 1e-8),
            row(name, "max |eta_escape - 0.4|", np.abs(r["eta_escape"] - .4), 1e-8),
            row(name, "max |guided|", np.abs(r["guided"]), 1e-8),
            row(name, "max |absorbed|", np.abs(r["absorbed"]), 1e-8)]


def absorbing_coating_power_budget(geometry, dipole):
    name = f"{geometry}, {dipole} dipole by an absorbing (n = 0.2 + 3i) layer"
    dims = [np.inf, .1, np.inf] if geometry == "films" else [.05, .1]
    n = [1, .2 + 3j, 1] if geometry == "films" else [1.5, .2 + 3j, 1]
    source = ps.PointDipole(-.1 if geometry == "films" else .2, dipole, 0 if geometry == "films" else 2)
    r = ps.solve_problem(ps.Problem(geometry, dims, n, .6, source, 1e-5, 30 if geometry == "cylinders" else None))
    return [flag(name, "converged", r["diagnostics"]["converged"]),
            row(name, "min absorbed", np.min(r["absorbed"]), -1e-8, ">="),
            close(name, "total against escape + guided + absorbed", r["total"], r["escape"] + r["guided"] + r["absorbed"], 1e-5, 1e-6)]


def guided_point_sources():
    rows = []
    for g, d, n, pos, layer in [("films", [np.inf, .2, np.inf], [1, 1.5, 1], .1, 1), ("cylinders", [.1], [1.5, 1], .03, 0)]:
        name = f"{g}: a point source inside a lossless guide"
        r = ps.solve_problem(ps.Problem(g, d, n, .6, ps.PointDipole(pos, "electric", layer), 1e-5, 20 if g == "cylinders" else None))
        rows += [flag(name, "converged", r["diagnostics"]["converged"]),
                 row(name, "max guided fraction (a 3D source, not a line source)", np.max(r["guided"]), .1, ">"),
                 row(name, "guided poles found", len(r["diagnostics"]["poles"]), 0, ">"),
                 row(name, "max |total - escape - guided|", np.abs(r["total"] - r["escape"] - r["guided"]), 1e-6)]
    return rows


def planar_guided_pole_uncertainty(tolerance):
    from pystratify.planar_emission import FilmSource

    name = f"film n = 3.5, 0.8 um thick (18 guided poles), tolerance {tolerance:g}"
    rates = FilmSource([1, 3.5, 1], [np.inf, .8, np.inf], .6, 1, .4).rates(tolerance)
    rows = [row(name, "guided poles", len(rates.poles), 18, "=="),
            row(name, "error / (tolerance max(1, |total|)) when converged",
                rates.error / (tolerance * max(1, np.max(abs(rates.total)))) if rates.converged else 0.0, 1),
            row(name, "max |total - escape - guided|", np.abs(rates.total - rates.escape - rates.guided), tolerance)]
    if tolerance == 1e-6:
        rows.insert(0, flag(name, "converged", rates.converged))
    return rows


def pattern_normalization():
    x, w = np.polynomial.legendre.leggauss(48)
    theta = np.arccos(x)[:, None]
    phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)[None, :]
    rows = []
    for g, d, n, pos, layer in [("films", [np.inf, .2, np.inf], [1, 1, 1], .1, 1), ("cylinders", [.1], [1, 1], .03, 0),
                                ("spheres", [.1], [1, 1], 0, 0)]:
        r = ps.solve_problem(ps.Problem(g, d, n, .6, ps.PointDipole(pos, "electric", layer), 1e-6, 15 if g == "cylinders" else None),
                             ("rates", "pattern"), theta=theta, phi=phi)
        power = np.sum(r["pattern"] * w[:, None, None], axis=(0, 1)) * 2 * np.pi / phi.size
        rows.append(row(f"{g}: emission pattern (Gauss-Legendre x 64 azimuths)", "max |integral - escape|",
                        np.abs(power - r["escape"]), 1e-7))
    r = ps.solve_problem(ps.Problem("films", [np.inf, .2, np.inf], [1, 1, 1], .6, ps.PointDipole(.1, "electric", 1)),
                         ("rates", "pattern"), theta=np.array([np.pi / 2]), phi=0)
    rows.append(row("film, grazing pattern of a perpendicular dipole in free space", "|p(90 deg) / (3/8pi) - 1|",
                    abs(r["pattern"][0, 0] / (3 / (8 * np.pi)) - 1), 1e-3))
    return rows


TREAMS_CYLINDERS = (([.1], [1.5, 1], 0), ([.05, .1], [1.5, .2 + 3j, 1], 2), ([.05, .1], [1.5, 2, 1], 8))


def oblique_cylinder_against_treams(radii, n, beta):
    import treams

    name = f"cylinder radii {radii}, n {[complex(v) if np.iscomplexobj(v) else v for v in n]}, beta = {beta:g}"
    tm = treams.TMatrixC.cylinder([beta], 12, 2 * np.pi / .6, radii, [treams.Material(v * v) for v in n])
    sol = solve_cylinder(radii, n, .6, beta=beta, m_max=12)
    own = cross_widths(sol)
    phi = np.linspace(0, 2 * np.pi, 2001)
    return [row(name, "|extinction - treams|", abs(own["extinction"] - tm.xw_ext_avg), 1e-12),
            row(name, "|scattering - treams|", abs(own["scattering"] - tm.xw_sca_avg), 1e-12),
            row(name, "|integral of the pattern - scattering|", abs(np.trapezoid(cylinder_pattern(sol, phi), phi) - own["scattering"]), 1e-12)]


def cylinder_homogeneous_fields(beta):
    n, wavelength = 1.5, .6
    k = 2 * np.pi * n / wavelength
    q = np.sqrt(k * k - beta * beta)
    s = solve_cylinder([.1], [n, n], wavelength, beta=beta, m_max=24)
    points = np.array([[0, 0, 0], [.02, .03, .01], [.18, -.02, .04]])
    phase = np.exp(1j * (q * points[:, 0] + beta * points[:, 2]))[:, None]
    rows = []
    for pol, direction in [("axial-electric", [-beta / k, 0, q / k]), ("axial-magnetic", [0, 1, 0])]:
        name = f"index-matched cylinder, beta = {beta:g}, {pol}: the plane wave"
        E, H = s.field(points, pol)
        rows += [row(name, "max |E - plane wave|", np.abs(E - phase * np.array(direction)), 1e-10),
                 row(name, "max ||E|^2 - 1|", np.abs(np.sum(abs(E) ** 2, axis=-1) - 1), 1e-10),
                 row(name, "max ||H|^2 - n^2|", np.abs(np.sum(abs(H) ** 2, axis=-1) - n * n), 1e-10)]
    return rows


def metal_cylinder_grazing_caps(dipole):
    from pystratify.cylinder_emission import CylinderSource
    from pystratify.integration import integrate

    name = f"10 nm metal cylinder (n = 0.97 + 1.87i), {dipole} dipole 5 nm away: grazing caps"
    source = CylinderSource([.01], [.97 + 1.87j, 1], .5, .015, dipole)
    rates = source.rates()
    rows = [flag(name, "converged", rates.converged), row(name, "evaluations", rates.evaluations, 2500, "<"),
            row(name, "grazing error charged", rates.grazing_error, 0, ">"),
            row(name, "error - grazing error", rates.error - rates.grazing_error, 0, ">="),
            row(name, "error / max total", rates.error / np.max(rates.total), 1e-6),
            close(name, "total against escape + absorbed", rates.total, rates.escape + rates.absorbed, 1e-6)]
    # moving the precision guard inward checks the charged error independently
    reference = CylinderSource([.01], [.97 + 1.87j, 1], .5, .015, dipole, m_max=rates.orders)
    original = reference.spectral
    reference.spectral = lambda b, *a, **k: original(1 + np.copysign(3e-6, b - 1) if abs(b - 1) < 3e-6 else b, *a, **k)

    def sample(b):
        green, escape, absorbed, tail = reference.spectral(b)
        return np.r_[green.real, escape, absorbed, tail]

    integral = integrate(sample, [0, .97, 1, reference.maximum], 1e-6)
    rows.append(flag(name, "the guard moved to 3e-6 converges", integral.converged))
    for label, actual, expected in [("total", rates.total, 1 + integral.value[:3]), ("escape", rates.escape, integral.value[3:6]),
                                    ("absorbed", rates.absorbed, integral.value[6:9])]:
        rows.append(row(name, f"max |{label} - guard at 3e-6| / grazing error", np.max(abs(actual - expected)) / rates.grazing_error, 1, "<"))
    return rows


def cylinder_grazing_uncertainty_at_a_tighter_tolerance():
    from pystratify.cylinder_emission import CylinderSource

    name = "the same metal cylinder at tolerance 1e-9"
    rates = CylinderSource([.01], [.97 + 1.87j, 1], .5, .015, m_max=40).rates(tolerance=1e-9)
    return [row(name, "error - grazing error", rates.error - rates.grazing_error, 0, ">="),
            row(name, "grazing error / (1e-9 max total)", rates.grazing_error / (1e-9 * np.max(rates.total)), 1, ">"),
            flag(name, "reports not converged", not rates.converged)]


CUTOFF_WAVELENGTHS = (.62, .64, .645, .67, .677)


def fibre_modes_near_cutoff(wavelength):
    """TE01 and TM01 of a step-index fibre (n = 2, R = 150 nm) against their closed-form dispersion
    relations (Snyder & Love, Table 12-4): J1(u)/(u J0(u)) + c K1(w)/(w K0(w)) = 0, c = 1 (TE) or (n2/n1)^2
    (TM)."""
    from scipy.optimize import brentq
    from scipy.special import j0, j1, k0, k1
    from pystratify.cylinder_emission import CylinderSource

    radius, n1, n2 = .15, 2.0, 1.0
    V = 2 * np.pi / wavelength * radius

    def relation(b, c):
        u, w = V * np.sqrt(n1 ** 2 - b ** 2), V * np.sqrt(b ** 2 - n2 ** 2)
        return j1(u) / (u * j0(u)) + (c * (n2 / n1) ** 2 if c else 1) * k1(w) / (w * k0(w))

    upper = np.sqrt(n1 ** 2 - (2.404825557695773 / V) ** 2)  # J0's first zero: the root lies below it
    exact = sorted(brentq(relation, n2 + 1e-12, upper - 1e-12, args=(c,), xtol=1e-15) for c in (0, 1))
    source = CylinderSource([radius], [n1, n2], wavelength, .25)
    poles = source.guided_poles()
    name = f"fibre n = 2, R = 150 nm, {wavelength:g} um: TE01, TM01, HE11"
    rows = [row(name, "guided poles", len(poles), 3, "=="), flag(name, "third pole is HE11 (n_eff > 1.4)", len(poles) == 3 and poles[2] > 1.4)]
    if len(poles) == 3:
        rows.append(row(name, "max |n_eff - dispersion relation| (TE01, TM01)", np.abs(np.array(poles[:2]) - exact), 1e-10))
    rates = source.rates(tolerance=1e-7)
    rows += [flag(name, "converged at 1e-7", rates.converged),
             close(name, "total against escape + guided", rates.total, rates.escape + rates.guided, 1e-7, 1e-8)]
    return rows


def rates_smooth_through_cutoff():
    wavelengths = np.linspace(.66, .70, 9)  # TE01 and TM01 cut off near 0.678 um
    total = np.array([ps.solve_problem(ps.Problem("cylinders", [.15], [2.0, 1], w, ps.PointDipole(.25, "electric", 1), 1e-7))["total"][:3]
                      for w in wavelengths])
    curvature = np.abs(total[2:] - 2 * total[1:-1] + total[:-2])
    return [row("fibre rates over 0.66-0.70 um, through the TE01/TM01 cutoff", "max second difference", curvature, 2e-3, "<")]


def light_line_power_balance():
    from pystratify.cylinder_emission import CylinderSource

    name = "core-shell 24/70 nm (n 1.898/2.168), magnetic dipole 43 nm out, 912 nm (a mode 1.7e-5 above cutoff)"
    source = CylinderSource([.024, .07], [1.898, 2.168, 1.0], .912, .113, "magnetic")
    rows = []
    for b in (.3, .9, .99):
        green, escape, absorbed, _ = source.spectral(b)
        direct = np.array([.75 * (1 + b * b), .75 * (1 + b * b), 1.5 * (1 - b * b)])
        rows.append(close(name, f"Re(reflected density) against escape + absorbed - direct at b = {b:g}", green.real,
                          escape + absorbed - direct, 1e-12, 1e-14))
    rates = source.rates(tolerance=1e-6)
    return rows + [
            flag(name, "converged", rates.converged),
            row(name, "|pole - (1 + 1.66769e-5)|", abs(rates.poles[0] - (1 + 1.66769e-5)), 1e-9),
            close(name, "total against escape + guided", rates.total, rates.escape + rates.guided, 1e-6)]


def all_rows(treams=True):
    rows = fresnel_and_opaque() + cylinder_bessel_boundary()
    for g in GEOMETRIES:
        for d in DIPOLES:
            rows += homogeneous_point_source(g, d)
    for g in GEOMETRIES:
        for d in DIPOLES:
            rows += absorbing_coating_power_budget(g, d)
    rows += guided_point_sources() + planar_guided_pole_uncertainty(1e-6) + planar_guided_pole_uncertainty(1e-8)
    rows += pattern_normalization()
    if treams:
        for case in TREAMS_CYLINDERS:
            rows += oblique_cylinder_against_treams(*case)
    rows += cylinder_homogeneous_fields(0) + cylinder_homogeneous_fields(3)
    for d in DIPOLES:
        rows += metal_cylinder_grazing_caps(d)
    rows += cylinder_grazing_uncertainty_at_a_tighter_tolerance()
    for w in CUTOFF_WAVELENGTHS:
        rows += fibre_modes_near_cutoff(w)
    return rows + rates_smooth_through_cutoff() + light_line_power_balance()


if __name__ == "__main__":
    try:
        import treams  # noqa: F401
        available = True
    except ImportError:
        available = False
    report("Films, cylinders and point sources: closed forms, limits and channel closure",
           "Generated by `python benchmarks/unified_checks.py`; `tests/test_unified.py` asserts every row (lengths in um,"
           " wavelength 0.6 um unless stated)." + ("" if available else " treams not installed: its rows are omitted."),
           all_rows(available))
