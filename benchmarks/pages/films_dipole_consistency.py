"""Thin films, point dipole: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_dipole_consistency.py                                                             #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-dipole-consistency.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
films_dipole_consistency.json (the curves the page draws, value for value) and films_dipole_consistency.png (needs matplotlib).
"""

from __future__ import annotations

import json  # noqa: F401
import sys
from pathlib import Path

import numpy as np
from numpy import inf  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "benchmarks", ROOT / "benchmarks" / "pages"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pystratify as ps  # noqa: E402,F401
from pystratify import planar  # noqa: E402,F401

from _page import figure, run, series, sig, worst  # noqa: E402,F401
from _shared import _guide_modes  # noqa: E402


@figure("sommerfeld-film-emitter", "Silver film", "height above the silver [nm]", "Γ / Γ₀", log=True)
def sommerfeld_film_emitter():
    import emitter_references as er

    n, d = er.STACKS["air above 60 nm silver on glass"]
    heights = np.geomspace(5, 400, 40)
    ours = np.array([er.stratify_rates(n, d, h) for h in heights])
    ref = np.array([er.reference_rates(n, d, h) for h in heights])
    return dict(x=sig(heights), series=[
        series("perp", "perpendicular, PyStratify", ours[:, 0], "pystratify"), series("par", "parallel, PyStratify", ours[:, 1], "pystratify"),
        series("perp_ref", "perpendicular, Sommerfeld integral", ref[:, 0], "reference"), series("par_ref", "parallel, Sommerfeld integral", ref[:, 1], "reference")],
        worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
        note="The reference integrates the Chance–Prock–Silbey integrands with Parratt's recursion for r_s and r_p, written without PyStratify; quenching rises as d⁻³ at contact and the surface plasmon of the silver sets the breakpoint of the evanescent integral.")


@figure("film-dipole-homogeneous", "Index-matched film", "depth in the film [nm]", "|Γ / Γ₀ − 1|", log=True)
def film_dipole_homogeneous():
    z = np.linspace(.005, .195, 39)
    out = {}
    for dipole in ("electric", "magnetic"):
        for zi in z:
            r = ps.solve_problem(ps.Problem("films", [inf, .2, inf], [1.5] * 3, .6, ps.PointDipole(zi, dipole, 1, .4), 1e-6))
            out.setdefault(dipole, []).append(max(float(np.max(np.abs(r["total"] - 1))), 1e-17))
    return dict(x=sig(z * 1e3), series=[series(d, f"{d} dipole, max over orientations, PyStratify", out[d], "pystratify") for d in out],
                worst=float(max(max(v) for v in out.values())), worst_label="largest |Γ / Γ₀ − 1|",
                note="A film of the surrounding index is no film: every rate is the free-space one, 1, wherever the dipole sits; the test also "
                     "checks escape = 1, no guided or absorbed power and the quantum yield 0.4 passed through. Floored at 1e-17 for the "
                     "logarithmic axis.")


def _film_coating_budget(dipole):
    h = np.geomspace(.005, .4, 30)
    total, channels = [], []
    for hi in h:
        r = ps.solve_problem(ps.Problem("films", [inf, .1, inf], [1, .2 + 3j, 1], .6, ps.PointDipole(-hi, dipole, 0), 1e-5))
        total.append(r["total"][:2]), channels.append((r["escape"] + r["guided"] + r["absorbed"])[:2])
    total, channels = np.array(total), np.array(channels)
    names = ("perpendicular", "parallel")
    return dict(x=sig(h * 1e3), series=[series(o, f"{o}, total (PyStratify)", total[:, i], "pystratify") for i, o in enumerate(names)]
                + [series(f"{o}_ref", f"{o}, escape + guided + absorbed", channels[:, i], "reference") for i, o in enumerate(names)],
                worst=worst(total, channels, relative=True), worst_label="largest relative difference",
                note=f"A {dipole} dipole in air above a free-standing 100 nm film of n = 0.2 + 3i, 600 nm. The total rate comes from the "
                     "reflected Green's function at the source; the channels separately, from the power escaping to either side, the power "
                     "in guided modes and the power absorbed in the film, so their sum is an independent budget. Quenching rises as h⁻³ at contact.")


for _dipole in ("electric", "magnetic"):
    def _make(dipole=_dipole):
        return _film_coating_budget(dipole)
    _make.__name__ = f"film_coating_budget_{_dipole}"
    figure(f"film-coating-budget-{_dipole}", f"Absorbing film, {_dipole} dipole",
           "height above the film [nm]", "Γ / Γ₀", log=True)(_make)


@figure("film-guided-pole-tolerance", "Thick high-index film", "tolerance asked for", "error", log=True)
def film_guided_pole_tolerance():
    from pystratify.planar_emission import FilmSource

    tolerances = np.geomspace(1e-3, 1e-8, 11)
    reported, budget, poles, converged = [], [], [], []
    for tol in tolerances:
        rates = FilmSource([1, 3.5, 1], [inf, .8, inf], .6, 1, .4).rates(tol)
        reported.append(rates.error), budget.append(tol * max(1, np.max(abs(rates.total)))), poles.append(len(rates.poles))
        converged.append(rates.converged)
    reported, budget, converged = np.array(reported), np.array(budget), np.array(converged)
    failed = ", ".join(f"{t:.0e}" for t in tolerances[~converged])
    return dict(x=[float(t) for t in tolerances], series=[series("error", "error reported, PyStratify", reported, "pystratify"),
                                                          series("asked", "tolerance × max(1, |Γ / Γ₀|), the accuracy asked for", budget, "reference")],
                worst=float(np.max((reported / budget)[converged])), worst_label="largest reported error / accuracy asked where it reports convergence (must stay below 1)",
                note="Eighteen guided poles sit on the real axis of the spectral integral; PyStratify takes their residues and charges "
                     "the uncertainty of each pole's position to the error. Where it reports convergence the error stays within the tolerance, "
                     f"with all {poles[-1]} poles found; where it cannot reach it ({failed or 'none'}) it says so, not converged, rather than "
                     "claiming the accuracy. The test asserts 1e-6 and checks total = escape + guided to the tolerance.")


@figure("film-guide-budget", "Lossless guide", "depth in the film [nm]", "Γ / Γ₀")
def film_guide_budget():
    z = np.linspace(.005, .195, 39)
    total, escape, guided = [], [], []
    for zi in z:
        r = ps.solve_problem(ps.Problem("films", [inf, .2, inf], [1, 1.5, 1], .6, ps.PointDipole(zi, "electric", 1), 1e-5))
        total.append(r["total"][2]), escape.append(r["escape"][2]), guided.append(r["guided"][2])
    total, escape, guided = map(np.array, (total, escape, guided))
    return dict(x=sig(z * 1e3), series=[series("total", "total, PyStratify", total, "pystratify"), series("escape", "escape", escape, "pystratify"),
                                        series("guided", "guided", guided, "pystratify"), series("total_ref", "escape + guided", escape + guided, "reference")],
                worst=worst(total, escape + guided, relative=True), worst_label="largest relative difference",
                note="An isotropic electric dipole inside the guide: a point source (unlike a line source) puts a finite share of its power "
                     f"into the guided modes, here up to {np.max(guided / total):.2f} of the total; the total from the Green's function must equal "
                     "what escapes plus what is guided, since nothing absorbs.")


@figure("film-pattern-free-space", "Free-space pattern", "polar angle θ [°]", "power per solid angle")
def film_pattern_free_space():
    theta = np.linspace(0, np.pi, 91)
    r = ps.solve_problem(ps.Problem("films", [inf, .2, inf], [1, 1, 1], .6, ps.PointDipole(.1, "electric", 1)), ("rates", "pattern"),
                         theta=theta[:, None], phi=np.array([[0.0]]))
    perp, par = r["pattern"][:, 0, 0], r["pattern"][:, 0, 1]
    exact_perp, exact_par = 3 / (8 * np.pi) * np.sin(theta) ** 2, 3 / (8 * np.pi) * np.cos(theta) ** 2
    deg = np.degrees(theta)
    return dict(x=sig(deg), series=[series("perp", "perpendicular, PyStratify", perp, "pystratify"), series("par", "parallel, in its plane (φ = 0), PyStratify", par, "pystratify"),
                                    series("perp_ref", "(3/8π) sin²θ", exact_perp, "reference"), series("par_ref", "(3/8π) cos²θ", exact_par, "reference")],
                worst=float(max(np.max(np.abs(perp - exact_perp)), np.max(np.abs(par - exact_par)))), worst_label="largest |difference|",
                note="Normalized to the free dipole's total power, the pattern of a dipole in an empty stack is the textbook one. The test "
                     "integrates PyStratify's pattern (Gauss–Legendre × 64 azimuths) for films, cylinders and spheres and compares it with "
                     "the escape rate (1e-7), and checks the grazing limit of the perpendicular dipole.")


@figure("image-dipole-mirror", "Perfect mirror", "kd = 4πh/λ", "Γ / Γ₀")
def image_dipole_mirror():
    import emitter_references as er

    x = np.linspace(0.3, 20, 80)
    heights = x * er.WAVELENGTH / (4 * np.pi)
    electric = np.array([er.reference_rates([1.0, 1.0], [inf, inf], h, mirror=True) for h in heights])
    magnetic = np.array([er.reference_rates([1.0, 1.0], [inf, inf], h, dipole="magnetic", mirror=True) for h in heights])
    exact = np.array([er.image_dipole(v) for v in x])
    out = [series("e_perp", "electric ⊥, Sommerfeld integrals", electric[:, 0], "pystratify"), series("e_par", "electric ∥, Sommerfeld integrals", electric[:, 1], "pystratify"),
           series("m_par", "magnetic ∥, Sommerfeld integrals", magnetic[:, 1], "pystratify"),
           series("e_perp_ref", "electric ⊥, image dipole", exact[:, 0], "reference"), series("e_par_ref", "electric ∥, image dipole", exact[:, 1], "reference"),
           series("m_par_ref", "magnetic ∥, 2 − image dipole", 2 - exact[:, 1], "reference")]
    return dict(x=sig(x), series=out, worst=float(max(np.max(np.abs(electric - exact)), np.max(np.abs(magnetic - (2 - exact))))), worst_label="largest |difference|",
                note="This row checks the reference, not PyStratify: the Sommerfeld integrals written without PyStratify (emitter_references.py), "
                     "given a perfect mirror (r_s = −1, r_p = 1), must give the image dipole's closed form, Drexhage's oscillations; a magnetic "
                     "dipole sees the mirror with the opposite sign. The same integrals are then PyStratify's reference for real stacks "
                     "(the Sommerfeld row, 1e-10).")


@figure("weak-guide-sommerfeld", "Weakly absorbing guide", "height above the film [nm]", "Γ / Γ₀")
def weak_guide_sommerfeld():
    import emitter_references as er
    from pystratify.planar_emission import FilmSource

    n, d = er.GUIDES["air above 200 nm TiO2 on glass"]
    poles = np.array(FilmSource(n, d, er.WAVELENGTH, 1, d[1] / 2).guided_poles()) * n[1]
    absorbing = er.guide_with_loss(n, 1e-3)
    h = np.geomspace(10, 400, 30)
    ours = np.array([er.stratify_rates(absorbing, d, hi) for hi in h])
    ref = np.array([er.reference_rates(absorbing, d, hi, poles=poles) for hi in h])
    return dict(x=sig(h), series=[series("perp", "perpendicular, PyStratify", ours[:, 0], "pystratify"), series("par", "parallel, PyStratify", ours[:, 1], "pystratify"),
                                  series("perp_ref", "perpendicular, Sommerfeld integrals", ref[:, 0], "reference"), series("par_ref", "parallel, Sommerfeld integrals", ref[:, 1], "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="A guide with a little loss turns its guided poles into sharp quasi-poles just off the real axis; the reference integrates "
                     "through them with their positions as breakpoints. The test asserts both guides at 50 nm to 1e-8.")


@figure("lossless-guide-limit", "Lossless guide limit", "height above the film [nm]", "Γ / Γ₀")
def lossless_guide_limit():
    import emitter_references as er

    n, d = er.GUIDES["air above 200 nm TiO2 on glass"]
    h = np.geomspace(10, 400, 30)
    lossless = np.array([er.stratify_rates(n, d, hi, tolerance=1e-8) for hi in h])
    limit = np.array([er.stratify_rates(er.guide_with_loss(n, 1e-7), d, hi, tolerance=1e-8) for hi in h])
    return dict(x=sig(h), series=[series("perp", "perpendicular, lossless guide", lossless[:, 0], "pystratify"), series("par", "parallel, lossless guide", lossless[:, 1], "pystratify"),
                                  series("perp_ref", "perpendicular, core loss k = 1e-7", limit[:, 0], "reference"), series("par_ref", "parallel, core loss k = 1e-7", limit[:, 1], "reference")],
                worst=worst(lossless, limit, relative=True), worst_label="largest relative difference",
                note="From outside a lossless guide the power coupled into its modes must still be counted: with real poles PyStratify takes "
                     "their residues; with a trace of loss the same power is absorbed along the guide instead. The two must agree (1e-5 in the "
                     "test, both guides, electric and magnetic, from above and below). Before 0.10.3 the guided power was dropped here.")


@figure("slab-dispersion-modes", "Slab modes", "thickness [nm]", "n_eff")
def slab_dispersion_modes():
    t, ours, theirs, diff, width = _guide_modes("analytic")
    return dict(x=sig(t), series=[series(f"m{i}", f"mode {i + 1}, PyStratify", ours[:, i], "pystratify") for i in range(width)]
                + [series(f"m{i}_ref", f"mode {i + 1}, dispersion relation", theirs[:, i], "reference") for i in range(width)],
                worst=diff, worst_label="largest |n_eff − n_eff(relation)|",
                note="κd = mπ + atan(ρ₁γ₁/κ) + atan(ρ₃γ₃/κ), ρ = 1 for TE and (n_core/n_clad)² for TM, solved by bisection (planar_codes.py "
                     "slab_modes).")


if __name__ == "__main__":
    run(__file__)
