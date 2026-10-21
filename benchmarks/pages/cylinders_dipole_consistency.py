"""Cylinders, point dipole: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/cylinders_dipole_consistency.py                                                         #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/cylinders-dipole-consistency.html, computed by this one file
from a PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE
and writes, beside itself, cylinders_dipole_consistency.json (the curves the page draws, value for value) and
cylinders_dipole_consistency.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

import pystratify as ps

ORIENTATIONS = ("radial", "azimuthal", "axial")


@figure("fibre-dispersion", "Fibre modes", "λ [µm]", "n_eff")
def plot_fibre_modes():
    from scipy.optimize import brentq
    from scipy.special import j0, j1, k0, k1

    from pystratify.cylinder_emission import CylinderSource

    radius, n1, n2 = .15, 2.0, 1.0
    lams = np.linspace(0.25, 0.38, 27)  # above 0.392 µm (V = 2.405) TE01 and TM01 are cut off and only HE11 guides
    te, tm, he, te_ref, tm_ref = [], [], [], [], []
    for lam in lams:
        V = 2 * np.pi / lam * radius

        def relation(b, c):
            u, w = V * np.sqrt(n1 ** 2 - b ** 2), V * np.sqrt(b ** 2 - n2 ** 2)
            return j1(u) / (u * j0(u)) + (c * (n2 / n1) ** 2 if c else 1) * k1(w) / (w * k0(w))

        upper = np.sqrt(n1 ** 2 - (2.404825557695773 / V) ** 2)
        exact = sorted(brentq(relation, n2 + 1e-12, upper - 1e-12, args=(c,), xtol=1e-15) for c in (0, 1))
        poles = np.sort(CylinderSource([radius], [n1, n2], lam, .25).guided_poles())
        # the poles nearest each relation's root are TE01 and TM01; HE11, with no cutoff, is the largest
        te.append(poles[np.argmin(np.abs(poles - exact[0]))]), tm.append(poles[np.argmin(np.abs(poles - exact[1]))])
        he.append(poles[-1]), te_ref.append(exact[0]), tm_ref.append(exact[1])
    return result(
        lams,
        [line("te", "TE₀₁, PyStratify", te), line("tm", "TM₀₁, PyStratify", tm), line("he", "HE₁₁, PyStratify", he),
         dots("te_ref", "TE₀₁, dispersion relation", te_ref), dots("tm_ref", "TM₀₁, dispersion relation", tm_ref)],
        worst=max(largest(te, te_ref), largest(tm, tm_ref)),
        worst_label="largest |n_eff − n_eff(relation)| for TE₀₁ and TM₀₁",
        note="Snyder & Love's TE₀₁ and TM₀₁ relations, J₁(u)/(u J₀(u)) + c K₁(w)/(w K₀(w)) = 0, solved by "
             "bisection, against the poles PyStratify finds behind a dipole's guided channel; HE₁₁ (no closed form "
             "of this kind, and no cutoff) is shown as found; the shorter wavelengths guide higher modes too, not "
             "drawn. TE₀₁ and TM₀₁ reach cutoff at 0.392 µm, where V = 2.405.")


def _metal_wire_figure(dipole):
    """The grazing error PyStratify charges near a metal wire against what moving the guard inward finds."""
    from unified_checks import metal_wire_rates

    lam = np.linspace(.4, .7, 16)
    bound, diff = [], []
    for w in lam:
        rates, guard = metal_wire_rates(dipole, w)
        bound.append(rates.grazing_error)
        diff.append(max(float(np.max(np.abs(getattr(rates, key) - guard[key])))
                        for key in ("total", "escape", "absorbed")))
    bound, diff = np.array(bound), np.array(diff)
    return result(
        lam * 1e3,
        [line("bound", "grazing error charged, PyStratify", bound),
         dots("diff", "|PyStratify − guard moved to 3e-6|", diff)],
        worst=np.max(diff / bound),
        worst_label="largest difference / charged error (must stay below 1)",
        note=f"A {dipole} dipole 5 nm from a 10 nm wire of n = 0.97 + 1.87i (no dispersion): near the light line "
             "the integrand is capped, and PyStratify charges the cap to its error. Integrating again with the "
             "precision guard moved inward to 3e-6 measures what the caps leave out; the largest difference over "
             "the total, escape and absorbed rates and the three orientations stays below the charge.")


@figure("cylinder-grazing-caps-electric", "Grazing caps, electric dipole", "λ [nm]", "error in Γ / Γ₀", log=True)
def plot_grazing_caps_electric_dipole():
    return _metal_wire_figure("electric")


@figure("cylinder-grazing-caps-magnetic", "Grazing caps, magnetic dipole", "λ [nm]", "error in Γ / Γ₀", log=True)
def plot_grazing_caps_magnetic_dipole():
    return _metal_wire_figure("magnetic")


@figure("cylinder-dipole-homogeneous", "Index-matched cylinder", "distance from the axis [nm]", "|Γ / Γ₀ − 1|",
        log=True)
def plot_index_matched_cylinder():
    r = np.r_[np.linspace(0, .095, 12), np.linspace(.105, .3, 20)]
    out = {}
    for dipole in ("electric", "magnetic"):
        for ri in r:
            source = ps.PointDipole(ri, dipole, int(ri > .1), .4)
            res = ps.solve_problem(ps.Problem("cylinders", [.1], [1.5, 1.5], .6, source, 1e-6, 15))
            out.setdefault(dipole, []).append(max(float(np.max(np.abs(res["total"] - 1))), 1e-17))
    return result(
        r * 1e3,
        [line(d, f"{d} dipole, max over orientations, PyStratify", out[d]) for d in out],
        worst=max(max(v) for v in out.values()),
        worst_label="largest |Γ / Γ₀ − 1|",
        note="A cylinder of the medium's own index changes nothing: every rate is the free-space one, 1, for any "
             "position and orientation, inside the cylinder (r < 100 nm) or out; the test also checks escape = 1, "
             "no guided or absorbed power and the quantum yield 0.4 passed through. Floored at 1e-17 for the "
             "logarithmic axis.")


def _coating_budget(dipole):
    """The total rate outside an absorbing coating against the sum of its channels."""
    r = np.linspace(.105, .4, 24)
    total, channels = [], []
    for ri in r:
        source = ps.PointDipole(ri, dipole, 2)
        res = ps.solve_problem(ps.Problem("cylinders", [.05, .1], [1.5, .2 + 3j, 1], .6, source, 1e-5, 30))
        total.append(res["total"][:3]), channels.append((res["escape"] + res["guided"] + res["absorbed"])[:3])
    total, channels = np.array(total), np.array(channels)
    return result(
        r * 1e3,
        [line(o, f"{o}, total (PyStratify)", total[:, i]) for i, o in enumerate(ORIENTATIONS)]
        + [dots(f"{o}_ref", f"{o}, escape + guided + absorbed", channels[:, i]) for i, o in enumerate(ORIENTATIONS)],
        worst=largest(total, channels, relative=True),
        worst_label="largest relative difference",
        note=f"A {dipole} dipole outside a cylinder of n = 1.5 (50 nm) under 50 nm of an absorbing n = 0.2 + 3i "
             "coating, 600 nm. The total rate comes from the Green's function at the source; the channels "
             "separately, from the power escaping to infinity, the power carried by guided modes and the power "
             "absorbed in the coating, so their sum is an independent budget.")


@figure("cylinder-coating-budget-electric", "Absorbing coating, electric dipole", "distance from the axis [nm]",
        "Γ / Γ₀", log=True)
def plot_absorbing_coating_electric_dipole():
    return _coating_budget("electric")


@figure("cylinder-coating-budget-magnetic", "Absorbing coating, magnetic dipole", "distance from the axis [nm]",
        "Γ / Γ₀", log=True)
def plot_absorbing_coating_magnetic_dipole():
    return _coating_budget("magnetic")


@figure("cylinder-guide-budget", "Lossless guide", "distance from the axis [nm]", "Γ / Γ₀")
def plot_lossless_guide():
    r = np.linspace(0, .095, 20)
    total, escape, guided = [], [], []
    for ri in r:
        source = ps.PointDipole(ri, "electric", 0)
        res = ps.solve_problem(ps.Problem("cylinders", [.1], [1.5, 1], .6, source, 1e-5, 20))
        total.append(res["total"][3]), escape.append(res["escape"][3]), guided.append(res["guided"][3])
    total, escape, guided = map(np.array, (total, escape, guided))
    return result(
        r * 1e3,
        [line("total", "total, PyStratify", total), line("escape", "escape", escape), line("guided", "guided", guided),
         dots("total_ref", "escape + guided", escape + guided)],
        worst=largest(total, escape + guided, relative=True),
        worst_label="largest relative difference",
        note="An electric dipole inside the guide, averaged over orientations: a point source (unlike a line "
             "source) sends a finite share of its power into the guided mode, here about a tenth of the free rate; "
             "the total from the Green's function must equal what escapes plus what is guided, since nothing "
             "absorbs.")


@figure("cylinder-grazing-tolerance", "Metal wire at grazing", "λ [nm]", "error in Γ / Γ₀", log=True)
def plot_metal_wire_at_grazing():
    from unified_checks import metal_wire_tight

    lam = np.linspace(.4, .7, 16)
    runs = [metal_wire_tight(w) for w in lam]
    grazing = np.array([rates.grazing_error for rates in runs])
    asked = np.array([1e-9 * np.max(rates.total) for rates in runs])
    unconverged = sum(not rates.converged for rates in runs)
    return result(
        lam * 1e3,
        [line("grazing", "grazing error charged, PyStratify", grazing),
         line("error", "total error reported, PyStratify", [rates.error for rates in runs]),
         dots("asked", "1e-9 × max Γ / Γ₀, the accuracy asked for", asked)],
        worst=np.min(grazing / asked),
        worst_label="smallest grazing error / accuracy asked (must stay above 1)",
        note="The light-line caps leave an error the integration cannot shrink; asked for 1e-9, PyStratify charges "
             f"it to the total error and reports the result as not converged ({unconverged} of {len(runs)} "
             "wavelengths), rather than claiming an accuracy it does not have. The grazing error is all but 0.3% of "
             "the reported error, so the two lines coincide. The same wire as the grazing-caps rows; 40 orders; no "
             "dispersion.")


@figure("fibre-rates-cutoff", "Fibre cutoff", "λ [µm]", "Γ / Γ₀")
def plot_fibre_cutoff():
    from unified_checks import fibre_rates

    lam = np.linspace(.62, .74, 49)
    every = 2  # the budget every 5 nm
    runs = [fibre_rates(w) for w in lam]
    total, escape, guided = (np.array([res[key][:3] for res in runs]) for key in ("total", "escape", "guided"))
    budget = (escape + guided)[::every]
    return result(
        lam,
        [line(o, f"{o}, total (PyStratify)", total[:, i]) for i, o in enumerate(ORIENTATIONS)]
        + [line(f"{o}_guided", f"{o}, guided (PyStratify)", guided[:, i]) for i, o in enumerate(ORIENTATIONS)]
        + [dots(f"{o}_ref", f"{o}, escape + guided", budget[:, i], every, size=lam.size)
           for i, o in enumerate(ORIENTATIONS)],
        worst=largest(total[::every], budget, relative=True),
        worst_label="largest relative difference, total against escape + guided",
        note="TE₀₁ and TM₀₁ are cut off near 0.678 µm: the guided power of the radial and azimuthal dipoles, which "
             "they carried, drops there and reappears as escaping power (HE₁₁ stays guided), while the total, from "
             "the Green's function, passes through smoothly (the test bounds its second difference over 0.66–0.70 "
             "µm by 2e-3). Every mode PyStratify finds is checked against the dispersion relations in the "
             "fibre-dispersion plot.")


if __name__ == "__main__":
    run(__file__)
