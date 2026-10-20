"""Thin films, plane wave: PyStratify against pyelli.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_plane_wave_pyelli.py                                                              #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-plane-wave-pyelli.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
films_plane_wave_pyelli.json (the curves the page draws, value for value) and films_plane_wave_pyelli.png (needs matplotlib).
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


@figure("pyelli-bragg-mirror", "Bragg stack", "λ [nm]", "R_s")
def pyelli_bragg_mirror():
    from planar_codes import CASES, elli_result, stratify

    case = CASES["Bragg mirror, 10 pairs"]
    lam = np.linspace(450, 850, 81)
    theta = np.radians(30)
    ours = np.array([stratify(case, w, theta, "s")["R"] for w in lam])
    two = np.array([elli_result(case, w, theta, "two")["Rs"] for w in lam])
    four = np.array([elli_result(case, w, theta, "four")["Rs"] for w in lam])
    four[1::2] = np.nan
    two[::2] = np.nan
    return dict(x=sig(lam), series=[series("R", "R_s, PyStratify", ours, "pystratify"), series("two", "R_s, pyElli 2×2", two, "reference"),
                                    series("four", "R_s, pyElli 4×4", four, "reference")],
                worst=float(np.nanmax(np.abs(np.r_[ours - two, ours - four]))), worst_label="largest |difference|",
                note="pyElli (an ellipsometry package) solves layered stacks with Berreman's 4×4 matrices and with 2×2 ones; the two solvers "
                     "are drawn at alternate points. The test runs nine stacks (an AR pair, this mirror, thin and opaque gold, a 20 µm absorber, "
                     "beyond critical, frustrated TIR) at four angles and three wavelengths for R, Ψ and Δ.")


@figure("pyelli-gold-ellipsometry", "Gold film", "angle of incidence [°]", "angle [°]")
def pyelli_gold_ellipsometry():
    from planar_codes import CASES, _split, _wrap, elli_result

    layers, d, _ = case = CASES["Au 20 nm on glass"]
    n, _ = _split(layers)
    angles = np.linspace(1, 85, 85)
    psi, delta, psi_ref, delta_ref = [], [], [], []
    for a in angles:
        e = planar.ellips(n, d, np.radians(a), 633.0)
        theirs = elli_result(case, 633.0, np.radians(a), "four")
        psi.append(np.degrees(e["psi"])), delta.append(_wrap(180.0 - np.degrees(e["Delta"])))
        psi_ref.append(theirs["psi"]), delta_ref.append(_wrap(theirs["delta"]))
    psi, delta, psi_ref, delta_ref = map(np.array, (psi, delta, psi_ref, delta_ref))
    return dict(x=sig(angles), series=[series("psi", "Ψ, PyStratify", psi, "pystratify"), series("delta", "Δ, PyStratify (pyElli's convention)", delta, "pystratify"),
                                       series("psi_ref", "Ψ, pyElli 4×4", psi_ref, "reference"), series("delta_ref", "Δ, pyElli 4×4", delta_ref, "reference")],
                worst=float(max(np.max(np.abs(psi - psi_ref)), np.max(np.abs(_wrap(delta - delta_ref))))), worst_label="largest difference [°]",
                note="pyElli reports Δ with the opposite sign convention for r_p, so PyStratify's is drawn as 180° − Δ, wrapped to ±180°, as "
                     "the test compares it (to 1e-10°).")


def _continuous_degrees(values, like=None):
    """An angle curve in degrees drawn without its ±180° or 360° jumps (np.unwrap over its finite points), shifted by
    whole turns to start on the curve it is compared with: the same values modulo 360°, which is how the tests compare
    them. Two codes may report Δ on different branches; drawn raw, agreeing curves would sit 360° apart."""
    v = np.asarray(values, float).copy()
    ok = np.isfinite(v)
    v[ok] = np.degrees(np.unwrap(np.radians(v[ok])))
    if like is not None:
        like = np.asarray(like, float)
        first = np.flatnonzero(ok & np.isfinite(like))[0]
        v[ok] -= 360.0 * np.round((v[first] - like[first]) / 360.0)
    return v


def _pyelli_dispersion_figure(name):
    from dispersion_codes import WAVELENGTHS_NM, cases

    model, theirs = cases()[name]
    ours, reference = model.epsilon(WAVELENGTHS_NM), np.asarray(theirs(), complex)
    return WAVELENGTHS_NM, ours, reference, float(np.max(np.abs(ours - reference)) / np.max(np.abs(reference)))


@figure("pyelli-tauc-lorentz", "Tauc–Lorentz model", "λ [nm]", "ε")
def pyelli_tauc_lorentz():
    lam, ours, ref, rel = _pyelli_dispersion_figure("Tauc-Lorentz")
    return dict(x=sig(lam), series=[series("e1", "ε₁, PyStratify", ours.real, "pystratify"), series("e2", "ε₂, PyStratify", ours.imag, "pystratify"),
                                    series("e1_ref", "ε₁, pyElli", ref.real, "reference"), series("e2_ref", "ε₂, pyElli", ref.imag, "reference")],
                worst=rel, worst_label="largest |Δε| / max |ε|",
                note="pystratify.dispersion and pyElli's dispersion models were written apart, so agreement checks the formulas and the "
                     "parameter conventions together (mapped in benchmarks/dispersion_codes.py). The test runs twelve models: Tauc-Lorentz, "
                     "Lorentz, Gaussian, poles, both Drude forms, Sellmeier, Cauchy, two Bruggeman mixtures, Maxwell Garnett and Cody-Lorentz.")


@figure("pyelli-drude", "Drude model", "λ [nm]", "ε")
def pyelli_drude():
    lam, ours, ref, rel = _pyelli_dispersion_figure("Drude (energy)")
    return dict(x=sig(lam), series=[series("e1", "ε₁, PyStratify", ours.real, "pystratify"), series("e2", "ε₂, PyStratify", ours.imag, "pystratify"),
                                    series("e1_ref", "ε₁, pyElli (sign corrected)", ref.real, "reference"), series("e2_ref", "ε₂, pyElli", ref.imag, "reference")],
                worst=rel, worst_label="largest |Δε| / max |ε|",
                note="pyElli's DrudeEnergy returns A / (E² − iγE), whose real part is positive at every energy, so a free-electron metal never "
                     "turns metallic; the free-electron ε = −E_p² / (E² + iΓE), which PyStratify returns, is negative below the plasma energy. "
                     "pyElli's curve is drawn as −conj(ε), as the test compares it; its ε₂ needs no correction.")


@figure("pyelli-dispersive-stack", "Rough Tauc–Lorentz film", "λ [nm]", "angle [°]")
def pyelli_dispersive_stack():
    from ellipsometry_codes import STACKS, WAVELENGTHS_NM, _elli, _elli_material, _wrap
    from pystratify import ellipsometry as el

    elli = _elli()
    stack = STACKS["Tauc-Lorentz film, 25% void roughness, on Si"]
    layers, substrate = stack[:-1], stack[-1][0]
    sample = el.Sample([el.Layer(model, thickness) for model, thickness in layers], substrate=substrate)
    structure = elli.Structure(elli.AIR, [elli.Layer(_elli_material(model), thickness) for model, thickness in layers], _elli_material(substrate))
    ours = el.simulate(sample, WAVELENGTHS_NM, 70.0)
    two = structure.evaluate(WAVELENGTHS_NM, 70.0, solver=elli.Solver2x2)
    four = structure.evaluate(WAVELENGTHS_NM, 70.0, solver=elli.Solver4x4)
    psi_ref, delta_ref = np.array(two.psi, float), np.array(two.delta, float)
    psi_ref[1::2], delta_ref[1::2] = np.asarray(four.psi)[1::2], np.asarray(four.delta)[1::2]
    worst_value = max(np.max(np.abs(ours.psi - psi_ref)), np.max(_wrap(ours.delta - delta_ref)))
    delta_drawn = _continuous_degrees(ours.delta)
    return dict(x=sig(WAVELENGTHS_NM), series=[series("psi", "Ψ, PyStratify", ours.psi, "pystratify"), series("delta", "Δ, PyStratify", delta_drawn, "pystratify"),
                                               series("psi_ref", "Ψ, pyElli (2×2 and 4×4 alternately)", psi_ref, "reference"),
                                               series("delta_ref", "Δ, pyElli (2×2 and 4×4 alternately)", _continuous_degrees(delta_ref, delta_drawn), "reference")],
                worst=float(worst_value), worst_label="largest difference [°]",
                note="Both codes get the same ε on the same grid (pyElli's TableEpsilon), so only the stack optics are compared; pyElli reports "
                     "Ψ, Δ in the same instrument convention as pystratify.ellipsometry. The two pyElli solvers are drawn at alternate "
                     "wavelengths. The test runs four stacks (this one, Cauchy on thermal oxide, a semi-transparent Drude metal, Cody-Lorentz "
                     "with Maxwell Garnett inclusions) at 45, 60, 70 and 75°. Δ is drawn continuous, modulo 360°.")


if __name__ == "__main__":
    run(__file__)
