"""Thin films, plane wave: PyStratify against pyElli.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_plane_wave_pyelli.py                                                              #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-plane-wave-pyelli.html, computed by this one file from a
PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, films_plane_wave_pyelli.json (the curves the page draws, value for value) and
films_plane_wave_pyelli.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run, spread

from pystratify import planar


@figure("pyelli-bragg-mirror", "Bragg stack", "λ [nm]", "R_s")
def plot_bragg_stack():
    from planar_codes import CASES, elli_result, stratify

    case = CASES["Bragg mirror, 10 pairs"]
    lam = np.linspace(450, 850, 81)
    theta = np.radians(30)
    every = 4  # each pyElli solver every 20 nm, the two alternating every 10 nm
    ours = np.array([stratify(case, w, theta, "s")["R"] for w in lam])
    four = np.array([elli_result(case, w, theta, "four")["Rs"] for w in lam[::every]])
    two = np.array([elli_result(case, w, theta, "two")["Rs"] for w in lam[every // 2::every]])
    return result(
        lam,
        [line("R", "R_s, PyStratify", ours),
         dots("two", "R_s, pyElli 2×2", two, every, every // 2, size=lam.size),
         dots("four", "R_s, pyElli 4×4", four, every, size=lam.size)],
        worst=max(largest(ours[every // 2::every], two), largest(ours[::every], four)),
        worst_label="largest |difference|",
        note="pyElli (an ellipsometry package) solves layered stacks with Berreman's 4×4 matrices and with 2×2 "
             "ones; the two solvers are drawn at alternate points. The test runs nine stacks (an AR pair, this "
             "mirror, thin and opaque gold, a 20 µm absorber, beyond critical, frustrated TIR) at four angles and "
             "three wavelengths for R, Ψ and Δ.")


@figure("pyelli-gold-ellipsometry", "Gold film", "angle of incidence [°]", "angle [°]")
def plot_gold_film():
    from planar_codes import CASES, _split, _wrap, elli_result

    layers, d, _ = case = CASES["Au 20 nm on glass"]
    n, _ = _split(layers)
    angles = np.linspace(1, 85, 85)
    every = 3  # pyElli every 3°
    ours = [planar.ellips(n, d, np.radians(a), 633.0) for a in angles]
    psi = np.array([np.degrees(e["psi"]) for e in ours])
    delta = np.array([_wrap(180.0 - np.degrees(e["Delta"])) for e in ours])
    theirs = [elli_result(case, 633.0, np.radians(a), "four") for a in angles[::every]]
    psi_ref = np.array([t["psi"] for t in theirs])
    delta_ref = np.array([_wrap(t["delta"]) for t in theirs])
    return result(
        angles,
        [line("psi", "Ψ, PyStratify", psi), line("delta", "Δ, PyStratify (pyElli's convention)", delta),
         dots("psi_ref", "Ψ, pyElli 4×4", psi_ref, every, size=angles.size),
         dots("delta_ref", "Δ, pyElli 4×4", delta_ref, every, size=angles.size)],
        worst=max(largest(psi[::every], psi_ref), float(np.max(np.abs(_wrap(delta[::every] - delta_ref))))),
        worst_label="largest difference [°]",
        note="pyElli reports Δ with the opposite sign convention for r_p, so PyStratify's is drawn as 180° − Δ, "
             "wrapped to ±180°, as the test compares it (to 1e-10°).")


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


def _pyelli_dispersion(name, ref_label):
    """A dispersion model's ε₁, ε₂ from PyStratify every 2.5 nm and from pyElli every 50 nm, 200 to 1700 nm."""
    from dispersion_codes import cases

    every = 20
    lam = np.linspace(200, 1700, 601)
    model, theirs = cases(lam[::every])[name]
    ours, ref = model.epsilon(lam), np.asarray(theirs(), complex)
    return lam, [line("e1", "ε₁, PyStratify", ours.real), line("e2", "ε₂, PyStratify", ours.imag),
                 dots("e1_ref", f"ε₁, {ref_label}", ref.real, every, size=lam.size),
                 dots("e2_ref", "ε₂, pyElli", ref.imag, every, size=lam.size)], \
        float(np.max(np.abs(ours[::every] - ref)) / np.max(np.abs(ref)))


@figure("pyelli-tauc-lorentz", "Tauc–Lorentz model", "λ [nm]", "ε")
def plot_tauc_lorentz_model():
    lam, curves, rel = _pyelli_dispersion("Tauc-Lorentz", "pyElli")
    return result(
        lam, curves, worst=rel, worst_label="largest |Δε| / max |ε|",
        note="pystratify.dispersion and pyElli's dispersion models were written apart, so agreement checks the "
             "formulas and the parameter conventions together (mapped in benchmarks/dispersion_codes.py). The test "
             "runs twelve models: Tauc-Lorentz, Lorentz, Gaussian, poles, both Drude forms, Sellmeier, Cauchy, two "
             "Bruggeman mixtures, Maxwell Garnett and Cody-Lorentz.")


@figure("pyelli-drude", "Drude model", "λ [nm]", "ε")
def plot_drude_model():
    lam, curves, rel = _pyelli_dispersion("Drude (energy)", "pyElli (sign corrected)")
    return result(
        lam, curves, worst=rel, worst_label="largest |Δε| / max |ε|",
        note="pyElli's DrudeEnergy returns A / (E² − iγE), whose real part is positive at every energy, so a "
             "free-electron metal never turns metallic; the free-electron ε = −E_p² / (E² + iΓE), which PyStratify "
             "returns, is negative below the plasma energy. pyElli's curve is drawn as −conj(ε), as the test "
             "compares it; its ε₂ needs no correction.")


@figure("pyelli-dispersive-stack", "Rough Tauc–Lorentz film", "λ [nm]", "angle [°]")
def plot_rough_tauc_lorentz_film():
    from ellipsometry_codes import STACKS, WAVELENGTHS_NM, _elli, _elli_material, _wrap

    from pystratify import ellipsometry as el

    elli = _elli()
    stack = STACKS["Tauc-Lorentz film, 25% void roughness, on Si"]
    layers, substrate = stack[:-1], stack[-1][0]
    sample = el.Sample([el.Layer(model, thickness) for model, thickness in layers], substrate=substrate)
    structure = elli.Structure(elli.AIR, [elli.Layer(_elli_material(model), thickness) for model, thickness in layers],
                               _elli_material(substrate))
    lam, every = WAVELENGTHS_NM, 3  # pyElli every 30 nm, its 2×2 and 4×4 solvers alternating
    ours = el.simulate(sample, lam, 70.0)
    two = structure.evaluate(lam[::2 * every], 70.0, solver=elli.Solver2x2)
    four = structure.evaluate(lam[every::2 * every], 70.0, solver=elli.Solver4x4)
    psi_ref, delta_ref = np.empty(lam[::every].size), np.empty(lam[::every].size)
    psi_ref[::2], delta_ref[::2] = two.psi, two.delta
    psi_ref[1::2], delta_ref[1::2] = four.psi, four.delta
    delta_drawn = _continuous_degrees(ours.delta)
    delta_ref_drawn = _continuous_degrees(spread(delta_ref, every, size=lam.size), delta_drawn)
    return result(
        lam,
        [line("psi", "Ψ, PyStratify", ours.psi), line("delta", "Δ, PyStratify", delta_drawn),
         dots("psi_ref", "Ψ, pyElli (2×2 and 4×4 alternately)", psi_ref, every, size=lam.size),
         dots("delta_ref", "Δ, pyElli (2×2 and 4×4 alternately)", delta_ref_drawn)],
        worst=max(largest(ours.psi[::every], psi_ref), float(np.max(_wrap(ours.delta[::every] - delta_ref)))),
        worst_label="largest difference [°]",
        note="Both codes get the same ε on the same grid (pyElli's TableEpsilon), so only the stack optics are "
             "compared; pyElli reports Ψ, Δ in the same instrument convention as pystratify.ellipsometry. The two "
             "pyElli solvers are drawn at alternate wavelengths. The test runs four stacks (this one, Cauchy on "
             "thermal oxide, a semi-transparent Drude metal, Cody-Lorentz with Maxwell Garnett inclusions) at 45, "
             "60, 70 and 75°. Δ is drawn continuous, modulo 360°.")


if __name__ == "__main__":
    run(__file__)
