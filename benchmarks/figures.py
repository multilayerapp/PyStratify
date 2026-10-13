"""Comparison figures for multilayer.app's Verification row pages (benchmarks/MATRIX.md, docs/verification.md).

    pip install -e ".[docs,benchmarks]"
    python benchmarks/figures.py            # writes docs/figures/<name>.json for every figure
    python benchmarks/figures.py NAME ...   # some of them
    python benchmarks/figures.py --env ENV  # the figures of one environment (bench, smuthi, meep, ngsolve, tmmax)
    python benchmarks/figures.py --list     # every figure and its environment

    python benchmarks/verify_all.py         # the whole study, every figure in its own environment (envs/README.md)

Each figure draws PyStratify against the reference of one matrix row, as curves on a common axis, so the row page
shows the comparison rather than a number: the closed form of a single interface against the planar engine, Mie's
series against the multilayered-sphere recursion, the 60-digit transfer matrices against double precision, treams
against PyStratify for spheres, cylinders and emitters, the Sommerfeld integrals against the film emitter, the fibre's
dispersion relations against the guided poles. The references are the ones the benchmark scripts beside this file use
(`planar_checks.py`, `sphere_checks.py`, `unified_checks.py`, `mie_codes.py`, `emitter_references.py`,
`mp_reference.py`, `pystratify.references`, and for the sphere rows also `gdm_spheres.py`, `mnpbem_spheres.py`,
`fdtd_meep.py`, `smuthi_spheres.py` and the papers' crosscheck classics); nothing here is asserted, the tests do that.
A figure needing a package that is not installed (treams, scattnlay, pyGDM2, smuthi, Meep) is skipped with a note;
the MNPBEM figures need GNU Octave (`mnpbem_octave.py`), and run one environment at a time: smuthi wants Python <= 3.10
with NumPy < 2, Meep its conda build, pyGDM2 NumPy < 2.4.

`docs/verification.md` names which row each figure belongs to; `python -m pystratify.docs build` carries the JSON into
`docs/manifest.json`, and the site draws it with the same chart core as its Learn examples (reference curves dashed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy import inf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import pystratify as ps  # noqa: E402
from pystratify import planar  # noqa: E402

OUT = ROOT / "docs" / "figures"
FIGURES = {}
# The environment each figure runs in (benchmarks/envs/README.md): "bench" holds PyStratify with treams, scattnlay,
# pyGDM2, PyMoosh, pyElli and meent (and drives GNU Octave for MNPBEM); smuthi, Meep, NGSolve and tmmax each need
# their own. benchmarks/verify_all.py runs every figure in its environment.
ENVS = {}


def figure(name, title, x_label, y_label, log=False, env="bench"):
    def register(fn):
        FIGURES[name] = (fn, dict(id=name, title=title, x_label=x_label, y_label=y_label, log=log))
        ENVS[name] = env
        return fn
    return register


def series(key, label, y, role):
    """A curve; NaN (a reference computed at only some of the points) is written as null."""
    return {"key": key, "label": label, "y": [float(v) if np.isfinite(v) else None for v in np.asarray(y, float)], "role": role}


def worst(ours, theirs, relative=False):
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    diff = np.max(np.abs(ours - theirs))
    return float(diff / np.max(np.abs(theirs))) if relative else float(diff)


def sig(values, digits=7):
    return [None if v is None else float(f"{v:.{digits}g}") for v in values]


# --- films, plane wave -------------------------------------------------------------------------------------------

@figure("fresnel-oblique", "A single interface, air to glass (n = 1.5) at 600 nm: the Fresnel equations", "angle of incidence [°]", "R")
def fresnel_oblique():
    from planar_checks import _forward_cos, _fresnel_amplitudes

    angles = np.linspace(0, 89.5, 180)
    n1, n2 = 1.0, 1.5
    ours, exact = {"s": [], "p": []}, {"s": [], "p": []}
    for a in angles:
        th = np.radians(a)
        kx = n1 * np.sin(th)
        cos_i, cos_f = _forward_cos(n1, kx), _forward_cos(n2, kx)
        for pol in ("s", "p"):
            ours[pol].append(planar.coh_tmm(pol, [n1, n2], [inf, inf], th, 0.6)["R"])
            exact[pol].append(abs(_fresnel_amplitudes(pol, n1, n2, cos_i, cos_f)[0]) ** 2)
    return dict(x=sig(angles), series=[
        series("R_s", "R_s, PyStratify", ours["s"], "pystratify"), series("R_p", "R_p, PyStratify", ours["p"], "pystratify"),
        series("R_s_ref", "R_s, Fresnel closed form", exact["s"], "reference"), series("R_p_ref", "R_p, Fresnel closed form", exact["p"], "reference")],
        worst=max(worst(ours["s"], exact["s"]), worst(ours["p"], exact["p"])), worst_label="largest |R − R_Fresnel| over both polarizations",
        note="Brewster's angle, where R_p vanishes, is arctan(1.5) = 56.3°; the two curves coincide to machine precision at every angle.")


@figure("airy-slab", "A 300 nm slab of n = 2 in air at normal incidence: the Airy formula", "λ [nm]", "R")
def airy_slab():
    lam = np.linspace(400, 800, 201)
    n, d = 2.0, 0.3
    ours = [planar.coh_tmm("s", [1.0, n, 1.0], [inf, d, inf], 0.0, w * 1e-3)["R"] for w in lam]
    r1 = ((n - 1) / (n + 1)) ** 2
    F = 4 * r1 / (1 - r1) ** 2
    delta = 2 * np.pi * n * d / (lam * 1e-3)
    airy = F * np.sin(delta) ** 2 / (1 + F * np.sin(delta) ** 2)
    return dict(x=sig(lam), series=[series("R", "R, PyStratify", ours, "pystratify"), series("R_ref", "R, Airy formula", airy, "reference")],
                worst=worst(ours, airy), worst_label="largest |R − R_Airy|",
                note="R = F sin²δ / (1 + F sin²δ) with F = 4R₁/(1 − R₁)², R₁ the single-surface reflectance and δ = 2πnd/λ: the fringes of a Fabry–Pérot slab.")


@figure("energy-conservation", "A lossless five-layer stack lit from glass, through the critical angle: 1 − R − T", "angle of incidence in glass [°]", "|1 − R − T|", log=True)
def energy_conservation():
    angles = np.linspace(0, 89, 179)
    n = [1.52, 2.3, 1.38, 2.3, 1.38, 2.3, 1.0]
    d = [inf, 0.08, 0.12, 0.08, 0.12, 0.08, inf]
    out = {}
    for pol in ("s", "p"):
        rows = [planar.coh_tmm(pol, n, d, np.radians(a), 0.55) for a in angles]
        out[pol] = np.maximum(np.abs(1 - np.array([r["R"] + r["T"] for r in rows])), 1e-17)
    return dict(x=sig(angles), series=[series("s", "|1 − R − T|, s polarization", out["s"], "pystratify"), series("p", "|1 − R − T|, p polarization", out["p"], "pystratify")],
                worst=float(max(out["s"].max(), out["p"].max())), worst_label="largest |1 − R − T|",
                note="Beyond the critical angle (41.1° in this glass) T is exactly zero and R is exactly one; the residual is floating-point rounding, floored at 10⁻¹⁷ for the logarithmic axis.")


# --- films, point dipole ----------------------------------------------------------------------------------------

@figure("sommerfeld-film-emitter", "A dipole in air above 60 nm of silver on glass, 633 nm: PyStratify against the Sommerfeld integrals", "height above the silver [nm]", "Γ / Γ₀", log=True)
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


# --- films, plane wave: closed forms, internal methods, other codes (2026-10-12, second batch) -----------------

@figure("opaque-film", "A metal film (n = 0.2 + 3i) in air at 600 nm, normal incidence: thickening it toward the bulk Fresnel reflectance", "thickness [nm]", "R, T")
def opaque_film():
    n = 0.2 + 3j
    t = np.linspace(1, 150, 150)
    rows = [planar.coh_tmm("s", [1, n, 1], [inf, w * 1e-3, inf], 0, 0.6) for w in t]
    R, T = np.array([r["R"] for r in rows]), np.array([r["T"] for r in rows])
    bulk = abs((1 - n) / (1 + n)) ** 2
    thick = planar.coh_tmm("s", [1, n, 1], [inf, 20.0, inf], 0, 0.6)
    return dict(x=sig(t), series=[series("R", "R, PyStratify", R, "pystratify"), series("T", "T, PyStratify", T, "pystratify"),
                                  series("R_ref", "|(1 − n)/(1 + n)|², the bulk Fresnel reflectance", np.full(t.size, bulk), "reference")],
                worst=float(abs(thick["R"] - bulk)), worst_label="|R − R_bulk| at 20 µm (the test's opaque film)",
                note="A thin metal film transmits; as it thickens T falls exponentially and R settles on the reflectance of the bulk metal. "
                     "The test takes the film to 20 µm, where T must be exactly 0 and R, the Poynting vector at 0, 10 and 20 µm finite (no "
                     "overflow of exp(Im(k_z) d)), and checks Fresnel's 0.04 and 0.96 for glass at normal incidence.")


def _bare_substrate(part):
    from planar_checks import bare_substrate_rho

    angles = np.linspace(1, 89, 89)
    cases = ((1.5 + 0j, "n = 1.5"), (1.5 + 0.3j, "n = 1.5 + 0.3i"), (0.2 + 3j, "n = 0.2 + 3i"))
    out, refs, diffs = [], [], []
    for i, (n_sub, label) in enumerate(cases):
        pairs = [bare_substrate_rho(n_sub, a) for a in angles]
        ours, theirs = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        f = (lambda z: np.degrees(np.arctan(np.abs(z)))) if part == "psi" else (lambda z: np.degrees(np.angle(z)) % 360)
        out.append(series(f"c{i}", f"{label}, PyStratify", f(ours), "pystratify"))
        refs.append(series(f"c{i}_ref", f"{label}, −r_p / r_s from Fresnel", f(theirs), "reference"))
        diffs.append(np.max(np.abs(ours - theirs)))
    return dict(x=sig(angles), series=out + refs, worst=float(max(diffs)), worst_label="largest |ρ − (−r_p / r_s)|",
                note="Ellipsometry's ratio ρ = tan Ψ e^{iΔ} of a bare substrate is the ratio of its Fresnel coefficients, −r_p / r_s in "
                     "PyStratify's sign convention; the test asserts four (substrate, angle) pairs to 1e-12, the plot sweeps the angle, at 633 nm. "
                     "For the lossless n = 1.5 Δ jumps by 180° at Brewster's angle, 56.3°.")


for _part, _what, _unit in (("psi", "Ψ", "Ψ [°]"), ("delta", "Δ", "Δ [°]")):
    def _make(part=_part):
        return _bare_substrate(part)
    _make.__name__ = f"bare_substrate_{_part}"
    figure(f"bare-substrate-{_part}", f"Bare substrates in air at 633 nm: the ellipsometric angle {_what} against the Fresnel ratio", "angle of incidence [°]", _unit)(_make)


def _smatrix_figure(stack, angles, what):
    from planar_checks import smatrix_rt

    n, d, lam = stack
    ours, theirs = {}, {}
    for pol in ("s", "p"):
        rows = [planar.coh_tmm(pol, n, d, np.radians(a), lam) for a in angles]
        ours[pol] = np.array([[r["R"], r["T"]] for r in rows])
        theirs[pol] = np.array([smatrix_rt(pol, n, d, np.radians(a), lam) for a in angles])
    out = [series(f"{q}_{p}", f"{q}_{p}, PyStratify", ours[p][:, i], "pystratify") for p in "sp" for i, q in enumerate("RT")]
    refs = [series(f"{q}_{p}_ref", f"{q}_{p}, S-matrix", theirs[p][:, i], "reference") for p in "sp" for i, q in enumerate("RT")]
    return dict(x=sig(angles), series=out + refs, worst=max(worst(ours[p], theirs[p]) for p in "sp"), worst_label="largest |R − R_S|, |T − T_S|",
                note=f"{what} The reference chains the interfaces' and layers' two-port scattering matrices by Redheffer's star product "
                     "(planar_checks.py), a different algebra from PyStratify's transfer matrices that never multiplies by exp(+Im(k_z) d); "
                     "the test also takes 120 µm of n = 2 + 0.5i, where T is below 1e-30 in both.")


@figure("smatrix-metal-stack", "Air / 100 nm silica / 40 nm metal (n = 0.05 + 3i) / glass at 600 nm: against the scattering-matrix star product", "angle of incidence [°]", "R, T")
def smatrix_metal_stack():
    from planar_checks import METAL_STACK
    return _smatrix_figure(METAL_STACK, np.linspace(0, 89, 90), "An absorbing stack, s and p.")


@figure("smatrix-beyond-critical", "Glass / 400 nm air / 130 nm of n = 2 / glass at 550 nm: through the critical angle against the scattering-matrix star product", "angle of incidence in glass [°]", "R, T")
def smatrix_beyond_critical():
    from planar_checks import BEYOND_CRITICAL
    return _smatrix_figure(BEYOND_CRITICAL, np.linspace(0, 89, 90), "Beyond 41.8° the air gap is evanescent and light tunnels through it to the n = 2 layer and the glass.")


INCOHERENT = ([1.0, 2.08 + 0.03j, 1.47 + 0.001j, 1.82 + 0.02j, 1.56], [inf, 0.09, 12.0, 0.11, inf], ["i", "c", "i", "c", "i"])


@figure("incoherent-phase-average", "Coherent films on both sides of a 12 µm incoherent layer, on glass, 640 nm: against an explicit average over its phase", "angle of incidence [°]", "R, T")
def incoherent_phase_average():
    from planar_checks import explicit_phase_average

    n, d, c = INCOHERENT
    angles = np.linspace(0, 80, 33)
    ours, ref = {}, {}
    for pol in ("s", "p"):
        rows = [planar.pim_tmm(pol, n, d, c, np.radians(a), 0.64) for a in angles]
        ours[pol] = np.array([[r["R"], r["T"]] for r in rows])
        sampled = [explicit_phase_average(pol, n, d, c, a, 0.64, samples_per_layer=301) for a in angles]
        ref[pol] = np.array([[e["reflectance"], e["transmittance"]] for e in sampled])
    out = [series(f"{q}_{p}", f"{q}_{p}, PyStratify", ours[p][:, i], "pystratify") for p in "sp" for i, q in enumerate("RT")]
    refs = [series(f"{q}_{p}_ref", f"{q}_{p}, phase average of coherent solves", ref[p][:, i], "reference") for p in "sp" for i, q in enumerate("RT")]
    return dict(x=sig(angles), series=out + refs, worst=max(worst(ours[p], ref[p]) for p in "sp"), worst_label="largest |difference|",
                note="An incoherent layer is a coherent one whose round-trip phase is averaged out: the reference solves the whole stack "
                     "coherently 301 times with that layer's phase spread over one period (its absorption kept) and averages R and T; "
                     "PyStratify's incoherent transfer matrix does it in closed form. The test also compares the absorption in each layer.")


@figure("inc-tmm-lineage", "The same stack: PyStratify's incoherent solver against inc_tmm, the lineage it replaced (GTMM in the backend tests)", "λ [nm]", "R, T, A")
def inc_tmm_lineage():
    n, d, c = INCOHERENT
    lam = np.linspace(450, 900, 91)
    ours, ref = [], []
    for w in lam:
        a = planar.pim_tmm("s", n, d, c, np.radians(31), w * 1e-3)
        b = planar.inc_tmm("s", n, d, c, np.radians(31), w * 1e-3)
        ours.append([a["R"], a["T"], 1 - a["R"] - a["T"]]), ref.append([b["R"], b["T"], 1 - b["R"] - b["T"]])
    ours, ref = np.array(ours), np.array(ref)
    labels = ("R", "T", "A")
    return dict(x=sig(lam), series=[series(q, f"{q}, PyStratify (pim_tmm)", ours[:, i], "pystratify") for i, q in enumerate(labels)]
                + [series(f"{q}_ref", f"{q}, inc_tmm", ref[:, i], "reference") for i, q in enumerate(labels)],
                worst=worst(ours, ref), worst_label="largest |difference|",
                note="inc_tmm is the incoherent transfer matrix of S. J. Byrnes' tmm package, from which PyStratify's planar engine "
                     "descends; it is kept as a lineage check, not an independent one. 31°, s polarization; the backend's tests run 120 random "
                     "stacks as well.")


@figure("rcwa-unpatterned", "A three-layer filter (n = 2.35 / 1.46 / 2.35, 95/110/95 nm) on n = 1.5: against meent's RCWA with zero patterning", "λ [nm]", "R, T")
def rcwa_unpatterned():
    from rcwa_meent import meent_rt, stratify_rt

    lam = np.linspace(400, 800, 81)
    ours = np.array([stratify_rt(w * 1e-3) for w in lam])
    theirs = np.array([meent_rt(w * 1e-3) for w in lam])
    return dict(x=sig(lam), series=[series("R", "R, PyStratify", ours[:, 0], "pystratify"), series("T", "T, PyStratify", ours[:, 1], "pystratify"),
                                    series("R_ref", "R, meent RCWA", theirs[:, 0], "reference"), series("T_ref", "T, meent RCWA", theirs[:, 1], "reference")],
                worst=worst(ours, theirs), worst_label="largest |difference|",
                note="Rigorous coupled-wave analysis, the calculator's grating engine (meent, MIT), with uniform layers and one Fourier order "
                     "must reduce to the transfer matrix. Normal incidence, s; rcwa_meent.py, which the test runs at 0° and 30°, s and p.")


@figure("position-resolved-absorption", "An absorbing four-layer stack at 30°: the absorption density integrated through each layer against its Poynting difference", "λ [nm]", "absorptance per layer")
def position_resolved_absorption_figure():
    from planar_checks import integrated_absorption

    lam = np.linspace(400, 1000, 61)
    pairs = [integrated_absorption("p", w * 1e-3) for w in lam]
    ours, ref = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    labels = ("120 nm n = 2.1 + 0.08i", "250 nm n = 1.46 + 0.01i", "20 nm n = 0.2 + 3.2i")
    return dict(x=sig(lam), series=[series(f"l{i}", f"{lab}, ∫ density (PyStratify)", ours[:, i], "pystratify") for i, lab in enumerate(labels)]
                + [series(f"l{i}_ref", f"{lab}, Poynting difference", ref[:, i], "reference") for i, lab in enumerate(labels)],
                worst=worst(ours, ref), worst_label="largest |difference|",
                note="position_resolved gives the field and the absorption density at any depth; integrated through a layer (200 Gauss–"
                     "Legendre points) it must equal the power the Poynting flux loses across that layer. p polarization, air above, glass "
                     "below. The focused-beam near field builds on the same fields (its Parseval check is the focused-beam row).")


# --- films, plane wave: other codes ----------------------------------------------------------------------------

@figure("pyelli-bragg-mirror", "A ten-pair Bragg mirror (n = 2.35 / 1.46) on glass at 30°: R_s against pyElli's 2×2 and 4×4 solvers", "λ [nm]", "R_s")
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


@figure("pyelli-gold-ellipsometry", "20 nm of gold (n = 0.27 + 2.93i) on glass at 633 nm: Ψ and Δ against pyElli's 4×4 solver", "angle of incidence [°]", "angle [°]")
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


@figure("pymoosh-bragg-mirror", "A ten-pair Bragg mirror (n = 2.35 / 1.46) on glass at 30°: R and T against PyMoosh", "λ [nm]", "R, T")
def pymoosh_bragg_mirror():
    from planar_codes import CASES, moosh, stratify

    case = CASES["Bragg mirror, 10 pairs"]
    lam = np.linspace(450, 850, 81)
    out, refs, diffs = [], [], []
    for pol in "sp":
        ours = np.array([[stratify(case, w, np.radians(30), pol)[q] for q in "RT"] for w in lam])
        theirs = np.array([[moosh(case, w, np.radians(30), pol)[q] for q in "RT"] for w in lam])
        out += [series(f"{q}_{pol}", f"{q}_{pol}, PyStratify", ours[:, i], "pystratify") for i, q in enumerate("RT")]
        refs += [series(f"{q}_{pol}_ref", f"{q}_{pol}, PyMoosh", theirs[:, i], "reference") for i, q in enumerate("RT")]
        diffs.append(worst(ours, theirs))
    return dict(x=sig(lam), series=out + refs, worst=float(max(diffs)), worst_label="largest |difference|",
                note="PyMoosh (Langevin, Moreau et al.) solves stacks with its own scattering matrices. The test runs nine stacks at four "
                     "angles and three wavelengths for R, T, r, t and the absorption in every layer.")


@figure("pymoosh-magnetic-layer", "A 150 nm layer with n = 1.5 and μ = 2 on glass at 633 nm: R and T against PyMoosh", "angle of incidence [°]", "R, T")
def pymoosh_magnetic_layer():
    from planar_codes import CASES, moosh, stratify

    case = CASES["magnetic layer (n 1.5, mu 2)"]
    angles = np.linspace(0, 89, 90)
    out, refs, diffs = [], [], []
    for pol in "sp":
        ours = np.array([[stratify(case, 633.0, np.radians(a), pol)[q] for q in "RT"] for a in angles])
        theirs = np.array([[moosh(case, 633.0, np.radians(a), pol)[q] for q in "RT"] for a in angles])
        out += [series(f"{q}_{pol}", f"{q}_{pol}, PyStratify", ours[:, i], "pystratify") for i, q in enumerate("RT")]
        refs += [series(f"{q}_{pol}_ref", f"{q}_{pol}, PyMoosh", theirs[:, i], "reference") for i, q in enumerate("RT")]
        diffs.append(worst(ours, theirs))
    return dict(x=sig(angles), series=out + refs, worst=float(max(diffs)), worst_label="largest |difference|",
                note="A magnetic layer changes the Fresnel coefficients through its impedance √(μ/ε), not only its index; PyMoosh takes ε and μ "
                     "separately, an independent check of PyStratify's permeability.")


@figure("pymoosh-field", "An AR pair (64 nm n = 2.35, 105 nm n = 1.46) on glass at 633 nm, 60°: the field through the stack against PyMoosh", "depth below the surface [nm]", "field")
def pymoosh_field():
    from planar_codes import CASES, moosh_field, stratify_field

    case = CASES["AR pair on glass"]
    _, d, _ = case
    out, refs, diffs = [], [], []
    for pol, label in (("s", "E_y (s)"), ("p", "H_y (p)")):
        samples, theirs = moosh_field(case, 633.0, np.radians(60), pol, step=2.0)
        ours = stratify_field(case, 633.0, np.radians(60), pol, samples)
        start = np.cumsum([0.0] + list(d[1:-1]))
        depth = np.array([start[layer - 1] + z for layer, z in samples])
        if pol == "s":
            x = depth
        out += [series(f"re_{pol}", f"Re {label}, PyStratify", ours.real, "pystratify"), series(f"im_{pol}", f"Im {label}, PyStratify", ours.imag, "pystratify")]
        refs += [series(f"re_{pol}_ref", f"Re {label}, PyMoosh", theirs.real, "reference"), series(f"im_{pol}_ref", f"Im {label}, PyMoosh", theirs.imag, "reference")]
        diffs.append(np.max(np.abs(ours - theirs)))
    return dict(x=sig(x), series=out + refs, worst=float(max(diffs)), worst_label="largest |difference|",
                note="PyMoosh's field routine with one Fourier order is a single plane wave; sampled every 2 nm through both layers and 200 nm "
                     "into the glass (E_y for s, H_y for p, as each code normalizes them). The test runs every stack at every angle and "
                     "wavelength every 5 nm.")


@figure("pymoosh-incoherent", "80 nm of TiO₂ (n = 2.3 + 0.01i) on 1 mm of incoherent glass, in air: R and T against PyMoosh's incoherent solver", "λ [nm]", "R, T")
def pymoosh_incoherent():
    from planar_codes import CASES, moosh, stratify

    case = CASES["TiO2 on 1 mm incoherent glass"]
    lam = np.linspace(400, 1000, 61)
    out, refs, diffs = [], [], []
    for pol, a in (("s", 0.0), ("p", 60.0)):
        ours = np.array([[stratify(case, w, np.radians(a), pol)[q] for q in "RT"] for w in lam])
        theirs = np.array([[moosh(case, w, np.radians(a), pol)[q] for q in "RT"] for w in lam])
        out += [series(f"{q}_{pol}", f"{q}, {a:g}° {pol}, PyStratify", ours[:, i], "pystratify") for i, q in enumerate("RT")]
        refs += [series(f"{q}_{pol}_ref", f"{q}, {a:g}° {pol}, PyMoosh", theirs[:, i], "reference") for i, q in enumerate("RT")]
        diffs.append(worst(ours, theirs))
    return dict(x=sig(lam), series=out + refs, worst=float(max(diffs)), worst_label="largest |difference|",
                note="A coherent film on a thick substrate whose back reflection adds in power, not in amplitude: PyMoosh's incoherent.py "
                     "is the first incoherent reference independent of PyStratify's lineage (inc_tmm).")


def _tmmax_figure(name, versions):
    from tmmax_films import STACKS, stratify_rt, tmmax_rt

    n, d = STACKS[name]
    lam = np.linspace(400, 900, 101)
    R0, _ = stratify_rt(n, d, "s", lam, (0.0,))
    out = [series("R", "R, PyStratify", R0[:, 0], "pystratify")]
    diffs = []
    for label, fixed in versions:
        R, _ = tmmax_rt(n, d, "s", fixed, lam, (0.0,))
        out.append(series(f"R_{int(fixed)}", f"R, tmmax 1.1.4 {label}", R[:, 0], "reference"))
        diffs.append(float(np.max(np.abs(R[:, 0] - R0[:, 0]))))
    return lam, out, diffs


@figure("tmmax-single-layer", "100 nm of n = 2.1 in air, normal incidence: R against tmmax (JAX, float32) as shipped", "λ [nm]", "R", env="tmmax")
def tmmax_single_layer():
    lam, out, diffs = _tmmax_figure("single layer n = 2.1 in air", (("as shipped", False),))
    return dict(x=sig(lam), series=out, worst=diffs[0], worst_label="largest |R − R_tmmax| (float32)",
                note="tmmax computes in single precision (its scan carries complex64), so agreement stops near 1e-7; single layers are "
                     "asserted as shipped, to 2e-6.")


@figure("tmmax-bragg-mirror", "A five-pair quarter-wave mirror (n = 2.35 / 1.46) on glass, normal incidence: R against tmmax, as shipped and with its scan step fixed", "λ [nm]", "R", env="tmmax")
def tmmax_bragg_mirror():
    lam, out, diffs = _tmmax_figure("quarter-wave mirror, 5 pairs", (("with the scan step fixed", True), ("as shipped", False)))
    return dict(x=sig(lam), series=out, worst=diffs[0], worst_label="largest |R − R_tmmax| (fixed step, float32)",
                note="tmmax 1.1.4, the version on PyPI, is wrong for two or more layers: its scan step broadcasts the 2×2 matrices as a "
                     "stack of column vectors (reported upstream, bahremsd/tmmax #44, #45; fixed on its main branch, not released). With the "
                     f"step rebuilt (tmmax_films.py) it agrees to float32; as shipped it is off by up to {diffs[1]:.2f}. The test asserts the "
                     "fixed step on multilayers.")


# --- films, point dipole ----------------------------------------------------------------------------------------

@figure("film-dipole-homogeneous", "A dipole inside a 200 nm film of n = 1.5 between half-spaces of n = 1.5, 600 nm: the rates of free space", "depth in the film [nm]", "|Γ / Γ₀ − 1|", log=True)
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
    figure(f"film-coating-budget-{_dipole}", f"{'An' if _dipole == 'electric' else 'A'} {_dipole} dipole above 100 nm of an absorbing film (n = 0.2 + 3i) in air, 600 nm: the total against its channels",
           "height above the film [nm]", "Γ / Γ₀", log=True)(_make)


@figure("film-guided-pole-tolerance", "A dipole in a 0.8 µm film of n = 3.5 in air (18 guided modes), 600 nm: the error PyStratify reports against the accuracy asked for", "tolerance asked for", "error", log=True)
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


@figure("film-guide-budget", "A dipole inside a lossless 200 nm film of n = 1.5 in air, 600 nm: the total against escape + guided", "depth in the film [nm]", "Γ / Γ₀")
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


@figure("film-pattern-free-space", "A dipole in a film of n = 1 in n = 1, 600 nm: the emission pattern against the free dipole's", "polar angle θ [°]", "power per solid angle")
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


@figure("image-dipole-mirror", "A dipole above a perfect mirror: the Sommerfeld reference against the image dipole (Drexhage; Chance, Prock & Silbey)", "kd = 4πh/λ", "Γ / Γ₀")
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


@figure("weak-guide-sommerfeld", "A dipole above 200 nm of TiO₂ (n = 2.3 + 0.001i) on glass, 633 nm: against the Sommerfeld integrals with the guide's quasi-poles", "height above the film [nm]", "Γ / Γ₀")
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


@figure("lossless-guide-limit", "A dipole above 200 nm of TiO₂ (n = 2.3) on glass, 633 nm: the lossless guide against the k → 0 limit of an absorbing one", "height above the film [nm]", "Γ / Γ₀")
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


def _guide_modes(reference):
    from planar_codes import GUIDES, guided_mode_sets, slab_modes
    from pystratify.planar_emission import FilmSource

    n, d, lam = GUIDES["symmetric slab, n 2.0 in air, 500 nm"]
    thickness = np.linspace(100, 600, 26)
    ours, theirs = [], []
    for t in thickness:
        dd = [inf, float(t), inf]
        if reference == "pymoosh":
            a, b, _ = guided_mode_sets((n, dd, lam))
        else:
            source = FilmSource(n, dd, lam, 1, t / 3)
            a, b = sorted(np.array(source.guided_poles()) * source.ns), slab_modes(n[0], n[1], n[2], t, lam)
        ours.append(sorted(a, reverse=True)), theirs.append(sorted(b, reverse=True))
    width = max(len(v) for v in ours + theirs)
    pad = lambda rows: np.array([v + [np.nan] * (width - len(v)) for v in rows])  # noqa: E731
    ours, theirs = pad(ours), pad(theirs)
    diffs = np.abs(ours - theirs)
    return thickness, ours, theirs, float(np.nanmax(diffs)), width


@figure("pymoosh-guided-modes", "A symmetric slab of n = 2 in air, 633 nm: the guided modes PyStratify finds against PyMoosh's mode solver", "thickness [nm]", "n_eff")
def pymoosh_guided_modes():
    t, ours, theirs, diff, width = _guide_modes("pymoosh")
    return dict(x=sig(t), series=[series(f"m{i}", f"mode {i + 1}, PyStratify", ours[:, i], "pystratify") for i in range(width)]
                + [series(f"m{i}_ref", f"mode {i + 1}, PyMoosh", theirs[:, i], "reference") for i in range(width)],
                worst=diff, worst_label="largest |n_eff − n_eff(PyMoosh)|",
                note="Every TE and TM mode, numbered from the highest index: the poles PyStratify finds behind a dipole's guided channel "
                     "against PyMoosh's guided_modes (its search stops at 1e-10). A mode appears at its cutoff, n_eff = 1. The test runs four guides.")


@figure("slab-dispersion-modes", "The same slab: the guided modes against the three-layer dispersion relation", "thickness [nm]", "n_eff")
def slab_dispersion_modes():
    t, ours, theirs, diff, width = _guide_modes("analytic")
    return dict(x=sig(t), series=[series(f"m{i}", f"mode {i + 1}, PyStratify", ours[:, i], "pystratify") for i in range(width)]
                + [series(f"m{i}_ref", f"mode {i + 1}, dispersion relation", theirs[:, i], "reference") for i in range(width)],
                worst=diff, worst_label="largest |n_eff − n_eff(relation)|",
                note="κd = mπ + atan(ρ₁γ₁/κ) + atan(ρ₃γ₃/κ), ρ = 1 for TE and (n_core/n_clad)² for TM, solved by bisection (planar_codes.py "
                     "slab_modes).")


@figure("smuthi-film-guide", "A dipole above 200 nm of TiO₂ (n = 2.3) on glass, 633 nm: total rates against smuthi", "height above the film [nm]", "Γ / Γ₀", env="smuthi")
def smuthi_film_guide():
    import emitter_references as er

    n, d = er.GUIDES["air above 200 nm TiO2 on glass"]
    h = np.linspace(30, 300, 28)
    ours = np.array([er.stratify_rates(n, d, hi, tolerance=1e-8) for hi in h])
    ref = np.full(ours.shape, np.nan)
    for i in range(0, h.size, 3):
        ref[i] = er.smuthi_rates(n, d, h[i])
    m = np.isfinite(ref[:, 0])
    return dict(x=sig(h), series=[series("perp", "perpendicular, PyStratify", ours[:, 0], "pystratify"), series("par", "parallel, PyStratify", ours[:, 1], "pystratify"),
                                  series("perp_ref", "perpendicular, smuthi", ref[:, 0], "reference"), series("par_ref", "parallel, smuthi", ref[:, 1], "reference")],
                worst=float(np.max(np.abs(ref[m] / ours[m] - 1))), worst_label="largest relative difference (smuthi's contour)",
                note="smuthi's DipoleSource.dissipated_power integrates along its own Sommerfeld contour deflected around the guide's poles "
                     "(resolution 2e-3), a method independent of PyStratify's residues; agreement is set by that contour's resolution.")


@figure("meep-axial-film", "A dipole 100 nm above 200 nm of n = 2.3 on glass: total decay rate against Meep (cylindrical coordinates, 100 px/µm)", "λ [µm]", "Γ / Γ₀", env="meep")
def meep_axial_film():
    import meep as mp
    from fdtd_meep import AXIAL_CASES, meep_ldos_axial

    case = "film: dipole 100 nm above 200 nm of n = 2.3 on glass"
    _, dimensions, n, source, names = AXIAL_CASES[case]
    mp.verbosity(0)
    coarse = 1 / np.linspace(0.85, 1.65, 17)
    theirs = {o: meep_ldos_axial(case, o, 100, wavelengths=coarse) / meep_ldos_axial(case, o, 100, False, wavelengths=coarse) for o in names}
    dense = 1 / np.linspace(0.85, 1.65, 161)
    ours = np.array([ps.solve_problem(ps.Problem("films", dimensions, n, w, ps.PointDipole(-source, "electric", 0), 1e-8))["total"][:2] for w in dense])
    out, refs, rel = [], [], []
    for i, o in enumerate(names):
        ref = np.full(dense.size, np.nan)
        ref[::10] = theirs[o]
        out.append(series(o, f"{o}, PyStratify", ours[:, i], "pystratify"))
        refs.append(series(f"{o}_ref", f"{o}, Meep", ref, "reference"))
        rel.append(np.max(np.abs(theirs[o] / ours[::10, i] - 1)))
    return dict(x=sig(dense), series=out + refs, worst=float(max(rel)), worst_label="largest relative difference (the grid's error)",
                note="Meep in cylindrical coordinates around the dipole (m = 0 perpendicular, m = 1 parallel), the film and substrate infinite "
                     "in r, 1 µm of PML; the local density of states over the same run without the film. The film is 20 pixels thick at 100 "
                     "px/µm and converges at about first order: worst 16.9% → 7.4% → 3.7% at 25, 50 and 100 px/µm (FDTD_MEEP.md); the "
                     "test asserts < 5% at 100 and closer than 25.")


@figure("lukosz-kunz-1977", "Dipoles in air near a lossless half-space of n = 1.5 and n = 3: total radiated power, Lukosz & Kunz 1977, Fig. 4", "distance z₀ / λ₁", "L / L∞")
def lukosz_kunz_1977():
    curve, paper = classic_curve("lukosz_kunz_1977_fig4"), classic_values("lukosz_kunz_1977")
    x = curve["z0_over_lambda1"]
    keys = (("electric perp n=1.5", "electric ⊥, n = 1.5"), ("electric par n=1.5", "electric ∥, n = 1.5"),
            ("electric par n=3.0", "electric ∥, n = 3"), ("magnetic perp n=1.5", "magnetic ⊥, n = 1.5"))
    e3 = curve["electric par n=3.0"]

    def turn(lo, hi, low=True):
        m = (x >= lo) & (x <= hi)
        return float(x[m][np.argmin(e3[m]) if low else np.argmax(e3[m])])

    refs = {"electric perp n=1.5": {float(x[0]): paper["e, perp, n = 1.5, at contact"], 0.4: paper["e, perp, n = 1.5, z0/λ1 = 0.4: back to ~1"]},
            "electric par n=1.5": {float(x[0]): paper["e, par, n = 1.5, at contact"]},
            "electric par n=3.0": {paper["e, par, n = 3: z0/λ1 of the first minimum"]: paper["e, par, n = 3: first minimum"],
                                   turn(0.25, 0.45, False): paper["e, par, n = 3: maximum near z0/λ1 = 0.35"],
                                   turn(0.5, 0.75): paper["e, par, n = 3: second minimum near z0/λ1 = 0.6"]},
            "magnetic perp n=1.5": {float(x[0]): paper["m, perp, n = 1.5, at contact"]}}
    return dict(x=sig(x), series=[series(f"s{i}", f"{lab}, PyStratify", curve[k], "pystratify") for i, (k, lab) in enumerate(keys)]
                + [series(f"s{i}_ref", f"{lab}, read off Figs. 4 and 6", at(x, refs[k]), "reference") for i, (k, lab) in enumerate(keys)],
                worst=_read_worst("lukosz_kunz_1977"), worst_label="largest relative difference from the read values",
                note="Lukosz & Kunz, J. Opt. Soc. Am. 67, 1607 (1977): the power a dipole radiates near a plane interface, against its "
                     "distance in units of the wavelength in its own medium. Values read off the figures to ±0.05; positions not printed are "
                     "marked where PyStratify's curve turns. From crosscheck/classics/lukosz_kunz_1977.py.")


@figure("neyts-1998-microcavity", "A dipole in 190 nm of ZnS (n = 2.3) on aluminium, 600 nm: emission into air and total rate across the layer, Neyts 1998, Figs. 9–11", "distance from the mirror [nm]", "fraction, rate")
def neyts_1998_microcavity():
    curve, paper = classic_curve("neyts_1998_fig10_column"), classic_values("neyts_1998")
    z = curve["z_from_mirror_um"] * 1e3
    e, f = curve["E_appl_isotropic"], curve["F_isotropic"]
    m = z > 15  # F diverges as z⁻³ at the mirror
    z, e, f = z[m], e[m], f[m]
    return dict(x=sig(z), series=[series("e", "E_appl, escape into air (PyStratify)", e, "pystratify"), series("f", "F, total rate (PyStratify)", f, "pystratify"),
                                  series("e_ref", "E_appl = 0.38, printed (dipole at the air side)", at(z, {float(z[-1]): paper["maximum E_appl (dipole at the air side)"]}), "reference"),
                                  series("f_ref", "F = 0.817, printed", at(z, {float(z[-1]): paper["F at that point"]}), "reference")],
                worst=float(max(abs(e[-1] - 0.38) / 0.38, abs(f[-1] - 0.817) / 0.817)), worst_label="largest relative difference from the printed values",
                note="Neyts, J. Opt. Soc. Am. A 15, 962 (1998): an isotropic dipole in the ZnS layer of an electroluminescent device on an "
                     "Al mirror (n = 1 + 6i), the layer at the optimum thickness n_e d_e / λ = 0.73 (PyStratify finds 0.725). The printed "
                     "maximum outcoupling 0.38 and F = 0.817 are for the dipole at the air side; F grows as z⁻³ toward the mirror (not drawn "
                     "below 15 nm). From crosscheck/classics/neyts_1998.py.")


# --- films, focused beam ----------------------------------------------------------------------------------------

def _film_cone(mode):
    from focused_checks import film_brute
    from pystratify import focused as F

    n = np.array([1.0, 1.55 + 0.03j, 1.42, 1.0 if mode == "transmission" else 0.3 + 18j])
    d = np.array([inf, 1.2, 3.0, inf])
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7 if mode == "transmission" else None,
                          collection_obscuration=0.35 if mode == "transmission" else None, mode=mode)
    lam = np.linspace(2.5, 6.0, 15)
    r = F.focused_films(n, d, lam, beam)
    ours = np.stack([np.ravel(r["reflectance"]), np.ravel(r["transmittance"]), np.ravel(r.detected)], -1)
    ref = np.array([film_brute(n, d, None, w, beam, (1.0, 0.0))[:3] for w in lam])
    keys = ("reflectance", "transmittance", "detected")
    return dict(x=sig(lam), series=[series(k, f"{k}, PyStratify", ours[:, i], "pystratify") for i, k in enumerate(keys)]
                + [series(f"{k}_ref", f"{k}, 2D pupil grid", ref[:, i], "reference") for i, k in enumerate(keys)],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note=f"In {mode}: a 1.2 µm film of n = 1.55 + 0.03i on 3 µm of n = 1.42, "
                     + ("in air, detected by a 0.35–0.7 condenser. " if mode == "transmission" else "on a metal (n = 0.3 + 18i), detected in reflection. ")
                     + "PyStratify reduces the beam to a one-dimensional quadrature over the cone angle; the reference sums a two-dimensional "
                       "pupil grid of x- and y-directed plane waves, each split into s and p and solved separately (focused_checks.py). The "
                       "test also compares every layer's absorption, for both pupil polarizations.")


for _mode in ("transmission", "reflection"):
    def _make(mode=_mode):
        return _film_cone(mode)
    _make.__name__ = f"film_cone_{_mode}"
    figure(f"film-cone-{_mode}", f"A two-layer film under a focused beam (NA 0.6/0.2), in {_mode}: the cone quadrature against a 2D pupil grid", "λ [µm]", "power / incident")(_make)


@figure("film-path-factor", "A weak absorber (α = 10 cm⁻¹, 1 µm) in n = 1.5 under an aplanatic NA 0.23–0.5 cone: the path factor", "λ [µm]", "ln10 A / (α d)")
def film_path_factor_figure():
    from pystratify import focused as F

    alpha, lam = 1e-5, np.linspace(2.0, 10.0, 33)
    kappa = alpha * lam / (4 * np.pi)
    n = np.stack((np.full(lam.size, 1.5), 1.5 + 1j * kappa, np.full(lam.size, 1.5)), 1)
    r = F.focused_films(n, [inf, 1.0, inf], lam, ps.FocusedBeam(0.5, 0.23), reference=([1.5, 1.5], [inf, inf]))
    factor = np.ravel(r.apparent_absorbance) * np.log(10) / alpha
    return dict(x=sig(lam), series=[series("factor", "ln10 A / (α d), PyStratify", factor, "pystratify"),
                                    series("factor_ref", "⟨1 / cos θ⟩ of the aplanatic cone, 1.0357", np.full(lam.size, 1.0357), "reference")],
                worst=float(np.max(np.abs(factor - 1.0357))), worst_label="largest |difference| (1.0357 is printed to 5 digits)",
                note="A focused beam crosses a film obliquely, so a weak absorber looks thicker by the cone's mean path factor ⟨1 / cos θ⟩, "
                     "1.0357 for an aplanatic NA 0.23–0.5 pupil in n = 1.5, the ruling the test pins. The test also checks that an empty stack, "
                     "or a sample that is its own reference, has zero absorbance and that Köhler illumination equals coherent for a film.")


@figure("film-koehler-coherent", "A film (1.2 µm, n = 1.55 + 0.03i) in air under NA 0.5/0.23: Köhler illumination against a coherent focus", "λ [µm]", "apparent absorbance")
def film_koehler_coherent():
    from pystratify import focused as F

    n, d = [1.0, 1.55 + 0.03j, 1.0], [inf, 1.2, inf]
    lam = np.linspace(2.5, 8.0, 23)
    beams = [ps.FocusedBeam(0.5, 0.23), ps.FocusedBeam(0.5, 0.23, illumination="kohler", field_stop=40.0)]
    r = F.focused_films(n, d, lam, beams, reference=([1.0, 1.0], [inf, inf]))
    coherent, kohler = np.ravel(r.apparent_absorbance[0]), np.ravel(r.apparent_absorbance[1])
    return dict(x=sig(lam), series=[series("coherent", "coherent focus, PyStratify", coherent, "pystratify"),
                                    series("kohler", "Köhler (field stop 40 µm)", kohler, "reference")],
                worst=float(np.max(np.abs(coherent - kohler))), worst_label="largest |difference|",
                note="A laterally uniform film cannot tell how the pupil's plane waves are phased: coherent and Köhler illumination see the same "
                     "cone and must give the same detected power.")


@figure("film-focused-near-field", "A two-layer film (0.8 µm n = 1.6 + 0.05i, 2 µm n = 1.42) in air under a focused beam (NA 0.6/0.2, focus off-axis), 3 µm: |E|² along the depth", "z [µm]", "|E|²")
def film_focused_near_field():
    from focused_checks import film_near_field_pair

    z = np.linspace(-1.5, 4.5, 121)
    field, pair = film_near_field_pair(np.column_stack([np.full_like(z, 0.1), np.zeros_like(z), z]))
    ours = np.sum(np.abs(np.asarray(field.e[0])) ** 2, -1)
    ref = np.sum(np.abs(pair[0]) ** 2, -1)
    return dict(x=sig(z), series=[series("e2", "|E|², PyStratify", ours, "pystratify"), series("e2_ref", "|E|², position-resolved plane waves summed", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Along x = 0.1, y = 0 µm from the air above, through both layers (0–0.8 and 0.8–2.8 µm) into the air below; x-polarized "
                     "pupil, focus at (0.3, −0.2, 0.5) µm. The reference sums a 48 × 48 pupil grid of plane waves, each from position_resolved. "
                     "The test also checks Parseval: the lateral integral of |E|² at a depth against the pupil's.")


# --- spheres, plane wave ----------------------------------------------------------------------------------------

@figure("mie-homogeneous", "A homogeneous sphere, m = 1.5: Mie's series, one shell and fifty identical shells", "size parameter x", "Q_ext")
def mie_homogeneous():
    from sphere_checks import mie_ab

    xs = np.linspace(0.2, 30, 150)
    lam, m = 1.0, 1.5
    radii = xs / (2 * np.pi)
    ours, fifty, exact = [], [], []
    for x, R in zip(xs, radii):
        lmax = int(x + 4 * x ** (1 / 3) + 4)
        sol = ps.solve([R], np.array([[m, 1.0]]), np.array([lam]), l_max=lmax)
        ours.append(ps.cross_sections(sol).q_ext[0])
        shells = np.linspace(R / 50, R, 50)
        sol50 = ps.solve(shells, np.array([[m] * 50 + [1.0]]), np.array([lam]), l_max=lmax)
        fifty.append(ps.cross_sections(sol50).q_ext[0])
        a, b = mie_ab(m, x, lmax)
        ell = np.arange(1, lmax + 1)
        exact.append(2 / x ** 2 * np.sum((2 * ell + 1) * np.real(a + b)))
    return dict(x=sig(xs), series=[series("q", "Q_ext, one shell (PyStratify)", ours, "pystratify"), series("q50", "Q_ext, fifty identical shells (PyStratify)", fifty, "pystratify"),
                                   series("q_ref", "Q_ext, Bohren & Huffman Eq. 4.53", exact, "reference")],
                worst=max(worst(ours, exact), worst(fifty, exact)), worst_label="largest |Q_ext − Q_Mie| over both curves",
                note="The textbook coefficients a_ℓ, b_ℓ (Riccati–Bessel functions from SciPy) summed directly, against the transfer-matrix recursion with one interface and with fifty interfaces between identical media.")


@figure("bhmie-large-spheres", "Large absorbing spheres, m = 1.5 + 0.1i: PyStratify against BHMIE to x = 5000", "size parameter x", "Q_ext")
def bhmie_large_spheres():
    from pystratify.references import bhmie

    xs = np.geomspace(1, 5000, 120)
    m = 1.5 + 0.1j
    ours, ref = [], []
    for x in xs:
        R = x / (2 * np.pi)
        lmax = int(x + 4 * x ** (1 / 3) + 2) + 20  # both codes 20 orders beyond Wiscombe's truncation
        sol = ps.solve([R], np.array([[m, 1.0]]), np.array([1.0]), l_max=lmax)
        ours.append(ps.cross_sections(sol).q_ext[0])
        ref.append(bhmie(x, m, lmax)[0])
    return dict(x=sig(xs), series=[series("q", "Q_ext, PyStratify", ours, "pystratify"), series("q_ref", "Q_ext, BHMIE (Bohren & Huffman, Appendix A)", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="BHMIE's downward recurrence for the logarithmic derivative, ported from the book's Appendix A, against the logarithmic transfer-matrix recursion, both summed 20 orders beyond Wiscombe's truncation; the x axis is logarithmic and the extinction tends to 2 with the ripple of the edge waves. For a weakly absorbing or lossless sphere BHMIE's double-precision recurrence drifts at large x (3e-4 in Q_ext at x = 1481 for m = 1.33 + 1e-4 i, 6e-4 for m = 1.33), where PyStratify agrees with the same series summed in 40-digit arithmetic (pystratify.references.sphere_extinction) to 2e-13: the absorbing case here is the one the test suite asserts.")


@figure("sixty-digit-coefficients", "A 5 nm metal nanoshell on a 15 nm silica core, 600 nm: Mie coefficients against 60-digit transfer matrices", "order ℓ", "|T_ℓ|", log=True)
def sixty_digit_coefficients():
    from mp_reference import coefficients

    radii, n = [15.0, 20.0], [np.sqrt(2.13), np.sqrt(-10 + 1j + 0j), 1.0]
    lmax = 40
    sol = ps.solve(radii, np.array([n]), np.array([600.0]), l_max=lmax)
    orders = np.arange(1, lmax + 1)
    ours = {ps.TM: np.abs(np.exp(sol.log_t[ps.TM, 0])), ps.TE: np.abs(np.exp(sol.log_t[ps.TE, 0]))}
    ref = {}
    for te in (False, True):
        vals = []
        for l in orders:
            regular, _ = coefficients(radii, n, [1, 1, 1], 600.0, int(l), te)
            vals.append(abs(complex(regular[-1][1])))
        ref[ps.TE if te else ps.TM] = np.array(vals)
    return dict(x=[int(l) for l in orders], series=[
        series("tm", "|T_ℓ|, electric (TM), PyStratify", ours[ps.TM], "pystratify"), series("te", "|T_ℓ|, magnetic (TE), PyStratify", ours[ps.TE], "pystratify"),
        series("tm_ref", "electric, 60 digits", ref[ps.TM], "reference"), series("te_ref", "magnetic, 60 digits", ref[ps.TE], "reference")],
        worst=max(worst(ours[ps.TM], ref[ps.TM], relative=True), worst(ours[ps.TE], ref[ps.TE], relative=True)), worst_label="largest relative difference over 40 orders",
        note="T_ℓ = B_host / A_host of the regular solution: the scattered amplitude of each order. The coefficients fall over fifty decades across the orders shown; double precision follows the 60-digit evaluation of the same recursion all the way down.")


@figure("treams-nanoshell", "A nanoshell (10 nm silica core, 10 nm shell of ε = −10 + i): PyStratify against treams", "λ [nm]", "Q")
def treams_nanoshell():
    import treams

    lam = np.linspace(400, 900, 101)
    radii, n = [10.0, 20.0], [np.sqrt(2.13), np.sqrt(-10 + 1j + 0j), 1.0]
    sol = ps.solve(radii, np.array([n] * lam.size), lam, l_max=12)
    cs = ps.cross_sections(sol)
    area = np.pi * radii[-1] ** 2
    ext, sca = [], []
    for w in lam:
        tm = treams.TMatrix.sphere(12, 2 * np.pi / w, radii, [treams.Material(complex(v) ** 2) for v in n])
        ext.append(np.real(tm.xs_ext_avg) / area), sca.append(np.real(tm.xs_sca_avg) / area)
    return dict(x=sig(lam), series=[series("ext", "Q_ext, PyStratify", cs.q_ext, "pystratify"), series("sca", "Q_sca, PyStratify", cs.q_sca, "pystratify"),
                                    series("ext_ref", "Q_ext, treams", ext, "reference"), series("sca_ref", "Q_sca, treams", sca, "reference")],
                worst=max(worst(cs.q_ext, ext, relative=True), worst(cs.q_sca, sca, relative=True)), worst_label="largest relative difference",
                note="treams (Beutel et al.) builds the T-matrix of a layered sphere from its own vector spherical waves; the fixed metal index keeps the shell's plasmon at one wavelength, so the comparison is of the two codes, not of a material.")


# The sphere rows' closed forms, extended-precision references and independent methods (2026-10-12): each calls the
# reference its test calls (sphere_checks.py, mie_codes.py, mp_reference.py, sphere_emitter_checks.py,
# focused_checks.py, pystratify.references) over a sweep instead of at a few points.

@figure("centre-field", "The field at the centre of a sphere of m = 3 + 0.01i in air, 600 nm: Bohren & Huffman's d₁ and c₁", "size parameter x", "|field at the centre|")
def centre_field():
    from mie_codes import bohren_huffman_centre

    xs = np.linspace(0.05, 4, 200)
    m, lam = 3 + 0.01j, 600.0
    e, h, e_ref, h_ref = [], [], [], []
    for x in xs:
        sol = ps.solve([x * lam / (2 * np.pi)], np.array([[m, 1.0]]), np.array([lam]), l_max=30)
        f = ps.near_field(sol, [0.0], [0.0], [0.0])
        d1, mc1 = bohren_huffman_centre(m, x)
        e.append(complex(np.ravel(f.e["x"])[0])), h.append(complex(np.ravel(f.h["y"])[0])), e_ref.append(d1), h_ref.append(mc1)
    e, h, e_ref, h_ref = map(np.array, (e, h, e_ref, h_ref))
    rel = max(np.max(np.abs(e / e_ref - 1)), np.max(np.abs(h / h_ref - 1)))
    return dict(x=sig(xs), series=[
        series("e", "|E(0)|, PyStratify", np.abs(e), "pystratify"), series("h", "|H(0)|, PyStratify", np.abs(h), "pystratify"),
        series("e_ref", "|d₁|, Bohren & Huffman", np.abs(e_ref), "reference"), series("h_ref", "|m c₁|, Bohren & Huffman", np.abs(h_ref), "reference")],
        worst=float(rel), worst_label="largest relative difference (complex fields)",
        note="Only the order ℓ = 1 survives at the centre, so E(0) = d₁ x̂ and H(0) = m c₁ ŷ exactly (Bohren & Huffman Eq. 4.53); "
             "PyStratify sums its full near field at r = 0. The peaks are the sphere's internal resonances.")


@figure("optical-theorem", "A 50 nm silica core with a 5 nm gold shell (n = 0.27 + 2.93i) in air: extinction and scattering two ways", "λ [nm]", "Q")
def optical_theorem():
    from scipy.integrate import trapezoid

    lam = np.linspace(450, 1000, 111)
    radii, n = [50.0, 55.0], [1.45, 0.27 + 2.93j, 1.0]
    th = np.linspace(0, np.pi, 4001)
    ext, sca, ext_ref, sca_ref = [], [], [], []
    area = np.pi * radii[-1] ** 2
    for w in lam:
        sol = ps.solve(radii, n, w, l_max=20)
        cs = ps.cross_sections(sol)
        k = 2 * np.pi / w
        forward, _ = ps.scattering_amplitudes(sol, np.array([0.0]))
        p, q = ps.scattering_amplitudes(sol, th)
        ext.append(cs.q_ext[0]), sca.append(cs.q_sca[0])
        ext_ref.append(4 * np.pi / k ** 2 * np.real(-forward[0, 0]) / area)
        sca_ref.append(np.pi / k ** 2 * trapezoid((np.abs(p[0]) ** 2 + np.abs(q[0]) ** 2) * np.sin(th), th) / area)
    return dict(x=sig(lam), series=[
        series("ext", "Q_ext, sum over orders (PyStratify)", ext, "pystratify"), series("sca", "Q_sca, sum over orders (PyStratify)", sca, "pystratify"),
        series("ext_ref", "Q_ext, optical theorem (forward amplitude)", ext_ref, "reference"), series("sca_ref", "Q_sca, integrated pattern", sca_ref, "reference")],
        worst=max(worst(ext, ext_ref, relative=True), worst(sca, sca_ref, relative=True)), worst_label="largest relative difference",
        note="The extinction from the coefficients against 4π/k² Re S(0), the forward amplitude; the scattering from the coefficients "
             "against the far-field pattern integrated over 4001 angles (the trapezoid rule limits that pair to about 1e-7). "
             "The fixed gold index keeps the comparison about the method, not a material.")


@figure("rayleigh-limit", "A small sphere of m = 1.5 + 0.2i: the electric dipole coefficient against Rayleigh's", "size parameter x", "|a₁| / x³")
def rayleigh_limit():
    xs = np.linspace(0.005, 1.0, 120)
    m, lam = 1.5 + 0.2j, 614.0
    ours = np.array([ps.solve([x * lam / (2 * np.pi)], [m, 1.0], lam, l_max=3).a[0, 0] for x in xs])
    ref = -2j * xs ** 3 / 3 * (m ** 2 - 1) / (m ** 2 + 2)
    small = xs <= 0.01
    return dict(x=sig(xs), series=[series("a1", "|a₁| / x³, PyStratify", np.abs(ours) / xs ** 3, "pystratify"),
                                   series("a1_ref", "Rayleigh: (2/3) |(m² − 1)/(m² + 2)|", np.abs(ref) / xs ** 3, "reference")],
                worst=float(np.max(np.abs(ours[small] / ref[small] - 1))), worst_label="relative difference for x ≤ 0.01",
                note="Rayleigh's a₁ = −(2i/3) x³ (m² − 1)/(m² + 2) is the first term in x, so divided by x³ it is a constant: PyStratify "
                     "meets it as x → 0 and leaves it as the sphere grows (11% at x = 1); the difference falls as x² (1e-7 at x = 1e-4 in the test).")


@figure("lossless-absorbs-nothing", "Random lossless eight-shell spheres in water: what they absorb", "size parameter x", "|Q_abs| / Q_ext", log=True)
def lossless_absorbs_nothing():
    rng = np.random.default_rng(3)
    fractions, n = np.sort(rng.uniform(0.05, 1.0, 8)), np.r_[rng.uniform(1.2, 3.5, 8), 1.33]
    xs = np.linspace(0.5, 300, 150)
    lam = 614.0
    out = []
    for x in xs:
        cs = ps.cross_sections(ps.solve(fractions * x * lam / (2 * np.pi), n, lam))
        out.append(max(abs(cs.q_abs[0]) / cs.q_ext[0], 1e-17))
    return dict(x=sig(xs), series=[series("abs", "|Q_abs| / Q_ext, PyStratify", out, "pystratify")],
                worst=float(np.max(out)), worst_label="largest |Q_abs| / Q_ext",
                note="Eight shells of random real indices 1.2–3.5: absorption is the difference of extinction and scattering, so "
                     "anything above rounding would be energy the recursion invents; it stays below 1e-12 (the test's bound) to x = 300, "
                     "growing slowly with the number of orders summed; floored at 1e-17 for the logarithmic axis.")


@figure("chiral-80-digits", "A chiral shell (κ = 0.05 + 0.01i) on a 40 nm core in water, 690 nm: the T-matrix against 80-digit transfer matrices", "order ℓ", "|T|", log=True)
def chiral_80_digits():
    from mp_reference import chiral_t_matrix

    radii, n, kappa, mu = [40.0, 55.0], [1.45 + 0.01j, 1.6 + 0.02j, 1.33], [0, 0.05 + 0.01j, 0], [1, 1, 1]
    L = 40
    sol = ps.solve_chiral(radii, n, kappa, 690.0, mu, l_max=L)
    ours = np.exp(sol.log_t_helicity[0])
    ref = np.array([np.exp(np.array(chiral_t_matrix(radii, n, kappa, mu, 690.0, l), dtype=complex)) for l in range(1, L + 1)])
    orders = list(range(1, L + 1))
    names = {(0, 0): "T₊₊", (1, 1): "T₋₋", (0, 1): "T₊₋"}
    out, refs = [], []
    for (i, j), label in names.items():
        out.append(series(f"t{i}{j}", f"|{label}|, PyStratify", np.abs(ours[:, i, j]), "pystratify"))
        refs.append(series(f"t{i}{j}_ref", f"|{label}|, 80 digits", np.abs(ref[:, i, j]), "reference"))
    return dict(x=orders, series=out + refs,
                worst=float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref))), worst_label="largest |T − T₈₀| / max |T₈₀|",
                note="The helicity blocks of the host T-matrix: direct 4×4 transfer matrices in 80-digit arithmetic (mp_reference.py) "
                     "against PyStratify's scaled recursion in double precision, as the elements fall over many decades.")


@figure("graphene-sphere-plasmon", "A graphene-coated sphere (ε = 2.1 in ε = 1.77, E_F = 0.5 eV): the dipole plasmon against the quasi-static resonance", "radius [nm]", "λ [µm]")
def graphene_sphere_plasmon():
    from scipy.optimize import brentq, minimize_scalar

    e1, e2, order = 2.1, 1.77, 1
    radii = np.linspace(5, 60, 23)
    peaks, qs = [], []
    for radius in radii:
        def quasi_static(lam):
            s_ = ps.graphene_conductivity(lam, 0.5, 1e-5, 1.0)
            return (e1 * order + e2 * (order + 1) + 1j * s_ * order * (order + 1) * lam / (2 * np.pi * radius)).real

        lam_qs = brentq(quasi_static, 800, 200000)

        def minus_t(lam):
            sol = ps.solve([radius], [np.sqrt(e1), np.sqrt(e2)], lam, l_max=order, sheets={0: ps.graphene_conductivity(lam, 0.5, 1e-5, 1.0)})
            return -abs(np.exp(sol.log_t[0, 0, order - 1]))

        peaks.append(minimize_scalar(minus_t, bounds=(0.9 * lam_qs, 1.2 * lam_qs), method="bounded", options={"xatol": 1e-6 * lam_qs}).x)
        qs.append(lam_qs)
    peaks, qs = np.array(peaks) / 1000, np.array(qs) / 1000
    return dict(x=sig(radii), series=[series("peak", "resonance, PyStratify (peak of |T₁|)", peaks, "pystratify"),
                                      series("qs", "quasi-static: ε₁ℓ + ε₂(ℓ + 1) + iσℓ(ℓ + 1)/(k₀R) = 0", qs, "reference")],
                worst=float(np.abs(peaks[0] / qs[0] - 1)), worst_label="relative shift at R = 5 nm",
                note="The full solution with a conducting sheet on the sphere against the quasi-static resonance condition: they meet as R → 0 "
                     "and part by retardation, too little to see here (5e-5 at 5 nm, 9e-4 at 60 nm); the test asserts the shift stays within (kR)².")


@figure("thin-film-sheet", "A 1 nm film of ε = 15 + 5i on a 200 nm sphere of n = 3.5 in water: the film as a 2D sheet", "λ [nm]", "Q_ext")
def thin_film_sheet():
    eps_film, d, radius, core = 15.0 + 5.0j, 1.0, 200.0, 3.5
    lam = np.linspace(500, 900, 161)
    ours, ref = [], []
    for w in lam:
        film = ps.solve([radius, radius + d], [core, np.sqrt(eps_film), 1.33], w, l_max=40)
        sheet = ps.Sheet.from_film(eps_film, d, w, 1.33 ** 2)
        full = ps.solve([radius], [core, 1.33], w, l_max=40, sheets={0: sheet})
        # Q relative to the same geometric section: the sheet's sphere is d smaller
        ours.append(ps.cross_sections(full).ext[0] / (np.pi * (radius + d) ** 2)), ref.append(ps.cross_sections(film).q_ext[0])
    return dict(x=sig(lam), series=[series("sheet", "Q_ext, 2D sheet (PyStratify)", ours, "pystratify"), series("film", "Q_ext, explicit 1 nm film", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Sheet.from_film turns a thin film into surface conductivities with the normal (out-of-plane) term; its error falls as d² "
                     "(halving d quarters it, the test's check), against the film solved as a shell of its own.")


@figure("near-field-truncation", "A 50 nm gold-like sphere (n = 0.25 + 3i) in water, 600 nm: |E|² outside it against 120 orders", "distance from the centre [nm]", "|E|²", log=True)
def near_field_truncation():
    r = np.linspace(50.5, 400, 150)
    sol = ps.solve([50.0], [0.25 + 3j, 1.33], 600.0, regime="near")
    ref = ps.solve([50.0], [0.25 + 3j, 1.33], 600.0, l_max=120)
    points = (r, 0 * r, 0 * r)  # along the polarisation, where the near field is strongest
    e2 = lambda f: sum(np.abs(np.ravel(f.e[c])) ** 2 for c in "xyz")  # noqa: E731
    ours, theirs = e2(ps.near_field(sol, *points)), e2(ps.near_field(ref, *points))
    return dict(x=sig(r), series=[series("e2", f"|E|², PyStratify near-field truncation ({sol.orders[-1]} orders)", ours, "pystratify"),
                                  series("e2_ref", "|E|², 120 orders", theirs, "reference")],
                worst=worst(ours, theirs, relative=True), worst_label="largest relative difference",
                note="The far-field truncation (Wiscombe's, 5 orders here) is enough for cross sections but not at the surface, where it "
                     "misses |E|² by 3e-5; regime='near', which the calculator uses for fields, adds the orders the distance needs.")


# The papers' rows (2026-10-12): PyStratify's curve with the published values as the circles. A crosscheck
# classic (crosscheck/classics/) writes its curve and the values it read off the paper to results/; the figure
# reads both, so a page shows exactly what the classic checked.
CLASSICS = ROOT / "crosscheck" / "classics" / "results"


def classic_values(name):
    """{check: the paper's value} of a crosscheck classic's results/<name>.json."""
    return {c["check"]: c["paper"] for c in json.loads((CLASSICS / f"{name}.json").read_text())}


def classic_curve(name):
    """{column: values} of a crosscheck classic's results/<name>.csv."""
    lines = (CLASSICS / f"{name}.csv").read_text().splitlines()
    columns = lines[0].split(",")
    data = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
    return {c: data[:, i] for i, c in enumerate(columns)}


def at(x, points):
    """A reference series on the grid x, null except at the published points {x: value} (nearest grid point)."""
    out = np.full(len(x), np.nan)
    for xp, value in points.items():
        out[int(np.argmin(np.abs(np.asarray(x) - xp)))] = value
    return out


@figure("bh-reference-values", "Bohren & Huffman's BHMIE case, m = 1.55 at 0.6328 µm: the printed Q_ext and Q_back", "radius [µm]", "Q")
def bh_reference_values():
    lam, m = 0.6328, 1.55
    a = np.linspace(0.45, 0.60, 151)
    q_ext, q_back = [], []
    for radius in a:
        sol = ps.solve([radius], [m, 1.0], lam, l_max=30)
        x = 2 * np.pi * radius / lam
        q_ext.append(ps.cross_sections(sol).q_ext[0])
        q_back.append(4 * np.pi * ps.scattering_pattern(sol, np.pi).differential_cross_section[0] / (np.pi * radius ** 2))
    q_ext, q_back = np.array(q_ext), np.array(q_back)
    i = int(np.argmin(np.abs(a - 0.525)))
    return dict(x=sig(a), series=[series("ext", "Q_ext, PyStratify", q_ext, "pystratify"), series("back", "Q_back, PyStratify", q_back, "pystratify"),
                                  series("ext_ref", "Q_ext = 3.10543, printed (a = 0.525 µm)", at(a, {0.525: 3.10543}), "reference"),
                                  series("back_ref", "Q_back = 2.92534, printed", at(a, {0.525: 2.92534}), "reference")],
                worst=float(max(abs(q_ext[i] - 3.10543), abs(q_back[i] - 2.92534))), worst_label="largest difference from the printed values (5 digits)",
                note="The book's Appendix A runs BHMIE for a sphere of m = 1.55, a = 0.525 µm at 0.6328 µm and prints Q_ext and Q_back to six "
                     "figures; PyStratify's curves through that radius, Q_back from its far-field pattern at 180°.")


@figure("many-identical-layers", "A sphere of m = 1.33 + 0.001i cut into 100 identical layers [W97]", "size parameter x", "Q_ext")
def many_identical_layers():
    xs = np.linspace(1, 100, 100)
    m, lam, N = 1.33 + 0.001j, 2 * np.pi, 100
    layered, single = [], []
    for x in xs:
        layered.append(ps.cross_sections(ps.solve(np.linspace(x / N, x, N), [m] * N + [1.0], lam)).q_ext[0])
        single.append(ps.cross_sections(ps.solve([x], [m, 1.0], lam)).q_ext[0])
    return dict(x=sig(xs), series=[series("layers", "Q_ext, 100 identical layers (PyStratify)", layered, "pystratify"),
                                   series("one", "Q_ext, one sphere", single, "reference")],
                worst=worst(layered, single, relative=True), worst_label="largest relative difference",
                note="Wu & Wang's and Yang's test for a layered recursion: 99 interfaces between identical media must change nothing. "
                     "The report repeats it with 1000 layers at x = 1000, and checks graded (Kai–Massoli, Luneburg) profiles absorb nothing.")


@figure("graded-droplet-rainbow", "The rainbow of a 250 µm droplet with a graded index (128 layers, n = 1.33 → 1.36), 514.5 nm [W97]", "scattering angle [°]", "|S₁|", log=True)
def graded_droplet_rainbow():
    lam, R, N = 0.5145, 125.0, 128
    radii = np.linspace(R / N, R, N)
    n = list(1.33 + 0.03 * (radii - R / N / 2) / R) + [1.0]
    sol = ps.solve(radii, n, lam)
    more = ps.solve(radii, n, lam, l_max=sol.orders.size + 200)
    theta = np.linspace(130, 150, 401)
    ours = np.abs(ps.scattering_amplitudes(sol, np.radians(theta))[1][0])
    ref = np.abs(ps.scattering_amplitudes(more, np.radians(theta))[1][0])
    return dict(x=sig(theta), series=[series("s", f"|S₁|, PyStratify ({sol.orders.size} orders)", ours, "pystratify"),
                                      series("s_ref", f"|S₁|, {sol.orders.size + 200} orders", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="x ≈ 1527 and 128 shells: the primary rainbow near 138° and its supernumerary fringes, from PyStratify's own "
                     "truncation against 200 orders more. A truncation that stopped too early would move the fringes.")


@figure("chiral-polarisabilities", "A small chiral sphere (k a = 0.001, ε = −4 + 0.3i, μ = 1): the dipole block against Guzatov & Klimov's quasi-static polarisabilities", "chirality χ", "|T| / k³a³")
def chiral_polarisabilities():
    from sphere_checks import small_chiral_t

    chis = np.linspace(-0.2, 0.2, 81)
    eps, mu = -4.0 + 0.3j, 1.0
    scale = (1e-3) ** 3
    ours, ref = [], []
    for chi in chis:
        t, (ee, hh, eh) = small_chiral_t(eps, mu, chi)
        ours.append([t[0, 0], t[1, 1], t[0, 1]]), ref.append([ee, hh, eh])
    ours, ref = np.abs(np.array(ours)) / scale, np.abs(np.array(ref)) / scale
    names = ("T_EE (electric)", "T_HH (magnetic)", "T_EH (cross, chiral)")
    return dict(x=sig(chis), series=[series(f"t{i}", f"|{nm}|, PyStratify", ours[:, i], "pystratify") for i, nm in enumerate(names)]
                + [series(f"t{i}_ref", f"|{nm}|, Eq. 48", ref[:, i], "reference") for i, nm in enumerate(names)],
                worst=float(np.max(np.abs(ours - ref).max(axis=0) / ref.max(axis=0))), worst_label="largest difference relative to each curve's peak",
                note="Guzatov & Klimov, New J. Phys. 14, 123009 (2012), Eq. 48: the electric, magnetic and cross polarisabilities of a "
                     "Drude–Born–Fedorov sphere, against the order-1 block of PyStratify's chiral T-matrix (agreement to (ka)² = 1e-6). "
                     "The cross term vanishes without chirality; the test also covers magnetic (μ ≠ 1) spheres.")


@figure("goncalves-feibelman", "A radial dipole beside a 5 nm gold sphere (n = 0.141 + 3.152i), 614 nm, with Feibelman d-parameters: against Gonçalves et al. 2020", "distance from the centre [nm]", "Γ / Γ₀", log=True)
def goncalves_feibelman():
    from sphere_emitter_checks import AU_GONCALVES, GONCALVES_CASES, _goncalves_perp

    r = np.geomspace(5.5, 15, 25)
    out, refs, diffs = [], [], []
    for d_perp, d_par in GONCALVES_CASES:
        rates = ps.decay_rates([5.0], [AU_GONCALVES, 1.0], 614.0, r, sheets={0: ps.Feibelman(d_perp, d_par)}, tol=1e-14)
        # the series converges as (R/r)^l: 300 orders near the surface, where 120 (the test's, at 6 and 7 nm) fall short
        ref = np.array([_goncalves_perp(5.0, AU_GONCALVES, 1.0, 614.0, ri, d_perp, d_par, orders=300 if ri < 6 else 120) for ri in r])
        label = f"d⊥ = {d_perp:g}, d∥ = {d_par:g} nm"
        out.append(series(f"c{len(out)}", f"{label}, PyStratify", rates.total[:, 0], "pystratify"))
        refs.append(series(f"c{len(refs)}_ref", f"{label}, Eqs. (4a), (13a)", ref, "reference"))
        diffs.append(np.max(np.abs(rates.total[:, 0] / ref - 1)))
    return dict(x=sig(r), series=out + refs, worst=float(max(diffs)), worst_label="largest relative difference",
                note="Gonçalves et al., Nat. Commun. 11, 366 (2020): quantum surface corrections enter the Mie coefficients through the "
                     "d-parameters; their closed form summed in mpmath (40 digits, 120 orders) against PyStratify's Feibelman boundary "
                     "conditions (300 orders within 1 nm of the surface). The d-parameters shift the quenching close to the surface.")


@figure("fu-2013-kerker", "A 75 nm silicon sphere in air: forward-to-backward scattering, Fu et al. 2013, Fig. 1a", "λ [nm]", "F / B")
def fu_2013_kerker():
    curve, paper = classic_curve("fu_2013_fig1a"), classic_values("fu_2013")
    nm, ratio = curve["wavelength_nm"], curve["F_over_B"]
    points = {paper["wavelength of the F/B maximum (nm)"]: paper["F/B maximum"],
              paper["F = B, electric dipole (nm)"]: 1.0, paper["F = B, magnetic dipole (nm)"]: 1.0}
    ref = at(nm, points)
    crossings = nm[1:][np.diff(np.sign(ratio - 1)) != 0]
    window = (nm > 620) & (nm < 720)
    shifts = [abs(nm[window][np.argmax(ratio[window])] - paper["wavelength of the F/B maximum (nm)"])]
    shifts += [abs(crossings[np.argmin(abs(crossings - x))] - x) for x in (paper["F = B, electric dipole (nm)"], paper["F = B, magnetic dipole (nm)"])]
    return dict(x=sig(nm), series=[series("fb", "F/B, PyStratify (Green 2008 silicon)", ratio, "pystratify"),
                                   series("fb_ref", "printed in the paper (p. 3)", ref, "reference")],
                worst=float(max(shifts)), worst_label="largest shift of a printed feature [nm]",
                note="Fu et al., Nat. Commun. 4, 1527 (2013): F/B peaks near 8 at 660 nm (the first Kerker condition) and F = B at 500 nm "
                     "(electric dipole) and 603 nm (magnetic dipole); the paper names no silicon data, so the peak height depends on the "
                     "table chosen. From crosscheck/classics/fu_2013.py, which also checks the minimum below 0.5 near 564 nm.")


# Other groups' codes (2026-10-12). treams, scattnlay and pyGDM2 install from PyPI ([benchmarks] extra; pyGDM2
# needs NumPy < 2.4); MNPBEM runs under GNU Octave (mnpbem_octave.py); a figure whose code is missing is skipped.

@figure("treams-chiral", "A chiral shell (κ = 0.05 + 0.01i, 40/55 nm) in water: the helicity T-matrix against treams", "λ [nm]", "|T|")
def treams_chiral():
    from mie_codes import CHIRAL, chiral_blocks

    radii, n, kappa, mu, _, l_max = CHIRAL["chiral shell"]
    lam = np.linspace(400, 900, 101)
    ours, theirs = zip(*(chiral_blocks(radii, n, kappa, mu, w, l_max) for w in lam))
    ours, theirs = np.array(ours), np.array(theirs)  # (W, L, 2, 2)
    pick = (("l1pp", "ℓ = 1, T₊₊", 0, 0, 0), ("l1mm", "ℓ = 1, T₋₋", 0, 1, 1), ("l2pp", "ℓ = 2, T₊₊", 1, 0, 0), ("l2mm", "ℓ = 2, T₋₋", 1, 1, 1))
    return dict(x=sig(lam), series=[series(k, f"|{lab}|, PyStratify", np.abs(ours[:, l, i, j]), "pystratify") for k, lab, l, i, j in pick]
                + [series(f"{k}_ref", f"|{lab}|, treams", np.abs(theirs[:, l, i, j]), "reference") for k, lab, l, i, j in pick],
                worst=float(np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs))), worst_label="largest |T − T_treams| / max |T|",
                note="treams (Beutel et al.) builds a chiral multilayer's T-matrix in the helicity basis from its own vector spherical waves; "
                     "the two helicities differ because the shell is chiral. The test covers three chiral multilayers (gold, magnetic, gain in κ).")


@figure("treams-near-field", "A nanoshell (10 nm silica core, 10 nm shell of ε = −10 + i) in air, 600 nm: the scattered field outside it against treams", "distance from the centre [nm]", "|E_sca|²", log=True)
def treams_near_field():
    from mie_codes import CASES, WAVELENGTH, stratify_fields, treams_scattered_fields

    radii, n = CASES["nanoshell, 10 nm shell"]
    r = np.linspace(20.5, 60, 80)
    polar, azimuth = np.radians(30.0), np.radians(40.0)
    points = r[:, None] * np.array([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)])
    l_max = ps.truncation_order(radii[-1], n[-1], WAVELENGTH, regime="near") + 20
    sol = ps.solve(radii, np.array([n], complex), np.array([WAVELENGTH]), l_max=l_max)
    e, _ = stratify_fields(sol, points, np.real(n[-1]), incident=False)
    e_ref, _ = treams_scattered_fields(radii, n, points, l_max)
    ours, ref = np.sum(np.abs(e) ** 2, 1), np.sum(np.abs(e_ref) ** 2, 1)
    return dict(x=sig(r), series=[series("e", "|E_sca|², PyStratify", ours, "pystratify"), series("e_ref", "|E_sca|², treams", ref, "reference")],
                worst=float(np.max(np.linalg.norm(e - e_ref, axis=1) / np.linalg.norm(e, axis=1))), worst_label="largest relative difference (vector field)",
                note="Along a line at 30° from the incidence axis: treams applies its T-matrix to the expanded plane wave and evaluates the "
                     "scattered waves; PyStratify its own near field without the incident wave. The test covers six spheres, 1.05–3 radii, five angles.")


@figure("scattnlay-matryoshka", "A gold-like matryoshka (ε = −10 + i / silica / ε = −10 + i, radii in proportion 1 : 2 : 3) in water, 600 nm: against scattnlay", "outer radius [nm]", "Q")
def scattnlay_matryoshka():
    from mie_codes import CASES, scattnlay_values, stratify

    radii, n = CASES["matryoshka metal/silica/metal in water"]
    scale = np.linspace(0.3, 4.0, 120)
    ours = [stratify(list(np.array(radii) * s), n) for s in scale]
    theirs = [scattnlay_values(list(np.array(radii) * s), n) for s in scale]
    keys = ("Q_ext", "Q_sca")
    got = {k: np.array([o[k] for o in ours]) for k in keys}
    ref = {k: np.array([float(t[k]) for t in theirs]) for k in keys}
    return dict(x=sig(scale * radii[-1]), series=[series(k, f"{k}, PyStratify", got[k], "pystratify") for k in keys]
                + [series(f"{k}_ref", f"{k}, scattnlay", ref[k], "reference") for k in keys],
                worst=max(worst(got[k], ref[k], relative=True) for k in keys), worst_label="largest relative difference",
                note="scattnlay (Peña & Pal; Ladutenko et al.) evaluates the multilayer Mie coefficients by its own recursion; both at the "
                     "default truncation. The test covers ten spheres to x = 1000, in double and multiple precision.")


@figure("scattnlay-pattern", "A lossless water-like sphere (m = 1.33) at x = 200: the scattering amplitudes against scattnlay", "scattering angle [°]", "|S|", log=True)
def scattnlay_pattern():
    from mie_codes import CASES, EXTRA_ORDERS, WAVELENGTH, scattnlay_values

    radii, n = CASES["large lossless, x = 200"]
    angles = np.linspace(0, 180, 721)
    l_max = ps.truncation_order(radii[-1], n[-1], WAVELENGTH) + EXTRA_ORDERS
    sol = ps.solve(radii, np.array([n], complex), np.array([WAVELENGTH]), l_max=l_max)
    s1, s2, _, _ = ps.amplitude_matrix(sol, np.radians(angles))
    theirs = scattnlay_values(radii, n, angles_deg=angles)
    ours = {"S1": np.abs(s1[0]), "S2": np.abs(s2[0])}
    ref = {k: np.abs(np.asarray(theirs[k])) for k in ours}
    return dict(x=sig(angles), series=[series(k, f"|{k}|, PyStratify", ours[k], "pystratify") for k in ours]
                + [series(f"{k}_ref", f"|{k}|, scattnlay", ref[k], "reference") for k in ours],
                worst=max(float(np.max(np.abs(ours[k] - ref[k]) / np.max(ref[k]))) for k in ours), worst_label="largest difference relative to the forward peak",
                note="Six decades from the forward peak to the backward glory, with the rainbow near 138°; PyStratify 20 orders beyond "
                     "Wiscombe's truncation, scattnlay at its own.")


@figure("scattnlay-near-field", "The matryoshka (ε = −10 + i / silica / ε = −10 + i, 10/20/30 nm) in water, 600 nm: |E|² through every layer against scattnlay", "x [nm]", "|E|²", log=True)
def scattnlay_near_field():
    from mie_codes import CASES, EXTRA_ORDERS, WAVELENGTH, scattnlay_fields, stratify_fields

    radii, n = CASES["matryoshka metal/silica/metal in water"]
    x = np.linspace(-60, 60, 241)
    points = np.column_stack([x, np.full_like(x, 1.0), np.zeros_like(x)])  # along the polarisation, 1 nm off the axis
    l_max = ps.truncation_order(radii[-1], n[-1], WAVELENGTH, regime="near") + EXTRA_ORDERS
    sol = ps.solve(radii, np.array([n], complex), np.array([WAVELENGTH]), l_max=l_max)
    e, _ = stratify_fields(sol, points, np.real(n[-1]))
    e_ref, _ = scattnlay_fields(radii, n, points, l_max)
    ours, ref = np.sum(np.abs(e) ** 2, 1), np.sum(np.abs(np.asarray(e_ref)) ** 2, 1)
    return dict(x=sig(x), series=[series("e", "|E|², PyStratify", ours, "pystratify"), series("e_ref", "|E|², scattnlay (fieldnlay)", ref, "reference")],
                worst=float(np.max(np.linalg.norm(e - e_ref, axis=1) / np.linalg.norm(e, axis=1))), worst_label="largest relative difference (vector field)",
                note="Through the gold core, the silica spacer, the gold shell and out into the water, along the polarisation. scattnlay's "
                     "fields are double precision; in the 10-layer graded sphere of the report it returns nan in the three inner layers, "
                     "where PyStratify's logarithmic coefficients stay finite (shown there, not asserted).")


@figure("gdm-spectrum", "An absorbing dielectric sphere (n = 2 + 0.1i, R = 40 nm) in air: cross sections against pyGDM2's volume discretization", "λ [nm]", "C [nm²]")
def gdm_spectrum():
    from gdm_spheres import gdm_simulation
    from pyGDM2 import linear

    lam = np.linspace(400, 800, 17)
    simulation, dipoles, r_eff = gdm_simulation(40.0, 2.0 + 0.1j, 6.0, lam)
    theirs = np.array([[float(v) for v in linear.extinct(simulation, i)][:2] for i in range(len(lam))])
    dense = np.linspace(400, 800, 161)
    cs = ps.cross_sections(ps.solve([r_eff], np.array([[2.0 + 0.1j, 1.0]] * dense.size), dense, l_max=20))
    area = np.pi * r_eff ** 2
    ext_ref, sca_ref = np.full(dense.size, np.nan), np.full(dense.size, np.nan)
    ext_ref[::10], sca_ref[::10] = theirs[:, 0], theirs[:, 1]
    ours_ext, ours_sca = cs.q_ext * area, cs.q_sca * area
    return dict(x=sig(dense), series=[series("ext", "C_ext, PyStratify", ours_ext, "pystratify"), series("sca", "C_sca, PyStratify", ours_sca, "pystratify"),
                                      series("ext_ref", f"C_ext, pyGDM2 ({len(dipoles)} dipoles)", ext_ref, "reference"),
                                      series("sca_ref", f"C_sca, pyGDM2 ({len(dipoles)} dipoles)", sca_ref, "reference")],
                worst=float(max(np.max(np.abs(theirs[:, 0] / ours_ext[::10] - 1)), np.max(np.abs(theirs[:, 1] / ours_sca[::10] - 1)))),
                worst_label="largest relative difference (the lattice's error at this mesh)",
                note="pyGDM2 (Wiecha) fills the sphere with point dipoles on a hexagonal lattice and solves their coupling; PyStratify is "
                     "evaluated for the sphere of the same volume (r_eff = %.2f nm). A volume lattice stays ~3%% high at the meshes this "
                     "machine affords; the test asserts a band, and the report follows four meshes." % r_eff)


@figure("mnpbem-spectrum", "A 40 nm sphere of ε = −10 + i in vacuum: cross sections against MNPBEM's boundary elements", "λ [nm]", "C [nm²]")
def mnpbem_spectrum_figure():
    from mnpbem_spheres import METAL, mnpbem_spectrum

    lam = np.linspace(400, 800, 161)
    faces, ext, sca = mnpbem_spectrum("sphere in vacuum", 1024, lam[::10])
    cs = ps.cross_sections(ps.solve([20.0], np.array([[np.sqrt(METAL + 0j), 1.0]] * lam.size), lam))
    area = np.pi * 20.0 ** 2
    ext_ref, sca_ref = np.full(lam.size, np.nan), np.full(lam.size, np.nan)
    ext_ref[::10], sca_ref[::10] = ext, sca
    return dict(x=sig(lam), series=[series("ext", "C_ext, PyStratify", cs.q_ext * area, "pystratify"), series("sca", "C_sca, PyStratify", cs.q_sca * area, "pystratify"),
                                    series("ext_ref", f"C_ext, MNPBEM ({faces} faces)", ext_ref, "reference"), series("sca_ref", f"C_sca, MNPBEM ({faces} faces)", sca_ref, "reference")],
                worst=float(max(np.max(np.abs(ext / (cs.q_ext[::10] * area) - 1)), np.max(np.abs(sca / (cs.q_sca[::10] * area) - 1)))),
                worst_label="largest relative difference (the mesh's error)",
                note="MNPBEM (Hohenester & Trügler) discretizes the surface into triangles and solves the retarded boundary integral "
                     "equations; a constant permittivity, so both codes see the same ε at every wavelength. Percent-level by design: the "
                     "test checks the difference shrinks as the mesh is refined (0.45–4.4% at 2044 faces).")


@figure("mnpbem-thin-nanoshell", "A thin nanoshell (15 nm silica core, 5 nm shell of ε = −10 + i) in vacuum: three Mie codes and MNPBEM", "λ [nm]", "C_ext [nm²]")
def mnpbem_thin_nanoshell():
    from mie_codes import treams_values, scattnlay_values
    import mie_codes
    from mnpbem_spheres import METAL, SILICA, mnpbem_spectrum

    radii, n = [15.0, 20.0], [np.sqrt(SILICA + 0j), np.sqrt(METAL + 0j), 1.0]
    lam = np.linspace(400, 800, 161)
    cs = ps.cross_sections(ps.solve(radii, np.array([n] * lam.size), lam))
    area = np.pi * radii[-1] ** 2
    ours = cs.q_ext * area
    t_ref, s_ref = np.full(lam.size, np.nan), np.full(lam.size, np.nan)
    for i in range(0, lam.size, 10):
        mie_codes.WAVELENGTH = lam[i]  # the Mie wrappers read the module's wavelength
        t_ref[i] = float(treams_values(radii, n)["Q_ext"]) * area
        s_ref[i] = float(scattnlay_values(radii, n)["Q_ext"]) * area
    mie_codes.WAVELENGTH = 600.0
    faces, ext, _ = mnpbem_spectrum("thin nanoshell", 256, lam[5::10])
    m_ref = np.full(lam.size, np.nan)
    m_ref[5::10] = ext
    mie = max(np.nanmax(np.abs(t_ref / ours - 1)), np.nanmax(np.abs(s_ref / ours - 1)))
    return dict(x=sig(lam), series=[series("ext", "C_ext, PyStratify", ours, "pystratify"),
                                    series("treams", "C_ext, treams", t_ref, "reference"), series("scattnlay", "C_ext, scattnlay", s_ref, "reference"),
                                    series("mnpbem", f"C_ext, MNPBEM ({faces} faces)", m_ref, "reference")],
                worst=float(mie), worst_label="largest relative difference among the Mie codes",
                note="Shown, not asserted. The three Mie codes agree to all printed digits; MNPBEM, with elements of 2–4 nm on surfaces 5 nm "
                     "apart, stays about 12% low and barely moves with the mesh: a resolution limit of the boundary elements at the "
                     "meshes this machine affords, not a disagreement about the physics.")


# --- spheres, point dipole --------------------------------------------------------------------------------------

@figure("sphere-decay-mpmath", "A dipole beside a 25 nm sphere of m = 0.05 + 3.5i at 600 nm: decay rates against the 40-digit series", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def sphere_decay_mpmath():
    from pystratify.references import sphere_decay_rates

    gaps = np.geomspace(1, 300, 36)
    radius, m, lam = 25.0, 0.05 + 3.5j, 600.0
    r = radius + gaps
    ours = ps.decay_rates([radius], np.array([m, 1.0]), lam, r, tol=1e-10)
    ref = np.array([sphere_decay_rates(radius, m, lam, rr, orders=400, dps=40) for rr in r])
    return dict(x=sig(gaps), series=[
        series("tot_perp", "total, radial, PyStratify", ours.total[:, 0], "pystratify"), series("tot_par", "total, tangential, PyStratify", ours.total[:, 1], "pystratify"),
        series("rad_perp", "radiative, radial, PyStratify", ours.radiative[:, 0], "pystratify"), series("rad_par", "radiative, tangential, PyStratify", ours.radiative[:, 1], "pystratify"),
        series("tot_perp_ref", "total, radial, 40 digits", ref[:, 0], "reference"), series("tot_par_ref", "total, tangential, 40 digits", ref[:, 1], "reference"),
        series("rad_perp_ref", "radiative, radial, 40 digits", ref[:, 2], "reference"), series("rad_par_ref", "radiative, tangential, 40 digits", ref[:, 3], "reference")],
        worst=worst(np.column_stack([ours.total, ours.radiative]), ref, relative=True), worst_label="largest relative difference over the four rates",
        note="Chew's sums over 400 orders in mpmath at 40 digits against PyStratify's adaptive order and logarithmic amplitudes; the total rate climbs as d⁻³ at contact (quenching) while the radiative rate stays finite.")


@figure("treams-dipole-nanoshell", "A dipole 5–60 nm from a nanoshell (10 nm silica core, 10 nm shell of ε = −10 + i), 600 nm: PyStratify against treams", "gap [nm]", "Γ / Γ₀", log=True)
def treams_dipole_nanoshell():
    import treams

    gaps = np.geomspace(5, 60, 24)
    radii, n = [0.010, 0.020], [np.sqrt(2.13), np.sqrt(-10 + 1j + 0j), 1.0]
    lam, l_max = 0.6, 50
    k0 = 2 * np.pi / lam
    materials = [treams.Material(complex(v) ** 2) for v in n]
    tmatrix = treams.TMatrix.sphere(l_max, k0, radii, materials, poltype="parity")
    ours, ref = [], []
    for gap in gaps:
        problem = ps.Problem("spheres", radii, n, lam, ps.PointDipole(radii[-1] + gap * 1e-3, "electric", len(radii)), 1e-10)
        ours.append(np.asarray(ps.solve_problem(problem)["total"][:2]))
        z0 = radii[-1] + gap * 1e-3
        source = treams.SphericalWaveBasis.default(1, positions=[[0, 0, z0]])
        rates = []
        for m, axis in ((0, 2), (1, 0)):
            c = np.zeros(len(source), complex)
            if m == 0:
                c[(source.m == 0) & (source.pol == 1)] = 1
            else:
                c[(source.m == -1) & (source.pol == 1)], c[(source.m == 1) & (source.pol == 1)] = 1, -1
            common = dict(basis=source, k0=k0, material=materials[-1], poltype="parity")
            own, regular = (treams.PhysicsArray(c, modetype=t, **common) for t in ("singular", "regular"))
            scattered = tmatrix @ own.expand(tmatrix.basis, "regular")
            at = np.array([[0, 0, z0]])
            e_s, e_j = np.asarray(scattered.efield(at))[0, axis], np.asarray(regular.efield(at))[0, axis]
            rates.append(1 + np.real(e_s / e_j))
        ref.append(rates)
    ours, ref = np.array(ours), np.array(ref)
    return dict(x=sig(gaps), series=[
        series("perp", "radial, PyStratify", ours[:, 0], "pystratify"), series("par", "tangential, PyStratify", ours[:, 1], "pystratify"),
        series("perp_ref", "radial, treams", ref[:, 0], "reference"), series("par_ref", "tangential, treams", ref[:, 1], "reference")],
        worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
        note="treams translates the dipole's ℓ = 1 singular wave to the sphere's centre, scatters it with the T-matrix (50 orders; closer than 5 nm its truncation no longer converges, PyStratify's adaptive order does) and reads the scattered field back at the dipole; PyStratify places the jump in its transfer-matrix recursion.")


@figure("silver-decay-mie", "A dipole 1–100 nm from a 50 nm silver sphere (n = 0.093 + 4i), 633 nm: the decay rates against Mie's series in 40 digits", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def silver_decay_mie():
    from pystratify.references import sphere_decay_rates

    radius, ag, lam = 50.0, 0.093 + 4j, 633.0
    gaps = np.geomspace(1, 100, 41)
    ours = ps.decay_rates([radius], [ag, 1.0], lam, radius + gaps, tol=1e-10)
    ref = np.array([sphere_decay_rates(radius, ag, lam, radius + g, orders=int(min(1400, 40 + 30 * radius / g))) for g in gaps])
    got = np.column_stack([ours.total, ours.radiative])
    return dict(x=sig(gaps), series=[
        series("tot_perp", "total, radial, PyStratify", ours.total[:, 0], "pystratify"), series("tot_par", "total, tangential, PyStratify", ours.total[:, 1], "pystratify"),
        series("rad_perp", "radiative, radial, PyStratify", ours.radiative[:, 0], "pystratify"), series("rad_par", "radiative, tangential, PyStratify", ours.radiative[:, 1], "pystratify"),
        series("tot_perp_ref", "total, radial, Mie (40 digits)", ref[:, 0], "reference"), series("tot_par_ref", "total, tangential, Mie (40 digits)", ref[:, 1], "reference"),
        series("rad_perp_ref", "radiative, radial, Mie (40 digits)", ref[:, 2], "reference"), series("rad_par_ref", "radiative, tangential, Mie (40 digits)", ref[:, 3], "reference")],
        worst=worst(got, ref, relative=True), worst_label="largest relative difference over the four rates",
        note="Majic & Le Ru's Eqs. 34–37 summed in mpmath (up to 1400 orders at 1 nm) against PyStratify's adaptive order; "
             "at contact the total rate is quenching, the nonradiative part, while the radiative rate stays finite.")


@figure("nanoshell-decay-layers", "An electric dipole inside and outside a nanoshell (50 nm silica core, 5 nm gold shell, in water), 614 nm: the rates against 60-digit transfer matrices", "distance from the centre [nm]", "Γ / Γ₀", log=True)
def nanoshell_decay_layers():
    from pystratify.references import layered_decay_rates

    radii, n, lam = [50.0, 55.0], [1.45, 0.27 + 2.93j, 1.33], 614.0
    r = np.r_[np.linspace(5, 48, 22), np.geomspace(57, 150, 22)]  # not in the gold: an emitter in an absorbing shell has no rate
    ours = ps.decay_rates(radii, n, lam, r, normalization="shell", tol=1e-10)

    def orders(rd):
        gap = min(abs(rd - v) for v in radii)
        return int(min(1200, 40 + 30 * radii[-1] / gap))

    ref = np.array([np.concatenate(layered_decay_rates(radii, n, lam, rd, orders(rd), 60)) for rd in r])
    got = np.column_stack([ours.total, ours.radiative])
    return dict(x=sig(r), series=[
        series("tot_perp", "total, radial, PyStratify", ours.total[:, 0], "pystratify"), series("tot_par", "total, tangential, PyStratify", ours.total[:, 1], "pystratify"),
        series("tot_perp_ref", "total, radial, 60 digits", ref[:, 0], "reference"), series("tot_par_ref", "total, tangential, 60 digits", ref[:, 1], "reference")],
        worst=worst(got, ref, relative=True), worst_label="largest relative difference (total and radiative)",
        note="Rates relative to the dipole in the layer it sits in (the silica core or the water), against transfer "
             "matrices evaluated in 60-digit arithmetic. The test also asserts 0.5–1 nm from the gold, with 1250–2400 orders.")


@figure("silicon-magnetic-decay", "A magnetic dipole outside a 100 nm silicon sphere (n = 3.7 + 0.005i), 800 nm: the rates against 60-digit transfer matrices", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def silicon_magnetic_decay():
    from pystratify.references import layered_decay_rates

    radius, n, lam = 100.0, [3.7 + 0.005j, 1.0], 800.0
    gaps = np.geomspace(2, 300, 41)
    ours = ps.decay_rates([radius], n, lam, radius + gaps, [1, 1], dipole="magnetic", normalization="shell", tol=1e-10)
    ref = np.array([np.concatenate(layered_decay_rates([radius], n, lam, radius + g, int(min(800, 40 + 30 * radius / g)), 60, mu=[1, 1], dipole="magnetic"))
                    for g in gaps])
    got = np.column_stack([ours.total, ours.radiative])
    return dict(x=sig(gaps), series=[
        series("tot_perp", "total, radial, PyStratify", ours.total[:, 0], "pystratify"), series("tot_par", "total, tangential, PyStratify", ours.total[:, 1], "pystratify"),
        series("tot_perp_ref", "total, radial, 60 digits", ref[:, 0], "reference"), series("tot_par_ref", "total, tangential, 60 digits", ref[:, 1], "reference")],
        worst=worst(got, ref, relative=True), worst_label="largest relative difference (total and radiative)",
        note="A magnetic dipole couples to the sphere's magnetic Mie mode; the reference is the same transfer-matrix formulation in 60-digit arithmetic.")


@figure("radiated-power-rates", "An emitter in a magnetic multilayer (50/70/90 nm; μ = 1.3, 1, 1.7 in μ = 1.1), 614 nm: far-field power against the radiative rate", "distance from the centre [nm]", "Γ_rad / Γ₀", log=True)
def radiated_power_rates():
    radii, n, mu = [50.0, 70.0, 90.0], [1.45, 0.27 + 2.93j, 1.6, 1.33], [1.3, 1.0, 1.7, 1.1]
    r = np.r_[np.linspace(5, 47, 12), np.linspace(73, 87, 6), np.geomspace(93, 400, 20)]
    out = {}
    for dipole in ("electric", "magnetic"):
        rates = ps.decay_rates(radii, n, 614.0, r, mu, dipole=dipole, tol=1e-10)
        power = np.array([ps.dipole_far_field(radii, n, 614.0, [0.0, 0.0, r0], [0.0, 0.0, 1.0], 0.0, 0.0, mu, dipole=dipole).power for r0 in r])
        out[dipole] = rates.radiative[:, 0], power
    return dict(x=sig(r), series=[
        series("e", "electric, radiative rate (PyStratify)", out["electric"][0], "pystratify"), series("m", "magnetic, radiative rate (PyStratify)", out["magnetic"][0], "pystratify"),
        series("e_ref", "electric, power in the far field", out["electric"][1], "reference"), series("m_ref", "magnetic, power in the far field", out["magnetic"][1], "reference")],
        worst=max(worst(*out["electric"], relative=True), worst(*out["magnetic"], relative=True)), worst_label="largest relative difference",
        note="Two routes to the same number for a radial dipole: the radiative rate from the decay-rate sums, and the power of the "
             "far field the dipole radiates through every layer. The gaps in r are the gold shell, where an emitter is not placed.")


@figure("two-point-green", "Points 56 and 60 nm from the centre of a gold nanoshell (50 nm silica core, 5 nm gold) in water, 614 nm: the two-point Green's dyadic per order against 80 digits", "order ℓ", "|g_ℓ|", log=True)
def two_point_green():
    from pystratify.solver import TE, TM
    from sphere_emitter_checks import two_point_terms

    terms = two_point_terms(56.0, 60.0)
    l = np.arange(1, 121)
    (tm, tm_ref), (te, te_ref) = terms[TM], terms[TE]
    w = 2 * l + 1
    return dict(x=[int(v) for v in l], series=[
        series("tm", "TM, PyStratify", np.abs(tm), "pystratify"), series("te", "TE, PyStratify", np.abs(te), "pystratify"),
        series("tm_ref", "TM, 80 digits", np.abs(tm_ref), "reference"), series("te_ref", "TE, 80 digits", np.abs(te_ref), "reference")],
        worst=float(max(np.sum(w * np.abs(tm - tm_ref)) / np.sum(w * np.abs(tm_ref)), np.sum(w * np.abs(te - te_ref)) / np.sum(w * np.abs(te_ref)))),
        worst_label="Σ(2ℓ+1)|g − g₈₀| / Σ(2ℓ+1)|g₈₀|",
        note="The radial part of the Green's dyadic between two points, per order and polarisation, which every dipole–dipole coupling near the sphere "
             "is built from: PyStratify's normalized form against transfer matrices in 80-digit arithmetic.")


@figure("quadrupole-coupling", "A quadrupole 47 nm from the centre (k = 0.013 nm⁻¹): its coupling to each multipole order against finite differences", "order ℓ", "|coupling|", log=True)
def quadrupole_coupling_figure():
    from sphere_emitter_checks import quadrupole_values

    orders = list(range(1, 9))
    rows = {key: ([], []) for key in (("psi", 1), ("xi", 1))}  # M - N has the same magnitudes for the axial family
    for l in orders:
        values = quadrupole_values(l)
        for (kind, sign), (ours, ref) in rows.items():
            got, exact = values[kind, (0, "e"), sign]
            ours.append(abs(got)), ref.append(abs(exact))
    label = {("psi", 1): "regular wave (ψ)", ("xi", 1): "outgoing wave (ξ)"}
    out = [series(f"{k}{s}", f"{label[k, s]}, PyStratify", rows[k, s][0], "pystratify") for k, s in rows]
    ref = [series(f"{k}{s}_ref", f"{label[k, s]}, finite differences", rows[k, s][1], "reference") for k, s in rows]
    every = [abs(g - r) / max(abs(r), 1e-300) for l in orders for g, r in quadrupole_values(l).values()]
    return dict(x=orders, series=out + ref, worst=float(max(every)), worst_label="largest relative difference over every family",
                note="(1/6) Σ Q_ij ∂_j W_i of the vector spherical waves W = M + N (Bohren & Huffman's, the axial m = 0 family shown; the "
                     "difference covers all five families and M − N), PyStratify's closed-form functional against central differences "
                     "of the 40-digit waves. The regular wave fades with order at this distance and the outgoing one grows.")


@figure("casimir-non-retarded", "An atom near a 200 nm Drude sphere: the Casimir–Polder potential against the non-retarded limit −C₃/d³", "distance from the surface [nm]", "|U| [J]", log=True)
def casimir_non_retarded():
    from scipy.integrate import quad
    from sphere_emitter_checks import _alpha, _eps
    from pystratify.casimir import HBAR

    f = lambda u: (lambda x: x * _alpha(x) * (_eps(x)[0] - 1) / (_eps(x)[0] + 1))(np.exp(u))  # noqa: E731
    c3 = HBAR / (4 * np.pi) * quad(f, np.log(1e6), np.log(1e21), limit=500, epsabs=0, epsrel=1e-12)[0]
    R = 200.0
    d = np.geomspace(0.5, 50, 25)
    out = ps.casimir_polder([R], _eps, R + d, _alpha, nodes=64)
    ref = c3 / d ** 3
    return dict(x=sig(d), series=[series("u", "|U|, PyStratify", np.abs(out.energy), "pystratify"), series("u_ref", "C₃/d³, non-retarded limit", ref, "reference")],
                worst=float(np.max(np.abs(-out.energy[:2] * d[:2] ** 3 / c3 - 1))), worst_label="relative difference within 0.6 nm",
                note="Within a nanometre the potential is the flat-surface, non-retarded −C₃/d³ with C₃ = (ħ/4π)∫α(iξ)(ε − 1)/(ε + 1) dξ; "
                     "further out retardation and the sphere's curvature weaken it, so the curves part by design: 0.3% at 0.5 nm, "
                     "3% at 5 nm, 37% at 50 nm.")


@figure("layered-magnetic-balance", "A dipole in the shell of a layered magnetic sphere (radii 1, 11; (ε, μ) = (2, 8), (1, 1), (8, 2); k₀ = 1) [Y24]: total against radiative", "distance from the centre", "Γ / Γ₀", log=True)
def layered_magnetic_balance():
    eps, mu = np.array([2, 1, 8.0]), np.array([8, 1, 2.0])
    r = np.geomspace(1.01, 10.99, 60)
    rates = ps.decay_rates([1.0, 11.0], np.sqrt(eps * mu), 2 * np.pi, r, mu=mu, tol=1e-9, normalization="shell")
    return dict(x=sig(r), series=[
        series("tot_perp", "total, radial (PyStratify, from the LDOS)", rates.total[:, 0], "pystratify"),
        series("tot_par", "total, tangential (PyStratify, from the LDOS)", rates.total[:, 1], "pystratify"),
        series("rad_perp", "radiative, radial (power reaching infinity)", rates.radiative[:, 0], "reference"),
        series("rad_par", "radiative, tangential (power reaching infinity)", rates.radiative[:, 1], "reference")],
        worst=worst(rates.total, rates.radiative, relative=True), worst_label="largest relative difference",
        note="Yuan, Zhu & Zhu's Table 1 configuration: every layer is lossless, so all the power the dipole emits reaches infinity "
             "and the total rate (from the local density of states) must equal the radiative one. Close to an interface the total "
             "carries the rounding of the reactive near field, as Majic & Le Ru note (the report: 5.5e-8 at 0.001 from it).")


@figure("guzatov-chiral-molecule", "A chiral molecule beside a small chiral sphere (ε = −4 + 0.3i, χ = 0.1, k₀a = 0.01): its radiative rate against Guzatov & Klimov's Eq. 46", "distance from the centre / radius", "Γ_rad / Γ₀")
def guzatov_chiral_molecule():
    from sphere_emitter_checks import guzatov_klimov_rate

    d = np.linspace(1.3, 4, 55)
    out, refs, diffs = [], [], []
    for xi in (0.1, -0.1):
        pairs = np.array([guzatov_klimov_rate(-4.0 + 0.3j, 1.0, 0.1, 0.01, di, xi) for di in d])
        label = f"ξ = {xi:+g}"
        out.append(series(f"x{len(out)}", f"{label}, PyStratify", pairs[:, 0], "pystratify"))
        refs.append(series(f"x{len(refs)}_ref", f"{label}, Eq. 46", pairs[:, 1], "reference"))
        diffs.append(np.max(np.abs(pairs[:, 0] / pairs[:, 1] - 1)))
    return dict(x=sig(d), series=out + refs, worst=float(max(diffs)), worst_label="largest relative difference",
                note="Guzatov & Klimov, New J. Phys. 14, 123009 (2012): a molecule with electric moment d₀ and magnetic moment −iξd₀ "
                     "(the two handednesses, ξ = ±0.1) beside a chiral plasmonic sphere radiates at different rates: chiral discrimination. "
                     "Their quasi-static Eq. 46 against PyStratify's full chiral solution; agreement to O((k₀a)²).")


@figure("anger-2006-excitation", "A molecule beside an 80 nm gold sphere (ε = −12.99 + 1.09i), 650 nm: excitation and quantum yield, Anger et al. 2006, Fig. 1a", "distance from the surface [nm]", "rate / free space")
def anger_2006_excitation():
    curve, paper = classic_curve("anger_2006_fig1"), classic_values("anger_2006")
    z = curve["z_um"] * 1e3
    exc, qy = curve["excitation d=80nm"], curve["quantum_yield d=80nm"]
    pick = lambda what: {float(m.group(1)): v for k, v in paper.items() if (m := __import__("re").match(rf"80 nm: {what} at z = (\d+) nm", k))}  # noqa: E731
    exc_ref, qy_ref = at(z, pick("excitation")), at(z, pick("quantum yield"))
    diffs = [abs(a - b) / b for a, b in zip(np.r_[exc, qy], np.r_[exc_ref, qy_ref]) if np.isfinite(b)]
    return dict(x=sig(z), series=[series("exc", "excitation |E|²/|E₀|², PyStratify", exc, "pystratify"), series("qy", "quantum yield, PyStratify", qy, "pystratify"),
                                  series("exc_ref", "excitation, read off Fig. 1a", exc_ref, "reference"), series("qy_ref", "quantum yield, read off Fig. 1a", qy_ref, "reference")],
                worst=float(max(diffs)), worst_label="largest relative difference from the read values (±5%)",
                note="Anger, Bharadwaj & Novotny, Phys. Rev. Lett. 96, 113002 (2006): the excitation rate grows toward the gold and the "
                     "quantum yield falls (quenching); values read off Fig. 1a to ±5%, from crosscheck/classics/anger_2006.py.")


@figure("anger-2006-fluorescence", "Fluorescence beside gold spheres of 20, 60, 80 and 100 nm, 650 nm: Anger et al. 2006, Fig. 1b", "distance from the surface [nm]", "fluorescence / free space")
def anger_2006_fluorescence():
    curve, paper = classic_curve("anger_2006_fig1"), classic_values("anger_2006")
    z = curve["z_um"] * 1e3
    out = [series(f"d{d}", f"{d} nm sphere, PyStratify", curve[f"fluorescence d={d}nm"], "pystratify") for d in (20, 60, 80, 100)]
    peak = at(z, {paper["80 nm: z of the maximum (nm)"]: paper["80 nm: fluorescence maximum"]})
    fl80 = curve["fluorescence d=80nm"]
    i = int(np.argmax(fl80))
    return dict(x=sig(z), series=out + [series("peak_ref", "80 nm maximum, read off Fig. 1b (z ≈ 10 nm)", peak, "reference")],
                worst=float(abs(fl80[i] / paper["80 nm: fluorescence maximum"] - 1)), worst_label="80 nm maximum, relative difference",
                note="The product of excitation and quantum yield peaks near 10 nm: enhancement turns into quenching. The paper prints the "
                     "maxima of all four sizes (5.3, 3.5, 2.3, 1.05 from the largest; PyStratify 5.27, 3.52, 2.20, 1.04) but the position "
                     "only for 80 nm, so that is the point marked.")


@figure("gdm-decay", "A dipole beside a dielectric sphere (n = 2, R = 40 nm) in air, 600 nm: total decay rates against pyGDM2", "distance from the surface [nm]", "Γ / Γ₀")
def gdm_decay():
    from gdm_spheres import gdm_rates, gdm_simulation

    gaps = np.linspace(10, 100, 10)
    simulation, dipoles, r_eff = gdm_simulation(40.0, 2.0, 6.0)
    radial, tangential = gdm_rates(simulation, dipoles, 40.0, gaps)
    dense = np.linspace(10, 100, 91)
    rates = ps.decay_rates([r_eff], [2.0, 1.0], 600.0, 40.0 + dense)
    rad_ref, tan_ref = np.full(dense.size, np.nan), np.full(dense.size, np.nan)
    rad_ref[::10], tan_ref[::10] = radial, tangential
    rel = max(np.max(np.abs(radial / rates.total[::10, 0] - 1)), np.max(np.abs(tangential / rates.total[::10, 1] - 1)))
    return dict(x=sig(dense), series=[series("rad", "radial, PyStratify", rates.total[:, 0], "pystratify"), series("tan", "tangential, PyStratify", rates.total[:, 1], "pystratify"),
                                      series("rad_ref", f"radial, pyGDM2 ({len(dipoles)} dipoles)", rad_ref, "reference"),
                                      series("tan_ref", f"tangential, pyGDM2 ({len(dipoles)} dipoles)", tan_ref, "reference")],
                worst=float(rel), worst_label="largest relative difference",
                note="pyGDM2's decay_rate couples the emitter to the dipole lattice through the Green dyadic; distances from the centroid of "
                     "the lattice, PyStratify for the sphere of the same volume. The lattice is coarse close to the surface (the test: 10% at "
                     "20 nm, 0.5% at 40 nm).")


@figure("mnpbem-decay", "A dipole 5–40 nm from a 40 nm sphere of ε = −10 + i in vacuum, 600 nm: decay rates against MNPBEM", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def mnpbem_decay():
    from mnpbem_spheres import METAL, mnpbem_rates

    gaps = np.geomspace(5, 40, 8)  # from the test's 5 nm: closer, the triangles (~2 nm) no longer resolve the near field
    faces, total, radiative = mnpbem_rates(1024, gaps)
    dense = np.geomspace(5, 40, 71)
    rates = ps.decay_rates([20.0], [np.sqrt(METAL + 0j), 1.0], 600.0, 20.0 + dense)
    def sparse(values):
        out = np.full(dense.size, np.nan)
        out[::10] = values
        return out
    rel = np.max(np.abs(total / rates.total[::10] - 1))
    return dict(x=sig(dense), series=[
        series("tot_rad", "total, radial, PyStratify", rates.total[:, 0], "pystratify"), series("tot_tan", "total, tangential, PyStratify", rates.total[:, 1], "pystratify"),
        series("rad_rad", "radiative, radial, PyStratify", rates.radiative[:, 0], "pystratify"),
        series("tot_rad_ref", f"total, radial, MNPBEM ({faces} faces)", sparse(total[:, 0]), "reference"),
        series("tot_tan_ref", "total, tangential, MNPBEM", sparse(total[:, 1]), "reference"),
        series("rad_rad_ref", "radiative, radial, MNPBEM", sparse(radiative[:, 0]), "reference")],
        worst=float(rel), worst_label="largest relative difference in the total rates (the mesh's error)",
        note="MNPBEM's dipoleret solves the boundary integral equations for the dipole's field; its triangles are ~2 nm, so the "
             "difference grows toward the surface (33% for the radial rate at 2 nm, not shown). The test follows four meshes at 5 nm (1.4% radial, 0.04% tangential at 2044 faces).")


@figure("meep-axial-sphere", "A dipole 100 nm outside an n = 2, 150 nm sphere, on its axis: total decay rate against Meep (FDTD, 100 px/µm)", "λ [µm]", "Γ / Γ₀", env="meep")
def meep_axial_sphere():
    import meep as mp
    from fdtd_meep import AXIAL_CASES, meep_ldos_axial

    case = "sphere n = 2, R = 150 nm, dipole 100 nm outside on its axis"
    _, dimensions, n, source, names = AXIAL_CASES[case]
    mp.verbosity(0)
    coarse = 1 / np.linspace(0.85, 1.65, 17)
    theirs = {o: meep_ldos_axial(case, o, 100, wavelengths=coarse) / meep_ldos_axial(case, o, 100, False, wavelengths=coarse) for o in names}
    dense = 1 / np.linspace(0.85, 1.65, 161)
    ours = np.array([ps.solve_problem(ps.Problem("spheres", dimensions, n, w, ps.PointDipole(source, "electric", 1), 1e-8))["total"][:2] for w in dense])
    out, refs, rel = [], [], []
    for i, o in enumerate(names):
        ref = np.full(dense.size, np.nan)
        ref[::10] = theirs[o]
        out.append(series(o, f"{o}, PyStratify", ours[:, i], "pystratify"))
        refs.append(series(f"{o}_ref", f"{o}, Meep", ref, "reference"))
        rel.append(np.max(np.abs(theirs[o] / ours[::10, i] - 1)))
    return dict(x=sig(dense), series=out + refs, worst=float(max(rel)), worst_label="largest relative difference (the grid's error)",
                note="Meep in cylindrical coordinates (m = 0 for the radial dipole, m = 1 for the tangential), a Gaussian pulse and its "
                     "local density of states at the dipole, over the same run without the sphere so the grid's own dispersion cancels. "
                     "A curved surface on a square grid converges in an oscillating way: 5.5% at 25 px/µm, 1.6–1.7% at 50–100 (the test).")


# smuthi (benchmarks/smuthi_spheres.py): each of its four cases swept outward from the case's own gap. smuthi's
# truncation l_max converges geometrically, as (R / (R + gap))^(2 l_max) close to the sphere and above the size
# parameter of the sphere in its host further away; it is taken from that estimate for ~1e-9, and capped at 40
# (the report's ladder: 1.8e-6 for the metal-like sphere at 5 nm, 2.6e-5 for the large one at 50 nm).
SMUTHI_SWEEPS = (
    # slug, case, gaps [nm], logarithmic axis, description
    ("dielectric", "dielectric, n = 2, R = 150 nm, 100 nm away", (100, 1000), False,
     "a dielectric sphere, n = 2, R = 150 nm, in air, 800 nm"),
    ("metal", "metal-like, eps = -10 + i, R = 20 nm, 5 nm away", (5, 60), True,
     "a metal-like sphere, ε = −10 + i, R = 20 nm, in air, 600 nm"),
    ("water", "in water, n = 1.5, R = 100 nm, 20 nm away", (20, 300), False,
     "a sphere of n = 1.5, R = 100 nm, in water (n = 1.33), 633 nm"),
    ("large", "large high-index, n = 3.5 + 0.01i, R = 500 nm, 50 nm away", (50, 1000), False,
     "a large high-index sphere, n = 3.5 + 0.01i, R = 500 nm, in air, 800 nm"),
)


def smuthi_l_max(case, gap):
    from smuthi_spheres import CASES

    wavelength, radius, _, host, _, _ = CASES[case]
    x = 2 * np.pi * host * radius / wavelength
    near = 9 * np.log(10) / (2 * np.log1p(gap / radius))
    return int(min(40, max(10, np.ceil(near), np.ceil(x + 4 * x ** (1 / 3) + 2))))


def smuthi_sweep(case, gaps):
    """PyStratify as a line (151 gaps) and smuthi at every tenth of them (16): the reference series is null between."""
    import smuthi_spheres as ss

    import smuthi  # noqa: F401  (ImportError: the figure is skipped)
    dense = np.geomspace(*gaps, 151)
    ours = np.array([ss.stratify(case, g) for g in dense])
    sampled = dense[::10]
    theirs = np.array([ss.smuthi_rates(case, smuthi_l_max(case, g), g) for g in sampled])
    rel = float(np.max(np.abs(theirs / ours[::10] - 1)))
    sparse = np.full_like(ours, np.nan)
    sparse[::10] = theirs
    return dict(x=sig(dense), series=[
        series("radial", "radial, PyStratify", ours[:, 0], "pystratify"), series("tangential", "tangential, PyStratify", ours[:, 1], "pystratify"),
        series("radial_ref", "radial, smuthi", sparse[:, 0], "reference"), series("tangential_ref", "tangential, smuthi", sparse[:, 1], "reference")],
        worst=rel, worst_label="largest relative difference",
        note=f"smuthi's T-matrix truncated at l_max = {smuthi_l_max(case, sampled[0])} nearest the sphere, "
             f"{smuthi_l_max(case, sampled[-1])} farthest; PyStratify at tolerance 1e-10.")


for _slug, _case, _gaps, _log, _what in SMUTHI_SWEEPS:
    def _make(case=_case, gaps=_gaps):
        return smuthi_sweep(case, gaps)
    _make.__name__ = f"smuthi_{_slug}"
    figure(f"smuthi-sphere-{_slug}", f"A dipole {_gaps[0]}–{_gaps[1]} nm from {_what}: PyStratify against smuthi",
           "gap [nm]", "Γ / Γ₀", log=_log, env="smuthi")(_make)


# --- spheres, focused beam --------------------------------------------------------------------------------------

@figure("focused-sphere-superposition", "A coated sphere (0.8/1.4 µm) under a focused beam (NA 0.6/0.2, collection 0.8/0.1): against plane-wave superposition", "λ [µm]", "power / incident")
def focused_sphere_superposition():
    from focused_checks import SPHERE, brute_sphere
    from pystratify import focused as F

    radii, n = SPHERE
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.8, collection_obscuration=0.1, polarization="x")
    lam = np.linspace(2.5, 6.0, 15)
    ours = F.focused_spheres(radii, n, lam, beam)
    ref = [brute_sphere(radii, n, w, beam) for w in lam]
    keys = ("extinction", "scattering", "detected")
    got = {k: np.ravel(ours[k]) for k in keys}
    exact = {k: np.array([r[k] for r in ref]) for k in keys}
    return dict(x=sig(lam), series=[series(k, f"{k}, PyStratify", got[k], "pystratify") for k in keys]
                + [series(f"{k}_ref", f"{k}, superposition of plane waves", exact[k], "reference") for k in keys],
                worst=max(worst(got[k], exact[k], relative=True) for k in keys), worst_label="largest relative difference",
                note="The reference sums the sphere's amplitude matrix over a grid of incident and scattered directions in the pupil "
                     "(focused_checks.py); PyStratify uses its closed-form beam coefficients. The incident power is 1.")


@figure("focused-na-limit", "The same coated sphere, λ = 3 µm: a focused beam as its NA closes, against the plane wave", "NA", "extinction / (|E(0)|² σ_ext)")
def focused_na_limit():
    from focused_checks import SPHERE
    from pystratify import focused as F

    radii, n = SPHERE
    lam = 3.0
    sigma = ps.cross_sections(ps.solve(radii, n, lam)).ext[0]
    nas = np.linspace(0.01, 0.6, 60)
    ratios = []
    for na in nas:
        out = F.focused_spheres(radii, n, lam, ps.FocusedBeam(na))
        focus = (1 / lam) ** 2 * F.focal_amplitude((0.0, na)) ** 2
        ratios.append(float(np.ravel(out["extinction"])[0] / focus / sigma))
    ratios = np.array(ratios)
    return dict(x=sig(nas), series=[series("ratio", "focused extinction / (|E(0)|² σ_ext), PyStratify", ratios, "pystratify"),
                                    series("plane", "plane wave (= 1)", np.ones_like(ratios), "reference")],
                worst=float(abs(ratios[0] - 1)), worst_label="difference from 1 at NA 0.01",
                note="A focused beam's extinction, divided by its intensity at the focus and the plane-wave cross section, tends to 1 as the NA "
                     "closes and the beam becomes a plane wave across the sphere; a tightly focused beam sees less of it.")


@figure("focused-axis-path", "The coated sphere at λ = 3 µm, focus moved along the axis: the fast on-axis path against the general one", "focus position z [µm]", "power / incident")
def focused_axis_path():
    from focused_checks import SPHERE
    from pystratify import focused as F

    radii, n = SPHERE
    lam = np.array([3.0])
    sol = ps.solve(radii, n, lam)
    k = 2 * np.pi / lam
    z = np.linspace(-3, 3, 61)
    axis, general = {}, {}
    for zi in z:
        a = F._sphere_axis(sol.t, k, sol.orders, (0.23, 0.5), (0.1, 0.7), zi)
        g = F._sphere_general(sol.t, k, sol.orders, (0.23, 0.5), (0.1, 0.7), (0.0, 0.0, zi), (1.0, 0.0), False)
        for key, value in (("extinction", lambda d: np.sum(d["ext_by_order"])), ("detected", lambda d: np.ravel(d["detected"])[0])):
            axis.setdefault(key, []).append(float(value(a))), general.setdefault(key, []).append(float(value(g)))
    return dict(x=sig(z), series=[series(k_, f"{k_}, on-axis path (PyStratify)", axis[k_], "pystratify") for k_ in ("extinction", "detected")]
                + [series(f"{k_}_ref", f"{k_}, general path", general[k_], "reference") for k_ in ("extinction", "detected")],
                worst=max(worst(axis[k_], general[k_], relative=True) for k_ in ("extinction", "detected")), worst_label="largest relative difference",
                note="On the axis the beam couples only to azimuthal orders m = ±1, and PyStratify takes a shortcut; the general path, for a focus "
                     "anywhere, must give the same numbers. NA 0.5/0.23 in, 0.7/0.1 collected.")


@figure("koehler-cone-average", "The coated sphere under Köhler illumination (NA 0.6/0.2, field stop 25 µm): against a cone average of plane waves", "λ [µm]", "power / incident")
def koehler_cone_average():
    from focused_checks import SPHERE, sphere_koehler_reference
    from pystratify import focused as F

    radii, n = SPHERE
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7, collection_obscuration=0.3, illumination="kohler", field_stop=25.0)
    lam = np.linspace(2.5, 6.0, 15)
    ours = F.focused_spheres(radii, n, lam, beam)
    ref = [sphere_koehler_reference(w) for w in lam]
    keys = ("extinction", "detected")
    got = {k: np.ravel(ours[k]) for k in keys}
    exact = {k: np.array([r[k] for r in ref]) for k in keys}
    return dict(x=sig(lam), series=[series(k, f"{k}, PyStratify", got[k], "pystratify") for k in keys]
                + [series(f"{k}_ref", f"{k}, cone average of plane waves", exact[k], "reference") for k in keys],
                worst=max(worst(got[k], exact[k], relative=True) for k in keys), worst_label="largest relative difference",
                note="Köhler illumination is incoherent across the pupil, so a sphere sees an average of plane waves over the illumination cone; "
                     "the reference does that average with the plane-wave cross sections and the amplitude matrix.")


@figure("focused-near-field", "A coated sphere (0.6/1.2 µm) under a focused beam (NA 0.6/0.2, y polarized, focus off-centre), λ = 3 µm: |E|² along a line against plane-wave near fields", "x [µm]", "|E|²")
def focused_near_field():
    from focused_checks import sphere_near_field_pair

    x = np.linspace(-2.5, 2.5, 101)
    points = np.column_stack([x, np.full_like(x, 0.2), np.full_like(x, 0.25)])
    field, E, _ = sphere_near_field_pair(points)
    ours = np.sum(np.abs(np.asarray(field.e[0])) ** 2, -1)
    ref = np.sum(np.abs(E) ** 2, -1)
    return dict(x=sig(x), series=[series("e2", "|E|², PyStratify", ours, "pystratify"), series("e2_ref", "|E|², superposition of plane-wave near fields", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Along y = 0.2, z = 0.25 µm through the sphere (core radius 0.6, shell 1.2 µm): the beam's field from PyStratify's beam "
                     "coefficients against 2016 plane waves from the pupil, each solved by the plane-wave near field and summed.")


# --- cylinders --------------------------------------------------------------------------------------------------

@figure("treams-cylinder-oblique", "A cylinder of radius 100 nm, n = 1.5 + 0.01i, at 60° to its axis: cross widths against treams", "λ [nm]", "width / (2a)")
def treams_cylinder_oblique():
    import treams

    lam = np.linspace(400, 900, 51)
    radii, n = [0.1], [1.5 + 0.01j, 1.0]
    ext, sca, ext_ref, sca_ref = [], [], [], []
    for w in lam:
        beta = 2 * np.pi / (w * 1e-3) * np.cos(np.radians(60))
        sol = ps.solve_cylinder(radii, n, w * 1e-3, beta=beta, m_max=12)
        own = ps.cross_widths(sol)
        tm = treams.TMatrixC.cylinder([beta], 12, 2 * np.pi / (w * 1e-3), radii, [treams.Material(complex(v) ** 2) for v in n])
        ext.append(own["efficiency_extinction"]), sca.append(own["efficiency_scattering"])
        ext_ref.append(tm.xw_ext_avg / (2 * radii[-1])), sca_ref.append(tm.xw_sca_avg / (2 * radii[-1]))
    return dict(x=sig(lam), series=[series("ext", "Q_ext, PyStratify", ext, "pystratify"), series("sca", "Q_sca, PyStratify", sca, "pystratify"),
                                    series("ext_ref", "Q_ext, treams", ext_ref, "reference"), series("sca_ref", "Q_sca, treams", sca_ref, "reference")],
                worst=max(worst(ext, ext_ref, relative=True), worst(sca, sca_ref, relative=True)), worst_label="largest relative difference",
                note="Oblique incidence couples the two polarizations at the surface; both codes average the two incident polarizations. treams' cylindrical T-matrix is per axial wavenumber, which a plane wave fixes.")


@figure("cylinder-bessel-direct", "A cylinder of n = 1.5 in air at 600 nm, normal incidence: the coefficients from the Bessel boundary equations", "radius [nm]", "Q_sca")
def cylinder_bessel_direct():
    from scipy.special import h1vp, hankel1, jv, jvp

    radii_nm = np.linspace(20, 600, 117)
    n, lam = 1.5, 0.6
    ours = {"e": [], "m": []}
    ref = {"e": [], "m": []}
    for R in radii_nm:
        x = 2 * np.pi * R * 1e-3 / lam
        s = ps.solve_cylinder([R * 1e-3], [n, 1.0], lam, m_max=16)
        ours["e"].append(ps.cross_widths(s, "axial-electric")["efficiency_scattering"])
        ours["m"].append(ps.cross_widths(s, "axial-magnetic")["efficiency_scattering"])
        for key, eta in (("e", n), ("m", 1 / n)):
            t = []
            for m in range(-16, 17):
                t.append(-(jvp(m, x) * jv(m, n * x) - eta * jv(m, x) * jvp(m, n * x)) / (h1vp(m, x) * jv(m, n * x) - eta * hankel1(m, x) * jvp(m, n * x)))
            ref[key].append(2 / x * np.sum(np.abs(t) ** 2))
    return dict(x=sig(radii_nm), series=[series("e", "E along the axis, PyStratify", ours["e"], "pystratify"), series("m", "H along the axis, PyStratify", ours["m"], "pystratify"),
                                         series("e_ref", "E along the axis, Bessel equations", ref["e"], "reference"), series("m_ref", "H along the axis, Bessel equations", ref["m"], "reference")],
                worst=max(worst(ours["e"], ref["e"], relative=True), worst(ours["m"], ref["m"], relative=True)), worst_label="largest relative difference",
                note="Each order's coefficient solved directly from the two boundary conditions in SciPy's Bessel functions (Bohren & Huffman §8.4), summed into Q_sca = (2/x) Σ|t_m|², against the cylinder transfer matrices.")


@figure("fibre-dispersion", "A step-index fibre, n = 2, radius 150 nm, in air: guided modes against the dispersion relations", "λ [µm]", "n_eff")
def fibre_dispersion():
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
        te.append(poles[np.argmin(np.abs(poles - exact[0]))]), tm.append(poles[np.argmin(np.abs(poles - exact[1]))]), he.append(poles[-1])
        te_ref.append(exact[0]), tm_ref.append(exact[1])
    return dict(x=sig(lams), series=[series("te", "TE₀₁, PyStratify", te, "pystratify"), series("tm", "TM₀₁, PyStratify", tm, "pystratify"), series("he", "HE₁₁, PyStratify", he, "pystratify"),
                                     series("te_ref", "TE₀₁, dispersion relation", te_ref, "reference"), series("tm_ref", "TM₀₁, dispersion relation", tm_ref, "reference")],
                worst=max(worst(te, te_ref), worst(tm, tm_ref)), worst_label="largest |n_eff − n_eff(relation)| for TE₀₁ and TM₀₁",
                note="Snyder & Love's TE₀₁ and TM₀₁ relations, J₁(u)/(u J₀(u)) + c K₁(w)/(w K₀(w)) = 0, solved by bisection, against the poles PyStratify finds behind a dipole's guided channel; HE₁₁ (no closed form of this kind, and no cutoff) is shown as found; the shorter wavelengths guide higher modes too, not drawn. TE₀₁ and TM₀₁ reach cutoff at 0.392 µm, where V = 2.405.")


@figure("cylinder-index-matched", "A cylinder of n = 1.5, radius 100 nm, in n = 1.5 at 600 nm, β = 2 µm⁻¹: its field is the plane wave", "x [nm]", "Re E")
def cylinder_index_matched():
    n, wavelength, beta = 1.5, .6, 2.0
    k = 2 * np.pi * n / wavelength
    q = np.sqrt(k * k - beta * beta)
    s = ps.solve_cylinder([.1], [n, n], wavelength, beta=beta, m_max=24)
    x = np.linspace(-.25, .25, 151)
    points = np.column_stack([x, np.full_like(x, .03), np.full_like(x, .01)])
    phase = np.exp(1j * (q * x + beta * .01))
    out, worst_ = [], 0.0
    for pol, label, component, amplitude in (("axial-electric", "E along the axis, E_z", 2, q / k), ("axial-magnetic", "H along the axis, E_y", 1, 1.0)):
        E, _ = s.field(points, pol)
        ours, ref = E[:, component].real, (amplitude * phase).real
        out += [(pol, label, ours, ref)]
        worst_ = max(worst_, float(np.max(np.abs(E[:, component] - amplitude * phase))))
    return dict(x=sig(x * 1e3), series=[series(p, f"{lab}, PyStratify", o, "pystratify") for p, lab, o, _ in out]
                + [series(f"{p}_ref", f"{lab}, the plane wave", r, "reference") for p, lab, _, r in out],
                worst=worst_, worst_label="largest |E − plane wave|",
                note="A cylinder of the same index as its surroundings is no cylinder: the field PyStratify builds from its Bessel expansion "
                     "inside and outside (24 orders) must be the incident plane wave, here at oblique incidence (β = 2 µm⁻¹) along y = 30 nm, "
                     "z = 10 nm across the 100 nm radius; the test also checks |E|² = 1 and |H|² = n² at three points.")


def _metal_wire_figure(dipole):
    from unified_checks import metal_wire_rates

    lam = np.linspace(.4, .7, 16)
    bound, diff = [], []
    for w in lam:
        rates, guard = metal_wire_rates(dipole, w)
        bound.append(rates.grazing_error)
        diff.append(max(float(np.max(np.abs(getattr(rates, key) - guard[key]))) for key in ("total", "escape", "absorbed")))
    bound, diff = np.array(bound), np.array(diff)
    return dict(x=sig(lam * 1e3), series=[series("bound", "grazing error charged, PyStratify", bound, "pystratify"),
                                          series("diff", "|PyStratify − guard moved to 3e-6|", diff, "reference")],
                worst=float(np.max(diff / bound)), worst_label="largest difference / charged error (must stay below 1)",
                note=f"A {dipole} dipole 5 nm from a 10 nm wire of n = 0.97 + 1.87i (no dispersion): near the light line the "
                     "integrand is capped, and PyStratify charges the cap to its error. Integrating again with the precision guard "
                     "moved inward to 3e-6 measures what the caps leave out; the largest difference over the total, escape and "
                     "absorbed rates and the three orientations stays below the charge.")


for _dipole in ("electric", "magnetic"):
    def _make(dipole=_dipole):
        return _metal_wire_figure(dipole)
    _make.__name__ = f"metal_wire_grazing_{_dipole}"
    figure(f"cylinder-grazing-caps-{_dipole}", f"{'An' if _dipole == 'electric' else 'A'} {_dipole} dipole 5 nm from a 10 nm metal wire (n = 0.97 + 1.87i): the grazing caps' error against moving the guard",
           "λ [nm]", "error in Γ / Γ₀", log=True)(_make)


def _bh_peak(x, y, lo=None, hi=None):
    m = np.ones_like(x, dtype=bool) if lo is None else (x >= lo) & (x <= hi)
    i = np.flatnonzero(m)[np.argmax(y[m])]
    return float(x[i])


def _read_worst(name):
    """Largest relative difference between a classic's computed and read values."""
    return float(max(abs(c["computed"] - c["paper"]) / abs(c["paper"]) for c in json.loads((CLASSICS / f"{name}.json").read_text())))


@figure("bohren-huffman-cylinder", "A non-absorbing cylinder of n = 1.55 in air at 0.6328 µm: scattering per unit volume, Bohren & Huffman Fig. 8.7", "diameter [µm]", "C_sca / V [10⁴ cm⁻¹]")
def bohren_huffman_cylinder():
    curve, paper = classic_curve("bohren_huffman_1983_fig8_7"), classic_values("bohren_huffman_1983")
    d, par, perp = curve["diameter_um"], curve["Csca_over_V_parallel_1e4_per_cm"], curve["Csca_over_V_perpendicular_1e4_per_cm"]
    par_ref = at(d, {_bh_peak(d, par, .08, .25): paper["parallel: first maximum"], _bh_peak(d, par, .25, .40): paper["parallel: second maximum"],
                     paper["parallel: diameter of the highest maximum (µm)"]: paper["parallel: highest maximum"],
                     _bh_peak(d, par, .55, .70): paper["parallel: shoulder after it"]})
    perp_ref = at(d, {paper["perpendicular: diameter of the maximum (µm)"]: paper["perpendicular: maximum"]})
    return dict(x=sig(d), series=[series("par", "parallel (E along the axis), PyStratify", par, "pystratify"),
                                  series("perp", "perpendicular, PyStratify", perp, "pystratify"),
                                  series("par_ref", "parallel, read off Fig. 8.7", par_ref, "reference"),
                                  series("perp_ref", "perpendicular, read off Fig. 8.7", perp_ref, "reference")],
                worst=_read_worst("bohren_huffman_1983"), worst_label="largest relative difference from the read values",
                note="Bohren & Huffman, Absorption and Scattering of Light by Small Particles (1983), §8.4: the scattering width over the "
                     "cross-sectional area, 4 Q_sca / (π d), the book's demonstration that a cylinder scatters the two polarizations "
                     "differently. Maxima read off the figure to ±0.2 and ±0.03 µm; the book prints the diameters of the highest ones only, "
                     "so the others are marked where PyStratify's curve peaks. From crosscheck/classics/bohren_huffman_1983.py.")


def _ruan_fan_widths(x, g_core, g_shell):
    """Ruan & Fan's three-layer rod (crosscheck/classics/ruan_fan_2010.py): scattering and absorption widths in units
    of 2λ/π, and |t_m|² of the axial-magnetic channel per order m = 1..4, at ω/ωp = x."""
    sys.path.insert(0, str(ROOT / "crosscheck" / "classics"))
    from _common import drude_nk

    lp = 0.1376
    radii = np.array([0.3485, 0.5623, 0.6370]) * lp
    out = []
    for lam in lp / np.asarray(x):
        s = ps.solve_cylinder(radii, [drude_nk(lam, lp, g_core), 3.6, drude_nk(lam, lp, g_shell), 1.0], lam)
        w = ps.cross_widths(s, "axial-magnetic")
        t = np.abs(s.t[:, 1, 1]) ** 2
        out.append([w["scattering"] / (2 * lam / np.pi), w["absorption"] / (2 * lam / np.pi)] + [t[s.orders == m].sum() for m in range(1, 5)])
    return np.array(out)


@figure("ruan-fan-superscattering", "A three-layer nanorod with four aligned resonances, lossless: superscattering, Ruan & Fan 2010, Fig. 1", "ω / ω_p", "width / (2λ/π)")
def ruan_fan_superscattering():
    paper = classic_values("ruan_fan_2010")
    fine = np.linspace(0.2535, 0.2550, 6001)  # the m = 3, 4 resonances are ~1e-6 ω_p wide
    w = _ruan_fan_widths(fine, 0.0, 0.0)
    keep = set(range(0, fine.size, 25)) | {fine.size - 1}
    for column in (0, 2, 3, 4, 5):  # every peak, with its flanks
        y = w[:, column]
        for i in np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1:
            keep |= set(range(max(i - 12, 0), min(i + 13, fine.size)))
    keep = np.array(sorted(keep))
    x, w = fine[keep], w[keep]
    ref = at(x, {paper["lossless: frequency of the peak (ω/ωp)"]: paper["lossless: total scattering peak (2λ/π)"]})
    return dict(x=[float(f"{v:.9g}") for v in x], series=[series("total", "total scattering, PyStratify", w[:, 0], "pystratify")]
                + [series(f"m{m}", f"channel m = ±{m}, PyStratify", w[:, 1 + m], "pystratify") for m in range(1, 5)]
                + [series("total_ref", "printed peak, 7.94 at ω = 0.2542 ω_p", ref, "reference")],
                worst=float(abs(np.max(w[:, 0]) - 7.94) / 7.94), worst_label="peak, relative difference from the printed 7.94",
                note="Ruan & Fan, Phys. Rev. Lett. 105, 013901 (2010): Drude core to 0.3485 λ_p, ε = 12.96 shell to 0.5623 λ_p, Drude shell "
                     "to 0.6370 λ_p, H along the axis. Each channel can scatter at most 2λ/π (the single-channel limit, 1 on this axis); "
                     "aligning four gives ~8. The m = 3 and 4 resonances are needles about 1e-6 ω_p wide, so the window is computed on "
                     "6001 points and drawn on a thinned grid that keeps every peak. From crosscheck/classics/ruan_fan_2010.py.")


@figure("ruan-fan-lossy", "The same nanorod with silver-like loss (γ = 0.004 ω_p core, 0.012 ω_p shell): Ruan & Fan 2010, Fig. 4a", "ω / ω_p", "width / (2λ/π)")
def ruan_fan_lossy():
    curve, paper = classic_curve("ruan_fan_2010_fig1_fig4a"), classic_values("ruan_fan_2010")
    x, sca, ab = curve["omega_over_omega_p"][::3], curve["sca_lossy"][::3], curve["abs_lossy"][::3]
    sca_ref = at(x, {float(x[np.argmax(sca)]): paper["lossy: total scattering peak (2λ/π)"]})
    abs_ref = at(x, {float(x[np.argmax(ab)]): paper["lossy: absorption peak (2λ/π)"]})
    return dict(x=sig(x), series=[series("sca", "scattering, PyStratify", sca, "pystratify"), series("abs", "absorption, PyStratify", ab, "pystratify"),
                                  series("sca_ref", "scattering peak, printed (1.92)", sca_ref, "reference"),
                                  series("abs_ref", "absorption peak, printed (0.73)", abs_ref, "reference")],
                worst=float(max(abs(np.max(sca) - 1.92) / 1.92, abs(np.max(ab) - 0.73) / 0.73)), worst_label="largest relative difference from the printed peaks",
                note="With silver's damping the needles broaden and the superscattering drops from ~8 to 1.92 single-channel limits; the "
                     "paper prints the peak heights (p. 3–4), not their frequencies, so they are marked where PyStratify's curves peak. "
                     "From crosscheck/classics/ruan_fan_2010.py.")


@figure("ruppin-sodium-wire", "A 1 nm sodium wire in vacuum and a 2 nm one in ε = 2, E across the wire: extinction, Ruppin 2001 (local curves)", "ω / ω_p", "Q_ext", log=True)
def ruppin_sodium_wire():
    curve, paper = classic_curve("ruppin_2001_fig1_fig4_local"), classic_values("ruppin_2001")
    x = curve["omega_over_omega_p"]
    one, two = curve["Qext_a1nm_host1"], curve["Qext_a2nm_host_eps2"]
    ref = at(x, {paper["a = 1 nm, ε_m = 1: local resonance (ω/ωp)"]: 10 ** paper["a = 1 nm: log10 of the local peak"],
                 1.4: 10 ** paper["a = 1 nm: log10 C_e at 1.4 ωp"]})
    i = int(np.argmax(one))
    return dict(x=sig(x), series=[series("a1", "a = 1 nm, ε_m = 1, PyStratify", one, "pystratify"), series("a2", "a = 2 nm, ε_m = 2, PyStratify", two, "pystratify"),
                                  series("a1_ref", "a = 1 nm: read off Fig. 1 (dashed) at the printed resonance ω_p/√2", ref, "reference")],
                worst=float(max(abs(np.log10(one[i]) - paper["a = 1 nm: log10 of the local peak"]), abs(np.log10(one[-1]) - paper["a = 1 nm: log10 C_e at 1.4 ωp"]))),
                worst_label="largest difference in log₁₀ Q_ext from the read values",
                note="Ruppin, Opt. Commun. 190, 205 (2001): Drude sodium (ω_p = 8.65e15 s⁻¹, γ = 0.01 ω_p). The calculator draws the paper's "
                     "dashed local curves; its solid ones add the hydrodynamic response. The printed resonances, ω_p/√(1 + ε_m), fall at 0.7071 "
                     "(PyStratify 0.707) and 0.5774 (0.576); the heights are read off the log axis. From crosscheck/classics/ruppin_2001.py.")


# --- cylinders, point dipole ------------------------------------------------------------------------------------

@figure("cylinder-dipole-homogeneous", "A dipole in and around a cylinder of n = 1.5 in n = 1.5 at 600 nm: the rates of free space", "distance from the axis [nm]", "|Γ / Γ₀ − 1|", log=True)
def cylinder_dipole_homogeneous():
    r = np.r_[np.linspace(0, .095, 12), np.linspace(.105, .3, 20)]
    out = {}
    for dipole in ("electric", "magnetic"):
        for ri in r:
            res = ps.solve_problem(ps.Problem("cylinders", [.1], [1.5, 1.5], .6, ps.PointDipole(ri, dipole, int(ri > .1), .4), 1e-6, 15))
            out.setdefault(dipole, []).append(max(float(np.max(np.abs(res["total"] - 1))), 1e-17))
    return dict(x=sig(r * 1e3), series=[series(d, f"{d} dipole, max over orientations, PyStratify", out[d], "pystratify") for d in out],
                worst=float(max(max(v) for v in out.values())), worst_label="largest |Γ / Γ₀ − 1|",
                note="A cylinder of the medium's own index changes nothing: every rate is the free-space one, 1, for any position and "
                     "orientation, inside the cylinder (r < 100 nm) or out; the test also checks escape = 1, no guided or absorbed power "
                     "and the quantum yield 0.4 passed through. Floored at 1e-17 for the logarithmic axis.")


def _coating_budget(dipole):
    r = np.linspace(.105, .4, 24)
    total, channels = [], []
    for ri in r:
        res = ps.solve_problem(ps.Problem("cylinders", [.05, .1], [1.5, .2 + 3j, 1], .6, ps.PointDipole(ri, dipole, 2), 1e-5, 30))
        total.append(res["total"][:3]), channels.append((res["escape"] + res["guided"] + res["absorbed"])[:3])
    total, channels = np.array(total), np.array(channels)
    names = ("radial", "azimuthal", "axial")
    return dict(x=sig(r * 1e3), series=[series(o, f"{o}, total (PyStratify)", total[:, i], "pystratify") for i, o in enumerate(names)]
                + [series(f"{o}_ref", f"{o}, escape + guided + absorbed", channels[:, i], "reference") for i, o in enumerate(names)],
                worst=worst(total, channels, relative=True), worst_label="largest relative difference",
                note=f"A {dipole} dipole outside a cylinder of n = 1.5 (50 nm) under 50 nm of an absorbing n = 0.2 + 3i coating, 600 nm. "
                     "The total rate comes from the Green's function at the source; the channels separately, from the power escaping to "
                     "infinity, the power carried by guided modes and the power absorbed in the coating, so their sum is an independent budget.")


for _dipole in ("electric", "magnetic"):
    def _make(dipole=_dipole):
        return _coating_budget(dipole)
    _make.__name__ = f"coating_budget_{_dipole}"
    figure(f"cylinder-coating-budget-{_dipole}", f"{'An' if _dipole == 'electric' else 'A'} {_dipole} dipole outside a coated cylinder (n = 1.5 core, absorbing n = 0.2 + 3i coating, 50/100 nm), 600 nm: the total against its channels",
           "distance from the axis [nm]", "Γ / Γ₀", log=True)(_make)


@figure("cylinder-guide-budget", "A dipole inside a lossless guide (n = 1.5, radius 100 nm) in air, 600 nm: the total against escape + guided", "distance from the axis [nm]", "Γ / Γ₀")
def cylinder_guide_budget():
    r = np.linspace(0, .095, 20)
    total, escape, guided = [], [], []
    for ri in r:
        res = ps.solve_problem(ps.Problem("cylinders", [.1], [1.5, 1], .6, ps.PointDipole(ri, "electric", 0), 1e-5, 20))
        total.append(res["total"][3]), escape.append(res["escape"][3]), guided.append(res["guided"][3])
    total, escape, guided = map(np.array, (total, escape, guided))
    return dict(x=sig(r * 1e3), series=[series("total", "total, PyStratify", total, "pystratify"), series("escape", "escape", escape, "pystratify"),
                                        series("guided", "guided", guided, "pystratify"),
                                        series("total_ref", "escape + guided", escape + guided, "reference")],
                worst=worst(total, escape + guided, relative=True), worst_label="largest relative difference",
                note="An electric dipole inside the guide, averaged over orientations: a point source (unlike a line source) sends a finite "
                     "share of its power into the guided mode, here about a tenth of the free rate; the total from the Green's function must "
                     "equal what escapes plus what is guided, since nothing absorbs.")


@figure("cylinder-grazing-tolerance", "An electric dipole 5 nm from a 10 nm metal wire (n = 0.97 + 1.87i), asked for 1e-9: the grazing error it cannot pass", "λ [nm]", "error in Γ / Γ₀", log=True)
def cylinder_grazing_tolerance():
    from unified_checks import metal_wire_tight

    lam = np.linspace(.4, .7, 16)
    grazing, error, asked, converged = [], [], [], []
    for w in lam:
        rates = metal_wire_tight(w)
        grazing.append(rates.grazing_error), error.append(rates.error), asked.append(1e-9 * np.max(rates.total)), converged.append(rates.converged)
    grazing, asked = np.array(grazing), np.array(asked)
    return dict(x=sig(lam * 1e3), series=[series("grazing", "grazing error charged, PyStratify", grazing, "pystratify"),
                                          series("error", "total error reported, PyStratify", error, "pystratify"),
                                          series("asked", "1e-9 × max Γ / Γ₀, the accuracy asked for", asked, "reference")],
                worst=float(np.min(grazing / asked)), worst_label="smallest grazing error / accuracy asked (must stay above 1)",
                note="The light-line caps leave an error the integration cannot shrink; asked for 1e-9, PyStratify charges it to the total "
                     f"error and reports the result as not converged ({sum(not c for c in converged)} of {len(converged)} wavelengths), rather than "
                     "claiming an accuracy it does not have. The grazing error is all but 0.3% of the reported error, so the two lines coincide. "
                     "The same wire as the grazing-caps rows; 40 orders; no dispersion.")


@figure("fibre-rates-cutoff", "An electric dipole 100 nm outside a step-index fibre (n = 2, radius 150 nm) through the TE₀₁/TM₀₁ cutoff", "λ [µm]", "Γ / Γ₀")
def fibre_rates_cutoff():
    from unified_checks import fibre_rates

    lam = np.linspace(.62, .74, 49)
    total, escape, guided = [], [], []
    for w in lam:
        res = fibre_rates(w)
        total.append(res["total"][:3]), escape.append(res["escape"][:3]), guided.append(res["guided"][:3])
    total, escape, guided = map(np.array, (total, escape, guided))
    names = ("radial", "azimuthal", "axial")
    return dict(x=sig(lam), series=[series(o, f"{o}, total (PyStratify)", total[:, i], "pystratify") for i, o in enumerate(names)]
                + [series(f"{o}_guided", f"{o}, guided (PyStratify)", guided[:, i], "pystratify") for i, o in enumerate(names)]
                + [series(f"{o}_ref", f"{o}, escape + guided", (escape + guided)[:, i], "reference") for i, o in enumerate(names)],
                worst=worst(total, escape + guided, relative=True), worst_label="largest relative difference, total against escape + guided",
                note="TE₀₁ and TM₀₁ are cut off near 0.678 µm: the guided power of the radial and azimuthal dipoles, which they carried, drops "
                     "there and reappears as escaping power (HE₁₁ stays guided), while the total, from the Green's function, passes through smoothly (the test bounds its second difference over 0.66–0.70 µm "
                     "by 2e-3). Every mode PyStratify finds is checked against the dispersion relations in the fibre-dispersion plot.")


@figure("meep-cylinder-fibre", "A dipole 100 nm outside a step-index fibre (n = 2, radius 150 nm) in air: total decay rates against Meep (3D FDTD, 60 px/µm)", "λ [µm]", "Γ / Γ₀", env="meep")
def meep_cylinder_fibre():
    import meep as mp
    from fdtd_meep import CASES, ORIENTATIONS, meep_ldos

    case = "fibre n = 2, R = 150 nm, dipole 100 nm outside"
    radii, n, source = CASES[case]
    mp.verbosity(0)
    coarse = 1 / np.linspace(0.85, 1.65, 17)
    theirs = {o: meep_ldos(case, o, 60, wavelengths=coarse) / meep_ldos(case, o, 60, False, wavelengths=coarse) for o in ORIENTATIONS}
    dense = 1 / np.linspace(0.85, 1.65, 161)
    ours = np.array([ps.solve_problem(ps.Problem("cylinders", radii, n, w, ps.PointDipole(source, "electric"), 1e-6))["total"][:3] for w in dense])
    out, refs, rel = [], [], []
    for i, o in enumerate(ORIENTATIONS):
        ref = np.full(dense.size, np.nan)
        ref[::10] = theirs[o]
        out.append(series(o, f"{o}, PyStratify", ours[:, i], "pystratify"))
        refs.append(series(f"{o}_ref", f"{o}, Meep", ref, "reference"))
        rel.append(np.max(np.abs(theirs[o] / ours[::10, i] - 1)))
    return dict(x=sig(dense), series=out + refs, worst=float(max(rel)), worst_label="largest relative difference (the grid's error)",
                note="Meep in 3D: the fibre runs through the perfectly matched layers, a 1.8 µm cell plus 1 µm of PML per side, two mirror "
                     "planes; a Gaussian pulse and the local density of states at the dipole, over the same run without the fibre so the "
                     "grid's own dispersion cancels. TE₀₁ and TM₀₁ are cut off at 0.678 µm. The difference shrinks with the grid, "
                     "6.9% → 3.2% → 1.7% → 0.74% at 20–60 px/µm (FDTD_MEEP.md); the test runs 20 and 30.")

@figure("ngsolve-cylinder-fibre", "A dipole 100 nm outside a step-index fibre (n = 2, radius 150 nm) in vacuum: total decay rates against NGSolve (3D finite elements)", "λ [µm]", "Γ / Γ₀", env="ngsolve")
def ngsolve_cylinder_fibre():
    import ngsolve  # noqa: F401
    from fem_ngsolve import CELL, MESHES, ORIENTATIONS, PML_THICKNESS, RADIUS, INDEX, SOURCE, ngsolve_rates

    points = (0.62, 0.645, 0.66, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2)  # both sides of the TE01/TM01 cutoff, 0.678 um
    theirs = {w: ngsolve_rates(w, *MESHES[1])[0] for w in points}
    dense = np.unique(np.round(np.r_[np.linspace(0.62, 1.2, 59), points], 6))  # NGSolve's wavelengths exactly on the grid
    ours = np.array([ps.solve_problem(ps.Problem("cylinders", [RADIUS], [INDEX, 1.0], w, ps.PointDipole(SOURCE, "electric"), 1e-6))["total"][:3] for w in dense])
    index = {w: int(np.flatnonzero(np.isclose(dense, w))[0]) for w in points}
    out, refs, rel = [], [], []
    for i, o in enumerate(ORIENTATIONS):
        out.append(series(o, f"{o}, PyStratify", ours[:, i], "pystratify"))
        refs.append(series(f"{o}_ref", f"{o}, NGSolve", at(dense, {w: theirs[w][i] for w in points}), "reference"))
        rel.append(max(abs(theirs[w][i] / ours[index[w], i] - 1) for w in points))
    return dict(x=sig(dense), series=out + refs, worst=float(max(rel)), worst_label="largest relative difference (the middle mesh)",
                note=f"Second-order Nédélec elements in a cylindrical cell (radius and half-height {CELL:g} µm) inside a {PML_THICKNESS:g} µm PML, "
                     "radial in ρ and Cartesian in z, which the fibre runs through; mirror symmetry halves it twice, so each orientation "
                     "is a quarter problem (~100k unknowns, the middle mesh: 0.25 µm in vacuum, 0.06 µm in the fibre). The rate is Im(p·E) at "
                     "the dipole over the same on the same mesh with the fibre's permittivity set to 1. TE₀₁ and TM₀₁ are cut off at 0.678 µm; "
                     "just above it their fields reach far into the vacuum. The difference shrinks with the mesh: worst 1.4% → 0.8% → 0.5% at 0.8 µm "
                     "and 4.4% → 1.3% → 0.8% at 0.645 µm (FEM_NGSOLVE.md). "
                     "An earlier spherical cell with a radial PML met the guided modes badly and was off by up to 8–11% here; Meep agrees "
                     "to 0.74% over the same band.")


# --- cylinders, focused beam ------------------------------------------------------------------------------------

@figure("focused-cylinder-rows", "A coated cylinder (0.7/1.2 µm) under a focused beam (NA 0.6/0.2, collection 0.8/0.1, x polarized), 3 µm: one pupil row at a time against plane-wave sums", "pupil row s_y", "power per row")
def focused_cylinder_rows():
    from focused_checks import CYLINDER, CYLINDER_ROW_BEAM, row_brute
    from pystratify import focused as F

    radii, n = CYLINDER
    rows_ = F._CylinderRows(radii, n, 3.0, [CYLINDER_ROW_BEAM], 14)
    sy = np.linspace(0.01, 0.59, 30)
    ours = np.array([rows_(s)[:3] / 2 for s in sy])
    ref = np.array([row_brute(radii, n, 3.0, s, CYLINDER_ROW_BEAM, (1.0, 0.0), 14)[:3] for s in sy])
    keys = ("extinction", "scattering", "detected")
    return dict(x=sig(sy), series=[series(k, f"{k}, PyStratify", ours[:, i], "pystratify") for i, k in enumerate(keys)]
                + [series(f"{k}_ref", f"{k}, plane waves summed", ref[:, i], "reference") for i, k in enumerate(keys)],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="A cylinder conserves the beam's direction along its axis, so the pupil splits into rows of fixed s_y, each an oblique "
                     "cylinder problem. PyStratify sums a row in closed form over Fourier orders; the reference solves every plane wave of the "
                     "row (60 nodes per interval), adds their far fields on observation angles and integrates the powers directly. Focus at "
                     "(0.4, 0, 0.3) µm; the y polarization and the mirror rows −s_y are checked too.")



def _cylinder_whole_beam_figure(pol):
    from focused_checks import cylinder_whole_beam_pair

    lam = np.linspace(2.5, 6.0, 15)
    keys = ("extinction", "scattering", "detected")
    ours, ref = [], []
    for w in lam:
        r, total, width = cylinder_whole_beam_pair(w)
        p = "xy".index(pol)
        bp = r.by_polarization[pol]
        ours.append([float(np.ravel(bp[k])[0]) for k in keys]), ref.append(total[p * width:p * width + 3])
    ours, ref = np.array(ours), np.array(ref)
    return dict(x=sig(lam), series=[series(k, f"{k}, PyStratify (adaptive)", ours[:, i], "pystratify") for i, k in enumerate(keys)]
                + [series(f"{k}_ref", f"{k}, fixed 40-node quadrature", ref[:, i], "reference") for i, k in enumerate(keys)],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note=f"The whole beam, {pol} polarized ({'across' if pol == 'x' else 'along'} the axis): PyStratify integrates its rows over s_y "
                     "adaptively to 1e-9; the reference sums the same rows with a fixed 40-node rule on each side of the obscuration. "
                     "The incident power is 1.")


for _pol in "xy":
    def _make(pol=_pol):
        return _cylinder_whole_beam_figure(pol)
    _make.__name__ = f"focused_cylinder_whole_beam_{_pol}"
    figure(f"focused-cylinder-whole-beam-{_pol}", f"The coated cylinder (0.7/1.2 µm) under a focused beam (NA 0.6/0.2, collection 0.8), {_pol} polarized: the whole beam against a fixed quadrature",
           "λ [µm]", "power / incident")(_make)


@figure("focused-cylinder-koehler", "The coated cylinder (0.7/1.2 µm) under Köhler illumination (NA 0.6/0.2, collection 0.7/0.3, field stop 30 µm): against a cone average of plane waves", "λ [µm]", "power / incident")
def focused_cylinder_koehler():
    from focused_checks import CYLINDER, cylinder_koehler_reference
    from pystratify import focused as F

    radii, n = CYLINDER
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7, collection_obscuration=0.3, illumination="kohler", field_stop=30.0)
    lam = np.linspace(2.5, 6.0, 15)
    ours, ref = [], []
    for w in lam:
        r = F.focused_cylinders(radii, n, w, beam, m_max=14, tolerance=1e-9)
        t = cylinder_koehler_reference(w, 30.0)
        ours.append([float(np.ravel(r.by_polarization[p][k])[0]) for p in "xy" for k in ("extinction", "detected")])
        ref.append([v for p in "xy" for v in (t[p][0], t[p][2] + t[p][1])])
    ours, ref = np.array(ours), np.array(ref)
    labels = ("extinction, x", "detected, x", "extinction, y", "detected, y")
    return dict(x=sig(lam), series=[series(f"s{i}", f"{lab}, PyStratify", ours[:, i], "pystratify") for i, lab in enumerate(labels)]
                + [series(f"s{i}_ref", f"{lab}, cone average", ref[:, i], "reference") for i, lab in enumerate(labels)],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Köhler illumination is incoherent across the pupil: the cylinder sees each plane wave of the illumination cone on its "
                     "own, over the chord the field stop lights. The reference averages the oblique plane-wave extinction widths and the "
                     "scattered power that lands in the collection annulus over the cone (focused_checks.py); x is across the axis.")


@figure("focused-cylinder-na-limit", "The coated cylinder (0.7/1.2 µm) at λ = 3 µm: a focused beam as its NA closes, against the plane wave", "NA", "extinction / (∫|E|² dl · C_ext)")
def focused_cylinder_na_limit():
    from focused_checks import cylinder_na_ratio

    nas = np.linspace(0.01, 0.3, 30)
    ratios = {p: np.array([cylinder_na_ratio(na, p) for na in nas]) for p in "xy"}
    return dict(x=sig(nas), series=[series(p, f"{p} polarized ({'across' if p == 'x' else 'along'} the axis), PyStratify", ratios[p], "pystratify") for p in "xy"]
                + [series("plane", "plane wave (= 1)", np.ones_like(nas), "reference")],
                worst=float(max(abs(ratios[p][0] - 1) for p in "xy")), worst_label="difference from 1 at NA 0.01",
                note="A focused beam's extinction, divided by its |E|² integrated along the cylinder's axis and the plane-wave extinction "
                     "width, tends to 1 as the NA closes; the test asserts it is within 2e-3 at NA 0.02 and closer than at 0.04, and that "
                     "a lossless cylinder absorbs nothing.")


@figure("focused-cylinder-near-field", "A coated cylinder (0.6/1.2 µm, axis along y) under a focused beam (NA 0.6/0.2, x polarized, focus off-centre), 3 µm: |E|² across it against plane-wave fields", "x [µm]", "|E|²")
def focused_cylinder_near_field():
    from focused_checks import cylinder_near_field_pair

    x = np.linspace(-2.5, 2.5, 101)
    field, E = cylinder_near_field_pair(np.column_stack([x, np.full_like(x, 0.2), np.full_like(x, 0.25)]))
    ours = np.sum(np.abs(np.asarray(field.e[0])) ** 2, -1)
    ref = np.sum(np.abs(E) ** 2, -1)
    return dict(x=sig(x), series=[series("e2", "|E|², PyStratify", ours, "pystratify"), series("e2_ref", "|E|², superposition of plane-wave cylinder fields", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Along y = 0.2, z = 0.25 µm, across the cylinder (core 0.6, shell 1.2 µm; n = 1.5 + 0.1i, 1.3 + 0.02i): PyStratify's field "
                     "from its row coefficients against the pupil's plane waves, each solved as an oblique cylinder and its field summed "
                     "(focused_checks.py). Focus at (0.3, 0.2, 0.25) µm.")


@figure("scalar-cylinder-coefficients", "The scalar model: a cylinder of radius 1.1 µm, n = 1.45 + 0.06i, row s_y = 0.4, 3 µm: coefficients against the scalar boundary equations", "order m", "|t_m|", log=True)
def scalar_cylinder_coefficients():
    from focused_checks import scalar_cylinder_coefficient

    lam, R, m_in, sy = 3.0, 1.1, 1.45 + 0.06j, 0.4
    k = 2 * np.pi / lam
    n_v, n_p = np.sqrt(1 - sy * sy), np.sqrt(m_in ** 2 - sy * sy)
    sol = ps.solve_cylinder([R], [n_p, n_v], lam, beta=0, m_max=12)
    expected = [scalar_cylinder_coefficient(abs(m), k * R * n_v, k * R * n_p, n_v, n_p) for m in sol.orders]
    t = sol.t[:, 0, 0]
    return dict(x=[float(m) for m in sol.orders], series=[series("t", "|t_m|, PyStratify", np.abs(t), "pystratify"),
                                                          series("t_ref", "|t_m|, scalar boundary equations", np.abs(expected), "reference")],
                worst=worst(np.abs(t), np.abs(expected), relative=True), worst_label="largest relative difference",
                note="The scalar model treats a row of the pupil as a scalar wave U = J_m + t_m H_m outside a cylinder of effective "
                     "indices √(n² − s_y²), with U and ∂U/∂ρ continuous; the coefficients from PyStratify's cylinder solver against those "
                     "two conditions solved directly in SciPy's Bessel functions.")


@figure("scalar-cylinder-rows", "The scalar model, the coated cylinder (0.7/1.2 µm) under NA 0.6/0.2, 3 µm: each pupil row against a direct superposition", "pupil row s_y", "power per row")
def scalar_cylinder_rows():
    from focused_checks import scalar_row_pair

    sy = np.linspace(0.01, 0.59, 30)
    ours, ref = [], []
    for s in sy:
        value, (sca, ext) = scalar_row_pair(s)
        ours.append([value[0], value[1]]), ref.append([ext, sca])
    ours, ref = np.array(ours), np.array(ref)
    return dict(x=sig(sy), series=[series("ext", "extinction, PyStratify", ours[:, 0], "pystratify"), series("sca", "scattering, PyStratify", ours[:, 1], "pystratify"),
                                   series("ext_ref", "extinction, direct superposition", ref[:, 0], "reference"),
                                   series("sca_ref", "scattering, direct superposition", ref[:, 1], "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Each row's scalar plane waves, with the aplanatic pupil amplitude and the focus at (0.3, 0, 0.2) µm, scattered one by one "
                     "with the effective-index coefficients and superposed on observation angles; the powers integrated directly. The test "
                     "also checks that a lossless cylinder absorbs nothing and that the scalar model has no polarization split.")


@figure("davis-2011-fibres", "Non-absorbing fibres (R = 5 µm, n = 1.477; R = 10 µm, n = 1.478) under a Cassegrain (NA 0.1–0.4): transmission, Davis, Carney & Bhargava 2011, Fig. 5", "wavenumber [cm⁻¹]", "transmission [%]")
def davis_2011_fibres():
    curve, paper = classic_curve("davis_2011_fig5"), classic_values("davis_2011")
    wn, t5, t10 = curve["wavenumber_cm-1"], curve["T_percent_R5um_n1.477"], curve["T_percent_R10um_n1.478"]
    window = lambda lo, hi: (wn >= lo) & (wn <= hi)  # noqa: E731
    w1, w2 = window(1900, 2600), window(2700, 3300)
    ref5 = at(wn, {paper["R = 5 µm: wavenumber of the minimum (cm-1)"]: paper["R = 5 µm: transmission minimum (%)"],
                   float(wn[w1][np.argmax(t5[w1])]): paper["R = 5 µm: maximum near the fit range, 1900-2600 cm-1 (%)"],
                   float(wn[w2][np.argmin(t5[w2])]): paper["R = 5 µm: dip near 3000 cm-1 (%)"], 4000.0: paper["R = 5 µm: at 4000 cm-1 (%)"]})
    ref10 = at(wn, {800.0: paper["R = 10 µm: at 800 cm-1 (%)"]})
    values = json.loads((CLASSICS / "davis_2011.json").read_text())
    return dict(x=sig(wn), series=[series("t5", "R = 5 µm, PyStratify", t5, "pystratify"), series("t10", "R = 10 µm, PyStratify", t10, "pystratify"),
                                   series("t5_ref", "R = 5 µm, read off Fig. 5a", ref5, "reference"), series("t10_ref", "R = 10 µm, read off Fig. 5b", ref10, "reference")],
                worst=float(max(abs(c["computed"] - c["paper"]) for c in values if "(%)" in c["check"])), worst_label="largest difference from a read transmission [percentage points]",
                note="Davis, Carney & Bhargava, Anal. Chem. 83, 525 (2011): a fibre that absorbs nothing still shows a strong, wavenumber-"
                     "dependent apparent absorbance, made by refraction and scattering out of the objective. Values read off Fig. 5 to ±3%; "
                     "the paper prints no position for the maximum and the dip, so they are marked where PyStratify's curve turns, and the "
                     "minimum sits 20 cm⁻¹ from the read 1100. The paper does not state Fig. 5's NA; 0.1–0.4 is its Fig. 2 pair. "
                     "From crosscheck/classics/davis_2011.py.")


# --- CLI ---------------------------------------------------------------------------------------------------------

def build(names=None, env=None):
    OUT.mkdir(exist_ok=True)
    written = []
    for name, (fn, meta) in FIGURES.items():
        if (names and name not in names) or (env and ENVS[name] != env):
            continue
        t = time.time()
        try:
            data = fn()
        except ImportError as error:
            print(f"skip {name}: {error}", flush=True)
            continue
        out = {**meta, **data, "script": "benchmarks/figures.py", "function": fn.__name__}
        out["x"] = [float(v) for v in out["x"]]
        for s in out["series"]:
            s["y"] = sig(s["y"], 8)
        (OUT / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
        written.append(name)
        print(f"{name}: {len(out['x'])} points, {len(out['series'])} series, worst {out['worst']:.2e} ({time.time() - t:.1f} s)", flush=True)
    return written


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--list"]:  # name and environment of every figure
        for name in FIGURES:
            print(name, ENVS[name])
    elif args[:1] == ["--env"]:  # every figure of one environment
        build(env=args[1])
    else:
        build(args or None)
