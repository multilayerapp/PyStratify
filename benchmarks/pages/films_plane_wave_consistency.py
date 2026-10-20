"""Thin films, plane wave: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_plane_wave_consistency.py                                                         #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-plane-wave-consistency.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
films_plane_wave_consistency.json (the curves the page draws, value for value) and films_plane_wave_consistency.png (needs matplotlib).
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
from _shared import INCOHERENT  # noqa: E402


@figure("fresnel-oblique", "Single interface", "angle of incidence [°]", "R")
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


@figure("airy-slab", "Fabry–Pérot slab", "λ [nm]", "R")
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


@figure("energy-conservation", "Lossless stack", "angle of incidence in glass [°]", "|1 − R − T|", log=True)
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


@figure("opaque-film", "Opaque metal film", "thickness [nm]", "R, T")
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
    figure(f"bare-substrate-{_part}", f"Bare substrates: {_what}", "angle of incidence [°]", _unit)(_make)


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


@figure("smatrix-metal-stack", "Metal stack", "angle of incidence [°]", "R, T")
def smatrix_metal_stack():
    from planar_checks import METAL_STACK
    return _smatrix_figure(METAL_STACK, np.linspace(0, 89, 90), "An absorbing stack, s and p.")


@figure("smatrix-beyond-critical", "Frustrated reflection", "angle of incidence in glass [°]", "R, T")
def smatrix_beyond_critical():
    from planar_checks import BEYOND_CRITICAL
    return _smatrix_figure(BEYOND_CRITICAL, np.linspace(0, 89, 90), "Beyond 41.8° the air gap is evanescent and light tunnels through it to the n = 2 layer and the glass.")


@figure("incoherent-phase-average", "Thick incoherent layer", "angle of incidence [°]", "R, T")
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


@figure("rcwa-unpatterned", "Three-layer filter", "λ [nm]", "R, T")
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


@figure("position-resolved-absorption", "Absorption per layer", "λ [nm]", "absorptance per layer")
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


@figure("kramers-kronig-closed-forms", "Kramers–Kronig", "E [eV]", "ε₁")
def kramers_kronig_closed_forms():
    from dispersion_codes import cauchy_weight_kk
    from pystratify import dispersion as dm

    energy = np.linspace(0.25, 9.0, 120)
    wavelength = dm.HC_EV_NM / energy
    tauc = dm.TaucLorentz(A=120, E0=3.6, C=2.4, Eg=1.6)
    cody = dm.CodyLorentz(A=100, E0=3.6, Gamma=2.4, Eg=1.6, Ep=0.8, Et=1.8, Eu=0.05)

    def tauc_eps2(e):  # zero below the gap by definition, so the quadrature may start at E = 0
        e = np.atleast_1d(np.asarray(e, float))
        out = np.zeros_like(e)
        above = e > tauc.Eg
        out[above] = tauc.epsilon(dm.HC_EV_NM / e[above]).imag
        return out

    ours_tl, ours_cl = tauc.epsilon(wavelength).real, cody.epsilon(wavelength).real
    ref_tl = np.array([cauchy_weight_kk(tauc_eps2, e, (1.6, 3.6, 20.0, 200.0)) for e in energy])
    ref_cl = np.full(energy.size, np.nan)
    for i in range(0, energy.size, 4):
        ref_cl[i] = cauchy_weight_kk(cody.eps2, energy[i], (1.6, 1.8, 3.6, 20.0, 200.0))
    scale = max(np.max(np.abs(ours_tl)), np.max(np.abs(ours_cl)))
    rel = float(max(np.max(np.abs(ours_tl - ref_tl)), np.nanmax(np.abs(ours_cl - ref_cl))) / scale)
    return dict(x=sig(energy), series=[series("tl", "Tauc-Lorentz ε₁, closed form", ours_tl, "pystratify"), series("cl", "Cody-Lorentz ε₁, closed form", ours_cl, "pystratify"),
                                       series("tl_ref", "Tauc-Lorentz, principal value of ε₂", ref_tl, "reference"),
                                       series("cl_ref", "Cody-Lorentz, principal value of ε₂", ref_cl, "reference")],
                worst=rel, worst_label="largest |Δε₁| / max |ε₁|",
                note="The reference is (2/π) P∫ E′ε₂ / (E′² − E²) dE′ by QUADPACK with the Cauchy weight on the interval holding E, split at "
                     "the kinks (benchmarks/dispersion_codes.py cauchy_weight_kk), independent of the module's own Kramers-Kronig subtraction. "
                     "The tests also check Gaussian and Lorentz ε₁ against dispersion.kramers_kronig_real, and that Cody-Lorentz ε₁ stays "
                     "continuous across E_g and E_t.")


if __name__ == "__main__":
    run(__file__)
