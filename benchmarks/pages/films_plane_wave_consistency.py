"""Thin films, plane wave: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/films_plane_wave_consistency.py                                                         #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-plane-wave-consistency.html, computed by this one file
from a PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE
and writes, beside itself, films_plane_wave_consistency.json (the curves the page draws, value for value) and
films_plane_wave_consistency.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np
from numpy import inf

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run
from _shared import INCOHERENT

from pystratify import planar


@figure("fresnel-oblique", "Single interface", "angle of incidence [°]", "R")
def plot_single_interface():
    from planar_checks import _forward_cos, _fresnel_amplitudes

    angles = np.linspace(0, 89.5, 180)
    every = 5  # the closed form every 2.5°
    n1, n2 = 1.0, 1.5

    def fresnel(pol, a):
        kx = n1 * np.sin(np.radians(a))
        return abs(_fresnel_amplitudes(pol, n1, n2, _forward_cos(n1, kx), _forward_cos(n2, kx))[0]) ** 2

    ours = {p: np.array([planar.coh_tmm(p, [n1, n2], [inf, inf], np.radians(a), 0.6)["R"] for a in angles])
            for p in "sp"}
    exact = {p: np.array([fresnel(p, a) for a in angles[::every]]) for p in "sp"}
    return result(
        angles,
        [line(f"R_{p}", f"R_{p}, PyStratify", ours[p]) for p in "sp"]
        + [dots(f"R_{p}_ref", f"R_{p}, Fresnel closed form", exact[p], every, size=angles.size) for p in "sp"],
        worst=max(largest(ours[p][::every], exact[p]) for p in "sp"),
        worst_label="largest |R − R_Fresnel| over both polarizations",
        note="Brewster's angle, where R_p vanishes, is arctan(1.5) = 56.3°; the two curves coincide to machine "
             "precision at every angle.")


@figure("airy-slab", "Fabry–Pérot slab", "λ [nm]", "R")
def plot_fabry_perot_slab():
    lam = np.linspace(400, 800, 201)
    every = 5  # the Airy formula every 10 nm
    n, d = 2.0, 0.3
    ours = np.array([planar.coh_tmm("s", [1.0, n, 1.0], [inf, d, inf], 0.0, w * 1e-3)["R"] for w in lam])
    r1 = ((n - 1) / (n + 1)) ** 2
    F = 4 * r1 / (1 - r1) ** 2
    delta = 2 * np.pi * n * d / (lam[::every] * 1e-3)
    airy = F * np.sin(delta) ** 2 / (1 + F * np.sin(delta) ** 2)
    return result(
        lam,
        [line("R", "R, PyStratify", ours), dots("R_ref", "R, Airy formula", airy, every, size=lam.size)],
        worst=largest(ours[::every], airy),
        worst_label="largest |R − R_Airy|",
        note="R = F sin²δ / (1 + F sin²δ) with F = 4R₁/(1 − R₁)², R₁ the single-surface reflectance and δ = 2πnd/λ: "
             "the fringes of a Fabry–Pérot slab.")


@figure("energy-conservation", "Lossless stack", "angle of incidence in glass [°]", "|1 − R − T|", log=True)
def plot_lossless_stack():
    angles = np.linspace(0, 89, 179)
    n = [1.52, 2.3, 1.38, 2.3, 1.38, 2.3, 1.0]
    d = [inf, 0.08, 0.12, 0.08, 0.12, 0.08, inf]
    out = {}
    for pol in ("s", "p"):
        rows = [planar.coh_tmm(pol, n, d, np.radians(a), 0.55) for a in angles]
        out[pol] = np.maximum(np.abs(1 - np.array([r["R"] + r["T"] for r in rows])), 1e-17)
    return result(
        angles,
        [line(p, f"|1 − R − T|, {p} polarization", out[p]) for p in "sp"],
        worst=max(out["s"].max(), out["p"].max()),
        worst_label="largest |1 − R − T|",
        note="Beyond the critical angle (41.1° in this glass) T is exactly zero and R is exactly one; the residual "
             "is floating-point rounding, floored at 10⁻¹⁷ for the logarithmic axis.")


@figure("opaque-film", "Opaque metal film", "thickness [nm]", "R, T")
def plot_opaque_metal_film():
    n = 0.2 + 3j
    t = np.linspace(1, 150, 150)
    every = 5  # the bulk value marked every 5 nm
    rows = [planar.coh_tmm("s", [1, n, 1], [inf, w * 1e-3, inf], 0, 0.6) for w in t]
    bulk = abs((1 - n) / (1 + n)) ** 2
    thick = planar.coh_tmm("s", [1, n, 1], [inf, 20.0, inf], 0, 0.6)
    return result(
        t,
        [line("R", "R, PyStratify", [r["R"] for r in rows]), line("T", "T, PyStratify", [r["T"] for r in rows]),
         dots("R_ref", "|(1 − n)/(1 + n)|², the bulk Fresnel reflectance", np.full(t[::every].size, bulk), every,
              size=t.size)],
        worst=abs(thick["R"] - bulk),
        worst_label="|R − R_bulk| at 20 µm (the test's opaque film)",
        note="A thin metal film transmits; as it thickens T falls exponentially and R settles on the reflectance of "
             "the bulk metal. The test takes the film to 20 µm, where T must be exactly 0 and R, the Poynting "
             "vector at 0, 10 and 20 µm finite (no overflow of exp(Im(k_z) d)), and checks Fresnel's 0.04 and 0.96 "
             "for glass at normal incidence.")


def _bare_substrate(part):
    """Ψ or Δ of three bare substrates against −r_p / r_s from Fresnel, 633 nm."""
    from planar_checks import bare_substrate_rho

    angles = np.linspace(1, 89, 89)
    every = 3  # Fresnel every 3°
    cases = ((1.5 + 0j, "n = 1.5"), (1.5 + 0.3j, "n = 1.5 + 0.3i"), (0.2 + 3j, "n = 0.2 + 3i"))
    if part == "psi":
        def drawn(z):
            return np.degrees(np.arctan(np.abs(z)))
    else:
        def drawn(z):
            return np.degrees(np.angle(z)) % 360
    lines, refs, diffs = [], [], []
    for i, (n_sub, label) in enumerate(cases):
        ours = np.array([bare_substrate_rho(n_sub, a)[0] for a in angles])
        theirs = np.array([bare_substrate_rho(n_sub, a)[1] for a in angles[::every]])
        lines.append(line(f"c{i}", f"{label}, PyStratify", drawn(ours)))
        refs.append(dots(f"c{i}_ref", f"{label}, −r_p / r_s from Fresnel", drawn(theirs), every, size=angles.size))
        diffs.append(np.max(np.abs(ours[::every] - theirs)))
    return result(
        angles, lines + refs, worst=max(diffs), worst_label="largest |ρ − (−r_p / r_s)|",
        note="Ellipsometry's ratio ρ = tan Ψ e^{iΔ} of a bare substrate is the ratio of its Fresnel coefficients, "
             "−r_p / r_s in PyStratify's sign convention; the test asserts four (substrate, angle) pairs to 1e-12, "
             "the plot sweeps the angle, at 633 nm. For the lossless n = 1.5 Δ jumps by 180° at Brewster's angle, "
             "56.3°.")


@figure("bare-substrate-psi", "Bare substrates: Ψ", "angle of incidence [°]", "Ψ [°]")
def plot_bare_substrates_psi():
    return _bare_substrate("psi")


@figure("bare-substrate-delta", "Bare substrates: Δ", "angle of incidence [°]", "Δ [°]")
def plot_bare_substrates_delta():
    return _bare_substrate("delta")


def _smatrix_figure(stack, what):
    """R and T, s and p, of a stack against the scattering-matrix reference, 0–89°."""
    from planar_checks import smatrix_rt

    n, d, lam = stack
    angles = np.linspace(0, 89, 90)
    every = 3  # the S-matrix every 3°
    ours, theirs = {}, {}
    for pol in ("s", "p"):
        rows = [planar.coh_tmm(pol, n, d, np.radians(a), lam) for a in angles]
        ours[pol] = np.array([[r["R"], r["T"]] for r in rows])
        theirs[pol] = np.array([smatrix_rt(pol, n, d, np.radians(a), lam) for a in angles[::every]])
    return result(
        angles,
        [line(f"{q}_{p}", f"{q}_{p}, PyStratify", ours[p][:, i]) for p in "sp" for i, q in enumerate("RT")]
        + [dots(f"{q}_{p}_ref", f"{q}_{p}, S-matrix", theirs[p][:, i], every, size=angles.size)
           for p in "sp" for i, q in enumerate("RT")],
        worst=max(largest(ours[p][::every], theirs[p]) for p in "sp"),
        worst_label="largest |R − R_S|, |T − T_S|",
        note=f"{what} The reference chains the interfaces' and layers' two-port scattering matrices by Redheffer's "
             "star product (planar_checks.py), a different algebra from PyStratify's transfer matrices that never "
             "multiplies by exp(+Im(k_z) d); the test also takes 120 µm of n = 2 + 0.5i, where T is below 1e-30 in "
             "both.")


@figure("smatrix-metal-stack", "Metal stack", "angle of incidence [°]", "R, T")
def plot_metal_stack():
    from planar_checks import METAL_STACK

    return _smatrix_figure(METAL_STACK, "An absorbing stack, s and p.")


@figure("smatrix-beyond-critical", "Frustrated reflection", "angle of incidence in glass [°]", "R, T")
def plot_frustrated_reflection():
    from planar_checks import BEYOND_CRITICAL

    return _smatrix_figure(BEYOND_CRITICAL, "Beyond 41.8° the air gap is evanescent and light tunnels through it to "
                                            "the n = 2 layer and the glass.")


@figure("incoherent-phase-average", "Thick incoherent layer", "angle of incidence [°]", "R, T")
def plot_thick_incoherent_layer():
    from planar_checks import explicit_phase_average

    n, d, c = INCOHERENT
    angles = np.linspace(0, 80, 33)
    ours, ref = {}, {}
    for pol in ("s", "p"):
        rows = [planar.pim_tmm(pol, n, d, c, np.radians(a), 0.64) for a in angles]
        ours[pol] = np.array([[r["R"], r["T"]] for r in rows])
        sampled = [explicit_phase_average(pol, n, d, c, a, 0.64, samples_per_layer=301) for a in angles]
        ref[pol] = np.array([[e["reflectance"], e["transmittance"]] for e in sampled])
    return result(
        angles,
        [line(f"{q}_{p}", f"{q}_{p}, PyStratify", ours[p][:, i]) for p in "sp" for i, q in enumerate("RT")]
        + [dots(f"{q}_{p}_ref", f"{q}_{p}, phase average of coherent solves", ref[p][:, i])
           for p in "sp" for i, q in enumerate("RT")],
        worst=max(largest(ours[p], ref[p]) for p in "sp"),
        worst_label="largest |difference|",
        note="An incoherent layer is a coherent one whose round-trip phase is averaged out: the reference solves "
             "the whole stack coherently 301 times with that layer's phase spread over one period (its absorption "
             "kept) and averages R and T; PyStratify's incoherent transfer matrix does it in closed form. The test "
             "also compares the absorption in each layer.")


@figure("rcwa-unpatterned", "Three-layer filter", "λ [nm]", "R, T")
def plot_three_layer_filter():
    from rcwa_meent import meent_rt, stratify_rt

    lam = np.linspace(400, 800, 81)
    every = 2  # meent every 10 nm
    ours = np.array([stratify_rt(w * 1e-3) for w in lam])
    theirs = np.array([meent_rt(w * 1e-3) for w in lam[::every]])
    return result(
        lam,
        [line(q, f"{q}, PyStratify", ours[:, i]) for i, q in enumerate("RT")]
        + [dots(f"{q}_ref", f"{q}, meent RCWA", theirs[:, i], every, size=lam.size) for i, q in enumerate("RT")],
        worst=largest(ours[::every], theirs),
        worst_label="largest |difference|",
        note="Rigorous coupled-wave analysis, the calculator's grating engine (meent, MIT), with uniform layers and "
             "one Fourier order must reduce to the transfer matrix. Normal incidence, s; rcwa_meent.py, which the "
             "test runs at 0° and 30°, s and p.")


@figure("position-resolved-absorption", "Absorption per layer", "λ [nm]", "absorptance per layer")
def plot_absorption_per_layer():
    from planar_checks import integrated_absorption

    lam = np.linspace(400, 1000, 61)
    every = 2  # the Poynting difference every 20 nm
    pairs = [integrated_absorption("p", w * 1e-3) for w in lam]
    ours, ref = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs[::every]])
    labels = ("120 nm n = 2.1 + 0.08i", "250 nm n = 1.46 + 0.01i", "20 nm n = 0.2 + 3.2i")
    return result(
        lam,
        [line(f"l{i}", f"{lab}, ∫ density (PyStratify)", ours[:, i]) for i, lab in enumerate(labels)]
        + [dots(f"l{i}_ref", f"{lab}, Poynting difference", ref[:, i], every, size=lam.size)
           for i, lab in enumerate(labels)],
        worst=largest(ours[::every], ref),
        worst_label="largest |difference|",
        note="position_resolved gives the field and the absorption density at any depth; integrated through a layer "
             "(200 Gauss–Legendre points) it must equal the power the Poynting flux loses across that layer. p "
             "polarization, air above, glass below. The focused-beam near field builds on the same fields (its "
             "Parseval check is the focused-beam row).")


@figure("kramers-kronig-closed-forms", "Kramers–Kronig", "E [eV]", "ε₁")
def plot_kramers_kronig():
    from dispersion_codes import cauchy_weight_kk

    from pystratify import dispersion as dm

    energy = np.linspace(0.25, 9.0, 176)  # every 0.05 eV
    every = 5  # the principal value every 0.25 eV
    tauc = dm.TaucLorentz(A=120, E0=3.6, C=2.4, Eg=1.6)
    cody = dm.CodyLorentz(A=100, E0=3.6, Gamma=2.4, Eg=1.6, Ep=0.8, Et=1.8, Eu=0.05)

    def tauc_eps2(e):  # zero below the gap by definition, so the quadrature may start at E = 0
        e = np.atleast_1d(np.asarray(e, float))
        out = np.zeros_like(e)
        above = e > tauc.Eg
        out[above] = tauc.epsilon(dm.HC_EV_NM / e[above]).imag
        return out

    ours_tl, ours_cl = tauc.epsilon(dm.HC_EV_NM / energy).real, cody.epsilon(dm.HC_EV_NM / energy).real
    ref_tl = np.array([cauchy_weight_kk(tauc_eps2, e, (1.6, 3.6, 20.0, 200.0)) for e in energy[::every]])
    ref_cl = np.array([cauchy_weight_kk(cody.eps2, e, (1.6, 1.8, 3.6, 20.0, 200.0)) for e in energy[::every]])
    scale = max(np.max(np.abs(ours_tl)), np.max(np.abs(ours_cl)))
    return result(
        energy,
        [line("tl", "Tauc-Lorentz ε₁, closed form", ours_tl), line("cl", "Cody-Lorentz ε₁, closed form", ours_cl),
         dots("tl_ref", "Tauc-Lorentz, principal value of ε₂", ref_tl, every, size=energy.size),
         dots("cl_ref", "Cody-Lorentz, principal value of ε₂", ref_cl, every, size=energy.size)],
        worst=max(largest(ours_tl[::every], ref_tl), largest(ours_cl[::every], ref_cl)) / scale,
        worst_label="largest |Δε₁| / max |ε₁|",
        note="The reference is (2/π) P∫ E′ε₂ / (E′² − E²) dE′ by QUADPACK with the Cauchy weight on the interval "
             "holding E, split at the kinks (benchmarks/dispersion_codes.py cauchy_weight_kk), independent of the "
             "module's own Kramers-Kronig subtraction. The tests also check Gaussian and Lorentz ε₁ against "
             "dispersion.kramers_kronig_real, and that Cody-Lorentz ε₁ stays continuous across E_g and E_t.")


if __name__ == "__main__":
    run(__file__)
