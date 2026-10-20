"""Spheres, point dipole: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_dipole_consistency.py                                                           #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-consistency.html, computed by this one file from a PyStratify clone: it
uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and writes, beside itself,
spheres_dipole_consistency.json (the curves the page draws, value for value) and spheres_dipole_consistency.png (needs matplotlib).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "benchmarks", ROOT / "benchmarks" / "pages"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pystratify as ps  # noqa: E402

from _page import figure, run, series, sig, worst  # noqa: E402


@figure("sphere-decay-mpmath", "Metal sphere", "distance from the surface [nm]", "Γ / Γ₀", log=True)
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


@figure("silver-decay-mie", "Silver sphere", "distance from the surface [nm]", "Γ / Γ₀", log=True)
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


@figure("nanoshell-decay-layers", "Gold nanoshell", "distance from the centre [nm]", "Γ / Γ₀", log=True)
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


@figure("silicon-magnetic-decay", "Silicon sphere", "distance from the surface [nm]", "Γ / Γ₀", log=True)
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


@figure("radiated-power-rates", "Magnetic multilayer", "distance from the centre [nm]", "Γ_rad / Γ₀", log=True)
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


@figure("two-point-green", "Two-point Green's dyadic", "order ℓ", "|g_ℓ|", log=True)
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


@figure("quadrupole-coupling", "Quadrupole coupling", "order ℓ", "|coupling|", log=True)
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


@figure("casimir-non-retarded", "Casimir–Polder potential", "distance from the surface [nm]", "|U| [J]", log=True)
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


if __name__ == "__main__":
    run(__file__)
