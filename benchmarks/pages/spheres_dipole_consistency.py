"""Spheres, point dipole: PyStratify consistency checks.

#####################################################################################################################
#                                                                                                                   #
#   git clone https://github.com/multilayerapp/PyStratify && cd PyStratify                                          #
#   pip install -e ".[benchmarks,references]" matplotlib                                                            #
#   python benchmarks/pages/spheres_dipole_consistency.py                                                           #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/spheres-dipole-consistency.html, computed by this one file from
a PyStratify clone: it uses the benchmark references beside it (benchmarks/*.py). It prints each figure's RMSE and
writes, beside itself, spheres_dipole_consistency.json (the curves the page draws, value for value) and
spheres_dipole_consistency.png (needs matplotlib).
"""

from __future__ import annotations

import numpy as np

# _page first: it puts the clone's src/ and benchmarks/ on the path
from _page import dots, figure, largest, line, result, run

import pystratify as ps

RATES = (("tot_perp", "total, radial"), ("tot_par", "total, tangential"),
         ("rad_perp", "radiative, radial"), ("rad_par", "radiative, tangential"))


def _rate_curves(ours, ref, reference, count=4, every=1):
    """PyStratify's decay rates (total then radiative, radial then tangential) against a reference's, the first
    `count` of them; ref holds the reference at every `every`-th point."""
    got = np.column_stack([ours.total, ours.radiative])
    size = got.shape[0]
    return ([line(key, f"{what}, PyStratify", got[:, i]) for i, (key, what) in enumerate(RATES[:count])]
            + [dots(f"{key}_ref", f"{what}, {reference}", ref[:, i], every, size=size)
               for i, (key, what) in enumerate(RATES[:count])])


@figure("sphere-decay-mpmath", "Metal sphere", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def plot_metal_sphere():
    from pystratify.references import sphere_decay_rates

    gaps = np.geomspace(1, 300, 36)
    radius, m, lam = 25.0, 0.05 + 3.5j, 600.0
    r = radius + gaps
    ours = ps.decay_rates([radius], np.array([m, 1.0]), lam, r, tol=1e-10)
    ref = np.array([sphere_decay_rates(radius, m, lam, rr, orders=400, dps=40) for rr in r])
    return result(
        gaps, _rate_curves(ours, ref, "40 digits"),
        worst=largest(np.column_stack([ours.total, ours.radiative]), ref, relative=True),
        worst_label="largest relative difference over the four rates",
        note="Chew's sums over 400 orders in mpmath at 40 digits against PyStratify's adaptive order and "
             "logarithmic amplitudes; the total rate climbs as d⁻³ at contact (quenching) while the radiative rate "
             "stays finite.")


@figure("silver-decay-mie", "Silver sphere", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def plot_silver_sphere():
    from pystratify.references import sphere_decay_rates

    radius, ag, lam = 50.0, 0.093 + 4j, 633.0
    gaps = np.geomspace(1, 100, 41)
    ours = ps.decay_rates([radius], [ag, 1.0], lam, radius + gaps, tol=1e-10)
    ref = np.array([sphere_decay_rates(radius, ag, lam, radius + g, orders=int(min(1400, 40 + 30 * radius / g)))
                    for g in gaps])
    return result(
        gaps, _rate_curves(ours, ref, "Mie (40 digits)"),
        worst=largest(np.column_stack([ours.total, ours.radiative]), ref, relative=True),
        worst_label="largest relative difference over the four rates",
        note="Majic & Le Ru's Eqs. 34–37 summed in mpmath (up to 1400 orders at 1 nm) against PyStratify's adaptive "
             "order; at contact the total rate is quenching, the nonradiative part, while the radiative rate stays "
             "finite.")


@figure("nanoshell-decay-layers", "Gold nanoshell", "distance from the centre [nm]", "Γ / Γ₀", log=True)
def plot_gold_nanoshell():
    from pystratify.references import layered_decay_rates

    radii, n, lam = [50.0, 55.0], [1.45, 0.27 + 2.93j, 1.33], 614.0
    # not in the gold: an emitter in an absorbing shell has no rate
    r = np.r_[np.linspace(5, 48, 22), np.geomspace(57, 150, 22)]
    every = 2  # 60 digits at every other point
    ours = ps.decay_rates(radii, n, lam, r, normalization="shell", tol=1e-10)

    def orders(rd):
        gap = min(abs(rd - v) for v in radii)
        return int(min(1200, 40 + 30 * radii[-1] / gap))

    ref = np.array([np.concatenate(layered_decay_rates(radii, n, lam, rd, orders(rd), 60)) for rd in r[::every]])
    return result(
        r, _rate_curves(ours, ref, "60 digits", count=2, every=every),
        worst=largest(np.column_stack([ours.total, ours.radiative])[::every], ref, relative=True),
        worst_label="largest relative difference (total and radiative)",
        note="Rates relative to the dipole in the layer it sits in (the silica core or the water), against transfer "
             "matrices evaluated in 60-digit arithmetic. The test also asserts 0.5–1 nm from the gold, with "
             "1250–2400 orders.")


@figure("silicon-magnetic-decay", "Silicon sphere", "distance from the surface [nm]", "Γ / Γ₀", log=True)
def plot_silicon_sphere():
    from pystratify.references import layered_decay_rates

    radius, n, lam = 100.0, [3.7 + 0.005j, 1.0], 800.0
    gaps = np.geomspace(2, 300, 41)
    ours = ps.decay_rates([radius], n, lam, radius + gaps, [1, 1], dipole="magnetic", normalization="shell",
                          tol=1e-10)
    ref = np.array([np.concatenate(layered_decay_rates([radius], n, lam, radius + g,
                                                       int(min(800, 40 + 30 * radius / g)), 60,
                                                       mu=[1, 1], dipole="magnetic")) for g in gaps])
    return result(
        gaps, _rate_curves(ours, ref, "60 digits", count=2),
        worst=largest(np.column_stack([ours.total, ours.radiative]), ref, relative=True),
        worst_label="largest relative difference (total and radiative)",
        note="A magnetic dipole couples to the sphere's magnetic Mie mode; the reference is the same "
             "transfer-matrix formulation in 60-digit arithmetic.")


@figure("radiated-power-rates", "Magnetic multilayer", "distance from the centre [nm]", "Γ_rad / Γ₀", log=True)
def plot_magnetic_multilayer():
    radii, n, mu = [50.0, 70.0, 90.0], [1.45, 0.27 + 2.93j, 1.6, 1.33], [1.3, 1.0, 1.7, 1.1]
    r = np.r_[np.linspace(5, 47, 12), np.linspace(73, 87, 6), np.geomspace(93, 400, 20)]
    out = {}
    for dipole in ("electric", "magnetic"):
        rates = ps.decay_rates(radii, n, 614.0, r, mu, dipole=dipole, tol=1e-10)
        power = np.array([ps.dipole_far_field(radii, n, 614.0, [0.0, 0.0, r0], [0.0, 0.0, 1.0], 0.0, 0.0, mu,
                                              dipole=dipole).power for r0 in r])
        out[dipole] = rates.radiative[:, 0], power
    return result(
        r,
        [line("e", "electric, radiative rate (PyStratify)", out["electric"][0]),
         line("m", "magnetic, radiative rate (PyStratify)", out["magnetic"][0]),
         dots("e_ref", "electric, power in the far field", out["electric"][1]),
         dots("m_ref", "magnetic, power in the far field", out["magnetic"][1])],
        worst=max(largest(*out["electric"], relative=True), largest(*out["magnetic"], relative=True)),
        worst_label="largest relative difference",
        note="Two routes to the same number for a radial dipole: the radiative rate from the decay-rate sums, and "
             "the power of the far field the dipole radiates through every layer. The gaps in r are the gold shell, "
             "where an emitter is not placed.")


@figure("two-point-green", "Two-point Green's dyadic", "order ℓ", "|g_ℓ|", log=True)
def plot_two_point_greens_dyadic():
    from sphere_emitter_checks import two_point_terms

    from pystratify.solver import TE, TM

    terms = two_point_terms(56.0, 60.0)  # the 80-digit recursion runs through every order to 120
    orders = np.arange(1, 121)
    every = 3  # drawn at every third order
    (tm, tm_ref), (te, te_ref) = terms[TM], terms[TE]
    w = 2 * orders + 1
    return result(
        orders,
        [line("tm", "TM, PyStratify", np.abs(tm)), line("te", "TE, PyStratify", np.abs(te)),
         dots("tm_ref", "TM, 80 digits", np.abs(tm_ref)[::every], every, size=orders.size),
         dots("te_ref", "TE, 80 digits", np.abs(te_ref)[::every], every, size=orders.size)],
        worst=max(np.sum(w * np.abs(tm - tm_ref)) / np.sum(w * np.abs(tm_ref)),
                  np.sum(w * np.abs(te - te_ref)) / np.sum(w * np.abs(te_ref))),
        worst_label="Σ(2ℓ+1)|g − g₈₀| / Σ(2ℓ+1)|g₈₀|",
        note="The radial part of the Green's dyadic between two points, per order and polarisation, which every "
             "dipole–dipole coupling near the sphere is built from: PyStratify's normalized form against transfer "
             "matrices in 80-digit arithmetic.")


@figure("quadrupole-coupling", "Quadrupole coupling", "order ℓ", "|coupling|", log=True)
def plot_quadrupole_coupling():
    from sphere_emitter_checks import quadrupole_values

    orders = np.arange(1, 9)
    values = [quadrupole_values(int(ell)) for ell in orders]
    waves = {"psi": "regular wave (ψ)", "xi": "outgoing wave (ξ)"}  # M − N has the same magnitudes, axial family
    ours = {k: np.array([abs(v[k, (0, "e"), 1][0]) for v in values]) for k in waves}
    ref = {k: np.array([abs(v[k, (0, "e"), 1][1]) for v in values]) for k in waves}
    return result(
        orders,
        [line(f"{k}1", f"{label}, PyStratify", ours[k]) for k, label in waves.items()]
        + [dots(f"{k}1_ref", f"{label}, finite differences", ref[k]) for k, label in waves.items()],
        worst=max(abs(g - r) / max(abs(r), 1e-300) for v in values for g, r in v.values()),
        worst_label="largest relative difference over every family",
        note="(1/6) Σ Q_ij ∂_j W_i of the vector spherical waves W = M + N (Bohren & Huffman's, the axial m = 0 "
             "family shown; the difference covers all five families and M − N), PyStratify's closed-form functional "
             "against central differences of the 40-digit waves. The regular wave fades with order at this distance "
             "and the outgoing one grows.")


@figure("casimir-non-retarded", "Casimir–Polder potential", "distance from the surface [nm]", "|U| [J]", log=True)
def plot_casimir_polder_potential():
    from scipy.integrate import quad
    from sphere_emitter_checks import _alpha, _eps

    from pystratify.casimir import HBAR

    def integrand(u):  # α(iξ)(ε − 1)/(ε + 1) ξ, on a logarithmic ξ = e^u
        x = np.exp(u)
        return x * _alpha(x) * (_eps(x)[0] - 1) / (_eps(x)[0] + 1)

    c3 = HBAR / (4 * np.pi) * quad(integrand, np.log(1e6), np.log(1e21), limit=500, epsabs=0, epsrel=1e-12)[0]
    R = 200.0
    d = np.geomspace(0.5, 50, 25)
    out = ps.casimir_polder([R], _eps, R + d, _alpha, nodes=64)
    return result(
        d,
        [line("u", "|U|, PyStratify", np.abs(out.energy)), dots("u_ref", "C₃/d³, non-retarded limit", c3 / d ** 3)],
        worst=np.max(np.abs(-out.energy[:2] * d[:2] ** 3 / c3 - 1)),
        worst_label="relative difference within 0.6 nm",
        note="Within a nanometre the potential is the flat-surface, non-retarded −C₃/d³ with C₃ = (ħ/4π)∫α(iξ)(ε − "
             "1)/(ε + 1) dξ; further out retardation and the sphere's curvature weaken it, so the curves part by "
             "design: 0.3% at 0.5 nm, 3% at 5 nm, 37% at 50 nm.")


if __name__ == "__main__":
    run(__file__)
