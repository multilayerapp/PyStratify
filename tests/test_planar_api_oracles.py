"""`planar.psi_delta` and `position_resolved`'s H against the multilayer.app API code they replace (WP07-D2).

The API's `serialize_ellipsometry_amplitudes` and `magnetic_field_components` are frozen verbatim in
`tests/oracles/multilayer_api_ef9c946.py` (commit ef9c946). Random stacks are drawn as the D2 evidence's section A
draws them (`umbrella-assurance/docs/assurance/evidence/wp07_d2_duplicates.py`): 1-8 layers, n 1.3-3.5 with or
without k up to 0.5, 5-500 nm, a substrate n 1.4-4.0 with or without k up to 0.3, 0-85 deg, 300-1200 nm; here a
quarter of the H stacks are magnetic too (the API serves mu != 1 for coherent stacks).
"""

import math

import numpy as np
import pytest

from pystratify import planar

from .oracles import multilayer_api_ef9c946 as api

PSI_STACKS = 3000
H_STACKS = 2000


def random_stack(rng, magnetic=False):
    layers = int(rng.integers(1, 9))
    n = [complex(rng.uniform(1.3, 3.5), rng.choice([0.0, rng.uniform(0, 0.5)])) for _ in range(layers)]
    d_nm = [float(rng.uniform(5, 500)) for _ in range(layers)]
    sub = complex(rng.uniform(1.4, 4.0), rng.choice([0.0, rng.uniform(0, 0.3)]))
    mu = [1.0, *(float(rng.uniform(0.6, 2.0)) for _ in range(layers)), 1.0] if magnetic else None
    return [1.0, *n, sub], [np.inf, *[d / 1000 for d in d_nm], np.inf], mu


def amplitudes(rng, cases):
    r_s, r_p = [], []
    for _ in range(cases):
        n_list, d_list, _ = random_stack(rng)
        angle, lam = math.radians(float(rng.uniform(0, 85))), float(rng.uniform(300, 1200)) / 1000
        r_s.append(planar.coh_tmm("s", n_list, d_list, angle, lam)["r"])
        r_p.append(planar.coh_tmm("p", n_list, d_list, angle, lam)["r"])
    return np.array(r_s), np.array(r_p)


def api_format(psi, delta, rounding=api.round_float):
    """What the API keeps after the switch: degrees, Delta in [0, 360), None for NaN, its serialization rounding."""
    out_psi = None if np.isnan(psi) else rounding(np.degrees(psi))
    out_delta = None
    if not np.isnan(delta):
        out_delta = rounding(np.degrees(delta) % 360.0)
        if out_delta >= 360.0:
            out_delta = 0.0
    return {"psi_deg": out_psi, "delta_deg": out_delta}


@pytest.fixture(scope="module")
def random_amplitudes():
    return amplitudes(np.random.default_rng(20261103), PSI_STACKS)


def test_psi_delta_formatted_is_the_api_on_random_stacks(random_amplitudes):
    r_s, r_p = random_amplitudes
    psi, delta = planar.psi_delta(r_s, r_p)
    for i in range(len(r_s)):
        assert api_format(psi[i], delta[i]) == api.serialize_ellipsometry_amplitudes(r_s[i], r_p[i])


def test_psi_delta_unrounded_matches_the_api_formula(random_amplitudes, monkeypatch):
    """The oracle with its serialization rounding switched off, to 1e-12 deg."""
    monkeypatch.setattr(api, "round_float", float)
    r_s, r_p = random_amplitudes
    psi, delta = planar.psi_delta(r_s, r_p)
    worst = 0.0
    for i in range(len(r_s)):
        ours = api_format(psi[i], delta[i], rounding=float)
        theirs = api.serialize_ellipsometry_amplitudes(r_s[i], r_p[i])
        worst = max(worst, abs(ours["psi_deg"] - theirs["psi_deg"]),
                    abs((ours["delta_deg"] - theirs["delta_deg"] + 180.0) % 360.0 - 180.0))
    assert worst <= 1e-12


def test_psi_delta_is_vectorised(random_amplitudes):
    r_s, r_p = random_amplitudes
    psi, delta = planar.psi_delta(r_s.reshape(30, -1), r_p.reshape(30, -1))
    assert psi.shape == delta.shape == (30, PSI_STACKS // 30)
    for i in range(0, PSI_STACKS, 97):
        one_psi, one_delta = planar.psi_delta(r_s[i], r_p[i])
        assert np.ndim(one_psi) == np.ndim(one_delta) == 0
        assert one_psi == psi.ravel()[i] and one_delta == delta.ravel()[i]


EPS = np.finfo(float).eps
EDGES = [(0.0, 0.0), (0.0, 0.3 + 0.1j), (-0.2, 0.0), (1e-15, 0.5), (0.5, 1e-15j), (64 * EPS, 0.3),
         (65 * EPS, -0.3), (0.3, 70 * EPS * 1j), (2.5, 2.5 * 63 * EPS), (2.5, 2.5 * 65 * EPS), (3.0 + 4.0j, -1e-13),
         (1e-300, 1e-300), (-0.2, 0.2), (-0.2, -0.2), (0.2j, -0.2)]


@pytest.mark.parametrize("r_s, r_p", EDGES)
def test_psi_delta_where_an_amplitude_vanishes(r_s, r_p):
    """The API's null rule at the floor 64 eps max(|r_s|, |r_p|, 1), on both sides of it."""
    psi, delta = planar.psi_delta(r_s, r_p)
    assert api_format(psi, delta) == api.serialize_ellipsometry_amplitudes(complex(r_s), complex(r_p))


def test_psi_delta_of_nan_is_nan():
    """Not compared with the oracle: its Python max(|r_s|, nan, 1) depends on argument order."""
    for r_s, r_p in ((np.nan, 0.3), (0.3, np.nan), (np.nan, np.nan)):
        psi, delta = planar.psi_delta(r_s, r_p)
        assert np.isnan(psi) and np.isnan(delta)


def test_ellips_at_brewster_returns_nan_delta_not_minus_pi():
    brewster = math.atan(1.5)
    card = planar.ellips([1.0, 1.5], [np.inf, np.inf], brewster, 0.6)
    assert card["psi"] < 1e-15 and np.isnan(card["Delta"])
    assert api_format(card["psi"], card["Delta"]) == {"psi_deg": 0.0, "delta_deg": None}


def test_ellips_is_unchanged_where_both_amplitudes_exist(random_amplitudes):
    """psi from atan2 against the old arctan(|r_p/r_s|), Delta from -r_p conj(r_s) against the old angle(-r_p/r_s)."""
    r_s, r_p = random_amplitudes
    psi, delta = planar.psi_delta(r_s, r_p)
    assert np.max(np.abs(psi - np.arctan(np.abs(r_p / r_s)))) <= 1e-15
    assert np.max(np.abs(np.angle(np.exp(1j * (delta - np.angle(-r_p / r_s)))))) <= 1e-14
    assert np.all(delta > -np.pi) and np.all(delta <= np.pi)


def test_ellips_passes_mu_to_coh_tmm():
    n_list, d_list, mu = [1.0, 1.8 + 0.05j, 1.5], [np.inf, 0.12, np.inf], [1.0, 1.6, 1.0]
    card = planar.ellips(n_list, d_list, 0.9, 0.6, mu_list=mu)
    psi, delta = planar.psi_delta(planar.coh_tmm("s", n_list, d_list, 0.9, 0.6, mu_list=mu)["r"],
                                  planar.coh_tmm("p", n_list, d_list, 0.9, 0.6, mu_list=mu)["r"])
    assert card == {"psi": psi, "Delta": delta}
    assert abs(card["psi"] - planar.ellips(n_list, d_list, 0.9, 0.6)["psi"]) > 1e-3


def test_h_is_the_api_formula_on_random_stacks():
    """position_resolved's Hx, Hy, Hz against magnetic_field_components (and |H|^2 against
    magnetic_field_intensity) in both half-spaces and at three depths of every layer."""
    rng = np.random.default_rng(20261104)
    worst = worst_intensity = 0.0
    points = 0
    for index in range(H_STACKS):
        n_list, d_list, mu = random_stack(rng, magnetic=index % 4 == 3)
        angle, lam = math.radians(float(rng.uniform(0, 85))), float(rng.uniform(300, 1200)) / 1000
        for pol in "sp":
            data = planar.coh_tmm(pol, n_list, d_list, angle, lam, mu_list=mu)
            last = len(n_list) - 1
            for layer in range(len(n_list)):
                depths = (-0.05, 0.0) if layer == 0 else (0.0, 0.2) if layer == last else (0.0, d_list[layer] / 2,
                                                                                            d_list[layer])
                for z in depths:
                    f = planar.position_resolved(layer, z, data)
                    ours = (f["Hx"], f["Hy"], f["Hz"])
                    theirs = api.magnetic_field_components(layer, z, data)
                    scale = max(1.0, *(abs(c) for c in theirs))
                    worst = max(worst, *(abs(a - b) / scale for a, b in zip(ours, theirs)))
                    h2 = sum(abs(c) ** 2 for c in ours)
                    worst_intensity = max(worst_intensity,
                                          abs(h2 - api.magnetic_field_intensity(layer, z, data)) / max(1.0, h2))
                    points += 1
    assert points > 20 * H_STACKS
    assert worst <= 1e-12 and worst_intensity <= 1e-12
