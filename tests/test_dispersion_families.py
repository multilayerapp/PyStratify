"""Tauc-Lorentz-Urbach, Forouhi-Bloomer and carrier-density Drude, ported from the harvester (WP07-D2),
against the frozen harvester code (tests/oracles/harvester_nk_model_table.py, thinfilm-harvester 36afd65)."""

import numpy as np
import pytest

from pystratify import dispersion as dm
from tests.oracles import harvester_nk_model_table as oracle

TLU_SETS = [
    dict(A=118.0, E0=3.9, C=2.2, Eg=1.9, Ec=2.1),
    dict(A=60.0, E0=4.5, C=1.0, Eg=3.0, Ec=3.3),
    dict(A=150.0, E0=3.4, C=3.1, Eg=1.1, Ec=1.4),
]
ENERGIES = np.concatenate([np.geomspace(0.05, 40.0, 700), np.linspace(0.4, 8.0, 301)])


def tlu_params(p, einf=0.0):
    return dict(A_eV=p["A"], E0_eV=p["E0"], C_eV=p["C"], Eg_eV=p["Eg"], Ec_eV=p["Ec"], eps1_inf=einf)


def scaled(a, b):
    """max |a - b| / |b| over the points (complex)."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(b)))


@pytest.mark.parametrize("p", TLU_SETS)
def test_tlu_matches_harvester(p):
    model = dm.TaucLorentzUrbach(**p)
    ec = p["Ec"]
    energy = np.sort(np.concatenate([ENERGIES, ec * (1 + np.array([-1e-3, -1e-6, -1e-9, 0, 1e-9, 1e-6, 1e-3]))]))
    old = oracle.tlu_eps(tlu_params(p), energy)
    assert scaled(model._epsilon(energy, None), old) <= 1e-9
    assert np.max(np.abs(model.eps2(energy) - oracle.tlu_eps2(tlu_params(p), energy))) == 0 or scaled(
        model.eps2(energy) + 0j, oracle.tlu_eps2(tlu_params(p), energy) + 0j
    ) <= 1e-9
    eu, au = model.urbach()
    old_eu, old_au = oracle.tlu_urbach(tlu_params(p))
    assert eu == pytest.approx(old_eu, rel=1e-12) and au == pytest.approx(old_au, rel=1e-12)
    # through the public path: wavelengths in nm
    wavelength = dm.HC_EV_NM / ENERGIES
    assert scaled(model.epsilon(wavelength), oracle.tlu_eps(tlu_params(p), dm.photon_energy(wavelength))) <= 1e-9


def test_tlu_eps1_inf_is_the_users():
    p = TLU_SETS[0]
    model = dm.EpsilonInf(1.7) + dm.TaucLorentzUrbach(**p)
    old = oracle.tlu_eps(tlu_params(p, einf=1.7), ENERGIES)
    assert scaled(model.epsilon(dm.HC_EV_NM / ENERGIES), old) <= 1e-9


@pytest.mark.parametrize("p", TLU_SETS)
def test_tlu_eps1_is_the_kramers_kronig_transform_of_eps2(p):
    """Independent check: eps1 vs the numerical principal value of eps2 (``kramers_kronig_real``)."""
    model = dm.TaucLorentzUrbach(**p)
    energy = np.array([0.3, 0.9, 1.5, p["Eg"], 0.5 * (p["Eg"] + p["Ec"]), p["Ec"] * 1.05, p["E0"], 5.5, 9.0, 20.0])
    numeric = dm.kramers_kronig_real(model.eps2, energy, [p["Eg"], p["Ec"], p["E0"]])
    closed = model.eps1(energy)
    # absolute tolerance relative to the model's own scale: |eps1| <= ~ A/E0 here
    assert np.max(np.abs(closed - numeric)) <= 1e-9 * max(1.0, np.max(np.abs(numeric)))


def test_tlu_continuity_at_ec():
    for p in TLU_SETS:
        m = dm.TaucLorentzUrbach(**p)
        h = 1e-6
        lo, hi = m.eps2(np.array([p["Ec"] - h])), m.eps2(np.array([p["Ec"] + h]))
        slope = (m.eps2(np.array([p["Ec"] + 2 * h])) - hi) / h
        assert abs(lo[0] - hi[0]) <= 3 * h * abs(slope[0])  # continuous: the jump is within a step of the slope
        d = (m.eps2(np.array([p["Ec"] + 2 * h])) - m.eps2(np.array([p["Ec"] + h]))) / h
        dl = (m.eps2(np.array([p["Ec"] - h])) - m.eps2(np.array([p["Ec"] - 2 * h]))) / h
        assert d[0] == pytest.approx(dl[0], rel=1e-3)


@pytest.mark.parametrize(
    "kw",
    [
        dict(A=-1.0), dict(E0=0.0), dict(C=-2.0), dict(Eg=0.0), dict(Eg=2.3), dict(C=6.0), dict(Eg=float("nan")),
        dict(E0=2.0, C=3.0),
    ],
)
def test_tlu_refuses(kw):
    base = dict(TLU_SETS[0])
    base.update(kw)
    with pytest.raises(ValueError):
        dm.TaucLorentzUrbach(**base)
    if np.isfinite(base["Eg"]):  # the harvester refuses the same sets (it does not check for NaN)
        with pytest.raises(oracle.ModelError):
            oracle.tlu_urbach(tlu_params(base))


FB_SETS = [
    dict(A=0.12, B=9.0, C=21.0, n_inf=1.6, Eg=1.5),
    dict(A=0.05, B=11.0, C=32.0, n_inf=1.4, Eg=3.0),
    dict(A=0.3, B=6.0, C=12.0, n_inf=1.8, Eg=0.0),
]


@pytest.mark.parametrize("p", FB_SETS)
def test_forouhi_bloomer_matches_harvester(p):
    model = dm.ForouhiBloomer(**p)
    params = dict(A=p["A"], B_eV=p["B"], C_eV2=p["C"], n_inf=p["n_inf"], Eg_eV=p["Eg"])
    energy = np.sort(np.concatenate([ENERGIES, p["Eg"] * (1 + np.array([-1e-9, 0, 1e-9]))]))
    assert scaled(model.nk(energy), oracle.fb_nk(params, energy)) <= 1e-9
    wavelength = dm.HC_EV_NM / ENERGIES
    assert scaled(model._epsilon(ENERGIES, wavelength), oracle.fb_nk(params, ENERGIES) ** 2) <= 1e-9
    assert np.all(model.nk(ENERGIES).imag >= 0)


@pytest.mark.parametrize("kw", [dict(A=0.0), dict(B=12.0), dict(Eg=-0.1), dict(C=float("inf"))])
def test_forouhi_bloomer_refuses(kw):
    base = dict(FB_SETS[0])
    base.update(kw)
    with pytest.raises(ValueError):
        dm.ForouhiBloomer(**base)


CARRIERS = [(1e21, 0.35, 0.08, 3.9), (3e20, 1.0, 0.2, 1.0), (5e22, 0.5, 0.03, 4.5)]


@pytest.mark.parametrize("N_cm3, mstar, gamma_ev, eps_inf", CARRIERS)
def test_drude_carrier_matches_harvester(N_cm3, mstar, gamma_ev, eps_inf, monkeypatch):
    """The harvester hard-codes truncated hbar, m_e and eps0 (CODATA 2018, 7.7e-11 / 4.7e-11 / 7e-10 off the
    scipy.constants values PyStratify uses).  Near the plasma edge eps -> 0 and the offset is amplified, so the
    comparison runs twice: with the harvester's own constants, scaled by the terms that cancel
    (|eps_inf| + |wp^2/w^2|) <= 3e-9, and with its constants replaced by PyStratify's, <= 1e-11 relative."""
    from scipy import constants

    wavelength = np.geomspace(300.0, 20000.0, 400)
    model = dm.EpsilonInf(eps_inf) + dm.DrudeCarrier(N=N_cm3 * 1e6, mstar=mstar, Gamma=gamma_ev)
    params = {"eps_inf": eps_inf, "N": [N_cm3, "cm-3"], "mstar": mstar, "gamma": [gamma_ev, "eV"]}
    energy = dm.photon_energy(wavelength)
    ep = dm.DrudeCarrier(N=N_cm3 * 1e6, mstar=mstar, Gamma=gamma_ev).plasma_energy
    old = oracle.evaluate("drude-n", params, wavelength)
    cancelling = eps_inf + (ep / energy) ** 2
    assert np.max(np.abs(model.epsilon(wavelength) - old**2) / cancelling) <= 3e-9
    monkeypatch.setattr(oracle, "HBAR_EV_S", constants.hbar / constants.e)
    monkeypatch.setattr(oracle, "ME", constants.m_e)
    monkeypatch.setattr(oracle, "EPS0", constants.epsilon_0)
    exact = oracle.evaluate("drude-n", params, wavelength)
    assert scaled(model.index(wavelength), exact) <= 1e-11
    assert scaled(model.epsilon(wavelength), exact**2) <= 1e-11
    carrier = dm.DrudeCarrier(N_cm3 * 1e6, mstar, gamma_ev)
    same = dm.DrudeEnergy(carrier.plasma_energy, gamma_ev)
    assert scaled(carrier.epsilon(wavelength), same.epsilon(wavelength)) <= 1e-14


def test_drude_carrier_refuses():
    for kw in (dict(N=0.0), dict(mstar=-1.0), dict(Gamma=-0.1)):
        base = dict(N=1e27, mstar=0.4, Gamma=0.1)
        base.update(kw)
        with pytest.raises(ValueError):
            dm.DrudeCarrier(**base)
