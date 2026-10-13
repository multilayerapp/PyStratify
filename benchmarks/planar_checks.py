"""Planar films: Fresnel at oblique incidence, the bare-substrate ellipsometric ratio, a scattering-matrix cross-check, and
incoherent stacks against an explicit phase average.

    python benchmarks/planar_checks.py        # prints PLANAR_CHECKS.md (~10 s)

The same checks run in the site's backend (``multilayer/tests/test_analytic_invariants.py``,
``test_layer_absorption.py``) through its ``tmm_core`` shim; here they call ``pystratify.planar`` itself,
so the package carries its own public copy. The incoherent oracle averages coherent solves over the phase
of every incoherent layer (301 samples per layer, Re(delta) over one pi-period, the physical Im(delta)
kept, so a lossy incoherent layer keeps its single-pass attenuation). The scattering-matrix reference (Redheffer's
star product of interface and propagation two-ports) never multiplies by exp(+Im(kz) d), so it also reaches opaque
layers where a transfer-matrix product would overflow. ``tests/test_planar_checks.py`` asserts every row.
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
from numpy import inf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checks import report, row  # noqa: E402
from pystratify.planar import (  # noqa: E402
    R_from_r, T_from_t, absorp_in_each_layer, coh_tmm, ellips, inc_tmm, interface_r, interface_t, list_snell, make_2x2_array,
    pim_absorp_in_each_layer, pim_tmm, position_resolved, power_entering_from_r,
)

LOSSLESS_STACK_N = [1.0, 1.46, 2.35, 1.38, 2.0, 1.7, 1.46, 2.35, 1.52]
LOSSLESS_STACK_D = [inf, 0.11, 0.06, 0.17, 0.05, 0.21, 0.08, 0.06, inf]


def _forward_cos(n, kx):
    """cos(theta) in medium n for conserved kx, on the branch where the forward wave decays."""
    cos_t = complex(np.lib.scimath.sqrt(1 - (kx / n) ** 2))
    q = n * cos_t
    if q.imag < 0 or (q.imag == 0 and q.real < 0):
        cos_t = -cos_t
    return cos_t


def _fresnel_amplitudes(pol, n_i, n_f, cos_i, cos_f):
    if pol == "s":
        den = n_i * cos_i + n_f * cos_f
        return (n_i * cos_i - n_f * cos_f) / den, 2 * n_i * cos_i / den
    den = n_f * cos_i + n_i * cos_f
    return (n_f * cos_i - n_i * cos_f) / den, 2 * n_i * cos_i / den


def _power_factor(pol, n_i, cos_i, n_f, cos_f):
    if pol == "s":
        return (n_f * cos_f).real / (n_i * cos_i).real
    return (n_f * np.conj(cos_f)).real / (n_i * np.conj(cos_i)).real


def _star(a, b):
    """Redheffer star product of scalar two-port S-matrices (s11, s12, s21, s22)."""
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    d = 1.0 / (1.0 - a22 * b11)
    return a11 + a12 * b11 * d * a21, a12 * d * b12, b21 * d * a21, b22 + b21 * a22 * d * b12


def smatrix_rt(pol, n_list, d_list, th_0, lam_vac):
    """R and T of a coherent stack from interface and propagation S-matrices (the backend's reference, kept here)."""
    n_list = [complex(n) for n in n_list]
    kx = n_list[0].real * np.sin(th_0)
    cosines = [_forward_cos(n, kx) for n in n_list]
    s = None
    for j in range(len(n_list) - 1):
        r_fwd, t_fwd = _fresnel_amplitudes(pol, n_list[j], n_list[j + 1], cosines[j], cosines[j + 1])
        r_rev, t_rev = _fresnel_amplitudes(pol, n_list[j + 1], n_list[j], cosines[j + 1], cosines[j])
        interface = (r_fwd, t_rev, t_fwd, r_rev)
        s = interface if s is None else _star(s, interface)
        if j + 1 < len(n_list) - 1:
            phase = np.exp(1j * 2 * np.pi * n_list[j + 1] * cosines[j + 1] * d_list[j + 1] / lam_vac)
            s = _star(s, (0.0, phase, phase, 0.0))
    r, t = s[0], s[2]
    return abs(r) ** 2, abs(t) ** 2 * _power_factor(pol, n_list[0], cosines[0], n_list[-1], cosines[-1])


METAL_STACK = ([1.0, 1.46, 0.05 + 3.0j, 1.52], [inf, 0.10, 0.04, inf], 0.6)  # n, d, wavelength
BEYOND_CRITICAL = ([1.5, 1.0, 2.0, 1.5], [inf, 0.4, 0.13, inf], 0.55)
OPAQUE = ([1.0, 2.0 + 0.5j, 1.52], [inf, 120.0, inf], 0.5)  # Im(kz) d ~ 754


def scattering_matrix_cross_check():
    rows = []
    for label, (n, d, lam), angles in (("air / silica / metal / glass", METAL_STACK, (0, 45, 70)),
                                       ("glass / air gap / n = 2 / glass, beyond critical", BEYOND_CRITICAL, (45, 60, 80))):
        for deg in angles:
            for pol in ("s", "p"):
                name = f"{label}, {deg} deg, {pol}: S-matrix"
                R, T = _coh_rt(pol, n, d, np.deg2rad(deg), lam)
                R_s, T_s = smatrix_rt(pol, n, d, np.deg2rad(deg), lam)
                rows += [row(name, "|R - R_S|", abs(R - R_s), 1e-10), row(name, "|T - T_S|", abs(T - T_s), 1e-10)]
    n, d, lam = OPAQUE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for deg in (0, 40):
            for pol in ("s", "p"):
                name = f"120 um of n = 2 + 0.5i at 0.5 um, {deg} deg, {pol}: S-matrix"
                R, T = _coh_rt(pol, n, d, np.deg2rad(deg), lam)
                R_s, T_s = smatrix_rt(pol, n, d, np.deg2rad(deg), lam)
                rows += [row(name, "|R - R_S|", abs(R - R_s), 1e-10), row(name, "T", T, 1e-30, "<"), row(name, "T_S", T_s, 1e-30, "<")]
    return rows


SUBSTRATES = ((1.5 + 0.0j, 50.0), (1.5 + 0.0j, 70.0), (1.5 + 0.3j, 65.0), (0.2 + 3.0j, 70.0))


def bare_substrate_rho(n_sub, deg, lam=0.633):
    """(tan Psi e^{i Delta} from pystratify.planar.ellips, -r_p / r_s from the Fresnel equations) of a bare substrate."""
    th = np.deg2rad(deg)
    data = ellips(np.array([1.0, n_sub], dtype=complex), np.array([inf, inf]), th, lam)
    kx = np.sin(th)
    cos_i, cos_f = _forward_cos(1.0, kx), _forward_cos(n_sub, kx)
    r_s, _ = _fresnel_amplitudes("s", 1.0, n_sub, cos_i, cos_f)
    r_p, _ = _fresnel_amplitudes("p", 1.0, n_sub, cos_i, cos_f)
    return np.tan(data["psi"]) * np.exp(1j * data["Delta"]), -r_p / r_s


def bare_substrate_ratio():
    return [row(f"bare substrate n = {n_sub:g} at {deg:g} deg", "|rho - (-r_p / r_s)|", abs(np.subtract(*bare_substrate_rho(n_sub, deg))), 1e-12)
            for n_sub, deg in SUBSTRATES]


def _coh_rt(pol, n_list, d_list, th_0, lam_vac):
    data = coh_tmm(pol, np.array(n_list, dtype=complex), np.array(d_list), th_0, lam_vac)
    return data["R"], data["T"]


def fresnel_oblique_brewster_normal():
    rows = []
    n1, n2, th = 1.0, 1.72, np.deg2rad(63.0)
    kx = n1 * np.sin(th)
    cos_i, cos_f = _forward_cos(n1, kx), _forward_cos(n2, kx)
    for pol in ("s", "p"):
        r, _ = _fresnel_amplitudes(pol, n1, n2, cos_i, cos_f)
        R, T = _coh_rt(pol, [n1, n2], [inf, inf], th, 0.55)
        name = f"n = 1 | 1.72 at 63 deg, {pol}: Fresnel"
        rows += [row(name, "|R - |r|^2|", abs(R - abs(r) ** 2), 1e-13), row(name, "|R + T - 1|", abs(R + T - 1), 1e-13)]
    R, _ = _coh_rt("p", [1.0, 1.5], [inf, inf], np.arctan(1.5), 0.6)
    rows.append(row("n = 1 | 1.5 at Brewster's angle, p", "R", R, 1e-20, "<"))
    Rs, Ts = _coh_rt("s", LOSSLESS_STACK_N, LOSSLESS_STACK_D, 0.0, 0.55)
    Rp, Tp = _coh_rt("p", LOSSLESS_STACK_N, LOSSLESS_STACK_D, 0.0, 0.55)
    name = "a seven-layer lossless stack at normal incidence"
    return rows + [row(name, "|R_s - R_p|", abs(Rs - Rp), 1e-14), row(name, "|T_s - T_p|", abs(Ts - Tp), 1e-14)]


def coherent_tmm_with_phase_overrides(pol, n_list, d_list, th_0, lam_vac, phase_overrides):
    """Coherent TMM with selected layer phases replaced: Re(delta) set, the physical Im(delta) kept."""
    n_list = np.array(n_list)
    d_list = np.array(d_list, dtype=float)
    num_layers = n_list.size
    th_list = list_snell(n_list, th_0)
    kz_list = 2 * np.pi * n_list * np.cos(th_list) / lam_vac
    olderr = np.seterr(invalid="ignore")
    delta = kz_list * d_list
    np.seterr(**olderr)
    for i in range(1, num_layers - 1):
        if delta[i].imag > 35:
            delta[i] = delta[i].real + 35j
        if i in phase_overrides:
            delta[i] = float(phase_overrides[i]) + 1j * delta[i].imag
    t_list = np.zeros((num_layers, num_layers), dtype=complex)
    r_list = np.zeros((num_layers, num_layers), dtype=complex)
    for i in range(num_layers - 1):
        t_list[i, i + 1] = interface_t(pol, n_list[i], n_list[i + 1], th_list[i], th_list[i + 1])
        r_list[i, i + 1] = interface_r(pol, n_list[i], n_list[i + 1], th_list[i], th_list[i + 1])
    m_list = np.zeros((num_layers, 2, 2), dtype=complex)
    for i in range(1, num_layers - 1):
        m_list[i] = (1 / t_list[i, i + 1]) * np.dot(make_2x2_array(np.exp(-1j * delta[i]), 0, 0, np.exp(1j * delta[i]), dtype=complex),
                                                    make_2x2_array(1, r_list[i, i + 1], r_list[i, i + 1], 1, dtype=complex))
    m_tilde = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for i in range(1, num_layers - 1):
        m_tilde = np.dot(m_tilde, m_list[i])
    m_tilde = np.dot(make_2x2_array(1, r_list[0, 1], r_list[0, 1], 1, dtype=complex) / t_list[0, 1], m_tilde)
    r, t = m_tilde[1, 0] / m_tilde[0, 0], 1 / m_tilde[0, 0]
    vw_list = np.zeros((num_layers, 2), dtype=complex)
    vw = np.array([[t], [0]])
    vw_list[-1, :] = np.transpose(vw)
    for i in range(num_layers - 2, 0, -1):
        vw = np.dot(m_list[i], vw)
        vw_list[i, :] = np.transpose(vw)
    return {"r": r, "t": t, "R": R_from_r(r), "T": T_from_t(pol, t, n_list[0], n_list[-1], th_0, th_list[-1]),
            "power_entering": power_entering_from_r(pol, r, n_list[0], th_0), "vw_list": vw_list, "kz_list": kz_list,
            "th_list": th_list, "pol": pol, "n_list": n_list, "d_list": d_list, "th_0": th_0, "lam_vac": lam_vac}


def explicit_phase_average(pol, n_list, d_list, c_list, angle_deg, wavelength, samples_per_layer):
    incoherent = [i for i, c in enumerate(c_list) if c == "i" and 0 < i < len(c_list) - 1]
    phases = [(s + 0.5) * np.pi / samples_per_layer for s in range(samples_per_layer)]
    weight = samples_per_layer ** len(incoherent)
    layers, R, T = np.zeros(len(d_list) - 2), 0.0, 0.0
    for phase_tuple in itertools.product(phases, repeat=len(incoherent)):
        data = coherent_tmm_with_phase_overrides(pol, n_list, d_list, np.deg2rad(angle_deg), wavelength, dict(zip(incoherent, phase_tuple)))
        layers += np.array(absorp_in_each_layer(data), dtype=float)[1:-1]
        R += float(data["R"])
        T += float(data["T"])
    return {"reflectance": R / weight, "transmittance": T / weight, "layer_absorptance": layers / weight}


def incoherent_stack_against_phase_average():
    n_list = [1.0, 2.08 + 0.03j, 1.47 + 0.001j, 1.82 + 0.02j, 1.56]
    d_list = [inf, 0.09, 12.0, 0.11, inf]
    c_list = ["i", "c", "i", "c", "i"]
    rows = []
    for pol in ("s", "p"):
        pim = pim_tmm(pol, n_list, d_list, c_list, np.deg2rad(31.0), 0.64)
        gtmm = inc_tmm(pol, n_list, d_list, c_list, np.deg2rad(31.0), 0.64)
        sampled = explicit_phase_average(pol, n_list, d_list, c_list, 31.0, 0.64, samples_per_layer=301)
        name = f"coherent / 12 um incoherent / coherent on glass at 31 deg, {pol}"
        pim_layers = np.asarray(pim_absorp_in_each_layer(pim), dtype=float)
        rows += [row(name, "|R(pim) - R(inc_tmm)|", abs(pim["R"] - gtmm["R"]), 1e-9),
                 row(name, "|T(pim) - T(inc_tmm)|", abs(pim["T"] - gtmm["T"]), 1e-9),
                 row(name, "max |layer A(pim) - layer A(inc_tmm)|", np.abs(pim_layers - np.asarray(pim_absorp_in_each_layer(gtmm), dtype=float)), 1e-9),
                 row(name, "|R - explicit phase average|", abs(pim["R"] - sampled["reflectance"]), 1e-9),
                 row(name, "|T - explicit phase average|", abs(pim["T"] - sampled["transmittance"]), 1e-9),
                 row(name, "max |layer A - explicit phase average|", np.abs(pim_layers[1:-1] - sampled["layer_absorptance"]), 1e-9)]
    return rows


ABSORBING_STACK = ([1.0, 2.1 + 0.08j, 1.46 + 0.01j, 0.2 + 3.2j, 1.52], [inf, 0.12, 0.25, 0.02, inf])


def integrated_absorption(pol, wavelength, angle_deg=30.0, stack=ABSORBING_STACK, nodes=200):
    """(per-layer absorption from integrating position_resolved's absorption density over each layer's depth
    (Gauss-Legendre), the same from absorp_in_each_layer's Poynting differences) for the finite layers."""
    n, d = stack
    data = coh_tmm(pol, n, d, np.deg2rad(angle_deg), wavelength)
    x, w = np.polynomial.legendre.leggauss(nodes)
    density = [sum(wi * position_resolved(layer, (xi + 1) * d[layer] / 2, data)["absor"] for xi, wi in zip(x, w)) * d[layer] / 2
               for layer in range(1, len(n) - 1)]
    return np.array(density, dtype=float), np.asarray(absorp_in_each_layer(data), dtype=float)[1:-1]


def position_resolved_absorption():
    rows = []
    for wavelength in (0.45, 0.6, 0.9):
        for pol in ("s", "p"):
            ours, flux = integrated_absorption(pol, wavelength)
            rows.append(row(f"absorbing four-layer stack, {wavelength:g} um, 30 deg, {pol}",
                            "max |integral of the absorption density - Poynting difference|", np.abs(ours - flux), 1e-10))
    return rows


def all_rows():
    return fresnel_oblique_brewster_normal() + bare_substrate_ratio() + scattering_matrix_cross_check() + position_resolved_absorption() + incoherent_stack_against_phase_average()


if __name__ == "__main__":
    report("Planar films: Fresnel at oblique incidence and incoherent stacks against an explicit phase average",
           "Generated by `python benchmarks/planar_checks.py`; `tests/test_planar_checks.py` asserts every row (lengths and "
           "wavelengths in um).", all_rows())
