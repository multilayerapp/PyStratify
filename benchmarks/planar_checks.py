"""Planar films: Fresnel at oblique incidence and incoherent stacks against an explicit phase average.

    python benchmarks/planar_checks.py        # prints PLANAR_CHECKS.md (~10 s)

The same checks run in the site's backend (``multilayer/tests/test_analytic_invariants.py``,
``test_layer_absorption.py``) through its ``tmm_core`` shim; here they call ``pystratify.planar`` itself,
so the package carries its own public copy. The incoherent oracle averages coherent solves over the phase
of every incoherent layer (301 samples per layer, Re(delta) over one pi-period, the physical Im(delta)
kept, so a lossy incoherent layer keeps its single-pass attenuation). ``tests/test_planar_checks.py``
asserts every row.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from numpy import inf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checks import report, row  # noqa: E402
from pystratify.planar import (  # noqa: E402
    R_from_r, T_from_t, absorp_in_each_layer, coh_tmm, inc_tmm, interface_r, interface_t, list_snell, make_2x2_array,
    pim_absorp_in_each_layer, pim_tmm, power_entering_from_r,
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


def all_rows():
    return fresnel_oblique_brewster_normal() + incoherent_stack_against_phase_average()


if __name__ == "__main__":
    report("Planar films: Fresnel at oblique incidence and incoherent stacks against an explicit phase average",
           "Generated by `python benchmarks/planar_checks.py`; `tests/test_planar_checks.py` asserts every row (lengths and "
           "wavelengths in um).", all_rows())
