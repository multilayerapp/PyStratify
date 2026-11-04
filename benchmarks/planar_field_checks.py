"""Planar films: the magnetic field of `position_resolved` and the field profile of stacks with incoherent layers
(`inc_field`) against Maxwell's equations, closed forms and a band-averaged coherent field (assurance WP07-D2, O-W7-12).

    python benchmarks/planar_field_checks.py      # prints PLANAR_FIELD_CHECKS.md (~5 s)

* **H against Maxwell.** On random coherent stacks (lossless, absorbing and magnetic layers, s and p, 0-80 deg),
  Re(E x H*)_z must be `position_resolved`'s Poynting flux, and the tangential E and H, the normal B = mu H_z (s) and
  the normal D = eps E_z (p) must be continuous across every interface.
* **The incoherent slab in closed form.** A lossless slab at normal incidence, its forward and backward powers V, W
  summed by hand from the Fresnel powers: |E|^2 = (V + W) n_0/n and |H|^2 = (V + W) n/n_0 inside it, 1 + R in the
  incident medium and T n_0/n in the exit medium.
* **The band-averaged coherent field** is the physical reference: what a source whose spectrum spans many free
  spectral ranges of the slab sees, computed by `coh_tmm` + `position_resolved` (E and H) and nothing incoherent.
  It is weighted by a Hann (raised-cosine) spectrum over a band in vacuum wavenumber 1/lambda that makes the slab's
  round trip, and the standing wave at every sampled depth (multiples of a quarter of the slab), advance by a whole
  number of cycles. A lossless slab's field is then a trigonometric polynomial in one phase and its average is the
  phase average exactly; the smooth weight keeps the leak of each harmonic, from whatever else changes across the
  band, at O(1/(jM)^3) instead of the O(1/(jM)) of a flat band (the D2 evidence's flat band left 1-5 %). An
  absorbing slab keeps its absorption coefficient across the band (kappa proportional to lambda, as a measured
  alpha would), so only its Fresnel coefficients drift (by kappa d lambda / lambda); a coherent film's own phase
  drifts with the band too, which a thicker slab (a narrower band) shrinks.
  The rows below the band average re-run the API's pre-fix conversion (powers plotted as |E|^2 and |H|^2, O-W7-12)
  against the same reference, so the reference is seen to reject it.
* **A coherent stack routed through `inc_tmm`** (every interior layer coherent) must give `position_resolved`'s
  |E|^2 and |H|^2 in every finite layer and in the exit medium (a lossy exit at oblique incidence has a complex
  angle).

``tests/test_planar_fields.py`` asserts every row.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numpy import inf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checks import report, row  # noqa: E402
from pystratify import planar  # noqa: E402

LAM0 = 0.6  # um, the band's centre
SEED = 20261103


def intensities(field):
    """|E|^2 and |H|^2 of a `position_resolved` result."""
    e2 = sum(abs(field[k]) ** 2 for k in ("Ex", "Ey", "Ez"))
    h2 = sum(abs(field[k]) ** 2 for k in ("Hx", "Hy", "Hz"))
    return e2, h2


# ------------------------------------------------------------------------------------------- H against Maxwell
def random_coherent_stack(rng, magnetic):
    layers = int(rng.integers(1, 7))
    n = [complex(rng.uniform(1.3, 3.5), rng.choice([0.0, rng.uniform(0, 0.5)])) for _ in range(layers)]
    d = [float(rng.uniform(0.005, 0.5)) for _ in range(layers)]
    sub = complex(rng.uniform(1.4, 4.0), rng.choice([0.0, rng.uniform(0, 0.3)]))
    mu = [1.0, *(float(rng.uniform(0.6, 2.0)) for _ in range(layers)), 1.0] if magnetic else None
    return [1.0, *n, sub], [inf, *d, inf], mu


def maxwell_rows(stacks=400):
    rng = np.random.default_rng(SEED)
    flux = tangential = normal = 0.0
    for index in range(stacks):
        n_list, d_list, mu = random_coherent_stack(rng, magnetic=index % 4 == 3)
        th_0, lam = math.radians(float(rng.uniform(0, 80))), float(rng.uniform(0.3, 1.2))
        eps = np.array(n_list) ** 2 / (np.ones(len(n_list)) if mu is None else np.array(mu))
        mu_all = np.ones(len(n_list)) if mu is None else np.array(mu)
        for pol in "sp":
            data = planar.coh_tmm(pol, n_list, d_list, th_0, lam, mu_list=mu)
            y_0 = data["y_list"][0]
            norm = abs(y_0) / (y_0 * (np.cos(th_0) if pol == "s" else np.conj(np.cos(th_0)))).real
            for layer in range(1, len(n_list) - 1):
                for z in np.linspace(0, d_list[layer], 5):
                    f = planar.position_resolved(layer, z, data)
                    s_z = (f["Ex"] * np.conj(f["Hy"]) - f["Ey"] * np.conj(f["Hx"])).real * norm
                    flux = max(flux, abs(s_z - f["poyn"]))
            for layer in range(0, len(n_list) - 1):
                top = planar.position_resolved(layer, 0.0 if layer == 0 else d_list[layer], data)
                bottom = planar.position_resolved(layer + 1, 0.0, data)
                scale = max(1.0, *(abs(top[k]) for k in ("Ex", "Ey", "Hx", "Hy")))
                for key in ("Ex", "Ey", "Hx", "Hy"):
                    tangential = max(tangential, abs(top[key] - bottom[key]) / scale)
                # normal B = mu H_z (s) and normal D = eps E_z (p)
                material, upper, lower = (mu_all, top["Hz"], bottom["Hz"]) if pol == "s" else (eps, top["Ez"],
                                                                                                bottom["Ez"])
                jump = abs(material[layer] * upper - material[layer + 1] * lower)
                normal = max(normal, jump / max(1.0, abs(material[layer] * upper)))
    name = f"{stacks} random coherent stacks (1-6 layers, lossless, absorbing, every 4th magnetic), 0-80 deg, s and p"
    return [
        row(name, "max |Re(E x H*)_z |y_0| / Re(y_0 cos th_0) - poyn|, 5 depths per layer", flux, 1e-12),
        row(name, "max jump of tangential Ex, Ey, Hx, Hy across an interface (relative)", tangential, 1e-12),
        row(name, "max jump of normal B = mu Hz (s), D = eps Ez (p) across an interface (relative)", normal, 1e-12),
    ]


# ------------------------------------------------------------------------- the incoherent slab in closed form
def analytic_rows():
    rows = []
    for n_0, n_1, n_2 in ((1.0, 1.5, 1.0), (1.0, 3.5, 1.0), (1.0, 1.5, 3.5), (1.33, 2.4, 1.52)):
        d = 37.0
        R01 = ((n_0 - n_1) / (n_0 + n_1)) ** 2
        R12 = ((n_1 - n_2) / (n_1 + n_2)) ** 2
        T01, T12 = 1 - R01, 1 - R12
        V = T01 / (1 - R01 * R12)          # forward power in the slab: R10 = R01 at normal incidence
        W = V * R12
        R = R01 + T01 ** 2 * R12 / (1 - R01 * R12)
        T = T01 * T12 / (1 - R01 * R12)
        worst = 0.0
        for pol in "sp":
            data = planar.inc_tmm(pol, [n_0, n_1, n_2], [inf, d, inf], ["i", "i", "i"], 0.0, LAM0)
            z = np.array([0.0, d / 3, d / 2, d])
            inside = planar.inc_field(1, z, data)
            expected = {"E2": (V + W) * n_0 / n_1, "H2": (V + W) * n_1 / n_0}
            front = planar.inc_field(0, np.array([-2.0, 0.0]), data)
            back = planar.inc_field(2, np.array([0.0, 5.0]), data)
            pairs = [(inside["E2"], expected["E2"]), (inside["H2"], expected["H2"]),
                     (front["E2"], 1 + R), (front["H2"], 1 + R),
                     (back["E2"], T * n_0 / n_2), (back["H2"], T * n_2 / n_0)]
            worst = max(worst, *(float(np.max(np.abs(np.asarray(a) / b - 1))) for a, b in pairs))
        rows.append(row(f"lossless slab n = {n_1:g} between n = {n_0:g} and {n_2:g}, normal incidence, s and p",
                        "max relative |E|^2, |H|^2 against (V + W) n_0/n, (V + W) n/n_0, 1 + R, T n_0/n_2", worst,
                        1e-14))
    return rows


# --------------------------------------------------------------------------------- the band-averaged reference
def indices_at(layers, lam):
    """Every layer's index at vacuum wavelength lam: real part fixed, absorption coefficient fixed (kappa ~ lam)."""
    return [complex(re, kappa * lam / LAM0) for re, kappa, _, _ in layers]


def forward_ncos(n, angle_deg):
    return (np.sqrt(complex(n) ** 2 - math.sin(math.radians(angle_deg)) ** 2)).real


def band_average(pol, layers, angle_deg, points, period, cycles, samples):
    """Hann-weighted mean coherent |E|^2, |H|^2 at each (layer, z) over a band of 1/lambda centred on 1/LAM0 in
    which an optical length ``period`` (Re(n cos th) times a geometric length) advances its round trip by
    ``cycles`` x 2 pi."""
    width = cycles / (2 * period)
    x = (np.arange(samples) + 0.5) / samples - 0.5
    weight = 1 + np.cos(2 * np.pi * x)
    weight /= weight.sum()
    d_list = [layer[2] for layer in layers]
    e2, h2 = np.zeros(len(points)), np.zeros(len(points))
    for xi, wi in zip(x, weight):
        lam = 1 / (1 / LAM0 + xi * width)
        data = planar.coh_tmm(pol, indices_at(layers, lam), d_list, math.radians(angle_deg), lam)
        for i, (layer, z) in enumerate(points):
            e, h = intensities(planar.position_resolved(layer, z, data))
            e2[i] += wi * e
            h2[i] += wi * h
    return e2, h2


def pre_fix_envelope(layer, z, inc):
    """The API's conversion before O-W7-12 (app.py 5462-5556 at commit ef9c946), re-run here only to show that the
    reference rejects it: an incoherent layer's |E|^2 = |H|^2 = V e^{-alpha z} + W e^{alpha z} (powers), and a
    coherent layer's F |E_fwd|^2 + B |E_bwd|^2 with each solution in units of its own incident wave."""
    if np.any(np.isnan(inc["stack_from_all"][layer])):
        V, W = inc["VW_list"][inc["inc_from_all"][layer]]
        n, th = inc["n_list"][layer], inc["th_list"][layer]
        p = max(math.exp(-4 * math.pi * (n * np.cos(th)).imag / LAM0 * z), 1e-30)
        return V * p + W / p, V * p + W / p
    stack, within = (int(v) for v in inc["stack_from_all"][layer])
    F, B = (float(np.real(v)) for v in inc["stackFB_list"][stack])
    fwd, bwd = inc["coh_tmm_data_list"][stack], inc["coh_tmm_bdata_list"][stack]
    e_f, h_f = intensities(planar.position_resolved(within, z, fwd))
    e_b, h_b = intensities(planar.position_resolved(len(bwd["d_list"]) - 1 - within, inc["d_list"][layer] - z, bwd))
    return F * e_f + B * e_b, F * h_f + B * h_b


SLAB_UM = 200.0
STACK_UM = 4000.0
SLABS = (  # label, slab index (real, kappa at LAM0), thickness
    ("lossless slab n = 1.5", 1.5, 0.0, SLAB_UM),
    ("lossless slab n = 3.5", 3.5, 0.0, SLAB_UM),
    ("absorbing slab n = 1.5 + 1e-4 i", 1.5, 1e-4, SLAB_UM),
    ("strongly absorbing slab n = 1.5 + 2e-3 i", 1.5, 2e-3, 50.0),
)
FILMS = (  # label, layers (index real, kappa at LAM0, thickness, coherence)
    ("100 nm n = 2 film on a 4 mm n = 1.5 slab (c_list i c i i)",
     [(1.0, 0.0, inf, "i"), (2.0, 0.0, 0.1, "c"), (1.5, 0.0, STACK_UM, "i"), (1.0, 0.0, inf, "i")]),
    ("films on both faces of a 4 mm absorbing slab (i c i c c i)",
     [(1.0, 0.0, inf, "i"), (2.0, 0.0, 0.1, "c"), (1.5, 2.5e-6, STACK_UM, "i"), (1.38, 0.0, 0.12, "c"),
      (2.3, 0.0, 0.05, "c"), (1.0, 0.0, inf, "i")]),
)
# About ten times what was observed (2026-11-03). Lossless: rounding (the slab's harmonics, up to the 60th round trip
# for n = 3.5 at 45 deg, alias only above 2048 samples; 512 left 3e-6). Absorbing: the Fresnel coefficients' drift with
# kappa across the band. Films: their own phase's drift, which falls as 1/d (9e-6 at 1 mm, 2e-6 at 4 mm, s at 45 deg).
# 2048 and 4096 wavelengths give the same rows.
TOLERANCE = {"lossless slab n = 1.5": 1e-12, "lossless slab n = 3.5": 1e-12,
             "absorbing slab n = 1.5 + 1e-4 i": 1e-8, "strongly absorbing slab n = 1.5 + 2e-3 i": 5e-7}
FILM_TOLERANCE = 5e-5
CYCLES, SLAB_SAMPLES, FILM_SAMPLES = 8, 2048, 2048


def sample_points(layers, angle_deg, slab_layer):
    """Depths whose standing waves advance a whole number of cycles over the band: multiples of a quarter of the
    slab inside it, and the same optical length into the half-spaces; three depths through every coherent film."""
    d = layers[slab_layer][2]
    period = forward_ncos(layers[slab_layer][0], angle_deg) * d / 4
    points = [(0, -period / forward_ncos(layers[0][0], angle_deg))]
    for layer in range(1, len(layers) - 1):
        thickness = layers[layer][2]
        if layer == slab_layer:
            points += [(layer, thickness / 4), (layer, thickness / 2), (layer, 3 * thickness / 4)]
        else:
            points += [(layer, 0.0), (layer, thickness / 2), (layer, thickness)]
    points.append((len(layers) - 1, period / forward_ncos(layers[-1][0], angle_deg)))
    return points, period


def band_case(pol, layers, angle_deg, slab_layer, samples):
    points, period = sample_points(layers, angle_deg, slab_layer)
    e_ref, h_ref = band_average(pol, layers, angle_deg, points, period, CYCLES, samples)
    inc = planar.inc_tmm(pol, indices_at(layers, LAM0), [layer[2] for layer in layers], [layer[3] for layer in layers],
                         math.radians(angle_deg), LAM0)
    ours = [planar.inc_field(layer, z, inc) for layer, z in points]
    old = [pre_fix_envelope(layer, z, inc) for layer, z in points]
    e_err = max(abs(f["E2"] / e - 1) for f, e in zip(ours, e_ref))
    h_err = max(abs(f["H2"] / h - 1) for f, h in zip(ours, h_ref))
    finite = [i for i, (layer, _) in enumerate(points) if 0 < layer < len(layers) - 1]
    old_e = [old[i][0] / e_ref[i] for i in finite]
    old_h = [old[i][1] / h_ref[i] for i in finite]
    return e_err, h_err, old_e, old_h, len(points)


def band_rows():
    rows = []
    for label, re, kappa, d in SLABS:
        layers = [(1.0, 0.0, inf, "i"), (re, kappa, d, "i"), (1.0, 0.0, inf, "i")]
        for pol in "sp":
            for angle in (0.0, 45.0):
                e_err, h_err, old_e, old_h, count = band_case(pol, layers, angle, 1, SLAB_SAMPLES)
                name = (f"{label}, {d:g} um, {pol}, {angle:g} deg: {count} depths against the coherent field over "
                        f"{CYCLES} quarter-slab cycles ({SLAB_SAMPLES} wavelengths, Hann)")
                rows.append(row(name, "max relative |E|^2, |H|^2 of inc_field - band average", max(e_err, h_err),
                                TOLERANCE[label]))
                if kappa == 0 and angle == 0 and pol == "s":
                    rows.append(row(name, "pre-fix |E|^2 / band average inside the slab (O-W7-12; n/n_0 expected)",
                                    min(old_e), 0.99 * re, ">="))
                    rows.append(row(name, "pre-fix |H|^2 / band average inside the slab (O-W7-12; n_0/n expected)",
                                    max(old_h), 1.01 / re))
    for label, layers in FILMS:
        for pol in "sp":
            for angle in (0.0, 45.0):
                e_err, h_err, old_e, old_h, count = band_case(pol, layers, angle, 2, FILM_SAMPLES)
                name = (f"{label}, {pol}, {angle:g} deg: {count} depths against the coherent field over {CYCLES} "
                        f"quarter-slab cycles ({FILM_SAMPLES} wavelengths, Hann)")
                rows.append(row(name, "max relative |E|^2, |H|^2 of inc_field - band average", max(e_err, h_err),
                                FILM_TOLERANCE))
                if pol == "s" and angle == 0:
                    film = [abs(r - 1) for r, (layer, _) in zip(old_e, sample_points(layers, angle, 2)[0][1:-1])
                            if layers[layer][3] == "c"]
                    rows.append(row(name, "pre-fix max relative |E|^2 error in the coherent films (O-W7-12: B taken "
                                    "as |E0|^2 of the slab)", max(film), 5e-3, ">="))
    return rows


# --------------------------------------------------------------------- a coherent stack routed through inc_tmm
def routing_rows(stacks=300):
    rng = np.random.default_rng(SEED + 1)
    worst = 0.0
    for _ in range(stacks):
        n_list, d_list, _ = random_coherent_stack(rng, magnetic=False)
        th_0, lam = math.radians(float(rng.uniform(0, 80))), float(rng.uniform(0.3, 1.2))
        c_list = ["i", *["c"] * (len(n_list) - 2), "i"]
        for pol in "sp":
            coherent = planar.coh_tmm(pol, n_list, d_list, th_0, lam)
            inc = planar.inc_tmm(pol, n_list, d_list, c_list, th_0, lam)
            for layer in range(1, len(n_list)):
                depths = [0.0, 0.3] if layer == len(n_list) - 1 else [0.0, d_list[layer] / 2, d_list[layer]]
                ours = planar.inc_field(layer, np.array(depths), inc)
                for z, e, h in zip(depths, ours["E2"], ours["H2"]):
                    e_ref, h_ref = intensities(planar.position_resolved(layer, z, coherent))
                    worst = max(worst, abs(e - e_ref) / max(e_ref, 1e-300), abs(h - h_ref) / max(h_ref, 1e-300))
    name = (f"{stacks} random coherent stacks (absorbing exits too) routed through inc_tmm with every interior layer "
            "coherent, 0-80 deg, s and p")
    return [row(name, "max relative |E|^2, |H|^2 of inc_field - position_resolved, finite layers and exit medium",
                worst, 1e-12)]


def all_rows():
    return maxwell_rows() + analytic_rows() + band_rows() + routing_rows()


if __name__ == "__main__":
    report("Planar films: H and the field of stacks with incoherent layers",
           "Generated by `python benchmarks/planar_field_checks.py`; `tests/test_planar_fields.py` asserts every row "
           "(lengths and wavelengths in um, the band centred on 0.6 um).", all_rows())
