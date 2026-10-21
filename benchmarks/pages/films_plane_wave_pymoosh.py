"""PyStratify against PyMoosh: thin films under a plane wave.

#####################################################################################################################
#                                                                                                                   #
#   pip install pystratify==0.11.0 PyMoosh==4.0.1 numpy matplotlib                                                  #
#   python films_plane_wave_pymoosh.py                                                                              #
#                                                                                                                   #
#####################################################################################################################

Every figure of https://multilayer.app/verification/films-plane-wave-pymoosh.html, computed by this one file. It prints
each figure's RMSE (PyStratify against PyMoosh, over every plotted point) and writes, beside itself,

* films_plane_wave_pymoosh.json -- the curves the page draws, value for value;
* films_plane_wave_pymoosh.png  -- the plots, PyStratify the lines and PyMoosh the circles (needs matplotlib).

PyMoosh (Langevin, Moreau et al., https://github.com/AnMoreau/PyMoosh) solves stacks with its own scattering
matrices; PyStratify (https://github.com/multilayerapp/PyStratify) with transfer matrices. Neither shares the other's
code. Three conventions differ, all handled below:

* PyMoosh takes a permittivity and a permeability per material, PyStratify an index n = sqrt(eps mu) and mu;
* PyMoosh's p-polarised t is the ratio of the magnetic fields, PyStratify's (as tmm's) of the electric fields: they
  differ by (n_in / mu_in) / (n_out / mu_out);
* PyMoosh's `field` returns E_y for s and H_y for p, a Gaussian beam expanded in 2 nmod + 1 plane waves. A waist
  wider than the window leaves one order, a single plane wave; with a zero-thickness superstrate its phase origin is
  the first interface, as in PyStratify. H_y = (n_j/mu_j)/(n_0/mu_0) (v e^{i kz z} + w e^{-i kz z}) from PyStratify's
  amplitudes.

The test suite (PyStratify tests/test_planar_codes.py) runs nine stacks at four angles and three wavelengths for R, T,
r, t, the absorption in every layer and the fields, to 1e-12; these four plots are a view of it.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from numpy import inf

from pystratify import planar

# name -> (layers, thicknesses in nm, incoherent?); a layer is n or (n, mu); the first and last are semi-infinite
CASES = {
    "AR pair on glass": ([1.0, 2.35, 1.46, 1.52], [inf, 64.0, 105.0, inf], False),
    "Bragg mirror, 10 pairs": ([1.0] + [2.35, 1.46] * 10 + [1.52], [inf] + [67.3, 108.4] * 10 + [inf], False),
    "TiO2 on 1 mm incoherent glass": ([1.0, 2.3 + 0.01j, 1.52 + 1e-6j, 1.0], [inf, 80.0, 1e6, inf], True),
}


# --- the two codes, on the same stack ----------------------------------------------------------------------------

def _split(layers):
    n = [complex(x[0]) if isinstance(x, tuple) else complex(x) for x in layers]
    mu = [complex(x[1]) if isinstance(x, tuple) else 1.0 for x in layers]
    return n, mu


def stratify(case, wavelength, theta, pol):
    """PyStratify: R, T (and r, t, the absorption per finite layer for a coherent stack) at wavelength [nm] and
    theta [rad]."""
    layers, d, incoherent = case
    n, mu = _split(layers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if incoherent:
            out = planar.inc_tmm(pol, n, d, ["i", "c", "i", "i"], theta, wavelength)
            return dict(R=out["R"], T=out["T"])
        magnetic = any(m != 1 for m in mu)
        out = planar.coh_tmm(pol, n, d, theta, wavelength, mu_list=mu) if magnetic else \
            planar.coh_tmm(pol, n, d, theta, wavelength)
        absorbed = np.asarray(planar.absorp_in_each_layer(out))[1:-1]
    return dict(R=out["R"], T=out["T"], r=out["r"], t=out["t"], A=absorbed)


def moosh(case, wavelength, theta, pol):
    """PyMoosh: the same quantities, its p-polarised t converted to the ratio of electric fields."""
    import PyMoosh as pm
    from PyMoosh.core import absorption_S
    from PyMoosh.incoherent import incoherent_coefficient_S

    layers, d, incoherent = case
    n, mu = _split(layers)
    materials = [[n[i] ** 2 / mu[i], mu[i]] if mu[i] != 1 else n[i] ** 2 for i in range(len(n))]
    structure = pm.Structure(materials, list(range(len(n))), [0.0] + list(d[1:-1]) + [0.0], verbose=False)
    polarization = 0 if pol == "s" else 1
    if incoherent:
        R, T = incoherent_coefficient_S(structure, True, wavelength, theta, polarization)
        return dict(R=R, T=T)
    absorbed, r, t, R, T = absorption_S(structure, wavelength, theta, polarization)
    if pol == "p":  # H-field ratio to E-field ratio
        t = t * (n[0] / mu[0]) / (n[-1] / mu[-1])
    return dict(R=R, T=T, r=r, t=t, A=np.asarray(absorbed)[1:-1])


def moosh_field(case, wavelength, theta, pol, step=5.0, substrate=200.0):
    """PyMoosh's field as one plane wave: (layer, depth within it) samples every `step` nm below the first interface."""
    import PyMoosh as pm
    from PyMoosh.core import field

    layers, d, _ = case
    n, mu = _split(layers)
    materials = [[n[i] ** 2 / mu[i], mu[i]] if mu[i] != 1 else n[i] ** 2 for i in range(len(n))]
    thickness = [0.0] + list(d[1:-1]) + [substrate]
    structure = pm.Structure(materials, list(range(len(n))), thickness, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        width = 50.0
        beam = pm.Beam(wavelength, theta, 0 if pol == "s" else 1, 1e9 * width)
        values = field(structure, beam, pm.Window(width, 0.5, width, step))[:, 0]
    samples = [(layer, (m + 1) * t / np.floor(t / step))
               for layer, t in enumerate(thickness) if layer > 0 for m in range(int(np.floor(t / step)))]
    return samples, values


def stratify_field(case, wavelength, theta, pol, samples):
    """PyStratify's E_y (s) or H_y (p) at PyMoosh's samples."""
    layers, d, _ = case
    n, mu = _split(layers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        magnetic = any(m != 1 for m in mu)
        data = planar.coh_tmm(pol, n, d, theta, wavelength, mu_list=mu) if magnetic else \
            planar.coh_tmm(pol, n, d, theta, wavelength)
        if pol == "s":
            return np.array([planar.position_resolved(layer, z, data)["Ey"] for layer, z in samples])
        out = []
        for layer, z in samples:
            v, w = data["vw_list"][layer]
            kz = data["kz_list"][layer]
            out.append((n[layer] / mu[layer]) / (n[0] / mu[0]) * (v * np.exp(1j * kz * z) + w * np.exp(-1j * kz * z)))
    return np.array(out)


# --- the figures -------------------------------------------------------------------------------------------------
# PyStratify is drawn as lines, PyMoosh as circles: one per value it computed, at most 41 per curve.

def series(key, label, y, role):
    """A curve; NaN (no value at that point) is written as null."""
    return {"key": key, "label": label, "y": [float(v) if np.isfinite(v) else None for v in np.asarray(y, float)],
            "role": role}


def line(key, label, y):
    return series(key, label, y, "pystratify")


def dots(key, label, y, every=1, size=None, start=0):
    """PyMoosh's values at every `every`-th point of a grid of `size` points from `start`, null between."""
    full = np.full(size, np.nan)
    full[start::every] = y
    return series(key, label, full, "reference")


def largest(ours, theirs):
    return float(np.max(np.abs(np.asarray(ours) - np.asarray(theirs))))


def sig(values, digits=7):
    return [None if v is None else float(f"{v:.{digits}g}") for v in values]


def plot_bragg_stack():
    case = CASES["Bragg mirror, 10 pairs"]
    lam = np.linspace(450, 850, 81)
    every = 2  # PyMoosh every 10 nm
    lines, refs, diffs = [], [], []
    for pol in "sp":
        ours = np.array([[stratify(case, w, np.radians(30), pol)[q] for q in "RT"] for w in lam])
        theirs = np.array([[moosh(case, w, np.radians(30), pol)[q] for q in "RT"] for w in lam[::every]])
        lines += [line(f"{q}_{pol}", f"{q}_{pol}, PyStratify", ours[:, i]) for i, q in enumerate("RT")]
        refs += [dots(f"{q}_{pol}_ref", f"{q}_{pol}, PyMoosh", theirs[:, i], every, lam.size)
                 for i, q in enumerate("RT")]
        diffs.append(largest(ours[::every], theirs))
    return dict(x=sig(lam), series=lines + refs, worst=max(diffs), worst_label="largest |difference|",
                note="Ten pairs of 67.3 nm with n = 2.35 and 108.4 nm with n = 1.46 on glass (n = 1.52), lit from "
                     "air at 30°: reflectance and transmittance for s and p polarization from 450 to 850 nm, "
                     "PyStratify every 5 nm and PyMoosh every 10 nm.")


def plot_anti_reflective_coating():
    case = CASES["AR pair on glass"]
    _, d, _ = case
    every, start = 10, 9  # PyMoosh's window is sampled every 1 nm; drawn every 10 nm, from 10 nm
    lines, refs, diffs = [], [], []
    for pol, label in (("s", "E_y (s)"), ("p", "H_y (p)")):
        samples, theirs = moosh_field(case, 633.0, np.radians(60), pol, step=1.0)
        ours = stratify_field(case, 633.0, np.radians(60), pol, samples)
        tops = np.cumsum([0.0] + list(d[1:-1]))  # each layer's top, nm below the surface
        depth = np.array([tops[layer - 1] + z for layer, z in samples])
        lines += [line(f"re_{pol}", f"Re {label}, PyStratify", ours.real),
                  line(f"im_{pol}", f"Im {label}, PyStratify", ours.imag)]
        refs += [dots(f"re_{pol}_ref", f"Re {label}, PyMoosh", theirs.real[start::every], every, depth.size,
                      start),
                 dots(f"im_{pol}_ref", f"Im {label}, PyMoosh", theirs.imag[start::every], every, depth.size,
                      start)]
        diffs.append(np.max(np.abs(ours - theirs)))
    return dict(x=sig(depth), series=lines + refs, worst=float(max(diffs)), worst_label="largest |difference|",
                note="64 nm with n = 2.35 and 105 nm with n = 1.46 on glass, lit from air at 633 nm and 60°: the "
                     "real and imaginary parts of E_y (s) and H_y (p), every 1 nm through both layers and 200 nm "
                     "into the glass, relative to the incident wave; PyMoosh drawn every 10 nm.")


def plot_incoherent_glass():
    case = CASES["TiO2 on 1 mm incoherent glass"]
    lam = np.linspace(400, 1000, 61)
    every = 2  # PyMoosh every 20 nm
    lines, refs, diffs = [], [], []
    for pol, a in (("s", 0.0), ("p", 60.0)):
        ours = np.array([[stratify(case, w, np.radians(a), pol)[q] for q in "RT"] for w in lam])
        theirs = np.array([[moosh(case, w, np.radians(a), pol)[q] for q in "RT"] for w in lam[::every]])
        lines += [line(f"{q}_{pol}", f"{q}, {a:g}° {pol}, PyStratify", ours[:, i]) for i, q in enumerate("RT")]
        refs += [dots(f"{q}_{pol}_ref", f"{q}, {a:g}° {pol}, PyMoosh", theirs[:, i], every, lam.size)
                 for i, q in enumerate("RT")]
        diffs.append(largest(ours[::every], theirs))
    return dict(x=sig(lam), series=lines + refs, worst=max(diffs), worst_label="largest |difference|",
                note="80 nm of TiO₂ on 1 mm of glass whose back reflection adds in power, not in amplitude, in air: "
                     "reflectance and transmittance for s at 0° and p at 60°, from 400 to 1000 nm. PyMoosh's "
                     "incoherent solver shares no code with PyStratify's.")


# id, title, x label, y label, the function computing it; in the page's order
FIGURES = [
    ("pymoosh-bragg-mirror", "Bragg stack", "λ [nm]", "R, T", plot_bragg_stack),
    ("pymoosh-field", "Anti-reflective coating", "depth below the surface [nm]", "field", plot_anti_reflective_coating),
    ("pymoosh-incoherent", "Incoherent glass", "λ [nm]", "R, T", plot_incoherent_glass),
]
SCRIPT = "benchmarks/pages/films_plane_wave_pymoosh.py"


def rmse(series_list):
    """The root-mean-square difference between each PyStratify curve and its reference, over every plotted point."""
    by_key = {s["key"]: s["y"] for s in series_list}
    diffs = [a - b for s in series_list if s["role"] == "pystratify"
             for a, b in zip(s["y"], by_key[s["key"] + "_ref"]) if a is not None and b is not None]
    return float(np.sqrt(np.mean(np.square(diffs))))


def record(name, title, x_label, y_label, fn):
    """One figure as the page stores it: every value in full double precision, and the RMSE between the codes."""
    out = {"id": name, "title": title, "x_label": x_label, "y_label": y_label, "log": False, **fn(),
           "script": SCRIPT, "function": fn.__name__}
    out["x"] = [float(v) for v in out["x"]]
    out["rmse"] = rmse(out["series"])
    return out


def plot(records, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # as the page draws them: R green, T blue, any other quantity the next colour; s solid, p dashed
    colors, others = {"R": "#009e73", "T": "#56b4e9"}, ["#cc79a7", "#e6c200", "#cc6677"]
    fig, axes = plt.subplots(1, len(records), figsize=(6.5 * len(records), 4.8), constrained_layout=True)
    for ax, r in zip(np.atleast_1d(axes), records):
        ours = [s for s in r["series"] if s["role"] == "pystratify"]
        extra = []
        for s in ours:
            ref = next(t for t in r["series"] if t["key"] == s["key"] + "_ref")
            quantity, pol = s["key"].rsplit("_", 1)
            if quantity not in colors and quantity not in extra:
                extra.append(quantity)
            color = colors.get(quantity) or others[extra.index(quantity) % len(others)]
            ax.plot(r["x"], [np.nan if v is None else v for v in s["y"]], color=color, lw=1.6,
                    ls="--" if pol == "p" else "-", label=s["label"])
            drawn = [(x, v) for x, v in zip(r["x"], ref["y"]) if v is not None]
            ax.plot(*zip(*drawn), "o", mfc="none", color=color, ms=5, label=ref["label"])
        ax.set(xlabel=r["x_label"], ylabel=r["y_label"])
        ax.set_title(f"{r['title']}\nRMSE {r['rmse']:.2e}", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=130)


def main():
    here = Path(sys.argv[0] if __name__ == "__main__" else __file__).resolve().parent
    stem = Path(SCRIPT).stem
    from importlib.metadata import version

    for package, wanted in (("pystratify", "0.11.0"), ("PyMoosh", "4.0.1")):
        installed = version(package)
        print(f"{package} {installed}" + ("" if installed == wanted else f" -- the page was computed with {wanted}"))
    records = []
    for spec in FIGURES:
        r = record(*spec)
        records.append(r)
        print(f"{r['id']}: {len(r['x'])} points, RMSE {r['rmse']:.2e}", flush=True)
    (here / f"{stem}.json").write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {stem}.json")
    try:
        plot(records, here / f"{stem}.png")
        print(f"wrote {stem}.png")
    except ImportError:
        print("matplotlib is not installed: no plot (pip install matplotlib)")


if __name__ == "__main__":
    main()
