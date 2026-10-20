"""Comparison figures for multilayer.app's Verification row pages (benchmarks/MATRIX.md, docs/verification.md).

    pip install -e ".[docs,benchmarks]"
    python benchmarks/figures.py            # writes docs/figures/<name>.json for every figure
    python benchmarks/figures.py NAME ...   # some of them
    python benchmarks/figures.py --env ENV  # the figures of one environment (bench, smuthi, meep, ngsolve, tmmax)
    python benchmarks/figures.py --list     # every figure and its environment

    python benchmarks/verify_all.py         # the whole study, every figure in its own environment (envs/README.md)

Each figure draws PyStratify against the reference of one matrix row, as curves on a common axis, so the row page
shows the comparison rather than a number: the closed form of a single interface against the planar engine, Mie's
series against the multilayered-sphere recursion, the 60-digit transfer matrices against double precision, treams
against PyStratify for spheres, cylinders and emitters, the Sommerfeld integrals against the film emitter, the fibre's
dispersion relations against the guided poles. The references are the ones the benchmark scripts beside this file use
(`planar_checks.py`, `sphere_checks.py`, `unified_checks.py`, `mie_codes.py`, `emitter_references.py`,
`mp_reference.py`, `pystratify.references`, and for the sphere rows also `gdm_spheres.py`, `mnpbem_spheres.py`,
`fdtd_meep.py`, `smuthi_spheres.py` and the papers' crosscheck classics); nothing here is asserted, the tests do that.
A figure needing a package that is not installed (treams, scattnlay, pyGDM2, smuthi, Meep) is skipped with a note;
the MNPBEM figures need GNU Octave (`mnpbem_octave.py`), and run one environment at a time: smuthi wants Python <= 3.10
with NumPy < 2, Meep its conda build, pyGDM2 NumPy < 2.4.

`docs/verification.md` names which row each figure belongs to; `python -m pystratify.docs build` carries the JSON into
`docs/manifest.json`, and the site draws it with the same chart core as its Learn examples (reference curves dashed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy import inf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import pystratify as ps  # noqa: E402
from pystratify import planar  # noqa: E402

OUT = ROOT / "docs" / "figures"
FIGURES = {}
# The environment each figure runs in (benchmarks/envs/README.md): "bench" holds PyStratify with treams, scattnlay,
# pyGDM2, PyMoosh, pyElli and meent (and drives GNU Octave for MNPBEM); smuthi, Meep, NGSolve and tmmax each need
# their own. benchmarks/verify_all.py runs every figure in its environment.
ENVS = {}
# A figure computed by a page script (benchmarks/pages/) names that script instead of this file, and may carry fields
# for the site beside its data (`calculator`: the setup the page sketches).
# A page script's figure is written as its own `record` returns it (full precision, its RMSE), never re-rounded here.
SCRIPTS, EXTRA, PAGE_RECORDS = {}, {}, {}


def figure(name, title, x_label, y_label, log=False, env="bench", script="benchmarks/figures.py", extra=None):
    def register(fn):
        FIGURES[name] = (fn, dict(id=name, title=title, x_label=x_label, y_label=y_label, log=log))
        ENVS[name] = env
        SCRIPTS[name] = script
        EXTRA[name] = extra or {}
        return fn
    return register


def series(key, label, y, role):
    """A curve; NaN (a reference computed at only some of the points) is written as null."""
    return {"key": key, "label": label, "y": [float(v) if np.isfinite(v) else None for v in np.asarray(y, float)], "role": role}


def worst(ours, theirs, relative=False):
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    diff = np.max(np.abs(ours - theirs))
    return float(diff / np.max(np.abs(theirs))) if relative else float(diff)


def sig(values, digits=7):
    return [None if v is None else float(f"{v:.{digits}g}") for v in values]


# --- films, plane wave -------------------------------------------------------------------------------------------


# --- films, point dipole ----------------------------------------------------------------------------------------


# --- films, plane wave: closed forms, internal methods, other codes (2026-10-12, second batch) -----------------


# --- films, plane wave: other codes ----------------------------------------------------------------------------


# Every plot of verification/films-plane-wave-pymoosh.html comes from one script a reader downloads and runs,
# benchmarks/pages/films_plane_wave_pymoosh.py; registered here, each figure only gains the calculator setup the page
# sketches its stack with (and opens, where the calculator can hold it: it has no permeability).
sys.path.insert(0, str(ROOT / "benchmarks" / "pages"))
import films_plane_wave_pymoosh as pymoosh_page  # noqa: E402
from figure_setups import ASIDES, SETUPS, SUMMARIES  # noqa: E402


def _film_calculator(case, labels, controls, link=True, theta=None):
    """A page script's stack as the calculator block a share token is built from: the first and last layers are the
    incident medium and the substrate, thicknesses go from nm to µm, an incoherent stack's thick layer stays incoherent."""
    layers, d, incoherent = case
    n, mu = pymoosh_page._split(layers)

    def medium(i):  # μ only where it is not 1: the calculator has none, the page's frozen table shows it
        out = {"label": labels[i], "n": float(n[i].real), "k": float(n[i].imag)}
        return {**out, "mu": float(complex(mu[i]).real)} if mu[i] != 1 else out

    finite = [{**medium(i), "thickness": round(d[i] / 1000, 9)} for i in range(1, len(n) - 1)]
    if incoherent:
        for layer in finite:
            if layer["thickness"] >= 100:
                layer["coherency"] = "i"
    # A stack that is one run repeated (a Bragg mirror) is declared as that run and its count, as the calculator holds
    # it in Bragg mode: never written out layer by layer.
    for size in range(2, len(finite) // 2 + 1):
        if len(finite) % size == 0 and finite == finite[:size] * (len(finite) // size):
            finite = [{"repeat": len(finite) // size, "layers": finite[:size]}]
            break
    out = {"film": {"incident": medium(0), "layers": finite, "substrate": medium(len(n) - 1), "controls": controls,
                    "view": {"polarization": "average"}}, "link": link}
    return {**out, "theta": theta} if theta else out


PYMOOSH_SETUPS = {
    "pymoosh-bragg-mirror": _film_calculator(pymoosh_page.CASES["Bragg mirror, 10 pairs"], ["Air"] + ["H", "L"] * 10 + ["Glass"],
                                             {"wavelength-start": 450, "wavelength-end": 850, "wavelength-step": 5, "angle-deg": 30}),
    "pymoosh-field": _film_calculator(pymoosh_page.CASES["AR pair on glass"], ["Air", "H", "L", "Glass"],
                                      {"wavelength-start": 633, "wavelength-end": 633, "angle-deg": 60}),
    "pymoosh-incoherent": _film_calculator(pymoosh_page.CASES["TiO2 on 1 mm incoherent glass"], ["Air", "TiO₂", "Glass", "Air"],
                                           {"wavelength-start": 400, "wavelength-end": 1000, "wavelength-step": 10},
                                           theta="0 (s), 60 (p)"),
}
for _spec in pymoosh_page.FIGURES:
    figure(*_spec[:4], script=pymoosh_page.SCRIPT, extra={"calculator": PYMOOSH_SETUPS[_spec[0]], "summary": SUMMARIES[_spec[0]]})(_spec[4])
    PAGE_RECORDS[_spec[0]] = (pymoosh_page.record, _spec)



# Every other verification page is computed by its own script too, benchmarks/pages/<page>.py; registered here, each
# figure gains the calculator setup its page freezes and sketches (benchmarks/figure_setups.py).
import _page  # noqa: E402
from _shared import INCOHERENT, _guide_modes, at  # noqa: E402,F401


def _site(name):
    """What a page figure carries for the site beside its data: its setup, its one-sentence summary, its aside."""
    out = {"calculator": SETUPS.get(name), "summary": SUMMARIES.get(name), "aside": ASIDES.get(name)}
    return {k: v for k, v in out.items() if v}

PAGE_MODULES = (
    "cylinders_dipole_consistency",
    "cylinders_dipole_meep",
    "cylinders_dipole_ngsolve",
    "cylinders_plane_wave_consistency",
    "cylinders_plane_wave_treams",
    "films_dipole_consistency",
    "films_dipole_meep",
    "films_dipole_pymoosh",
    "films_dipole_smuthi",
    "films_plane_wave_consistency",
    "films_plane_wave_pyelli",
    "films_plane_wave_tmm",
    "films_plane_wave_tmmax",
    "spheres_dipole_consistency",
    "spheres_dipole_meep",
    "spheres_dipole_mnpbem",
    "spheres_dipole_pygdm2",
    "spheres_dipole_smuthi",
    "spheres_dipole_treams",
    "spheres_plane_wave_bhmie",
    "spheres_plane_wave_consistency",
    "spheres_plane_wave_mnpbem",
    "spheres_plane_wave_pygdm2",
    "spheres_plane_wave_scattnlay",
    "spheres_plane_wave_treams",
)
for _module in PAGE_MODULES:
    __import__(_module)
    for _name, _title, _x, _y, _log, _env, _fn in _page.FIGURES[_module]:
        figure(_name, _title, _x, _y, log=_log, env=_env, script=f"benchmarks/pages/{_module}.py",
               extra=_site(_name))(_fn)
        PAGE_RECORDS[_name] = (_page.record, (_name, _title, _x, _y, _fn, _log, f"benchmarks/pages/{_module}.py"))

# --- films, plane wave: dispersion models and ellipsometry (0.11.0) ----------------------------------------------


# --- films, point dipole ----------------------------------------------------------------------------------------


@figure("lukosz-kunz-1977", "Dipoles in air near a lossless half-space of n = 1.5 and n = 3: total radiated power, Lukosz & Kunz 1977, Fig. 4", "distance z₀ / λ₁", "L / L∞")
def lukosz_kunz_1977():
    curve, paper = classic_curve("lukosz_kunz_1977_fig4"), classic_values("lukosz_kunz_1977")
    x = curve["z0_over_lambda1"]
    keys = (("electric perp n=1.5", "electric ⊥, n = 1.5"), ("electric par n=1.5", "electric ∥, n = 1.5"),
            ("electric par n=3.0", "electric ∥, n = 3"), ("magnetic perp n=1.5", "magnetic ⊥, n = 1.5"))
    e3 = curve["electric par n=3.0"]

    def turn(lo, hi, low=True):
        m = (x >= lo) & (x <= hi)
        return float(x[m][np.argmin(e3[m]) if low else np.argmax(e3[m])])

    refs = {"electric perp n=1.5": {float(x[0]): paper["e, perp, n = 1.5, at contact"], 0.4: paper["e, perp, n = 1.5, z0/λ1 = 0.4: back to ~1"]},
            "electric par n=1.5": {float(x[0]): paper["e, par, n = 1.5, at contact"]},
            "electric par n=3.0": {paper["e, par, n = 3: z0/λ1 of the first minimum"]: paper["e, par, n = 3: first minimum"],
                                   turn(0.25, 0.45, False): paper["e, par, n = 3: maximum near z0/λ1 = 0.35"],
                                   turn(0.5, 0.75): paper["e, par, n = 3: second minimum near z0/λ1 = 0.6"]},
            "magnetic perp n=1.5": {float(x[0]): paper["m, perp, n = 1.5, at contact"]}}
    return dict(x=sig(x), series=[series(f"s{i}", f"{lab}, PyStratify", curve[k], "pystratify") for i, (k, lab) in enumerate(keys)]
                + [series(f"s{i}_ref", f"{lab}, read off Figs. 4 and 6", at(x, refs[k]), "reference") for i, (k, lab) in enumerate(keys)],
                worst=_read_worst("lukosz_kunz_1977"), worst_label="largest relative difference from the read values",
                note="Lukosz & Kunz, J. Opt. Soc. Am. 67, 1607 (1977): the power a dipole radiates near a plane interface, against its "
                     "distance in units of the wavelength in its own medium. Values read off the figures to ±0.05; positions not printed are "
                     "marked where PyStratify's curve turns. From crosscheck/classics/lukosz_kunz_1977.py.")


@figure("neyts-1998-microcavity", "A dipole in 190 nm of ZnS (n = 2.3) on aluminium, 600 nm: emission into air and total rate across the layer, Neyts 1998, Figs. 9–11", "distance from the mirror [nm]", "fraction, rate")
def neyts_1998_microcavity():
    curve, paper = classic_curve("neyts_1998_fig10_column"), classic_values("neyts_1998")
    z = curve["z_from_mirror_um"] * 1e3
    e, f = curve["E_appl_isotropic"], curve["F_isotropic"]
    m = z > 15  # F diverges as z⁻³ at the mirror
    z, e, f = z[m], e[m], f[m]
    return dict(x=sig(z), series=[series("e", "E_appl, escape into air (PyStratify)", e, "pystratify"), series("f", "F, total rate (PyStratify)", f, "pystratify"),
                                  series("e_ref", "E_appl = 0.38, printed (dipole at the air side)", at(z, {float(z[-1]): paper["maximum E_appl (dipole at the air side)"]}), "reference"),
                                  series("f_ref", "F = 0.817, printed", at(z, {float(z[-1]): paper["F at that point"]}), "reference")],
                worst=float(max(abs(e[-1] - 0.38) / 0.38, abs(f[-1] - 0.817) / 0.817)), worst_label="largest relative difference from the printed values",
                note="Neyts, J. Opt. Soc. Am. A 15, 962 (1998): an isotropic dipole in the ZnS layer of an electroluminescent device on an "
                     "Al mirror (n = 1 + 6i), the layer at the optimum thickness n_e d_e / λ = 0.73 (PyStratify finds 0.725). The printed "
                     "maximum outcoupling 0.38 and F = 0.817 are for the dipole at the air side; F grows as z⁻³ toward the mirror (not drawn "
                     "below 15 nm). From crosscheck/classics/neyts_1998.py.")


# --- films, focused beam ----------------------------------------------------------------------------------------

def _film_cone(mode):
    from focused_checks import film_brute
    from pystratify import focused as F

    n = np.array([1.0, 1.55 + 0.03j, 1.42, 1.0 if mode == "transmission" else 0.3 + 18j])
    d = np.array([inf, 1.2, 3.0, inf])
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7 if mode == "transmission" else None,
                          collection_obscuration=0.35 if mode == "transmission" else None, mode=mode)
    lam = np.linspace(2.5, 6.0, 15)
    r = F.focused_films(n, d, lam, beam)
    ours = np.stack([np.ravel(r["reflectance"]), np.ravel(r["transmittance"]), np.ravel(r.detected)], -1)
    ref = np.array([film_brute(n, d, None, w, beam, (1.0, 0.0))[:3] for w in lam])
    keys = ("reflectance", "transmittance", "detected")
    return dict(x=sig(lam), series=[series(k, f"{k}, PyStratify", ours[:, i], "pystratify") for i, k in enumerate(keys)]
                + [series(f"{k}_ref", f"{k}, 2D pupil grid", ref[:, i], "reference") for i, k in enumerate(keys)],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note=f"In {mode}: a 1.2 µm film of n = 1.55 + 0.03i on 3 µm of n = 1.42, "
                     + ("in air, detected by a 0.35–0.7 condenser. " if mode == "transmission" else "on a metal (n = 0.3 + 18i), detected in reflection. ")
                     + "PyStratify reduces the beam to a one-dimensional quadrature over the cone angle; the reference sums a two-dimensional "
                       "pupil grid of x- and y-directed plane waves, each split into s and p and solved separately (focused_checks.py). The "
                       "test also compares every layer's absorption, for both pupil polarizations.")


for _mode in ("transmission", "reflection"):
    def _make(mode=_mode):
        return _film_cone(mode)
    _make.__name__ = f"film_cone_{_mode}"
    figure(f"film-cone-{_mode}", f"A two-layer film under a focused beam (NA 0.6/0.2), in {_mode}: the cone quadrature against a 2D pupil grid", "λ [µm]", "power / incident")(_make)


@figure("film-path-factor", "A weak absorber (α = 10 cm⁻¹, 1 µm) in n = 1.5 under an aplanatic NA 0.23–0.5 cone: the path factor", "λ [µm]", "ln10 A / (α d)")
def film_path_factor_figure():
    from pystratify import focused as F

    alpha, lam = 1e-5, np.linspace(2.0, 10.0, 33)
    kappa = alpha * lam / (4 * np.pi)
    n = np.stack((np.full(lam.size, 1.5), 1.5 + 1j * kappa, np.full(lam.size, 1.5)), 1)
    r = F.focused_films(n, [inf, 1.0, inf], lam, ps.FocusedBeam(0.5, 0.23), reference=([1.5, 1.5], [inf, inf]))
    factor = np.ravel(r.apparent_absorbance) * np.log(10) / alpha
    return dict(x=sig(lam), series=[series("factor", "ln10 A / (α d), PyStratify", factor, "pystratify"),
                                    series("factor_ref", "⟨1 / cos θ⟩ of the aplanatic cone, 1.0357", np.full(lam.size, 1.0357), "reference")],
                worst=float(np.max(np.abs(factor - 1.0357))), worst_label="largest |difference| (1.0357 is printed to 5 digits)",
                note="A focused beam crosses a film obliquely, so a weak absorber looks thicker by the cone's mean path factor ⟨1 / cos θ⟩, "
                     "1.0357 for an aplanatic NA 0.23–0.5 pupil in n = 1.5, the ruling the test pins. The test also checks that an empty stack, "
                     "or a sample that is its own reference, has zero absorbance and that Köhler illumination equals coherent for a film.")


@figure("film-koehler-coherent", "A film (1.2 µm, n = 1.55 + 0.03i) in air under NA 0.5/0.23: Köhler illumination against a coherent focus", "λ [µm]", "apparent absorbance")
def film_koehler_coherent():
    from pystratify import focused as F

    n, d = [1.0, 1.55 + 0.03j, 1.0], [inf, 1.2, inf]
    lam = np.linspace(2.5, 8.0, 23)
    beams = [ps.FocusedBeam(0.5, 0.23), ps.FocusedBeam(0.5, 0.23, illumination="kohler", field_stop=40.0)]
    r = F.focused_films(n, d, lam, beams, reference=([1.0, 1.0], [inf, inf]))
    coherent, kohler = np.ravel(r.apparent_absorbance[0]), np.ravel(r.apparent_absorbance[1])
    return dict(x=sig(lam), series=[series("coherent", "coherent focus, PyStratify", coherent, "pystratify"),
                                    series("kohler", "Köhler (field stop 40 µm)", kohler, "reference")],
                worst=float(np.max(np.abs(coherent - kohler))), worst_label="largest |difference|",
                note="A laterally uniform film cannot tell how the pupil's plane waves are phased: coherent and Köhler illumination see the same "
                     "cone and must give the same detected power.")


@figure("film-focused-near-field", "A two-layer film (0.8 µm n = 1.6 + 0.05i, 2 µm n = 1.42) in air under a focused beam (NA 0.6/0.2, focus off-axis), 3 µm: |E|² along the depth", "z [µm]", "|E|²")
def film_focused_near_field():
    from focused_checks import film_near_field_pair

    z = np.linspace(-1.5, 4.5, 121)
    field, pair = film_near_field_pair(np.column_stack([np.full_like(z, 0.1), np.zeros_like(z), z]))
    ours = np.sum(np.abs(np.asarray(field.e[0])) ** 2, -1)
    ref = np.sum(np.abs(pair[0]) ** 2, -1)
    return dict(x=sig(z), series=[series("e2", "|E|², PyStratify", ours, "pystratify"), series("e2_ref", "|E|², position-resolved plane waves summed", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Along x = 0.1, y = 0 µm from the air above, through both layers (0–0.8 and 0.8–2.8 µm) into the air below; x-polarized "
                     "pupil, focus at (0.3, −0.2, 0.5) µm. The reference sums a 48 × 48 pupil grid of plane waves, each from position_resolved. "
                     "The test also checks Parseval: the lateral integral of |E|² at a depth against the pupil's.")


# --- spheres, plane wave ----------------------------------------------------------------------------------------


# The sphere rows' closed forms, extended-precision references and independent methods (2026-10-12): each calls the
# reference its test calls (sphere_checks.py, mie_codes.py, mp_reference.py, sphere_emitter_checks.py,
# focused_checks.py, pystratify.references) over a sweep instead of at a few points.


# The papers' rows (2026-10-12): PyStratify's curve with the published values as the circles. A crosscheck
# classic (crosscheck/classics/) writes its curve and the values it read off the paper to results/; the figure
# reads both, so a page shows exactly what the classic checked.
CLASSICS = ROOT / "crosscheck" / "classics" / "results"


def classic_values(name):
    """{check: the paper's value} of a crosscheck classic's results/<name>.json."""
    return {c["check"]: c["paper"] for c in json.loads((CLASSICS / f"{name}.json").read_text())}


def classic_curve(name):
    """{column: values} of a crosscheck classic's results/<name>.csv."""
    lines = (CLASSICS / f"{name}.csv").read_text().splitlines()
    columns = lines[0].split(",")
    data = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
    return {c: data[:, i] for i, c in enumerate(columns)}


@figure("bh-reference-values", "Bohren & Huffman's BHMIE case, m = 1.55 at 0.6328 µm: the printed Q_ext and Q_back", "radius [µm]", "Q")
def bh_reference_values():
    lam, m = 0.6328, 1.55
    a = np.linspace(0.45, 0.60, 151)
    q_ext, q_back = [], []
    for radius in a:
        sol = ps.solve([radius], [m, 1.0], lam, l_max=30)
        x = 2 * np.pi * radius / lam
        q_ext.append(ps.cross_sections(sol).q_ext[0])
        q_back.append(4 * np.pi * ps.scattering_pattern(sol, np.pi).differential_cross_section[0] / (np.pi * radius ** 2))
    q_ext, q_back = np.array(q_ext), np.array(q_back)
    i = int(np.argmin(np.abs(a - 0.525)))
    return dict(x=sig(a), series=[series("ext", "Q_ext, PyStratify", q_ext, "pystratify"), series("back", "Q_back, PyStratify", q_back, "pystratify"),
                                  series("ext_ref", "Q_ext = 3.10543, printed (a = 0.525 µm)", at(a, {0.525: 3.10543}), "reference"),
                                  series("back_ref", "Q_back = 2.92534, printed", at(a, {0.525: 2.92534}), "reference")],
                worst=float(max(abs(q_ext[i] - 3.10543), abs(q_back[i] - 2.92534))), worst_label="largest difference from the printed values (5 digits)",
                note="The book's Appendix A runs BHMIE for a sphere of m = 1.55, a = 0.525 µm at 0.6328 µm and prints Q_ext and Q_back to six "
                     "figures; PyStratify's curves through that radius, Q_back from its far-field pattern at 180°.")


@figure("many-identical-layers", "A sphere of m = 1.33 + 0.001i cut into 100 identical layers [W97]", "size parameter x", "Q_ext")
def many_identical_layers():
    xs = np.linspace(1, 100, 100)
    m, lam, N = 1.33 + 0.001j, 2 * np.pi, 100
    layered, single = [], []
    for x in xs:
        layered.append(ps.cross_sections(ps.solve(np.linspace(x / N, x, N), [m] * N + [1.0], lam)).q_ext[0])
        single.append(ps.cross_sections(ps.solve([x], [m, 1.0], lam)).q_ext[0])
    return dict(x=sig(xs), series=[series("layers", "Q_ext, 100 identical layers (PyStratify)", layered, "pystratify"),
                                   series("one", "Q_ext, one sphere", single, "reference")],
                worst=worst(layered, single, relative=True), worst_label="largest relative difference",
                note="Wu & Wang's and Yang's test for a layered recursion: 99 interfaces between identical media must change nothing. "
                     "The report repeats it with 1000 layers at x = 1000, and checks graded (Kai–Massoli, Luneburg) profiles absorb nothing.")


@figure("graded-droplet-rainbow", "The rainbow of a 250 µm droplet with a graded index (128 layers, n = 1.33 → 1.36), 514.5 nm [W97]", "scattering angle [°]", "|S₁|", log=True)
def graded_droplet_rainbow():
    lam, R, N = 0.5145, 125.0, 128
    radii = np.linspace(R / N, R, N)
    n = list(1.33 + 0.03 * (radii - R / N / 2) / R) + [1.0]
    sol = ps.solve(radii, n, lam)
    more = ps.solve(radii, n, lam, l_max=sol.orders.size + 200)
    theta = np.linspace(130, 150, 401)
    ours = np.abs(ps.scattering_amplitudes(sol, np.radians(theta))[1][0])
    ref = np.abs(ps.scattering_amplitudes(more, np.radians(theta))[1][0])
    return dict(x=sig(theta), series=[series("s", f"|S₁|, PyStratify ({sol.orders.size} orders)", ours, "pystratify"),
                                      series("s_ref", f"|S₁|, {sol.orders.size + 200} orders", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="x ≈ 1527 and 128 shells: the primary rainbow near 138° and its supernumerary fringes, from PyStratify's own "
                     "truncation against 200 orders more. A truncation that stopped too early would move the fringes.")


@figure("chiral-polarisabilities", "A small chiral sphere (k a = 0.001, ε = −4 + 0.3i, μ = 1): the dipole block against Guzatov & Klimov's quasi-static polarisabilities", "chirality χ", "|T| / k³a³")
def chiral_polarisabilities():
    from sphere_checks import small_chiral_t

    chis = np.linspace(-0.2, 0.2, 81)
    eps, mu = -4.0 + 0.3j, 1.0
    scale = (1e-3) ** 3
    ours, ref = [], []
    for chi in chis:
        t, (ee, hh, eh) = small_chiral_t(eps, mu, chi)
        ours.append([t[0, 0], t[1, 1], t[0, 1]]), ref.append([ee, hh, eh])
    ours, ref = np.abs(np.array(ours)) / scale, np.abs(np.array(ref)) / scale
    names = ("T_EE (electric)", "T_HH (magnetic)", "T_EH (cross, chiral)")
    return dict(x=sig(chis), series=[series(f"t{i}", f"|{nm}|, PyStratify", ours[:, i], "pystratify") for i, nm in enumerate(names)]
                + [series(f"t{i}_ref", f"|{nm}|, Eq. 48", ref[:, i], "reference") for i, nm in enumerate(names)],
                worst=float(np.max(np.abs(ours - ref).max(axis=0) / ref.max(axis=0))), worst_label="largest difference relative to each curve's peak",
                note="Guzatov & Klimov, New J. Phys. 14, 123009 (2012), Eq. 48: the electric, magnetic and cross polarisabilities of a "
                     "Drude–Born–Fedorov sphere, against the order-1 block of PyStratify's chiral T-matrix (agreement to (ka)² = 1e-6). "
                     "The cross term vanishes without chirality; the test also covers magnetic (μ ≠ 1) spheres.")


@figure("goncalves-feibelman", "A radial dipole beside a 5 nm gold sphere (n = 0.141 + 3.152i), 614 nm, with Feibelman d-parameters: against Gonçalves et al. 2020", "distance from the centre [nm]", "Γ / Γ₀", log=True)
def goncalves_feibelman():
    from sphere_emitter_checks import AU_GONCALVES, GONCALVES_CASES, _goncalves_perp

    r = np.geomspace(5.5, 15, 25)
    out, refs, diffs = [], [], []
    for d_perp, d_par in GONCALVES_CASES:
        rates = ps.decay_rates([5.0], [AU_GONCALVES, 1.0], 614.0, r, sheets={0: ps.Feibelman(d_perp, d_par)}, tol=1e-14)
        # the series converges as (R/r)^l: 300 orders near the surface, where 120 (the test's, at 6 and 7 nm) fall short
        ref = np.array([_goncalves_perp(5.0, AU_GONCALVES, 1.0, 614.0, ri, d_perp, d_par, orders=300 if ri < 6 else 120) for ri in r])
        label = f"d⊥ = {d_perp:g}, d∥ = {d_par:g} nm"
        out.append(series(f"c{len(out)}", f"{label}, PyStratify", rates.total[:, 0], "pystratify"))
        refs.append(series(f"c{len(refs)}_ref", f"{label}, Eqs. (4a), (13a)", ref, "reference"))
        diffs.append(np.max(np.abs(rates.total[:, 0] / ref - 1)))
    return dict(x=sig(r), series=out + refs, worst=float(max(diffs)), worst_label="largest relative difference",
                note="Gonçalves et al., Nat. Commun. 11, 366 (2020): quantum surface corrections enter the Mie coefficients through the "
                     "d-parameters; their closed form summed in mpmath (40 digits, 120 orders) against PyStratify's Feibelman boundary "
                     "conditions (300 orders within 1 nm of the surface). The d-parameters shift the quenching close to the surface.")


@figure("fu-2013-kerker", "A 75 nm silicon sphere in air: forward-to-backward scattering, Fu et al. 2013, Fig. 1a", "λ [nm]", "F / B")
def fu_2013_kerker():
    curve, paper = classic_curve("fu_2013_fig1a"), classic_values("fu_2013")
    nm, ratio = curve["wavelength_nm"], curve["F_over_B"]
    points = {paper["wavelength of the F/B maximum (nm)"]: paper["F/B maximum"],
              paper["F = B, electric dipole (nm)"]: 1.0, paper["F = B, magnetic dipole (nm)"]: 1.0}
    ref = at(nm, points)
    crossings = nm[1:][np.diff(np.sign(ratio - 1)) != 0]
    window = (nm > 620) & (nm < 720)
    shifts = [abs(nm[window][np.argmax(ratio[window])] - paper["wavelength of the F/B maximum (nm)"])]
    shifts += [abs(crossings[np.argmin(abs(crossings - x))] - x) for x in (paper["F = B, electric dipole (nm)"], paper["F = B, magnetic dipole (nm)"])]
    return dict(x=sig(nm), series=[series("fb", "F/B, PyStratify (Green 2008 silicon)", ratio, "pystratify"),
                                   series("fb_ref", "printed in the paper (p. 3)", ref, "reference")],
                worst=float(max(shifts)), worst_label="largest shift of a printed feature [nm]",
                note="Fu et al., Nat. Commun. 4, 1527 (2013): F/B peaks near 8 at 660 nm (the first Kerker condition) and F = B at 500 nm "
                     "(electric dipole) and 603 nm (magnetic dipole); the paper names no silicon data, so the peak height depends on the "
                     "table chosen. From crosscheck/classics/fu_2013.py, which also checks the minimum below 0.5 near 564 nm.")


# Other groups' codes (2026-10-12). treams, scattnlay and pyGDM2 install from PyPI ([benchmarks] extra; pyGDM2
# needs NumPy < 2.4); MNPBEM runs under GNU Octave (mnpbem_octave.py); a figure whose code is missing is skipped.


# --- spheres, point dipole --------------------------------------------------------------------------------------


@figure("layered-magnetic-balance", "A dipole in the shell of a layered magnetic sphere (radii 1, 11; (ε, μ) = (2, 8), (1, 1), (8, 2); k₀ = 1) [Y24]: total against radiative", "distance from the centre", "Γ / Γ₀", log=True)
def layered_magnetic_balance():
    eps, mu = np.array([2, 1, 8.0]), np.array([8, 1, 2.0])
    r = np.geomspace(1.01, 10.99, 60)
    rates = ps.decay_rates([1.0, 11.0], np.sqrt(eps * mu), 2 * np.pi, r, mu=mu, tol=1e-9, normalization="shell")
    return dict(x=sig(r), series=[
        series("tot_perp", "total, radial (PyStratify, from the LDOS)", rates.total[:, 0], "pystratify"),
        series("tot_par", "total, tangential (PyStratify, from the LDOS)", rates.total[:, 1], "pystratify"),
        series("rad_perp", "radiative, radial (power reaching infinity)", rates.radiative[:, 0], "reference"),
        series("rad_par", "radiative, tangential (power reaching infinity)", rates.radiative[:, 1], "reference")],
        worst=worst(rates.total, rates.radiative, relative=True), worst_label="largest relative difference",
        note="Yuan, Zhu & Zhu's Table 1 configuration: every layer is lossless, so all the power the dipole emits reaches infinity "
             "and the total rate (from the local density of states) must equal the radiative one. Close to an interface the total "
             "carries the rounding of the reactive near field, as Majic & Le Ru note (the report: 5.5e-8 at 0.001 from it).")


@figure("guzatov-chiral-molecule", "A chiral molecule beside a small chiral sphere (ε = −4 + 0.3i, χ = 0.1, k₀a = 0.01): its radiative rate against Guzatov & Klimov's Eq. 46", "distance from the centre / radius", "Γ_rad / Γ₀")
def guzatov_chiral_molecule():
    from sphere_emitter_checks import guzatov_klimov_rate

    d = np.linspace(1.3, 4, 55)
    out, refs, diffs = [], [], []
    for xi in (0.1, -0.1):
        pairs = np.array([guzatov_klimov_rate(-4.0 + 0.3j, 1.0, 0.1, 0.01, di, xi) for di in d])
        label = f"ξ = {xi:+g}"
        out.append(series(f"x{len(out)}", f"{label}, PyStratify", pairs[:, 0], "pystratify"))
        refs.append(series(f"x{len(refs)}_ref", f"{label}, Eq. 46", pairs[:, 1], "reference"))
        diffs.append(np.max(np.abs(pairs[:, 0] / pairs[:, 1] - 1)))
    return dict(x=sig(d), series=out + refs, worst=float(max(diffs)), worst_label="largest relative difference",
                note="Guzatov & Klimov, New J. Phys. 14, 123009 (2012): a molecule with electric moment d₀ and magnetic moment −iξd₀ "
                     "(the two handednesses, ξ = ±0.1) beside a chiral plasmonic sphere radiates at different rates: chiral discrimination. "
                     "Their quasi-static Eq. 46 against PyStratify's full chiral solution; agreement to O((k₀a)²).")


@figure("anger-2006-excitation", "A molecule beside an 80 nm gold sphere (ε = −12.99 + 1.09i), 650 nm: excitation and quantum yield, Anger et al. 2006, Fig. 1a", "distance from the surface [nm]", "rate / free space")
def anger_2006_excitation():
    curve, paper = classic_curve("anger_2006_fig1"), classic_values("anger_2006")
    z = curve["z_um"] * 1e3
    exc, qy = curve["excitation d=80nm"], curve["quantum_yield d=80nm"]
    pick = lambda what: {float(m.group(1)): v for k, v in paper.items() if (m := __import__("re").match(rf"80 nm: {what} at z = (\d+) nm", k))}  # noqa: E731
    exc_ref, qy_ref = at(z, pick("excitation")), at(z, pick("quantum yield"))
    diffs = [abs(a - b) / b for a, b in zip(np.r_[exc, qy], np.r_[exc_ref, qy_ref]) if np.isfinite(b)]
    return dict(x=sig(z), series=[series("exc", "excitation |E|²/|E₀|², PyStratify", exc, "pystratify"), series("qy", "quantum yield, PyStratify", qy, "pystratify"),
                                  series("exc_ref", "excitation, read off Fig. 1a", exc_ref, "reference"), series("qy_ref", "quantum yield, read off Fig. 1a", qy_ref, "reference")],
                worst=float(max(diffs)), worst_label="largest relative difference from the read values (±5%)",
                note="Anger, Bharadwaj & Novotny, Phys. Rev. Lett. 96, 113002 (2006): the excitation rate grows toward the gold and the "
                     "quantum yield falls (quenching); values read off Fig. 1a to ±5%, from crosscheck/classics/anger_2006.py.")


@figure("anger-2006-fluorescence", "Fluorescence beside gold spheres of 20, 60, 80 and 100 nm, 650 nm: Anger et al. 2006, Fig. 1b", "distance from the surface [nm]", "fluorescence / free space")
def anger_2006_fluorescence():
    curve, paper = classic_curve("anger_2006_fig1"), classic_values("anger_2006")
    z = curve["z_um"] * 1e3
    out = [series(f"d{d}", f"{d} nm sphere, PyStratify", curve[f"fluorescence d={d}nm"], "pystratify") for d in (20, 60, 80, 100)]
    peak = at(z, {paper["80 nm: z of the maximum (nm)"]: paper["80 nm: fluorescence maximum"]})
    fl80 = curve["fluorescence d=80nm"]
    i = int(np.argmax(fl80))
    return dict(x=sig(z), series=out + [series("peak_ref", "80 nm maximum, read off Fig. 1b (z ≈ 10 nm)", peak, "reference")],
                worst=float(abs(fl80[i] / paper["80 nm: fluorescence maximum"] - 1)), worst_label="80 nm maximum, relative difference",
                note="The product of excitation and quantum yield peaks near 10 nm: enhancement turns into quenching. The paper prints the "
                     "maxima of all four sizes (5.3, 3.5, 2.3, 1.05 from the largest; PyStratify 5.27, 3.52, 2.20, 1.04) but the position "
                     "only for 80 nm, so that is the point marked.")


# --- spheres, focused beam --------------------------------------------------------------------------------------

@figure("focused-sphere-superposition", "A coated sphere (0.8/1.4 µm) under a focused beam (NA 0.6/0.2, collection 0.8/0.1): against plane-wave superposition", "λ [µm]", "power / incident")
def focused_sphere_superposition():
    from focused_checks import SPHERE, brute_sphere
    from pystratify import focused as F

    radii, n = SPHERE
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.8, collection_obscuration=0.1, polarization="x")
    lam = np.linspace(2.5, 6.0, 15)
    ours = F.focused_spheres(radii, n, lam, beam)
    ref = [brute_sphere(radii, n, w, beam) for w in lam]
    keys = ("extinction", "scattering", "detected")
    got = {k: np.ravel(ours[k]) for k in keys}
    exact = {k: np.array([r[k] for r in ref]) for k in keys}
    return dict(x=sig(lam), series=[series(k, f"{k}, PyStratify", got[k], "pystratify") for k in keys]
                + [series(f"{k}_ref", f"{k}, superposition of plane waves", exact[k], "reference") for k in keys],
                worst=max(worst(got[k], exact[k], relative=True) for k in keys), worst_label="largest relative difference",
                note="The reference sums the sphere's amplitude matrix over a grid of incident and scattered directions in the pupil "
                     "(focused_checks.py); PyStratify uses its closed-form beam coefficients. The incident power is 1.")


@figure("focused-na-limit", "The same coated sphere, λ = 3 µm: a focused beam as its NA closes, against the plane wave", "NA", "extinction / (|E(0)|² σ_ext)")
def focused_na_limit():
    from focused_checks import SPHERE
    from pystratify import focused as F

    radii, n = SPHERE
    lam = 3.0
    sigma = ps.cross_sections(ps.solve(radii, n, lam)).ext[0]
    nas = np.linspace(0.01, 0.6, 60)
    ratios = []
    for na in nas:
        out = F.focused_spheres(radii, n, lam, ps.FocusedBeam(na))
        focus = (1 / lam) ** 2 * F.focal_amplitude((0.0, na)) ** 2
        ratios.append(float(np.ravel(out["extinction"])[0] / focus / sigma))
    ratios = np.array(ratios)
    return dict(x=sig(nas), series=[series("ratio", "focused extinction / (|E(0)|² σ_ext), PyStratify", ratios, "pystratify"),
                                    series("plane", "plane wave (= 1)", np.ones_like(ratios), "reference")],
                worst=float(abs(ratios[0] - 1)), worst_label="difference from 1 at NA 0.01",
                note="A focused beam's extinction, divided by its intensity at the focus and the plane-wave cross section, tends to 1 as the NA "
                     "closes and the beam becomes a plane wave across the sphere; a tightly focused beam sees less of it.")


@figure("focused-axis-path", "The coated sphere at λ = 3 µm, focus moved along the axis: the fast on-axis path against the general one", "focus position z [µm]", "power / incident")
def focused_axis_path():
    from focused_checks import SPHERE
    from pystratify import focused as F

    radii, n = SPHERE
    lam = np.array([3.0])
    sol = ps.solve(radii, n, lam)
    k = 2 * np.pi / lam
    z = np.linspace(-3, 3, 61)
    axis, general = {}, {}
    for zi in z:
        a = F._sphere_axis(sol.t, k, sol.orders, (0.23, 0.5), (0.1, 0.7), zi)
        g = F._sphere_general(sol.t, k, sol.orders, (0.23, 0.5), (0.1, 0.7), (0.0, 0.0, zi), (1.0, 0.0), False)
        for key, value in (("extinction", lambda d: np.sum(d["ext_by_order"])), ("detected", lambda d: np.ravel(d["detected"])[0])):
            axis.setdefault(key, []).append(float(value(a))), general.setdefault(key, []).append(float(value(g)))
    return dict(x=sig(z), series=[series(k_, f"{k_}, on-axis path (PyStratify)", axis[k_], "pystratify") for k_ in ("extinction", "detected")]
                + [series(f"{k_}_ref", f"{k_}, general path", general[k_], "reference") for k_ in ("extinction", "detected")],
                worst=max(worst(axis[k_], general[k_], relative=True) for k_ in ("extinction", "detected")), worst_label="largest relative difference",
                note="On the axis the beam couples only to azimuthal orders m = ±1, and PyStratify takes a shortcut; the general path, for a focus "
                     "anywhere, must give the same numbers. NA 0.5/0.23 in, 0.7/0.1 collected.")


@figure("koehler-cone-average", "The coated sphere under Köhler illumination (NA 0.6/0.2, field stop 25 µm): against a cone average of plane waves", "λ [µm]", "power / incident")
def koehler_cone_average():
    from focused_checks import SPHERE, sphere_koehler_reference
    from pystratify import focused as F

    radii, n = SPHERE
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7, collection_obscuration=0.3, illumination="kohler", field_stop=25.0)
    lam = np.linspace(2.5, 6.0, 15)
    ours = F.focused_spheres(radii, n, lam, beam)
    ref = [sphere_koehler_reference(w) for w in lam]
    keys = ("extinction", "detected")
    got = {k: np.ravel(ours[k]) for k in keys}
    exact = {k: np.array([r[k] for r in ref]) for k in keys}
    return dict(x=sig(lam), series=[series(k, f"{k}, PyStratify", got[k], "pystratify") for k in keys]
                + [series(f"{k}_ref", f"{k}, cone average of plane waves", exact[k], "reference") for k in keys],
                worst=max(worst(got[k], exact[k], relative=True) for k in keys), worst_label="largest relative difference",
                note="Köhler illumination is incoherent across the pupil, so a sphere sees an average of plane waves over the illumination cone; "
                     "the reference does that average with the plane-wave cross sections and the amplitude matrix.")


@figure("focused-near-field", "A coated sphere (0.6/1.2 µm) under a focused beam (NA 0.6/0.2, y polarized, focus off-centre), λ = 3 µm: |E|² along a line against plane-wave near fields", "x [µm]", "|E|²")
def focused_near_field():
    from focused_checks import sphere_near_field_pair

    x = np.linspace(-2.5, 2.5, 101)
    points = np.column_stack([x, np.full_like(x, 0.2), np.full_like(x, 0.25)])
    field, E, _ = sphere_near_field_pair(points)
    ours = np.sum(np.abs(np.asarray(field.e[0])) ** 2, -1)
    ref = np.sum(np.abs(E) ** 2, -1)
    return dict(x=sig(x), series=[series("e2", "|E|², PyStratify", ours, "pystratify"), series("e2_ref", "|E|², superposition of plane-wave near fields", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Along y = 0.2, z = 0.25 µm through the sphere (core radius 0.6, shell 1.2 µm): the beam's field from PyStratify's beam "
                     "coefficients against 2016 plane waves from the pupil, each solved by the plane-wave near field and summed.")


# --- cylinders --------------------------------------------------------------------------------------------------


def _bh_peak(x, y, lo=None, hi=None):
    m = np.ones_like(x, dtype=bool) if lo is None else (x >= lo) & (x <= hi)
    i = np.flatnonzero(m)[np.argmax(y[m])]
    return float(x[i])


def _read_worst(name):
    """Largest relative difference between a classic's computed and read values."""
    return float(max(abs(c["computed"] - c["paper"]) / abs(c["paper"]) for c in json.loads((CLASSICS / f"{name}.json").read_text())))


@figure("bohren-huffman-cylinder", "A non-absorbing cylinder of n = 1.55 in air at 0.6328 µm: scattering per unit volume, Bohren & Huffman Fig. 8.7", "diameter [µm]", "C_sca / V [10⁴ cm⁻¹]")
def bohren_huffman_cylinder():
    curve, paper = classic_curve("bohren_huffman_1983_fig8_7"), classic_values("bohren_huffman_1983")
    d, par, perp = curve["diameter_um"], curve["Csca_over_V_parallel_1e4_per_cm"], curve["Csca_over_V_perpendicular_1e4_per_cm"]
    par_ref = at(d, {_bh_peak(d, par, .08, .25): paper["parallel: first maximum"], _bh_peak(d, par, .25, .40): paper["parallel: second maximum"],
                     paper["parallel: diameter of the highest maximum (µm)"]: paper["parallel: highest maximum"],
                     _bh_peak(d, par, .55, .70): paper["parallel: shoulder after it"]})
    perp_ref = at(d, {paper["perpendicular: diameter of the maximum (µm)"]: paper["perpendicular: maximum"]})
    return dict(x=sig(d), series=[series("par", "parallel (E along the axis), PyStratify", par, "pystratify"),
                                  series("perp", "perpendicular, PyStratify", perp, "pystratify"),
                                  series("par_ref", "parallel, read off Fig. 8.7", par_ref, "reference"),
                                  series("perp_ref", "perpendicular, read off Fig. 8.7", perp_ref, "reference")],
                worst=_read_worst("bohren_huffman_1983"), worst_label="largest relative difference from the read values",
                note="Bohren & Huffman, Absorption and Scattering of Light by Small Particles (1983), §8.4: the scattering width over the "
                     "cross-sectional area, 4 Q_sca / (π d), the book's demonstration that a cylinder scatters the two polarizations "
                     "differently. Maxima read off the figure to ±0.2 and ±0.03 µm; the book prints the diameters of the highest ones only, "
                     "so the others are marked where PyStratify's curve peaks. From crosscheck/classics/bohren_huffman_1983.py.")


def _ruan_fan_widths(x, g_core, g_shell):
    """Ruan & Fan's three-layer rod (crosscheck/classics/ruan_fan_2010.py): scattering and absorption widths in units
    of 2λ/π, and |t_m|² of the axial-magnetic channel per order m = 1..4, at ω/ωp = x."""
    sys.path.insert(0, str(ROOT / "crosscheck" / "classics"))
    from _common import drude_nk

    lp = 0.1376
    radii = np.array([0.3485, 0.5623, 0.6370]) * lp
    out = []
    for lam in lp / np.asarray(x):
        s = ps.solve_cylinder(radii, [drude_nk(lam, lp, g_core), 3.6, drude_nk(lam, lp, g_shell), 1.0], lam)
        w = ps.cross_widths(s, "axial-magnetic")
        t = np.abs(s.t[:, 1, 1]) ** 2
        out.append([w["scattering"] / (2 * lam / np.pi), w["absorption"] / (2 * lam / np.pi)] + [t[s.orders == m].sum() for m in range(1, 5)])
    return np.array(out)


@figure("ruan-fan-superscattering", "A three-layer nanorod with four aligned resonances, lossless: superscattering, Ruan & Fan 2010, Fig. 1", "ω / ω_p", "width / (2λ/π)")
def ruan_fan_superscattering():
    paper = classic_values("ruan_fan_2010")
    fine = np.linspace(0.2535, 0.2550, 6001)  # the m = 3, 4 resonances are ~1e-6 ω_p wide
    w = _ruan_fan_widths(fine, 0.0, 0.0)
    keep = set(range(0, fine.size, 25)) | {fine.size - 1}
    for column in (0, 2, 3, 4, 5):  # every peak, with its flanks
        y = w[:, column]
        for i in np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1:
            keep |= set(range(max(i - 12, 0), min(i + 13, fine.size)))
    keep = np.array(sorted(keep))
    x, w = fine[keep], w[keep]
    ref = at(x, {paper["lossless: frequency of the peak (ω/ωp)"]: paper["lossless: total scattering peak (2λ/π)"]})
    return dict(x=[float(f"{v:.9g}") for v in x], series=[series("total", "total scattering, PyStratify", w[:, 0], "pystratify")]
                + [series(f"m{m}", f"channel m = ±{m}, PyStratify", w[:, 1 + m], "pystratify") for m in range(1, 5)]
                + [series("total_ref", "printed peak, 7.94 at ω = 0.2542 ω_p", ref, "reference")],
                worst=float(abs(np.max(w[:, 0]) - 7.94) / 7.94), worst_label="peak, relative difference from the printed 7.94",
                note="Ruan & Fan, Phys. Rev. Lett. 105, 013901 (2010): Drude core to 0.3485 λ_p, ε = 12.96 shell to 0.5623 λ_p, Drude shell "
                     "to 0.6370 λ_p, H along the axis. Each channel can scatter at most 2λ/π (the single-channel limit, 1 on this axis); "
                     "aligning four gives ~8. The m = 3 and 4 resonances are needles about 1e-6 ω_p wide, so the window is computed on "
                     "6001 points and drawn on a thinned grid that keeps every peak. From crosscheck/classics/ruan_fan_2010.py.")


@figure("ruan-fan-lossy", "The same nanorod with silver-like loss (γ = 0.004 ω_p core, 0.012 ω_p shell): Ruan & Fan 2010, Fig. 4a", "ω / ω_p", "width / (2λ/π)")
def ruan_fan_lossy():
    curve, paper = classic_curve("ruan_fan_2010_fig1_fig4a"), classic_values("ruan_fan_2010")
    x, sca, ab = curve["omega_over_omega_p"][::3], curve["sca_lossy"][::3], curve["abs_lossy"][::3]
    sca_ref = at(x, {float(x[np.argmax(sca)]): paper["lossy: total scattering peak (2λ/π)"]})
    abs_ref = at(x, {float(x[np.argmax(ab)]): paper["lossy: absorption peak (2λ/π)"]})
    return dict(x=sig(x), series=[series("sca", "scattering, PyStratify", sca, "pystratify"), series("abs", "absorption, PyStratify", ab, "pystratify"),
                                  series("sca_ref", "scattering peak, printed (1.92)", sca_ref, "reference"),
                                  series("abs_ref", "absorption peak, printed (0.73)", abs_ref, "reference")],
                worst=float(max(abs(np.max(sca) - 1.92) / 1.92, abs(np.max(ab) - 0.73) / 0.73)), worst_label="largest relative difference from the printed peaks",
                note="With silver's damping the needles broaden and the superscattering drops from ~8 to 1.92 single-channel limits; the "
                     "paper prints the peak heights (p. 3–4), not their frequencies, so they are marked where PyStratify's curves peak. "
                     "From crosscheck/classics/ruan_fan_2010.py.")


@figure("ruppin-sodium-wire", "A 1 nm sodium wire in vacuum and a 2 nm one in ε = 2, E across the wire: extinction, Ruppin 2001 (local curves)", "ω / ω_p", "Q_ext", log=True)
def ruppin_sodium_wire():
    curve, paper = classic_curve("ruppin_2001_fig1_fig4_local"), classic_values("ruppin_2001")
    x = curve["omega_over_omega_p"]
    one, two = curve["Qext_a1nm_host1"], curve["Qext_a2nm_host_eps2"]
    ref = at(x, {paper["a = 1 nm, ε_m = 1: local resonance (ω/ωp)"]: 10 ** paper["a = 1 nm: log10 of the local peak"],
                 1.4: 10 ** paper["a = 1 nm: log10 C_e at 1.4 ωp"]})
    i = int(np.argmax(one))
    return dict(x=sig(x), series=[series("a1", "a = 1 nm, ε_m = 1, PyStratify", one, "pystratify"), series("a2", "a = 2 nm, ε_m = 2, PyStratify", two, "pystratify"),
                                  series("a1_ref", "a = 1 nm: read off Fig. 1 (dashed) at the printed resonance ω_p/√2", ref, "reference")],
                worst=float(max(abs(np.log10(one[i]) - paper["a = 1 nm: log10 of the local peak"]), abs(np.log10(one[-1]) - paper["a = 1 nm: log10 C_e at 1.4 ωp"]))),
                worst_label="largest difference in log₁₀ Q_ext from the read values",
                note="Ruppin, Opt. Commun. 190, 205 (2001): Drude sodium (ω_p = 8.65e15 s⁻¹, γ = 0.01 ω_p). The calculator draws the paper's "
                     "dashed local curves; its solid ones add the hydrodynamic response. The printed resonances, ω_p/√(1 + ε_m), fall at 0.7071 "
                     "(PyStratify 0.707) and 0.5774 (0.576); the heights are read off the log axis. From crosscheck/classics/ruppin_2001.py.")


# --- cylinders, point dipole ------------------------------------------------------------------------------------


# --- cylinders, focused beam ------------------------------------------------------------------------------------

@figure("focused-cylinder-rows", "A coated cylinder (0.7/1.2 µm) under a focused beam (NA 0.6/0.2, collection 0.8/0.1, x polarized), 3 µm: one pupil row at a time against plane-wave sums", "pupil row s_y", "power per row")
def focused_cylinder_rows():
    from focused_checks import CYLINDER, CYLINDER_ROW_BEAM, row_brute
    from pystratify import focused as F

    radii, n = CYLINDER
    rows_ = F._CylinderRows(radii, n, 3.0, [CYLINDER_ROW_BEAM], 14)
    sy = np.linspace(0.01, 0.59, 30)
    ours = np.array([rows_(s)[:3] / 2 for s in sy])
    ref = np.array([row_brute(radii, n, 3.0, s, CYLINDER_ROW_BEAM, (1.0, 0.0), 14)[:3] for s in sy])
    keys = ("extinction", "scattering", "detected")
    return dict(x=sig(sy), series=[series(k, f"{k}, PyStratify", ours[:, i], "pystratify") for i, k in enumerate(keys)]
                + [series(f"{k}_ref", f"{k}, plane waves summed", ref[:, i], "reference") for i, k in enumerate(keys)],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="A cylinder conserves the beam's direction along its axis, so the pupil splits into rows of fixed s_y, each an oblique "
                     "cylinder problem. PyStratify sums a row in closed form over Fourier orders; the reference solves every plane wave of the "
                     "row (60 nodes per interval), adds their far fields on observation angles and integrates the powers directly. Focus at "
                     "(0.4, 0, 0.3) µm; the y polarization and the mirror rows −s_y are checked too.")


def _cylinder_whole_beam_figure(pol):
    from focused_checks import cylinder_whole_beam_pair

    lam = np.linspace(2.5, 6.0, 15)
    keys = ("extinction", "scattering", "detected")
    ours, ref = [], []
    for w in lam:
        r, total, width = cylinder_whole_beam_pair(w)
        p = "xy".index(pol)
        bp = r.by_polarization[pol]
        ours.append([float(np.ravel(bp[k])[0]) for k in keys]), ref.append(total[p * width:p * width + 3])
    ours, ref = np.array(ours), np.array(ref)
    return dict(x=sig(lam), series=[series(k, f"{k}, PyStratify (adaptive)", ours[:, i], "pystratify") for i, k in enumerate(keys)]
                + [series(f"{k}_ref", f"{k}, fixed 40-node quadrature", ref[:, i], "reference") for i, k in enumerate(keys)],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note=f"The whole beam, {pol} polarized ({'across' if pol == 'x' else 'along'} the axis): PyStratify integrates its rows over s_y "
                     "adaptively to 1e-9; the reference sums the same rows with a fixed 40-node rule on each side of the obscuration. "
                     "The incident power is 1.")


for _pol in "xy":
    def _make(pol=_pol):
        return _cylinder_whole_beam_figure(pol)
    _make.__name__ = f"focused_cylinder_whole_beam_{_pol}"
    figure(f"focused-cylinder-whole-beam-{_pol}", f"The coated cylinder (0.7/1.2 µm) under a focused beam (NA 0.6/0.2, collection 0.8), {_pol} polarized: the whole beam against a fixed quadrature",
           "λ [µm]", "power / incident")(_make)


@figure("focused-cylinder-koehler", "The coated cylinder (0.7/1.2 µm) under Köhler illumination (NA 0.6/0.2, collection 0.7/0.3, field stop 30 µm): against a cone average of plane waves", "λ [µm]", "power / incident")
def focused_cylinder_koehler():
    from focused_checks import CYLINDER, cylinder_koehler_reference
    from pystratify import focused as F

    radii, n = CYLINDER
    beam = ps.FocusedBeam(0.6, 0.2, collection_na=0.7, collection_obscuration=0.3, illumination="kohler", field_stop=30.0)
    lam = np.linspace(2.5, 6.0, 15)
    ours, ref = [], []
    for w in lam:
        r = F.focused_cylinders(radii, n, w, beam, m_max=14, tolerance=1e-9)
        t = cylinder_koehler_reference(w, 30.0)
        ours.append([float(np.ravel(r.by_polarization[p][k])[0]) for p in "xy" for k in ("extinction", "detected")])
        ref.append([v for p in "xy" for v in (t[p][0], t[p][2] + t[p][1])])
    ours, ref = np.array(ours), np.array(ref)
    labels = ("extinction, x", "detected, x", "extinction, y", "detected, y")
    return dict(x=sig(lam), series=[series(f"s{i}", f"{lab}, PyStratify", ours[:, i], "pystratify") for i, lab in enumerate(labels)]
                + [series(f"s{i}_ref", f"{lab}, cone average", ref[:, i], "reference") for i, lab in enumerate(labels)],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Köhler illumination is incoherent across the pupil: the cylinder sees each plane wave of the illumination cone on its "
                     "own, over the chord the field stop lights. The reference averages the oblique plane-wave extinction widths and the "
                     "scattered power that lands in the collection annulus over the cone (focused_checks.py); x is across the axis.")


@figure("focused-cylinder-na-limit", "The coated cylinder (0.7/1.2 µm) at λ = 3 µm: a focused beam as its NA closes, against the plane wave", "NA", "extinction / (∫|E|² dl · C_ext)")
def focused_cylinder_na_limit():
    from focused_checks import cylinder_na_ratio

    nas = np.linspace(0.01, 0.3, 30)
    ratios = {p: np.array([cylinder_na_ratio(na, p) for na in nas]) for p in "xy"}
    return dict(x=sig(nas), series=[series(p, f"{p} polarized ({'across' if p == 'x' else 'along'} the axis), PyStratify", ratios[p], "pystratify") for p in "xy"]
                + [series("plane", "plane wave (= 1)", np.ones_like(nas), "reference")],
                worst=float(max(abs(ratios[p][0] - 1) for p in "xy")), worst_label="difference from 1 at NA 0.01",
                note="A focused beam's extinction, divided by its |E|² integrated along the cylinder's axis and the plane-wave extinction "
                     "width, tends to 1 as the NA closes; the test asserts it is within 2e-3 at NA 0.02 and closer than at 0.04, and that "
                     "a lossless cylinder absorbs nothing.")


@figure("focused-cylinder-near-field", "A coated cylinder (0.6/1.2 µm, axis along y) under a focused beam (NA 0.6/0.2, x polarized, focus off-centre), 3 µm: |E|² across it against plane-wave fields", "x [µm]", "|E|²")
def focused_cylinder_near_field():
    from focused_checks import cylinder_near_field_pair

    x = np.linspace(-2.5, 2.5, 101)
    field, E = cylinder_near_field_pair(np.column_stack([x, np.full_like(x, 0.2), np.full_like(x, 0.25)]))
    ours = np.sum(np.abs(np.asarray(field.e[0])) ** 2, -1)
    ref = np.sum(np.abs(E) ** 2, -1)
    return dict(x=sig(x), series=[series("e2", "|E|², PyStratify", ours, "pystratify"), series("e2_ref", "|E|², superposition of plane-wave cylinder fields", ref, "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Along y = 0.2, z = 0.25 µm, across the cylinder (core 0.6, shell 1.2 µm; n = 1.5 + 0.1i, 1.3 + 0.02i): PyStratify's field "
                     "from its row coefficients against the pupil's plane waves, each solved as an oblique cylinder and its field summed "
                     "(focused_checks.py). Focus at (0.3, 0.2, 0.25) µm.")


@figure("scalar-cylinder-coefficients", "The scalar model: a cylinder of radius 1.1 µm, n = 1.45 + 0.06i, row s_y = 0.4, 3 µm: coefficients against the scalar boundary equations", "order m", "|t_m|", log=True)
def scalar_cylinder_coefficients():
    from focused_checks import scalar_cylinder_coefficient

    lam, R, m_in, sy = 3.0, 1.1, 1.45 + 0.06j, 0.4
    k = 2 * np.pi / lam
    n_v, n_p = np.sqrt(1 - sy * sy), np.sqrt(m_in ** 2 - sy * sy)
    sol = ps.solve_cylinder([R], [n_p, n_v], lam, beta=0, m_max=12)
    expected = [scalar_cylinder_coefficient(abs(m), k * R * n_v, k * R * n_p, n_v, n_p) for m in sol.orders]
    t = sol.t[:, 0, 0]
    return dict(x=[float(m) for m in sol.orders], series=[series("t", "|t_m|, PyStratify", np.abs(t), "pystratify"),
                                                          series("t_ref", "|t_m|, scalar boundary equations", np.abs(expected), "reference")],
                worst=worst(np.abs(t), np.abs(expected), relative=True), worst_label="largest relative difference",
                note="The scalar model treats a row of the pupil as a scalar wave U = J_m + t_m H_m outside a cylinder of effective "
                     "indices √(n² − s_y²), with U and ∂U/∂ρ continuous; the coefficients from PyStratify's cylinder solver against those "
                     "two conditions solved directly in SciPy's Bessel functions.")


@figure("scalar-cylinder-rows", "The scalar model, the coated cylinder (0.7/1.2 µm) under NA 0.6/0.2, 3 µm: each pupil row against a direct superposition", "pupil row s_y", "power per row")
def scalar_cylinder_rows():
    from focused_checks import scalar_row_pair

    sy = np.linspace(0.01, 0.59, 30)
    ours, ref = [], []
    for s in sy:
        value, (sca, ext) = scalar_row_pair(s)
        ours.append([value[0], value[1]]), ref.append([ext, sca])
    ours, ref = np.array(ours), np.array(ref)
    return dict(x=sig(sy), series=[series("ext", "extinction, PyStratify", ours[:, 0], "pystratify"), series("sca", "scattering, PyStratify", ours[:, 1], "pystratify"),
                                   series("ext_ref", "extinction, direct superposition", ref[:, 0], "reference"),
                                   series("sca_ref", "scattering, direct superposition", ref[:, 1], "reference")],
                worst=worst(ours, ref, relative=True), worst_label="largest relative difference",
                note="Each row's scalar plane waves, with the aplanatic pupil amplitude and the focus at (0.3, 0, 0.2) µm, scattered one by one "
                     "with the effective-index coefficients and superposed on observation angles; the powers integrated directly. The test "
                     "also checks that a lossless cylinder absorbs nothing and that the scalar model has no polarization split.")


@figure("davis-2011-fibres", "Non-absorbing fibres (R = 5 µm, n = 1.477; R = 10 µm, n = 1.478) under a Cassegrain (NA 0.1–0.4): transmission, Davis, Carney & Bhargava 2011, Fig. 5", "wavenumber [cm⁻¹]", "transmission [%]")
def davis_2011_fibres():
    curve, paper = classic_curve("davis_2011_fig5"), classic_values("davis_2011")
    wn, t5, t10 = curve["wavenumber_cm-1"], curve["T_percent_R5um_n1.477"], curve["T_percent_R10um_n1.478"]
    window = lambda lo, hi: (wn >= lo) & (wn <= hi)  # noqa: E731
    w1, w2 = window(1900, 2600), window(2700, 3300)
    ref5 = at(wn, {paper["R = 5 µm: wavenumber of the minimum (cm-1)"]: paper["R = 5 µm: transmission minimum (%)"],
                   float(wn[w1][np.argmax(t5[w1])]): paper["R = 5 µm: maximum near the fit range, 1900-2600 cm-1 (%)"],
                   float(wn[w2][np.argmin(t5[w2])]): paper["R = 5 µm: dip near 3000 cm-1 (%)"], 4000.0: paper["R = 5 µm: at 4000 cm-1 (%)"]})
    ref10 = at(wn, {800.0: paper["R = 10 µm: at 800 cm-1 (%)"]})
    values = json.loads((CLASSICS / "davis_2011.json").read_text())
    return dict(x=sig(wn), series=[series("t5", "R = 5 µm, PyStratify", t5, "pystratify"), series("t10", "R = 10 µm, PyStratify", t10, "pystratify"),
                                   series("t5_ref", "R = 5 µm, read off Fig. 5a", ref5, "reference"), series("t10_ref", "R = 10 µm, read off Fig. 5b", ref10, "reference")],
                worst=float(max(abs(c["computed"] - c["paper"]) for c in values if "(%)" in c["check"])), worst_label="largest difference from a read transmission [percentage points]",
                note="Davis, Carney & Bhargava, Anal. Chem. 83, 525 (2011): a fibre that absorbs nothing still shows a strong, wavenumber-"
                     "dependent apparent absorbance, made by refraction and scattering out of the objective. Values read off Fig. 5 to ±3%; "
                     "the paper prints no position for the maximum and the dip, so they are marked where PyStratify's curve turns, and the "
                     "minimum sits 20 cm⁻¹ from the read 1100. The paper does not state Fig. 5's NA; 0.1–0.4 is its Fig. 2 pair. "
                     "From crosscheck/classics/davis_2011.py.")


# --- CLI ---------------------------------------------------------------------------------------------------------

def build(names=None, env=None):
    OUT.mkdir(exist_ok=True)
    written = []
    for name, (fn, meta) in FIGURES.items():
        if (names and name not in names) or (env and ENVS[name] != env):
            continue
        t = time.time()
        try:
            if name in PAGE_RECORDS:
                record, spec = PAGE_RECORDS[name]
                out = record(*spec)
            else:
                data = fn()
        except ImportError as error:
            print(f"skip {name}: {error}", flush=True)
            continue
        if name not in PAGE_RECORDS:
            out = {**meta, **data, "script": SCRIPTS[name], "function": fn.__name__}
            out["x"] = [float(v) for v in out["x"]]
            for s in out["series"]:
                s["y"] = sig(s["y"], 8)
        out.update(EXTRA[name])
        (OUT / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
        written.append(name)
        print(f"{name}: {len(out['x'])} points, {len(out['series'])} series, worst {out['worst']:.2e} ({time.time() - t:.1f} s)", flush=True)
    return written


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--list"]:  # name and environment of every figure
        for name in FIGURES:
            print(name, ENVS[name])
    elif args[:1] == ["--env"]:  # every figure of one environment
        build(env=args[1])
    elif args[:1] == ["--meta"]:  # a page script's figures: title, labels, script and setup refreshed, nothing recomputed
        for name in args[1:] or [n for n in PAGE_RECORDS]:
            path = OUT / f"{name}.json"
            fn, meta = FIGURES[name]
            out = {**json.loads(path.read_text()), **meta, "script": SCRIPTS[name], "function": fn.__name__}
            out.pop("calculator", None)
            out.update(EXTRA[name])
            path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(f"{name}: meta refreshed", flush=True)
    else:
        build(args or None)
