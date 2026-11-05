"""Read docs/, compute the examples, write docs/manifest.json (schema in docs/README.md)."""

import copy
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SCHEMA = 1
GEOMETRIES = ("films", "cylinders", "spheres")
LEARN_GEOMETRIES = GEOMETRIES + ("common",)
CLASSIC_FIELDS = ("id", "kind", "position", "geometry", "title", "cite", "doi", "figure", "shows", "inputs",
                  "perturb", "script", "results", "calculator")
# A classic's reproduction figure (R-CLASSICS-CONTENT, Ilia 2026-11-05): its script writes the computed curves beside its
# checks, the build copies them to docs/classics/figures/<id>.json. At most this many numbers per classic (x and every y
# of every panel), about 100 kB of JSON inline on its Learn page: decimate the script's grid to plotting resolution.
FIGURE_BUDGET = 12_000
FIGURE_FOLDER = "classics/figures"
LEARN_FIELDS = ("id", "title", "geometry", "description", "summary", "references")
# Search engines show about 160 characters of a description; the index cards show it whole.
DESCRIPTION_LIMIT = 160
EXAMPLE_FIELDS = ("id", "geometry", "source", "title", "caption", "plot", "calculator")
AXES = ("wavelength", "angle")
# What an example can plot, per geometry and source (the website maps each onto the API's response).
SERIES = {
    ("spheres", "plane_wave"): ("q_ext", "q_sca", "q_abs", "md_sca", "ed_sca", "mq_sca", "eq_sca"),
    ("films", "plane_wave"): ("R", "T", "A", "R_s", "R_p", "T_s", "T_p"),
    # cylinders at normal incidence: unpolarized, E along the axis (_e; Bohren & Huffman's case I) and H along it (_m)
    ("cylinders", "plane_wave"): ("q_ext", "q_sca", "q_abs", "q_ext_e", "q_sca_e", "q_abs_e", "q_ext_m", "q_sca_m", "q_abs_m"),
}
# A sweep precomputes every slider position (the page reads the grid instead of calling the API): at most this
# many numbers per example, about half a megabyte of JSON. Coarsen a slider's step, or set `sweep: false`, to get under it.
SWEEP_BUDGET = 60_000
LAYER_SERIES = re.compile(r"^A_(\d+)$")  # films: the unpolarized absorptance of layer i (1-based, repeats expanded)
MAX_L = 1500  # the API's ceiling on the multipole order (multilayer backend/spheres.py)


class ContentError(ValueError):
    """docs/ holds a record the manifest cannot be built from."""


def _yaml():
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit('pystratify.docs needs PyYAML: pip install "pystratify[docs]"') from error
    return yaml


def load_yaml(path):
    return _yaml().safe_load(Path(path).read_text())


def _require(record, fields, where):
    missing = [f for f in fields if f not in record]
    if missing:
        raise ContentError(f"{where}: missing {', '.join(missing)}")


def _sig(value, digits=6):
    return float(f"{float(value):.{digits}g}")


# --- materials ------------------------------------------------------------------------------------

def _rii_data(path):
    """The DATA block of a refractiveindex.info file: ('tabulated nk', rows) or ('formula 1', coefficients)."""
    kind, rows, coeffs, inside = None, [], None, False
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("- type:"):
            kind = s.split(":", 1)[1].strip()
        elif s.startswith("coefficients:"):
            coeffs = [float(v) for v in s.split(":", 1)[1].split()]
        elif s.startswith("data: |"):
            inside = True
        elif inside:
            try:
                row = [float(p) for p in s.split()]
            except ValueError:
                row = []
            if len(row) == 3:
                rows.append(row)
            else:
                inside = False
    return kind, np.array(rows) if rows else coeffs


def material(path, wavelength_um):
    """n + ik of a refractiveindex.info page (tabulated n,k or Sellmeier formula 1) at the wavelengths (µm),
    interpolated linearly as the multilayer.app API does."""
    kind, data = _rii_data(path)
    lam = np.asarray(wavelength_um, dtype=float)
    if kind == "tabulated nk":
        if lam.min() < data[0, 0] - 1e-12 or lam.max() > data[-1, 0] + 1e-12:
            raise ContentError(f"{path} covers {data[0, 0]}-{data[-1, 0]} µm only")
        return np.interp(lam, data[:, 0], data[:, 1]) + 1j * np.interp(lam, data[:, 0], data[:, 2])
    if kind == "formula 1":
        c = data
        n2 = 1 + c[0] + sum(c[i] * lam**2 / (lam**2 - c[i + 1] ** 2) for i in range(1, len(c), 2))
        return np.sqrt(n2) + 0j
    raise ContentError(f"unsupported refractiveindex.info type {kind!r}")


def _drude_rows(lo, hi, count, lambda_p, gamma):
    rows = []
    for i in range(count):
        lam = lo + (hi - lo) * i / (count - 1)
        x = lambda_p / lam
        eps = 1 - 1 / complex(x * x, gamma * x)
        n = eps ** 0.5
        if n.imag < 0:
            n = -n
        rows.append((lam, n.real, n.imag))
    return rows


def _lorentz_rows(wn_lo, wn_hi, count, eps_inf, d_eps, w_to, gamma):
    rows = []
    for i in range(count):
        lam = 1e4 / wn_hi + (1e4 / wn_lo - 1e4 / wn_hi) * i / (count - 1)
        w = 1e4 / lam
        eps = eps_inf + d_eps * w_to**2 / complex(w_to**2 - w**2, -gamma * w)
        n = eps ** 0.5
        if n.imag < 0:
            n = -n
        rows.append((lam, n.real, n.imag))
    return rows


def _table(spec, where):
    """An n,k table as the calculator uploads one: {name, rows: [[λ µm, n, k], ...]} to six decimals."""
    if "rows" in spec:
        rows = spec["rows"]
    elif "file" in spec:
        kind, data = _rii_data(Path(spec["root"]) / spec["file"])
        if kind != "tabulated nk":
            raise ContentError(f"{where}: {spec['file']} is not a tabulated n,k page")
        rows = [tuple(r) for r in data.tolist()]
    elif "drude" in spec:
        d = spec["drude"]
        rows = _drude_rows(d["from_um"], d["to_um"], d["count"], d["lambda_p_um"], d["gamma"])
    elif "lorentz" in spec:
        d = spec["lorentz"]
        rows = _lorentz_rows(d["from_cm1"], d["to_cm1"], d["count"], d["eps_inf"], d["d_eps"], d["w_to_cm1"], d["gamma_cm1"])
    else:
        raise ContentError(f"{where}: a table needs rows, drude or lorentz")
    out = {"name": spec["name"], "rows": [[round(w, 6), round(n, 6), round(k, 6)] for w, n, k in rows]}
    if "file" in spec:
        out["file"] = spec["file"]
    return out


class Materials:
    """docs/materials.yaml: the refractiveindex.info pages the content cites, at the commit the calculator pins."""

    def __init__(self, root):
        self.root = root
        spec = load_yaml(root / "docs" / "materials.yaml")
        self.rii = spec["rii"]

    def resolve(self, medium, where):
        """The medium with `rii` expanded to the calculator's reference and `table` to rows."""
        out = dict(medium)
        _require(out, ("label", "n"), where)
        out.setdefault("k", 0.0)
        if "rii" in out:
            if out["rii"] not in self.rii:
                raise ContentError(f"{where}: {out['rii']!r} is not in docs/materials.yaml")
            out["rii"] = {"key": out["rii"], **self.rii[out["rii"]]}
        if "table" in out:
            # A worked example cites a refractiveindex.info page or a constant, never an uploaded table (Ilia,
            # 2026-10-16); a classic may still carry the paper's own data.
            if where.startswith("docs/examples/"):
                raise ContentError(f"{where}: {out['label']!r} is an uploaded n,k table; cite a page from "
                                   "docs/materials.yaml (`rii:`) or give a constant n, k")
            out["table"] = _table(dict(out["table"], root=str(self.root)), where)
        return out

    def index(self, medium, wavelength_um):
        """n + ik at the wavelengths, as the API resolves the medium: database page, uploaded table, or constant."""
        lam = np.asarray(wavelength_um, dtype=float)
        if "rii" in medium:
            return material(self.root / medium["rii"]["file"], lam)
        if "table" in medium:
            rows = np.asarray(medium["table"]["rows"], dtype=float)
            if lam.min() < rows[0, 0] - 1e-12 or lam.max() > rows[-1, 0] + 1e-12:
                # np.interp would hold the end value flat; the API refuses, and so does the build
                raise ContentError(f"{medium['table']['name']} covers {rows[0, 0]}-{rows[-1, 0]} µm only")
            return np.interp(lam, rows[:, 0], rows[:, 1]) + 1j * np.interp(lam, rows[:, 0], rows[:, 2])
        return np.full(lam.shape, complex(medium["n"], medium["k"]))


def _resolve_layers(layers, materials, where):
    out = []
    for i, layer in enumerate(layers):
        if "repeat" in layer:
            out.append({"repeat": layer["repeat"], "layers": _resolve_layers(layer["layers"], materials, f"{where} repeat")})
        else:
            out.append(materials.resolve(layer, f"{where} layer {i + 1}"))
    return out


def resolve_calculator(spec, materials, where):
    """A calculator block with every material expanded; the website turns it into a share token."""
    spec = copy.deepcopy(spec)
    kinds = [k for k in ("film", "sphere", "cylinder") if k in spec]
    if len(kinds) != 1:
        raise ContentError(f"{where}: the calculator block needs exactly one of film, sphere, cylinder")
    body = spec[kinds[0]]
    if kinds[0] == "film":
        body["incident"] = materials.resolve(body["incident"], f"{where} incident")
        body["substrate"] = materials.resolve(body["substrate"], f"{where} substrate")
        body["layers"] = _resolve_layers(body["layers"], materials, where)
    else:
        body["host"] = materials.resolve(body["host"], f"{where} host")
        body["shells"] = _resolve_layers(body["shells"], materials, where)
    return spec


# --- examples ------------------------------------------------------------------------------------

def _sphere_plane_wave(example, materials):
    """Spectra of a sphere under a plane wave, computed the way the API's /spheres/calculate computes them."""
    import pystratify as ps

    sphere = example["calculator"]["sphere"]
    changes = sphere.get("changes", {})
    if changes.get("spectralUnit", "nm") != "nm":
        raise ContentError(f"example {example['id']}: only nm spectral axes are computed so far")
    start, end, step = sphere["start"] / 1000, sphere["end"] / 1000, sphere["step"] / 1000
    count = int(np.floor((end - start) / step + 1e-9))
    lam = start + np.arange(count + 1) * step
    shells = sphere["shells"]
    if any("repeat" in s for s in shells):
        raise ContentError(f"example {example['id']}: sphere shells do not repeat")
    radii = np.cumsum([s["thickness"] for s in shells])
    ref = np.column_stack([materials.index(s, lam) for s in shells] + [materials.index(sphere["host"], lam)])
    # The API's default order (backend spheres.default_l, far field): Wiscombe on the host size parameter,
    # raised to cover the largest internal |n k0 r|.
    host = float(np.max(np.abs(ref[:, -1])))
    L = ps.truncation_order(radii[-1], host, float(np.min(lam)), "far")
    k0r = 2 * np.pi * np.max(np.abs(ref[:, :-1]) * radii[None, :] / lam[:, None])
    L = int(min(MAX_L, max(3, max(L, int(math.ceil(k0r + 4 * k0r ** (1 / 3) + 2))))))
    cs = ps.cross_sections(ps.solve(radii, ref, lam, l_max=L))
    g = cs.geometric
    curves = {
        "q_ext": cs.ext_by_order.sum(axis=(1, 2)) / g,
        "q_sca": cs.sca_by_order.sum(axis=(1, 2)) / g,
        "q_abs": (cs.ext_by_order - cs.sca_by_order).sum(axis=(1, 2)) / g,
        "md_sca": cs.sca_by_order[:, ps.TE, 0] / g,
        "ed_sca": cs.sca_by_order[:, ps.TM, 0] / g,
        "mq_sca": cs.sca_by_order[:, ps.TE, 1] / g if L > 1 else np.zeros_like(lam),
        "eq_sca": cs.sca_by_order[:, ps.TM, 1] / g if L > 1 else np.zeros_like(lam),
    }
    return lam * 1000, curves


def _expand(layers):
    out = []
    for layer in layers:
        out.extend(_expand(layer["layers"]) * layer["repeat"] if "repeat" in layer else [layer])
    return out


WAVELENGTH_UNIT = {"nm": 1e-3, "um": 1.0}  # the calculator's film units: controls in nm or µm
API_MAX_ANGLE_POINTS = 181  # multilayer backend MAX_ANGLE_POINTS: an example the live page cannot recompute is refused


def _film_unit(example):
    unit = example["calculator"]["film"].get("units", {}).get("wavelength", "nm")
    if unit not in WAVELENGTH_UNIT:
        raise ContentError(f"example {example['id']}: wavelengths in {unit!r} are not computed (nm or um)")
    return WAVELENGTH_UNIT[unit]


def _film_axes(example):
    """(wavelengths µm, angles deg) of a film example, from its calculator controls (nm or µm)."""
    film = example["calculator"]["film"]
    to_um = _film_unit(example)
    c = film.get("controls", {})
    if example.get("axis", "wavelength") == "angle":
        lam = np.array([c["wavelength-start"] * to_um])
        count = int(np.floor((c["angle-end"] - c["angle-start"]) / c["angle-step"] + 1e-9))
        if count + 1 > API_MAX_ANGLE_POINTS:
            raise ContentError(f"example {example['id']}: {count + 1} angles; the API computes at most {API_MAX_ANGLE_POINTS}")
        return lam, c["angle-start"] + np.arange(count + 1) * c["angle-step"]
    start, end, step = c["wavelength-start"] * to_um, c["wavelength-end"] * to_um, c["wavelength-step"] * to_um
    count = int(np.floor((end - start) / step + 1e-9))
    return start + np.arange(count + 1) * step, np.array([float(c.get("angle-deg", 0))])


def _film_plane_wave(example, materials):
    """R and T of a film stack, computed with the planar engine the API's /calculate runs (pystratify.planar)."""
    from .. import planar

    film = example["calculator"]["film"]
    lam, angles = _film_axes(example)
    layers = _expand(film["layers"])
    media = [film["incident"]] + layers + [film["substrate"]]
    n = np.column_stack([materials.index(m, lam) for m in media])
    d = [np.inf] + [float(layer["thickness"]) for layer in layers] + [np.inf]
    c = ["i"] + [layer.get("coherency", "c") for layer in layers] + ["i"]
    incoherent = "i" in c[1:-1]
    out = {}
    for pol in ("s", "p"):
        R = np.empty((angles.size, lam.size))
        T = np.empty_like(R)
        layer_a = np.empty((len(layers), angles.size, lam.size))
        for a, angle in enumerate(np.radians(angles)):
            for w, wavelength in enumerate(lam):
                if incoherent:
                    r = planar.inc_tmm(pol, list(n[w]), d, c, angle, wavelength)
                    layer_a[:, a, w] = planar.inc_absorp_in_each_layer(r)[1:-1]
                else:
                    r = planar.coh_tmm(pol, list(n[w]), d, angle, wavelength)
                    layer_a[:, a, w] = planar.absorp_in_each_layer(r)[1:-1]
                R[a, w], T[a, w] = r["R"], r["T"]
        out[pol] = (R.ravel(), T.ravel(), layer_a.reshape(len(layers), angles.size * lam.size))
    curves = {"R_s": out["s"][0], "R_p": out["p"][0], "T_s": out["s"][1], "T_p": out["p"][1]}
    curves["R"] = 0.5 * (curves["R_s"] + curves["R_p"])
    curves["T"] = 0.5 * (curves["T_s"] + curves["T_p"])
    curves["A"] = np.clip(1 - curves["R"] - curves["T"], 0, None)
    for i in range(len(layers)):
        curves[f"A_{i + 1}"] = 0.5 * (out["s"][2][i] + out["p"][2][i])
    x = angles if example.get("axis", "wavelength") == "angle" else lam / _film_unit(example)
    return np.asarray(x, dtype=float), curves


def _medium_code(medium, where):
    if "rii" in medium:
        return f"material({medium['rii']['file']!r}, lam)"
    if "table" in medium:
        raise ContentError(f"{where}: examples with uploaded n,k tables have no script form yet")
    return f"np.full(lam.shape, {complex(medium['n'], medium['k'])!r})"


def _sphere_plane_wave_code(example):
    """The example as a PyStratify script, run from the repository root (database pages are paths in it)."""
    sphere = example["calculator"]["sphere"]
    shells = sphere["shells"]
    radii = ", ".join(repr(round(float(r), 9)) for r in np.cumsum([s["thickness"] for s in shells]))
    columns = ", ".join(_medium_code(m, f"example {example['id']}") for m in shells + [sphere["host"]])
    count = int(np.floor((sphere["end"] - sphere["start"]) / sphere["step"] + 1e-9)) + 1
    lines = [
        "import numpy as np",
        "import pystratify as ps",
        "from pystratify.docs import material  # n + ik of a refractiveindex.info page",
        "",
        f"lam = ({sphere['start']} + {sphere['step']} * np.arange({count})) * 1e-3  # wavelength, µm",
        f"N = np.column_stack([{columns}])  # n + ik: core{', shells' if len(shells) > 1 else ''}, then the host",
        f"cs = ps.cross_sections(ps.solve([{radii}], N, lam))  # outer radii, µm",
        "q_ext, q_sca = cs.q_ext, cs.q_sca",
        "b1 = cs.sca_by_order[:, ps.TE, 0] / cs.geometric  # magnetic dipole part of Q_sca",
    ]
    return "\n".join(lines) + "\n"


def _film_plane_wave_code(example):
    """The example as a PyStratify script, run from the repository root (database pages are paths in it)."""
    film = example["calculator"]["film"]
    c = film.get("controls", {})
    layers = _expand(film["layers"])
    media = [film["incident"]] + layers + [film["substrate"]]
    if any("table" in m and "file" not in m["table"] for m in media):
        raise ContentError(f"example {example['id']}: examples with inline n,k tables have no script form yet")

    def index(m):
        if "rii" in m:
            return f"material({m['rii']['file']!r}, wl)"
        if "table" in m:
            return f"material({m['table']['file']!r}, wl)"
        return repr(complex(m["n"], m["k"])) if m["k"] else repr(m["n"])

    n = "[" + ", ".join(index(m) for m in media) + "]"
    d = "[" + ", ".join(["np.inf"] + [repr(layer["thickness"]) for layer in layers] + ["np.inf"]) + "]"
    coherent = all(layer.get("coherency", "c") == "c" for layer in layers)
    coherency = ["i"] + [layer.get("coherency", "c") for layer in layers] + ["i"]
    tmm = "planar.coh_tmm(pol, n(wl), d, th, wl)" if coherent else f"planar.inc_tmm(pol, n(wl), d, {coherency!r}, th, wl)"
    lam, angles = _film_axes(example)
    lines = ["import numpy as np", "from pystratify import planar"]
    if any("rii" in m or "table" in m for m in media):
        lines.append("from pystratify.docs import material  # n + ik of a tabulated page (refractiveindex.info format)")
    lines += ["", f"n = lambda wl: {n}  # incident medium, layers, substrate", f"d = {d}  # thicknesses, µm"]
    if example.get("axis", "wavelength") == "angle":
        scale = "" if _film_unit(example) == 1.0 else " * 1e-3"
        lines += [f"wl = {c['wavelength-start']}{scale}  # wavelength, µm",
                  f"angles = {c['angle-start']} + {c['angle-step']} * np.arange({angles.size})  # degrees",
                  "R = {pol: np.array([" + tmm.replace(", th,", ", np.radians(a),") + "['R'] for a in angles]) for pol in ('s', 'p')}"]
    else:
        scale = "" if _film_unit(example) == 1.0 else " * 1e-3"
        lines += [f"lam = ({c['wavelength-start']} + {c['wavelength-step']} * np.arange({lam.size})){scale}  # wavelength, µm",
                  f"th = np.radians({c.get('angle-deg', 0)})",
                  "R, T = {}, {}",
                  "for pol in ('s', 'p'):",
                  f"    rows = [{tmm} for wl in lam]",
                  "    R[pol], T[pol] = np.array([r['R'] for r in rows]), np.array([r['T'] for r in rows])",
                  "R_avg, T_avg = (R['s'] + R['p']) / 2, (T['s'] + T['p']) / 2  # unpolarized"]
    return "\n".join(lines) + "\n"


def _cylinder_axis(example):
    cylinder = example["calculator"]["cylinder"]
    for key in ("start", "end", "step"):
        if key not in cylinder:
            raise ContentError(f"example {example['id']}: a cylinder example sets start, end and step (nm) on its cylinder block")
    start, end, step = cylinder["start"] / 1000, cylinder["end"] / 1000, cylinder["step"] / 1000
    count = int(np.floor((end - start) / step + 1e-9))
    return start + np.arange(count + 1) * step


def _cylinder_plane_wave(example, materials):
    """Efficiencies of an infinite cylinder under a plane wave at normal incidence (pystratify.solve_cylinder, the
    engine the API's cylinder jobs run), unpolarized and for each polarization."""
    import pystratify as ps

    cylinder = example["calculator"]["cylinder"]
    lam = _cylinder_axis(example)
    shells = cylinder["shells"]
    if any("repeat" in s for s in shells):
        raise ContentError(f"example {example['id']}: cylinder shells do not repeat")
    radii = np.cumsum([s["thickness"] for s in shells])
    ref = np.column_stack([materials.index(s, lam) for s in shells] + [materials.index(cylinder["host"], lam)])
    curves = {k: np.empty(lam.size) for k in SERIES[("cylinders", "plane_wave")]}
    for w, wavelength in enumerate(lam):
        sol = ps.solve_cylinder(radii, ref[w], wavelength)
        for suffix, polarization in (("", "unpolarized"), ("_e", "axial-electric"), ("_m", "axial-magnetic")):
            widths = ps.cross_widths(sol, polarization)
            curves["q_ext" + suffix][w] = widths["efficiency_extinction"]
            curves["q_sca" + suffix][w] = widths["efficiency_scattering"]
            curves["q_abs" + suffix][w] = widths["efficiency_absorption"]
    return lam * 1000, curves


def _cylinder_plane_wave_code(example):
    """The example as a PyStratify script, run from the repository root (database pages are paths in it)."""
    cylinder = example["calculator"]["cylinder"]
    shells = cylinder["shells"]
    radii = ", ".join(repr(round(float(r), 9)) for r in np.cumsum([s["thickness"] for s in shells]))
    columns = ", ".join(_medium_code(m, f"example {example['id']}") for m in shells + [cylinder["host"]])
    count = _cylinder_axis(example).size
    lines = [
        "import numpy as np",
        "import pystratify as ps",
        "from pystratify.docs import material  # n + ik of a refractiveindex.info page",
        "",
        f"lam = ({cylinder['start']} + {cylinder['step']} * np.arange({count})) * 1e-3  # wavelength, µm",
        f"N = np.column_stack([{columns}])  # n + ik: core{', shells' if len(shells) > 1 else ''}, then the host",
        f"radii = [{radii}]  # outer radii, µm",
        "q_ext = {pol: np.array([ps.cross_widths(ps.solve_cylinder(radii, N[i], wl), pol)['efficiency_extinction']",
        "                        for i, wl in enumerate(lam)]) for pol in ('unpolarized', 'axial-electric', 'axial-magnetic')}",
    ]
    return "\n".join(lines) + "\n"


COMPUTE = {("spheres", "plane_wave"): _sphere_plane_wave, ("films", "plane_wave"): _film_plane_wave,
           ("cylinders", "plane_wave"): _cylinder_plane_wave}
CODE = {("spheres", "plane_wave"): _sphere_plane_wave_code, ("films", "plane_wave"): _film_plane_wave_code,
        ("cylinders", "plane_wave"): _cylinder_plane_wave_code}
# Examples the live page can also recompute on the API (films and spheres answer at once; a cylinder is a job of
# minutes on the optics service, so cylinder examples are precomputed only).
LIVE = {("spheres", "plane_wave"), ("films", "plane_wave")}


def _measure(measure, x, curves, where):
    """peak / minimum: x of the series' extremum (optionally within [lo, hi]); at: the series' value at x;
    crosses: the first x where the series reaches a level."""
    if "crosses" in measure:
        series, level = measure["crosses"]
        hits = np.flatnonzero(np.asarray(curves[series]) >= level)
        if not hits.size:
            raise ContentError(f"{where}: {series} never reaches {level}")
        return float(x[hits[0]])
    if "max" in measure:
        return float(np.max(curves[measure["max"]]))
    if "product" in measure:
        return float(np.prod([_measure({"at": pair}, x, curves, where) for pair in measure["product"]]))
    if "mean" in measure:
        series, (lo, hi) = measure["mean"]
        m = (x >= lo - 1e-9) & (x <= hi + 1e-9)
        return float(np.mean(np.asarray(curves[series])[m]))
    if "balance" in measure:
        layers = sorted((k for k in curves if LAYER_SERIES.match(k)), key=lambda k: int(k[2:]))
        total = sum(np.asarray(curves[k]) for k in layers)
        return float(np.max(np.abs(1 - np.asarray(curves["R"]) - np.asarray(curves["T"]) - total)))
    if "at" in measure:
        series, at = measure["at"]
        i = int(np.argmin(np.abs(x - at)))
        if abs(x[i] - at) > 1e-9 * max(1.0, abs(at)):
            raise ContentError(f"{where}: {at} is not on the example's grid")
        return float(curves[series][i])
    kind = "peak" if "peak" in measure else "minimum" if "minimum" in measure else None
    if kind is None:
        raise ContentError(f"{where}: unknown measure {measure!r}")
    y = curves[measure[kind]]
    m = np.ones_like(x, dtype=bool)
    lo, hi = measure.get("within", (None, None))
    if lo is not None:
        m &= x >= lo
    if hi is not None:
        m &= x <= hi
    pick = np.argmax if kind == "peak" else np.argmin
    return float(x[np.flatnonzero(m)[pick(y[m])]])


def _get(block, path, where):
    node = block
    for key in path.split("."):
        node = node[int(key)] if isinstance(node, list) else node.get(key) if isinstance(node, dict) else None
        if node is None:
            raise ContentError(f"{where}: no {path!r} in the calculator block")
    return node


def build_example(example, materials):
    where = f"docs/examples/{example['id']}.yaml"
    _require(example, EXAMPLE_FIELDS, where)
    if example["geometry"] not in GEOMETRIES:
        raise ContentError(f"{where}: geometry must be one of {GEOMETRIES}")
    compute = COMPUTE.get((example["geometry"], example["source"]))
    if compute is None:
        raise ContentError(f"{where}: {example['geometry']} / {example['source']} examples are not computed yet")
    out = dict(example)
    out["calculator"] = resolve_calculator(example["calculator"], materials, where)
    keys = [s["key"] for s in example["plot"]["series"]]
    known = SERIES[(example["geometry"], example["source"])]
    unknown = [k for k in keys if k not in known and not (example["geometry"] == "films" and LAYER_SERIES.match(k))]
    if unknown:
        raise ContentError(f"{where}: unknown series {unknown}; known: {known}")
    for p in example.get("params", []):
        _require(p, ("id", "label", "unit", "target", "scale", "min", "max", "step", "value"), f"{where} param")
        for target in ([p["target"]] if isinstance(p["target"], str) else p["target"]):
            current = _get(out["calculator"], target, where)
            if abs(current - p["value"] * p["scale"]) > 1e-9 * max(1.0, abs(current)):
                raise ContentError(f"{where}: param {p['id']} = {p['value']} × {p['scale']} but {target} holds {current}")
            if target.endswith((".n", ".k")) and {"rii", "table"} & set(_get(out["calculator"], target.rsplit(".", 1)[0], where)):
                raise ContentError(f"{where}: param {p['id']} sets an index the medium takes from a database or table")
    x, curves = compute(out, materials)
    out["data"] = {"x": [_sig(v) for v in x], "series": {k: [_sig(v) for v in curves[k]] for k in keys}}
    out["live"] = (example["geometry"], example["source"]) in LIVE
    out["sweep"], sweep_data = _sweep(out, x, keys, compute, materials, where)
    if sweep_data is not None:
        out["_sweep_data"] = sweep_data
    readouts = []
    for r in example.get("readouts", []):
        _require(r, ("label", "measure"), f"{where} readout")
        values = r["measure"].values()
        for key in [v for v in values if isinstance(v, str)] + [v[0] for v in values if isinstance(v, list) and isinstance(v[0], str)]:
            if key not in keys:
                raise ContentError(f"{where}: readout {r['label']!r} reads {key}, which the plot does not carry")
        readouts.append(dict(r, value=_sig(_measure(r["measure"], x, curves, where))))
    out["readouts"] = readouts
    checks = []
    for c in example.get("checks", []):
        _require(c, ("check", "reference", "tolerance", "source", "measure"), f"{where} check")
        computed = _measure(c["measure"], x, curves, where)
        checks.append({"check": c["check"], "reference": c["reference"], "computed": _sig(computed),
                       "tolerance": c["tolerance"], "pass": bool(abs(computed - c["reference"]) <= c["tolerance"]),
                       "source": c["source"]})
    out["checks"] = checks
    out["code"] = CODE[(example["geometry"], example["source"])](out)
    return out


def _grid(param, where):
    """A slider's positions, min to max in steps (the HTML range input's positions), as the slider prints them."""
    count = int(round((param["max"] - param["min"]) / param["step"]))
    if count < 1 or abs(param["min"] + count * param["step"] - param["max"]) > 1e-9 * max(1.0, abs(param["max"])):
        raise ContentError(f"{where}: param {param['id']} runs {param['min']}..{param['max']} in steps of {param['step']}, "
                           "which does not land on max")
    decimals = max(0, -int(np.floor(np.log10(param["step"]))) + 2)
    return [float(round(param["min"] + i * param["step"], decimals)) for i in range(count + 1)]


def _set_paths(block, paths, value, where):
    for path in paths:
        head, _, key = path.rpartition(".")
        node = _get(block, head, where)
        node[int(key) if isinstance(node, list) else key] = value


def _sweep(out, x, keys, compute, materials, where):
    """Every slider position computed in advance: (the manifest's sweep block, the grid's data) or (None, None).

    The page reads a slider move from this grid, so a reader never waits for the API and the site needs no physics at
    run time. The grid is the product of the sliders' positions; its data goes to docs/sweeps/<id>.json, and the
    manifest carries the axes and the file's SHA-256 (the site cache-busts with it). `sweep: false` on an example
    leaves it to the API (a film or sphere example only)."""
    params = out.get("params", [])
    if not params or out.get("sweep") is False:
        if not params or out["live"]:
            return None, None
        raise ContentError(f"{where}: a {out['geometry']} example cannot recompute on the API; drop `sweep: false`")
    axes = [{"id": p["id"], "values": _grid(p, where)} for p in params]
    shape = [len(a["values"]) for a in axes]
    total = int(np.prod(shape)) * len(x) * len(keys)
    if total > SWEEP_BUDGET:
        raise ContentError(f"{where}: the sweep holds {total} numbers, over the budget of {SWEEP_BUDGET}; "
                           "coarsen a slider's step, narrow its range, or plot fewer series")
    series = {k: [] for k in keys}
    for index in np.ndindex(*shape):
        setup = copy.deepcopy(out["calculator"])
        for p, a, i in zip(params, axes, index):
            _set_paths(setup, [p["target"]] if isinstance(p["target"], str) else p["target"], a["values"][i] * p["scale"], where)
        x_i, curves = compute(dict(out, calculator=setup), materials)
        if x_i.shape != x.shape or np.max(np.abs(x_i - x)) > 1e-9:
            raise ContentError(f"{where}: the sweep moved the x grid (a slider may not set the axis)")
        for k in keys:
            series[k].append([_sig(v) for v in curves[k]])
    data = {"id": out["id"], "axes": axes, "shape": shape, "x": out["data"]["x"], "series": series}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    digest = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
    meta = {"axes": axes, "shape": shape, "points": int(np.prod(shape)), "file": f"docs/sweeps/{out['id']}.json", "sha256": digest}
    return meta, text


# --- records -------------------------------------------------------------------------------------

def _front_matter(text, where):
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not match:
        raise ContentError(f"{where}: a Learn section starts with a --- front matter block")
    return _yaml().safe_load(match.group(1)), match.group(2).lstrip("\n")


def _ordered_name(path):
    match = re.match(r"^(\d+)-(.+)$", path.stem if path.suffix else path.name)
    if not match:
        raise ContentError(f"{path}: Learn folders and files are named NN-slug")
    return int(match.group(1)), match.group(2)


def _review_date(value, where):
    """Ilia's review of a section: empty until he has read it, then the date (YAML may hand back a date)."""
    if value in (None, ""):
        return None
    text = str(value)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ContentError(f"{where}: reviewed is a date, YYYY-MM-DD, or empty")
    return text


REVIEW_LINES = re.compile(r"^reviewed(?:_fingerprint)?:.*(?:\n|$)", re.M)


def review_fingerprint(root, path):
    """What a review covers (Ilia, 2026-10-31): 16 hex digits of SHA-256 over the section's Markdown with its
    `reviewed` and `reviewed_fingerprint` lines taken out, then the records of the examples and classics it places.
    Any edit to them, a typo included, changes it, and the page stops saying it was reviewed."""
    text = path.read_text()
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not match:
        raise ContentError(f"{path}: a Learn section starts with a --- front matter block")
    meta = _yaml().safe_load(match.group(1))
    digest = __import__("hashlib").sha256()
    digest.update(f"---\n{REVIEW_LINES.sub('', match.group(1))}\n---\n{match.group(2)}".encode("utf-8"))
    for kind, folder in (("examples", "examples"), ("classics", "classics")):
        for record in meta.get(kind, []):
            source = root / "docs" / folder / f"{record}.yaml"
            digest.update(f"\0{folder}/{record}\0".encode("utf-8") + (source.read_bytes() if source.exists() else b""))
    return digest.hexdigest()[:16]


def load_learn(root):
    sections = []
    base = root / "docs" / "learn"
    for path in sorted(base.glob("*/*/*.md")):
        where = str(path.relative_to(root))
        meta, body = _front_matter(path.read_text(), where)
        _require(meta, LEARN_FIELDS, where)
        geometry = path.parts[-3]
        if geometry not in LEARN_GEOMETRIES or meta["geometry"] != geometry:
            raise ContentError(f"{where}: lives under learn/{geometry}/ but says geometry {meta['geometry']!r}")
        chapter_order, chapter_slug = _ordered_name(path.parent)
        order, slug = _ordered_name(path)
        sections.append({
            "id": meta["id"], "kind": "learn", "geometry": geometry, "title": meta["title"], "summary": meta["summary"],
            "chapter": meta.get("chapter", chapter_slug.replace("-", " ").capitalize()), "chapter_order": chapter_order,
            "order": order, "slug": slug, "url": f"learn/{geometry}/{slug}.html",
            "topic": meta.get("topic", chapter_slug), "sources": meta.get("sources", []),
            "examples": meta.get("examples", []), "classics": meta.get("classics", []), "references": meta["references"],
            "status": meta.get("status", "published"), "body": body,
            "description": " ".join(str(meta["description"]).split()), "reviewed": _review_date(meta.get("reviewed"), where),
        })
        stamp = meta.get("reviewed_fingerprint")
        if stamp and not sections[-1]["reviewed"]:
            raise ContentError(f"{where}: reviewed_fingerprint without a reviewed date")
        # The review holds while the page reads as it did; an edit since, or a date written by hand without
        # `python -m pystratify.docs review`, leaves the date but not the claim.
        sections[-1]["review_current"] = bool(stamp) and str(stamp) == review_fingerprint(root, path)
        if len(sections[-1]["description"]) > DESCRIPTION_LIMIT:
            raise ContentError(f"{where}: description is {len(sections[-1]['description'])} characters, at most {DESCRIPTION_LIMIT}")
        for kind, listed in (("example", sections[-1]["examples"]), ("classic", sections[-1]["classics"])):
            placed = re.findall(rf"^::: {kind} (\S+)\s*$", body, flags=re.M)
            if sorted(placed) != sorted(listed):
                raise ContentError(f"{where}: front matter lists {kind}s {listed} but the text places {placed}")
        if sections[-1]["status"] not in ("draft", "published"):
            raise ContentError(f"{where}: status is draft or published")
    sections.sort(key=lambda s: (LEARN_GEOMETRIES.index(s["geometry"]), s["chapter_order"], s["order"]))
    for geometry in LEARN_GEOMETRIES:
        chain = [s for s in sections if s["geometry"] == geometry]
        for i, s in enumerate(chain):
            s["prev"] = chain[i - 1]["id"] if i else None
            s["next"] = chain[i + 1]["id"] if i + 1 < len(chain) else None
    return sections


def _run_classics(root, scripts):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg", PYTHONPATH=str(root / "src"))
    for script in scripts:
        subprocess.run([sys.executable, Path(script).name], cwd=root / Path(script).parent, env=env, check=True,
                       stdout=subprocess.DEVNULL)


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _digits(value):
    """The significant digits of a number as the paper prints it: 466 -> 3, 0.38 -> 2, 1.0 -> 2, 0.0 -> 1."""
    mantissa = repr(float(value) if isinstance(value, float) else value).lower().split("e")[0]
    return max(1, len(mantissa.lstrip("-").replace(".", "").lstrip("0")))


def shown(value, digits=None):
    """A number as a classic's card prints it: as written when `digits` is None (a paper value, a tolerance), else to
    that many significant digits; a minus sign, and a power of ten outside 1e-3 to 1e5."""
    value = float(value) if digits is not None or isinstance(value, float) else value
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        mantissa, exponent = f"{value:.{max(0, (digits or _digits(value)) - 1)}e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".") if "." in mantissa else mantissa
        text = ("" if mantissa in ("1", "-1") else mantissa + "×") + "10" + str(int(exponent)).translate(SUPERSCRIPT)
        text = ("-" + text if mantissa == "-1" else text)
    else:
        text = f"{value:.{digits}g}" if digits is not None else repr(value)
    return text.replace("-", "−")


def computed_shown(check):
    """A check's computed value: one significant digit more than its paper value, at least three, at most six."""
    return shown(check["computed"], min(6, max(3, _digits(check["paper"]) + 1)))


def agreement(checks):
    """The agreement sentence of a classic, generated from its checks (R-CLASSICS-CONTENT, Ilia 2026-11-05): how many
    agree within their tolerances, every one that does not, and how many are shown without a tolerance. Its numbers
    cannot drift from the checks; the record's `note` frames it in prose, its numbers placeholders (`expand_note`)."""
    graded = [c for c in checks if c["tolerance"] is not None]
    passed = [c for c in graded if c["pass"]]
    failed = [c for c in graded if c["pass"] is False]
    free = len(checks) - len(graded)
    within = " with a tolerance agree within it" if free else " agree within their tolerances"
    if not graded:
        sentence = "No check carries a tolerance"
    elif not failed:
        sentence = (("The one check with a tolerance agrees within it" if free else "The one check agrees within its tolerance")
                    if len(graded) == 1 else f"All {len(graded)} checks{within}")
    else:
        sentence = (f"{len(passed)} of {len(graded)} checks{within}; "
                    + "; ".join(f"“{c['check']}” does not: {computed_shown(c)} against {shown(c['paper'])} ± "
                                f"{shown(c['tolerance'])}" for c in failed))
    if free:
        sentence += f"; {free} more {'is' if free == 1 else 'are'} shown without one"
    return sentence + "."


# A placeholder in a classic's note: {<check>} is its computed value as the card prints it, {<check>|paper} the
# paper's value, {<check>|tolerance} its tolerance -- the check named exactly as its script names it.
PLACEHOLDER = re.compile(r"\{([^{}|]+)(?:\|(paper|computed|tolerance))?\}")


def expand_note(text, checks, where):
    """A classic's note with every placeholder filled from its checks; prose in the note carries no number of its own
    that a check holds, so the card's numbers all come from the checks."""
    def fill(match):
        name, field = " ".join(match.group(1).split()), match.group(2) or "computed"
        found = [c for c in checks if c["check"] == name]
        if len(found) != 1:
            raise ContentError(f"{where}: the note names {name!r}, which is not exactly one of the checks")
        return computed_shown(found[0]) if field == "computed" else shown(found[0][field])

    out = PLACEHOLDER.sub(fill, " ".join(str(text or "").split()))
    if "{" in out or "}" in out:
        raise ContentError(f"{where}: an unmatched brace in the note")
    return out


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContentError(f"{where}: {value!r} is not a finite number")
    return value


def resolve_figure(figure, checks, where):
    """A classic's reproduction figure, checked against its checks: every panel's curves on its x grid, and every mark
    placed at its check's paper value -- a point (`at`: x), a level over a range (`at`: [lo, hi]) or, with no `at`,
    a vertical rule at that value (a peak's or a crossing's position); `log10` when the check is the base-10 log of
    the plotted quantity. The mark's coordinates are recomputed from the check on every build, so a figure cannot show
    a paper value its checks do not hold."""
    by_name = {}
    for c in checks:
        by_name.setdefault(c["check"], []).append(c)
    panels = figure.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ContentError(f"{where}: a reproduction figure needs panels")
    out, ids, numbers = [], set(), 0
    for p in panels:
        here = f"{where}: panel {p.get('id')!r}"
        _require(p, ("id", "title", "x_label", "y_label", "x", "series"), here)
        if p["id"] in ids:
            raise ContentError(f"{here}: duplicate panel id")
        ids.add(p["id"])
        x = [_number(v, here) for v in p["x"]]
        steps = np.diff(x)
        if len(x) < 2 or not (np.all(steps > 0) or np.all(steps < 0)):
            raise ContentError(f"{here}: x must run monotonically over at least two points")
        lo, hi = min(x), max(x)
        keys = [s["key"] for s in p["series"]]
        if not keys or len(set(keys)) != len(keys):
            raise ContentError(f"{here}: series keys must be present and distinct")
        for s in p["series"]:
            _require(s, ("key", "label", "y"), f"{here}, series {s.get('key')!r}")
            if len(s["y"]) != len(x) or any(v is not None and _number(v, here) is None for v in s["y"]):
                raise ContentError(f"{here}, series {s['key']!r}: y must hold one number (or null) per x")
        numbers += len(x) * (1 + len(keys))
        marks = []
        for m in p.get("marks", []):
            found = by_name.get(m.get("check"), [])
            if len(found) != 1:
                raise ContentError(f"{here}: a mark names {m.get('check')!r}, which is not exactly one of the checks")
            check = found[0]
            if m.get("series") is not None and m["series"] not in keys:
                raise ContentError(f"{here}: a mark of {check['check']!r} names no series of the panel ({m['series']!r})")
            value = _number(check["paper"], f"{here}, mark {check['check']!r}")
            value = 10 ** value if m.get("log10") else value
            at = m.get("at")
            mark = {k: m[k] for k in ("check", "series", "label", "at", "log10") if m.get(k) is not None}
            if at is None:
                mark.update(kind="rule", x=value)
                xs = [value]
            elif isinstance(at, list) and len(at) == 2:
                mark.update(kind="level", x=_number(at[0], here), x2=_number(at[1], here), y=value)
                xs = at
            else:
                mark.update(kind="point", x=_number(at, here), y=value)
                xs = [at]
            if any(not (lo - 1e-9 * (hi - lo) <= v <= hi + 1e-9 * (hi - lo)) for v in xs):
                raise ContentError(f"{here}: the mark of {check['check']!r} lies outside x = {lo:g}..{hi:g}")
            if p.get("log") and mark.get("y", 1) <= 0:
                raise ContentError(f"{here}: the mark of {check['check']!r} is not positive on a log axis")
            mark["pass"] = check["pass"]
            marks.append(mark)
        out.append({"id": p["id"], "title": p["title"], "x_label": p["x_label"], "y_label": p["y_label"],
                    "log": bool(p.get("log", False)), "x": x,
                    "series": [{k: s[k] for k in ("key", "label", "quantity", "dash", "y") if k in s} for s in p["series"]],
                    "marks": marks})
    if numbers > FIGURE_BUDGET:
        raise ContentError(f"{where}: {numbers} numbers, at most {FIGURE_BUDGET}: decimate the curves to plotting resolution")
    default = figure.get("default", 0)
    if not isinstance(default, int) or not 0 <= default < len(out):
        raise ContentError(f"{where}: `default` must index a panel")
    return {"title": figure.get("title", ""), "note": figure.get("note", ""), "script": figure.get("script", ""),
            "default": default, "panels": out}


def figure_text(figure):
    return json.dumps(figure, ensure_ascii=False, separators=(",", ":")) + "\n"


def load_classics(root, materials, run_classics=False, previous=None):
    paths = sorted((root / "docs" / "classics").glob("*.yaml"))
    records = [(p, load_yaml(p)) for p in paths]
    if run_classics:
        _run_classics(root, [r["script"] for _, r in records])
    out = []
    for path, record in records:
        where = str(path.relative_to(root))
        _require(record, CLASSIC_FIELDS, where)
        if record["id"] != path.stem or record["geometry"] not in GEOMETRIES:
            raise ContentError(f"{where}: the id must be the file name and the geometry one of {GEOMETRIES}")
        if "agreement" in record:
            raise ContentError(f"{where}: the agreement sentence is generated from the checks (R-CLASSICS-CONTENT); "
                               "write any framing prose, without the checks' numbers, as `note`")
        if not (root / record["script"]).exists():
            raise ContentError(f"{where}: no script {record['script']}")
        results = root / record["results"]
        if results.exists():
            checks = json.loads(results.read_text())
        elif previous and record["id"] in previous:
            checks = previous[record["id"]]["checks"]
        else:
            raise ContentError(f"{where}: no {record['results']}; run the script, or build with --run-classics")
        # Each check as the card prints it, so the table, the sentence and the note show one number one way.
        checks = [dict({k: v for k, v in c.items() if k != "display"},
                       display={"paper": shown(c["paper"]), "computed": computed_shown(c),
                                "tolerance": None if c["tolerance"] is None else shown(c["tolerance"])}) for c in checks]
        # The reproduction figure: the script's dataset beside its checks (results/<paper>.figure.json, git-ignored),
        # else -- a fresh clone -- the tracked copy the last build wrote; either way checked against the checks.
        written = results.with_name(results.stem + ".figure.json")
        tracked = root / "docs" / FIGURE_FOLDER / f"{record['id']}.json"
        source = written if written.exists() else tracked
        if not source.exists():
            raise ContentError(f"{where}: no {written.relative_to(root)}; run the script, or build with --run-classics")
        figure = dict(resolve_figure(json.loads(source.read_text()), checks, str(source.relative_to(root))), id=record["id"])
        text = figure_text(figure)
        reproduction = {"file": f"{FIGURE_FOLDER}/{record['id']}.json", "sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
                        "panels": [p["title"] for p in figure["panels"]], "default": figure["default"]}
        record = dict(record, calculator=resolve_calculator(record["calculator"], materials, where), checks=checks,
                      agreement=agreement(checks), note=expand_note(record.get("note"), checks, where),
                      reproduction=reproduction, _figure=text)
        out.append(record)
    out.sort(key=lambda r: r["position"])
    if [r["position"] for r in out] != list(range(1, len(out) + 1)):
        raise ContentError("docs/classics: positions must run 1..N without gaps")
    return out


def build_manifest(root, run_classics=False, previous=None):
    """The manifest of `root`'s docs/. `previous` (a manifest) supplies a classic's checks when its git-ignored
    results file is absent, as on a fresh clone; the CLI never passes it."""
    root = Path(root).resolve()
    from .. import __version__

    materials = Materials(root)
    for key, ref in materials.rii.items():
        if not (root / ref["file"]).exists():
            raise ContentError(f"docs/materials.yaml: {key} names a missing file {ref['file']}")
    examples, sweeps = {}, {}
    for path in sorted((root / "docs" / "examples").glob("*.yaml")):
        example = load_yaml(path)
        if example.get("id") != path.stem:
            raise ContentError(f"{path.relative_to(root)}: the id must be the file name")
        examples[example["id"]] = build_example(example, materials)
        if "_sweep_data" in examples[example["id"]]:
            sweeps[example["id"]] = examples[example["id"]].pop("_sweep_data")
    learn = load_learn(root)
    earlier = {r["id"]: r for r in previous["records"] if r["kind"] == "classic"} if previous else None
    classics = load_classics(root, materials, run_classics, earlier)
    ids = [r["id"] for r in learn + classics]  # benchmark ids are geometry-source-row and cannot collide
    duplicate = sorted({i for i in ids if ids.count(i) > 1})
    if duplicate:
        raise ContentError(f"duplicate record ids: {duplicate}")
    for section in learn:
        for example_id in section["examples"]:
            if example_id not in examples:
                raise ContentError(f"learn section {section['id']}: no example {example_id!r} in docs/examples/")
    placed = {c for section in learn for c in section["classics"]}
    unplaced = [c["id"] for c in classics if c["id"] not in placed]
    if unplaced:
        raise ContentError(f"classics not placed in any Learn section (`::: classic <id>`): {unplaced}")
    for section in learn:
        for classic_id in section["classics"]:
            if classic_id not in {c["id"] for c in classics}:
                raise ContentError(f"learn section {section['id']}: no classic {classic_id!r} in docs/classics/")
    verification, benchmarks = load_verification(root, examples)
    figures = {c["reproduction"]["file"]: c.pop("_figure") for c in classics}
    return {"schema": SCHEMA, "generator": f"pystratify {__version__}", "records": learn + classics + benchmarks,
            "examples": examples, "verification": verification, "_sweeps": sweeps, "_figures": figures}


def sweep_files(manifest):
    """{docs-relative path: JSON text} of the sweeps a build computed (written beside the manifest)."""
    return {manifest["examples"][i]["sweep"]["file"]: text for i, text in manifest.get("_sweeps", {}).items()}


def figure_files(manifest):
    """{docs-relative path: JSON text} of the classics' reproduction figures (docs/classics/figures/<id>.json)."""
    return dict(manifest.get("_figures", {}))


def load_verification(root, examples):
    """benchmarks/MATRIX.md's rows as `benchmark` records, and docs/verification.md's introduction."""
    from .verification import parse_matrix

    matrix = parse_matrix((root / "benchmarks" / "MATRIX.md").read_text())
    meta, body = _front_matter((root / "docs" / "verification.md").read_text(), "docs/verification.md")
    _require(meta, ("title", "summary"), "docs/verification.md")
    rows = {(r["geometry"], r["source"], r["reference"]): r for r in matrix["records"]}
    for link in meta.get("examples", []):
        key = (link["geometry"], link["source"], link["reference"])
        if key not in rows:
            raise ContentError(f"docs/verification.md: no MATRIX.md row {key}")
        if link["example"] not in examples:
            raise ContentError(f"docs/verification.md: no example {link['example']!r} in docs/examples/")
        rows[key]["example"] = link["example"]
    # Comparison figures (benchmarks/figures.py writes docs/figures/<name>.json): a row names one, or a list, one per
    # case it compares. The row carries `figures` (the list) and `figure` (the first).
    figures = {}
    for link in meta.get("figures", []):
        key = (link["geometry"], link["source"], link["reference"])
        names = [link["figure"]] if isinstance(link["figure"], str) else list(link["figure"])
        if key not in rows:
            raise ContentError(f"docs/verification.md: no MATRIX.md row {key} for figure {names!r}")
        for name in names:
            path = root / "docs" / "figures" / f"{name}.json"
            if not path.exists():
                raise ContentError(f"docs/verification.md: no {path.relative_to(root)}; run `python benchmarks/figures.py {name}`")
            data = json.loads(path.read_text())
            for field in ("id", "title", "x_label", "y_label", "x", "series", "worst", "worst_label", "note", "script", "function"):
                if field not in data:
                    raise ContentError(f"{path.relative_to(root)}: missing {field}")
            if any(len(s["y"]) != len(data["x"]) for s in data["series"]):
                raise ContentError(f"{path.relative_to(root)}: a series does not match the x grid")
            figures[name] = data
        rows[key]["figure"], rows[key]["figures"] = names[0], names
    page = {"title": meta["title"], "summary": meta["summary"], "body": body,
            **{k: matrix[k] for k in ("kinds", "statuses", "coverage", "findings")}, "figures": figures}
    return page, matrix["records"]


def dumps(manifest):
    return json.dumps({k: v for k, v in manifest.items() if not k.startswith("_")}, indent=1, ensure_ascii=False) + "\n"


def _generated(manifest):
    """{folder: {file name: text}} of everything a build writes beside the manifest: the sweeps and the figures."""
    out = {"sweeps": {Path(name).name: text + "\n" for name, text in sweep_files(manifest).items()},
           FIGURE_FOLDER: {}}
    for name, text in figure_files(manifest).items():
        out[str(Path(name).parent)][Path(name).name] = text
    return out


def write_manifest(manifest, path):
    """docs/manifest.json and, beside it, docs/sweeps/<id>.json for every precomputed sweep and
    docs/classics/figures/<id>.json for every classic's reproduction figure (stale files removed)."""
    path = Path(path)
    path.write_text(dumps(manifest))
    for folder, wanted in _generated(manifest).items():
        folder = path.parent / folder
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.json"):
            if stale.name not in wanted:
                stale.unlink()
        for name, text in wanted.items():
            (folder / name).write_text(text)


def stale(manifest, path):
    """What `write_manifest` would change: the manifest, sweep and figure files whose content differs (for --check)."""
    path = Path(path)
    out = []
    if (path.read_text() if path.exists() else "") != dumps(manifest):
        out.append(str(path))
    for folder, wanted in _generated(manifest).items():
        folder = path.parent / folder
        for name, text in wanted.items():
            if not (folder / name).exists() or (folder / name).read_text() != text:
                out.append(str(folder / name))
        out += [str(f) for f in (folder.glob("*.json") if folder.exists() else []) if f.name not in wanted]
    return out
