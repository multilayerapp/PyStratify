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
CLASSIC_FIELDS = ("id", "kind", "position", "geometry", "title", "cite", "doi", "figure", "shows", "inputs", "agreement",
                  "perturb", "script", "results", "calculator")
LEARN_FIELDS = ("id", "title", "geometry", "summary", "references")
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
        })
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
        if not (root / record["script"]).exists():
            raise ContentError(f"{where}: no script {record['script']}")
        results = root / record["results"]
        if results.exists():
            checks = json.loads(results.read_text())
        elif previous and record["id"] in previous:
            checks = previous[record["id"]]["checks"]
        else:
            raise ContentError(f"{where}: no {record['results']}; run the script, or build with --run-classics")
        record = dict(record, calculator=resolve_calculator(record["calculator"], materials, where), checks=checks)
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
    return {"schema": SCHEMA, "generator": f"pystratify {__version__}", "records": learn + classics + benchmarks,
            "examples": examples, "verification": verification, "_sweeps": sweeps}


def sweep_files(manifest):
    """{docs-relative path: JSON text} of the sweeps a build computed (written beside the manifest)."""
    return {manifest["examples"][i]["sweep"]["file"]: text for i, text in manifest.get("_sweeps", {}).items()}


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
    page = {"title": meta["title"], "summary": meta["summary"], "body": body,
            **{k: matrix[k] for k in ("kinds", "statuses", "coverage", "findings")}}
    return page, matrix["records"]


def dumps(manifest):
    return json.dumps({k: v for k, v in manifest.items() if not k.startswith("_")}, indent=1, ensure_ascii=False) + "\n"


def write_manifest(manifest, path):
    """docs/manifest.json and, beside it, docs/sweeps/<id>.json for every precomputed sweep (stale files removed)."""
    path = Path(path)
    path.write_text(dumps(manifest))
    folder = path.parent / "sweeps"
    folder.mkdir(exist_ok=True)
    wanted = {Path(name).name: text for name, text in sweep_files(manifest).items()}
    for stale in folder.glob("*.json"):
        if stale.name not in wanted:
            stale.unlink()
    for name, text in wanted.items():
        (folder / name).write_text(text + "\n")


def stale(manifest, path):
    """What `write_manifest` would change: the manifest and sweep files whose content differs (for --check)."""
    path = Path(path)
    out = []
    if (path.read_text() if path.exists() else "") != dumps(manifest):
        out.append(str(path))
    folder = path.parent / "sweeps"
    wanted = {Path(name).name: text + "\n" for name, text in sweep_files(manifest).items()}
    for name, text in wanted.items():
        if not (folder / name).exists() or (folder / name).read_text() != text:
            out.append(str(folder / name))
    out += [str(f) for f in (folder.glob("*.json") if folder.exists() else []) if f.name not in wanted]
    return out
