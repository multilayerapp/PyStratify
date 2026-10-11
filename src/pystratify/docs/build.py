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
# What a sphere plane-wave example can plot, and how the API's /spheres/calculate response names it.
SPHERE_SERIES = ("q_ext", "q_sca", "q_abs", "md_sca", "ed_sca")
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
    elif "drude" in spec:
        d = spec["drude"]
        rows = _drude_rows(d["from_um"], d["to_um"], d["count"], d["lambda_p_um"], d["gamma"])
    elif "lorentz" in spec:
        d = spec["lorentz"]
        rows = _lorentz_rows(d["from_cm1"], d["to_cm1"], d["count"], d["eps_inf"], d["d_eps"], d["w_to_cm1"], d["gamma_cm1"])
    else:
        raise ContentError(f"{where}: a table needs rows, drude or lorentz")
    return {"name": spec["name"], "rows": [[round(w, 6), round(n, 6), round(k, 6)] for w, n, k in rows]}


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
            out["table"] = _table(out["table"], where)
        return out

    def index(self, medium, wavelength_um):
        """n + ik at the wavelengths, as the API resolves the medium: database page, uploaded table, or constant."""
        lam = np.asarray(wavelength_um, dtype=float)
        if "rii" in medium:
            return material(self.root / medium["rii"]["file"], lam)
        if "table" in medium:
            rows = np.asarray(medium["table"]["rows"], dtype=float)
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
    }
    return lam * 1000, curves


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


COMPUTE = {("spheres", "plane_wave"): _sphere_plane_wave}
CODE = {("spheres", "plane_wave"): _sphere_plane_wave_code}


def _measure(measure, x, curves, where):
    if "peak" in measure:
        y = curves[measure["peak"]]
        m = np.ones_like(x, dtype=bool)
        lo, hi = measure.get("within", (None, None))
        if lo is not None:
            m &= x >= lo
        if hi is not None:
            m &= x <= hi
        i = np.flatnonzero(m)[np.argmax(y[m])]
        return float(x[i])
    raise ContentError(f"{where}: unknown measure {measure!r}")


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
    unknown = [k for k in keys if k not in SPHERE_SERIES]
    if unknown:
        raise ContentError(f"{where}: unknown series {unknown}; known: {SPHERE_SERIES}")
    x, curves = compute(out, materials)
    out["data"] = {"x": [_sig(v) for v in x], "series": {k: [_sig(v) for v in curves[k]] for k in keys}}
    checks = []
    for c in example.get("checks", []):
        _require(c, ("check", "reference", "tolerance", "source", "measure"), f"{where} check")
        computed = _measure(c["measure"], x, curves, where)
        checks.append({"check": c["check"], "reference": c["reference"], "computed": round(computed, 4),
                       "tolerance": c["tolerance"], "pass": bool(abs(computed - c["reference"]) <= c["tolerance"]),
                       "source": c["source"]})
    out["checks"] = checks
    out["code"] = CODE[(example["geometry"], example["source"])](out)
    for p in example.get("params", []):
        _require(p, ("id", "label", "unit", "target", "scale", "min", "max", "step", "value"), f"{where} param")
    return out


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
            "examples": meta.get("examples", []), "references": meta["references"],
            "status": meta.get("status", "published"), "body": body,
        })
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
    examples = {}
    for path in sorted((root / "docs" / "examples").glob("*.yaml")):
        example = load_yaml(path)
        if example.get("id") != path.stem:
            raise ContentError(f"{path.relative_to(root)}: the id must be the file name")
        examples[example["id"]] = build_example(example, materials)
    learn = load_learn(root)
    earlier = {r["id"]: r for r in previous["records"] if r["kind"] == "classic"} if previous else None
    classics = load_classics(root, materials, run_classics, earlier)
    ids = [r["id"] for r in learn + classics]
    duplicate = sorted({i for i in ids if ids.count(i) > 1})
    if duplicate:
        raise ContentError(f"duplicate record ids: {duplicate}")
    for section in learn:
        for example_id in section["examples"]:
            if example_id not in examples:
                raise ContentError(f"learn section {section['id']}: no example {example_id!r} in docs/examples/")
    return {"schema": SCHEMA, "generator": f"pystratify {__version__}", "records": learn + classics, "examples": examples}


def dumps(manifest):
    return json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"


def write_manifest(manifest, path):
    Path(path).write_text(dumps(manifest))
