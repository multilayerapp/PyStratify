"""docs/manifest.json is what `python -m pystratify.docs build` writes from docs/ (the site builds from it)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")

import pystratify as ps  # noqa: E402
from pystratify.docs import build_manifest, material  # noqa: E402
from pystratify.docs.build import ContentError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = json.loads((ROOT / "docs" / "manifest.json").read_text())


@pytest.fixture(scope="module")
def manifest():
    # A fresh clone has no crosscheck/classics/results/ (git-ignored): the committed checks stand in for them.
    return build_manifest(ROOT, previous=COMMITTED)


def test_manifest_is_current(manifest):
    built, committed = dict(manifest), dict(COMMITTED)
    for name, example in committed["examples"].items():
        ours = built["examples"][name]
        assert ours["data"]["x"] == example["data"]["x"]
        for key, ys in example["data"]["series"].items():
            np.testing.assert_allclose(ours["data"]["series"][key], ys, rtol=1e-5, atol=1e-9)
        assert {k: v for k, v in ours.items() if k != "data"} == {k: v for k, v in example.items() if k != "data"}
    built.pop("examples"), committed.pop("examples")
    built = {k: v for k, v in built.items() if not k.startswith("_")}  # the sweeps' data is written beside the manifest
    assert built == committed, "docs/ changed: run `python -m pystratify.docs build` and commit docs/manifest.json"


def test_sweep_files_are_current(manifest):
    """Every example with sliders has its grid in docs/sweeps/<id>.json, byte for byte what a build writes."""
    from pystratify.docs.build import sweep_files

    files = sweep_files(manifest)
    with_params = {name for name, e in manifest["examples"].items() if e.get("params")}
    assert {Path(f).stem for f in files} == with_params
    for name, text in files.items():
        assert (ROOT / name).read_text() == text + "\n", f"{name} is stale: run `python -m pystratify.docs build`"
        meta = manifest["examples"][Path(name).stem]["sweep"]
        data = json.loads(text)
        assert data["shape"] == meta["shape"] == [len(a["values"]) for a in data["axes"]]
        assert all(len(ys) == meta["points"] for ys in data["series"].values())
    assert set(p.name for p in (ROOT / "docs" / "sweeps").glob("*.json")) == {Path(f).name for f in files}


def test_a_sweep_point_is_a_fresh_computation(manifest):
    """The grid's slice at a slider position equals the example rebuilt with that value: the page's sliders read
    exactly what PyStratify computes for the setup they describe."""
    from pystratify.docs.build import Materials, build_example, load_yaml

    materials = Materials(ROOT)
    for name in ("bragg-mirror", "si-sphere-mie", "glass-cylinder"):
        example = load_yaml(ROOT / "docs" / "examples" / f"{name}.yaml")
        grid = json.loads(manifest["_sweeps"][name])
        axis, i = grid["axes"][0], -1  # the slider's last position
        value = axis["values"][i]
        param = example["params"][0]
        param["value"] = value
        targets = [param["target"]] if isinstance(param["target"], str) else param["target"]
        for target in targets:
            node = example["calculator"]
            keys = target.split(".")
            for key in keys[:-1]:
                node = node[int(key)] if isinstance(node, list) else node[key]
            node[keys[-1]] = value * param["scale"]
        fresh = build_example(example, materials)
        assert fresh["data"]["x"] == grid["x"]
        for key, ys in fresh["data"]["series"].items():
            np.testing.assert_allclose(grid["series"][key][i], ys, rtol=1e-5, atol=1e-9)


def test_cylinder_example_is_a_plain_pystratify_solve(manifest):
    example = manifest["examples"]["glass-cylinder"]
    lam = np.asarray(example["data"]["x"]) * 1e-3
    q = [ps.cross_widths(ps.solve_cylinder([0.25], [1.55, 1.0], wl), "axial-magnetic")["efficiency_scattering"] for wl in lam]
    np.testing.assert_allclose(example["data"]["series"]["q_sca_m"], q, rtol=1e-5)
    assert example["live"] is False  # a cylinder job takes minutes on the API: the page reads the sweep only


def test_every_classic_is_placed_in_one_learn_section(manifest):
    sections = [r for r in manifest["records"] if r["kind"] == "learn"]
    placed = [c for s in sections for c in s["classics"]]
    assert sorted(placed) == sorted(r["id"] for r in manifest["records"] if r["kind"] == "classic")


def test_every_example_check_passes(manifest):
    for example in manifest["examples"].values():
        for check in example["checks"]:
            assert check["pass"], f"{example['id']}: {check['check']} {check['computed']} vs {check['reference']}"


def test_sphere_example_is_a_plain_pystratify_solve(manifest):
    example = manifest["examples"]["si-sphere-mie"]
    lam = np.asarray(example["data"]["x"]) * 1e-3
    n = material(ROOT / "crosscheck" / "classics" / "data" / "Si_Green-2008.yml", lam)
    cs = ps.cross_sections(ps.solve([0.075], np.column_stack([n, np.ones_like(lam)]), lam))
    np.testing.assert_allclose(example["data"]["series"]["q_ext"], cs.q_ext, rtol=1e-5)
    np.testing.assert_allclose(example["data"]["series"]["md_sca"], cs.sca_by_order[:, ps.TE, 0] / cs.geometric, rtol=1e-5, atol=1e-9)


def test_learn_sections_link_in_order(manifest):
    sections = [r for r in manifest["records"] if r["kind"] == "learn"]
    by_id = {s["id"]: s for s in sections}
    for s in sections:
        assert s["url"] == f"learn/{s['geometry']}/{s['slug']}.html"
        if s["next"]:
            assert by_id[s["next"]]["prev"] == s["id"]
        for example in s["examples"]:
            assert example in manifest["examples"]


def test_a_broken_record_is_refused(tmp_path):
    for part in ("docs", "crosscheck/classics/data"):
        shutil.copytree(ROOT / part, tmp_path / part, ignore=shutil.ignore_patterns("manifest.json", "results"))
    (tmp_path / "src").mkdir()
    example = tmp_path / "docs" / "examples" / "si-sphere-mie.yaml"
    example.write_text(example.read_text().replace("  - key: md_sca", "  - key: not_a_series"))
    with pytest.raises(ContentError, match="unknown series"):
        build_manifest(tmp_path, previous=COMMITTED)


def test_every_matrix_row_is_one_benchmark_record(manifest):
    from pystratify.docs.verification import table_rows

    rows = [r for r in manifest["records"] if r["kind"] == "benchmark"]
    assert len(rows) == table_rows((ROOT / "benchmarks" / "MATRIX.md").read_text())
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["geometry"] for r in rows} == {"films", "cylinders", "spheres"}
    assert all(r["status"] and r["reference"] for r in rows)
    assert {r["reference_kind"] for r in rows} >= {"analytic", "precision", "method", "code", "paper", "lineage"}
    assert sum("example" in r for r in rows) == 3  # docs/verification.md's links, each to an existing example


def test_an_example_outside_its_data_is_refused(tmp_path):
    # The tutorials' silver table ends at 0.8 µm; their etalon plotted to 1.0 µm on silently extrapolated values.
    for part in ("docs", "crosscheck/classics/data"):
        shutil.copytree(ROOT / part, tmp_path / part, ignore=shutil.ignore_patterns("manifest.json", "results"))
    (tmp_path / "src").mkdir()
    example = tmp_path / "docs" / "examples" / "fabry-perot-etalon.yaml"
    example.write_text(example.read_text().replace("wavelength-end: 800", "wavelength-end: 1000"))
    with pytest.raises(ContentError, match="covers"):
        build_manifest(tmp_path, previous=COMMITTED)
