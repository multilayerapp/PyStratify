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
        # A check's computed value can be a rounding residual (an energy balance at 2e-16), which differs between
        # platforms' BLAS/libm in its last bits: compare it to a tolerance, everything else exactly.
        assert len(ours.get("checks", [])) == len(example.get("checks", []))
        for mine, theirs in zip(ours.get("checks", []), example.get("checks", [])):
            np.testing.assert_allclose(mine["computed"], theirs["computed"], rtol=1e-5, atol=1e-12)
            assert {k: v for k, v in mine.items() if k != "computed"} == {k: v for k, v in theirs.items() if k != "computed"}
        assert ({k: v for k, v in ours.items() if k not in ("data", "checks")}
                == {k: v for k, v in example.items() if k not in ("data", "checks")})
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
    figures = manifest["verification"]["figures"]
    assert {name for r in rows for name in r.get("figures", [])} == set(figures)
    assert all(r["figure"] == r["figures"][0] for r in rows if "figure" in r)
    for name, f in figures.items():
        assert (ROOT / "docs" / "figures" / f"{name}.json").exists()
        assert f["worst"] >= 0 and all(len(s["y"]) == len(f["x"]) for s in f["series"])
        assert {s["role"] for s in f["series"]} <= {"pystratify", "reference"}


def _copy_content(tmp_path):
    for part in ("docs", "crosscheck/classics/data"):
        shutil.copytree(ROOT / part, tmp_path / part, ignore=shutil.ignore_patterns("manifest.json", "results"))
    (tmp_path / "src").mkdir()
    return tmp_path / "docs" / "examples" / "fabry-perot-etalon.yaml"


def test_an_example_outside_its_data_is_refused(tmp_path):
    # Johnson and Christy's silver ends at 1.94 µm. (The former tutorials' resampled table ended at 0.8 µm, and their
    # etalon plotted to 1.0 µm on silently extrapolated values.)
    example = _copy_content(tmp_path)
    example.write_text(example.read_text().replace("wavelength-end: 800", "wavelength-end: 2500"))
    with pytest.raises(ContentError, match="covers"):
        build_manifest(tmp_path, previous=COMMITTED)


def test_an_example_with_an_uploaded_table_is_refused(tmp_path):
    # Ilia, 2026-10-16: a worked example cites a refractiveindex.info page or a constant, never an uploaded table.
    example = _copy_content(tmp_path)
    example.write_text(example.read_text().replace("rii: Ag_Johnson", 'table: {name: "Ag", rows: [[0.3, 0.05, 3.0], [1.0, 0.05, 6.0]]}', 1))
    with pytest.raises(ContentError, match="uploaded n,k table"):
        build_manifest(tmp_path, previous=COMMITTED)


def test_learn_descriptions_fit_a_search_snippet_and_reviews_are_dates(manifest, tmp_path):
    """Every section has a one-line description (the index card, the meta description, llms.txt) that fits the
    ~160 characters a search engine shows; `reviewed` is a date or empty, and anything else is refused."""
    from pystratify.docs.build import DESCRIPTION_LIMIT, load_learn

    sections = [r for r in manifest["records"] if r["kind"] == "learn"]
    assert all(0 < len(s["description"]) <= DESCRIPTION_LIMIT for s in sections)
    assert all(s["reviewed"] is None or len(s["reviewed"]) == 10 for s in sections)
    shutil.copytree(ROOT / "docs" / "learn", tmp_path / "docs" / "learn")
    page = tmp_path / "docs" / "learn" / "films" / "01-interfaces" / "01-fresnel-equations.md"
    text = page.read_text()
    page.write_text(text.replace("\nreviewed:\n", "\nreviewed: 2026-10-20\n"))
    assert next(s for s in load_learn(tmp_path) if s["id"] == "fresnel-equations")["reviewed"] == "2026-10-20"
    page.write_text(text.replace("\nreviewed:\n", "\nreviewed: soon\n"))
    with pytest.raises(ContentError, match="reviewed is a date"):
        load_learn(tmp_path)
    page.write_text(text.replace('\ndescription: "', '\ndescription: "' + "x" * DESCRIPTION_LIMIT))
    with pytest.raises(ContentError, match="at most 160"):
        load_learn(tmp_path)


def test_a_review_holds_until_the_page_is_edited(tmp_path):
    """D6 (Ilia, 2026-10-31): `python -m pystratify.docs review` stamps the date and a fingerprint of what was read.
    An edit to the text, or to an example the page places, ends the claim but keeps the date; a date written by hand
    without the stamp makes no claim; a fingerprint without a date is refused."""
    from pystratify.docs.__main__ import main
    from pystratify.docs.build import load_learn

    for folder in ("learn", "examples", "classics"):
        shutil.copytree(ROOT / "docs" / folder, tmp_path / "docs" / folder)
    page = tmp_path / "docs" / "learn" / "films" / "01-interfaces" / "01-fresnel-equations.md"
    original = page.read_text()
    review = ["review", "fresnel-equations", "--date", "2026-11-02", "--root", str(tmp_path)]

    def section():
        return next(s for s in load_learn(tmp_path) if s["id"] == "fresnel-equations")

    assert section()["reviewed"] is None and not section()["review_current"]
    assert main(review) == 0
    assert section()["reviewed"] == "2026-11-02" and section()["review_current"]
    stamped = page.read_text()
    assert main(review) == 0 and page.read_text() == stamped  # restamping an unchanged page changes nothing
    page.write_text(stamped.replace("half-wave loss", "half-wave-loss"))
    assert section()["reviewed"] == "2026-11-02" and not section()["review_current"]
    assert main(review) == 0 and section()["review_current"]
    example = tmp_path / "docs" / "examples" / "fresnel-air-glass.yaml"
    example.write_text(example.read_text() + "\n")
    assert not section()["review_current"]
    page.write_text(original.replace("\nreviewed:\n", "\nreviewed: 2026-11-02\n"))
    assert section()["reviewed"] == "2026-11-02" and not section()["review_current"]
    page.write_text(original.replace("\nreviewed:\n", '\nreviewed:\nreviewed_fingerprint: "0123456789abcdef"\n'))
    with pytest.raises(ContentError, match="without a reviewed date"):
        load_learn(tmp_path)


def test_the_build_command_hands_the_committed_manifest_to_the_build(monkeypatch, capsys):
    # The command must pass the committed manifest as `previous`, or a clean checkout (no classics results/) cannot
    # build: it did not until 2026-11-05, while the tests above, which pass it themselves, stayed green.
    from pystratify.docs import __main__ as cli

    seen = {}

    def fake_build(root, run_classics=False, previous=None):
        seen["previous"] = previous
        return COMMITTED

    monkeypatch.setattr(cli, "build_manifest", fake_build)
    monkeypatch.setattr(cli, "stale", lambda manifest, target: [])
    assert cli.main(["build", "--check", "--root", str(ROOT)]) == 0
    assert seen["previous"] == COMMITTED



def test_classic_figure_files_are_current(manifest):
    """Every classic has its reproduction figure in docs/classics/figures/<id>.json, byte for byte what a build writes
    (from the script's results/<paper>.figure.json, or on a fresh clone the tracked file itself)."""
    import hashlib

    from pystratify.docs.build import figure_files

    files = figure_files(manifest)
    classics = [r for r in manifest["records"] if r["kind"] == "classic"]
    assert {Path(f).stem for f in files} == {c["id"] for c in classics}
    for c in classics:
        text = files[c["reproduction"]["file"]]
        assert (ROOT / "docs" / c["reproduction"]["file"]).read_text() == text, f"{c['id']}: run `python -m pystratify.docs build`"
        assert c["reproduction"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert {p.name for p in (ROOT / "docs" / "classics" / "figures").glob("*.json")} == {Path(f).name for f in files}


def test_a_classic_card_takes_every_number_from_its_checks(manifest):
    """R-CLASSICS-CONTENT (Ilia, 2026-11-05): the agreement sentence is generated from the checks, the note's numbers
    are filled from them, and every mark of the figure sits at its check's paper value (recomputed here, not read
    back), on a series of its panel and inside the plotted range."""
    from pystratify.docs.build import FIGURE_BUDGET, agreement, computed_shown, figure_files, shown

    files = figure_files(manifest)
    for c in (r for r in manifest["records"] if r["kind"] == "classic"):
        checks = {k["check"]: k for k in c["checks"]}
        for k in c["checks"]:
            assert k["display"] == {"paper": shown(k["paper"]), "computed": computed_shown(k),
                                    "tolerance": None if k["tolerance"] is None else shown(k["tolerance"])}
        assert c["agreement"] == agreement(c["checks"])
        assert "{" not in c["note"] and "}" not in c["note"]
        figure = json.loads(files[c["reproduction"]["file"]])
        assert figure["id"] == c["id"] and 0 <= figure["default"] == c["reproduction"]["default"] < len(figure["panels"])
        assert c["reproduction"]["panels"] == [p["title"] for p in figure["panels"]]
        numbers = 0
        for p in figure["panels"]:
            numbers += len(p["x"]) * (1 + len(p["series"]))
            lo, hi = min(p["x"]), max(p["x"])
            keys = {s["key"] for s in p["series"]}
            for m in p["marks"]:
                paper = checks[m["check"]]["paper"]
                value = 10 ** paper if m.get("log10") else paper
                assert m["pass"] == checks[m["check"]]["pass"]
                assert m.get("series") is None or m["series"] in keys
                if m["kind"] == "rule":
                    assert m["x"] == value and "at" not in m
                else:
                    assert m["y"] == value
                    assert [m["x"], m["x2"]] == m["at"] if m["kind"] == "level" else m["x"] == m["at"]
                assert all(lo <= x <= hi for x in (m["x"], m.get("x2", m["x"])))
        assert numbers <= FIGURE_BUDGET


def _copy_classics(tmp_path):
    for part in ("docs", "crosscheck/classics/data"):
        shutil.copytree(ROOT / part, tmp_path / part, ignore=shutil.ignore_patterns("manifest.json", "results"))
    for script in (ROOT / "crosscheck" / "classics").glob("*.py"):
        shutil.copy(script, tmp_path / "crosscheck" / "classics" / script.name)
    return {r["id"]: r for r in COMMITTED["records"] if r["kind"] == "classic"}


def test_a_fresh_clone_builds_the_tracked_figures(tmp_path):
    # No crosscheck/classics/results/ (git-ignored): the committed checks and the tracked figures stand in, and the
    # build writes the tracked figures back unchanged.
    from pystratify.docs.build import Materials, figure_text, load_classics

    earlier = _copy_classics(tmp_path)
    for c in load_classics(tmp_path, Materials(tmp_path), previous=earlier):
        assert c["_figure"] == (ROOT / "docs" / c["reproduction"]["file"]).read_text()
        assert c["agreement"] == earlier[c["id"]]["agreement"] and c["note"] == earlier[c["id"]]["note"]
        assert figure_text(json.loads(c["_figure"])) == c["_figure"]


@pytest.mark.parametrize("edit, refusal", [
    (lambda fig, rec: fig.write_text(fig.read_text().replace('"check":"80 nm: z of the maximum (nm)"', '"check":"80 nm: z of the peak"')),
     "not exactly one of the checks"),
    (lambda fig, rec: fig.write_text(fig.read_text().replace('"series":"d80","label":"max"', '"series":"d90","label":"max"')),
     "names no series"),
    (lambda fig, rec: rec.write_text(rec.read_text().replace("{80 nm: fluorescence maximum}", "{80 nm: fluorescence peak}", 1)),
     "the note names"),
    (lambda fig, rec: rec.write_text(rec.read_text().replace("note: >-", "agreement: Hand-written, 3.5 near 12 nm.\nnote: >-")),
     "generated from the checks"),
    (lambda fig, rec: fig.write_text(fig.read_text().replace('"at":20.0', '"at":200.0', 1)),
     "outside"),
])
def test_a_classic_figure_or_note_that_does_not_match_its_checks_is_refused(tmp_path, edit, refusal):
    from pystratify.docs.build import Materials, load_classics

    earlier = _copy_classics(tmp_path)
    edit(tmp_path / "docs" / "classics" / "figures" / "anger-2006.json", tmp_path / "docs" / "classics" / "anger-2006.yaml")
    with pytest.raises(ContentError, match=refusal):
        load_classics(tmp_path, Materials(tmp_path), previous=earlier)
