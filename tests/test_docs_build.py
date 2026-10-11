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
    assert built == committed, "docs/ changed: run `python -m pystratify.docs build` and commit docs/manifest.json"


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
