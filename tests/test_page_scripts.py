"""Each Verification page is computed by one script (benchmarks/pages/), and the site draws exactly what it computes.

Every figure a page script declares is published under that script's name (docs/figures/<name>.json, written by
benchmarks/figures.py from the script's own records); the PyMoosh page, fast and self-contained, is recomputed here
and compared value for value. The others are recomputed by benchmarks/verify_all.py in their environments.
"""

import importlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "pages"))
sys.path.insert(0, str(ROOT / "benchmarks"))
PAGES = sorted(p.stem for p in (ROOT / "benchmarks" / "pages").glob("*.py") if not p.stem.startswith("_"))


@pytest.mark.parametrize("name", [p for p in PAGES if p != "films_plane_wave_pymoosh"])
def test_page_script_publishes_its_figures(name):
    import _page

    importlib.import_module(name)
    declared = _page.FIGURES[name]
    assert declared, f"{name} declares no figure"
    for figure, *_, fn in declared:
        published = json.loads((ROOT / "docs" / "figures" / f"{figure}.json").read_text())
        assert published["script"] == f"benchmarks/pages/{name}.py", figure
        assert published["function"] == fn.__name__, figure


@pytest.mark.parametrize("path", sorted((ROOT / "docs" / "figures").glob("*.json")), ids=lambda p: p.stem)
def test_every_circle_is_a_computed_reference_value(path):
    """A reference is drawn as one circle per value its code returned, never thinned for display: each page script
    computes it on a sparse grid of at most 41 points (benchmarks/pages/_page.py MAX_DOTS)."""
    import _page

    figure = json.loads(path.read_text())
    assert figure["script"].startswith("benchmarks/pages/"), "every published figure comes from a page script"
    assert figure["function"] == _page.plot_name(figure["title"]), "a figure's function is plot_<its title>"
    for s in figure["series"]:
        assert len(s["y"]) == len(figure["x"]), s["key"]
        if s["role"] == "reference":
            assert 0 < sum(v is not None for v in s["y"]) <= 41, s["key"]


# The figures are published from the maintainer's Mac; GitHub's Linux runners reproduce them to a few eps, not bit
# for bit (first seen on the benchmarks job, 2026-11-05: R and T to 2e-15, the agreement's `worst` 1.63e-14 there
# against 1.88e-14 here). Floats are compared to this many eps, relative to max(1, |value|); everything else exactly.
FIGURE_EPS = 64


def _deviation(ours, published, path):
    """The largest float difference in units of eps * max(1, |value|); asserts the rest is identical."""
    if isinstance(ours, float) or isinstance(published, float):
        assert isinstance(ours, (int, float)) and isinstance(published, (int, float)), path
        if math.isnan(ours) and math.isnan(published):
            return 0.0
        return abs(ours - published) / (sys.float_info.epsilon * max(1.0, abs(ours), abs(published)))
    if isinstance(ours, dict):
        assert isinstance(published, dict) and ours.keys() == published.keys(), path
        return max((_deviation(ours[k], published[k], f"{path}.{k}") for k in ours), default=0.0)
    if isinstance(ours, (list, tuple)):
        assert isinstance(published, list) and len(ours) == len(published), path
        return max((_deviation(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(ours, published))), default=0.0)
    assert ours == published, path
    return 0.0


def test_pymoosh_page_reproduces_its_figures():
    pytest.importorskip("PyMoosh")
    page = importlib.import_module("films_plane_wave_pymoosh")
    assert page.SCRIPT == "benchmarks/pages/films_plane_wave_pymoosh.py"
    deviation = {}
    for spec in page.FIGURES:
        ours = page.record(*spec)
        published = json.loads((ROOT / "docs" / "figures" / f"{ours['id']}.json").read_text())
        deviation[ours["id"]] = _deviation(ours, {k: published[k] for k in ours}, ours["id"])
    assert max(deviation.values()) <= FIGURE_EPS, f"eps per figure: {deviation}"
