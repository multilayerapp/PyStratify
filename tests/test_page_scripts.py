"""Each Verification page is computed by one script (benchmarks/pages/), and the site draws exactly what it computes.

Every figure a page script declares is published under that script's name (docs/figures/<name>.json, written by
benchmarks/figures.py from the script's own records); the PyMoosh page, fast and self-contained, is recomputed here
and compared value for value. The others are recomputed by benchmarks/verify_all.py in their environments.
"""

import importlib
import json
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


def test_pymoosh_page_reproduces_its_figures():
    pytest.importorskip("PyMoosh")
    page = importlib.import_module("films_plane_wave_pymoosh")
    assert page.SCRIPT == "benchmarks/pages/films_plane_wave_pymoosh.py"
    for spec in page.FIGURES:
        ours = page.record(*spec)
        published = json.loads((ROOT / "docs" / "figures" / f"{ours['id']}.json").read_text())
        assert {k: published[k] for k in ours} == ours, ours["id"]
