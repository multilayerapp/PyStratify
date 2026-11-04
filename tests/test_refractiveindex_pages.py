"""benchmarks/refractiveindex_pages.py on the small fixture database (the full run is REFRACTIVEINDEX_PAGES.md).

Skipped unless PyYAML is importable; the pip `refractiveindex` comparison runs when that package is installed."""

import sys
import warnings
from pathlib import Path

import pytest

pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import refractiveindex_pages  # noqa: E402
from checks import assert_rows  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "rii"


def test_fixture_pages_agree_with_the_api_oracle_and_the_package():
    try:
        import refractiveindex  # noqa: F401

        with_package = True
    except ImportError:
        with_package = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = refractiveindex_pages.compare(FIXTURE, with_package=with_package)
    assert_rows(refractiveindex_pages.rows(results))
    assert sum("dn" in p for p in results["pages"]) >= 10
