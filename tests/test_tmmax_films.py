"""Films against tmmax through benchmarks/tmmax_films.py; skipped unless tmmax is installed.

Single layers are asserted as shipped; multilayers with tmmax's scan step fixed (its shipped step mis-broadcasts
a (2, 2, 1) array; see the script). The shipped multilayer error is reported, not asserted.
"""

import sys
import warnings
from pathlib import Path

import pytest

pytest.importorskip("tmmax")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import tmmax_films  # noqa: E402
from checks import assert_rows  # noqa: E402


def test_films_agree_with_tmmax_to_float32():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_rows(tmmax_films.rows(tmmax_films.compare()))
