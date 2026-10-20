"""Helpers more than one page script (and benchmarks/figures.py) uses."""

from __future__ import annotations

import json  # noqa: F401
import sys
from pathlib import Path

import numpy as np
from numpy import inf  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "benchmarks"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pystratify as ps  # noqa: E402,F401
from pystratify import planar  # noqa: E402,F401


INCOHERENT = ([1.0, 2.08 + 0.03j, 1.47 + 0.001j, 1.82 + 0.02j, 1.56], [inf, 0.09, 12.0, 0.11, inf], ["i", "c", "i", "c", "i"])


def _guide_modes(reference):
    from planar_codes import GUIDES, guided_mode_sets, slab_modes
    from pystratify.planar_emission import FilmSource

    n, d, lam = GUIDES["symmetric slab, n 2.0 in air, 500 nm"]
    thickness = np.linspace(100, 600, 26)
    ours, theirs = [], []
    for t in thickness:
        dd = [inf, float(t), inf]
        if reference == "pymoosh":
            a, b, _ = guided_mode_sets((n, dd, lam))
        else:
            source = FilmSource(n, dd, lam, 1, t / 3)
            a, b = sorted(np.array(source.guided_poles()) * source.ns), slab_modes(n[0], n[1], n[2], t, lam)
        ours.append(sorted(a, reverse=True)), theirs.append(sorted(b, reverse=True))
    width = max(len(v) for v in ours + theirs)
    pad = lambda rows: np.array([v + [np.nan] * (width - len(v)) for v in rows])  # noqa: E731
    ours, theirs = pad(ours), pad(theirs)
    diffs = np.abs(ours - theirs)
    return thickness, ours, theirs, float(np.nanmax(diffs)), width


def at(x, points):
    """A reference series on the grid x, null except at the published points {x: value} (nearest grid point)."""
    out = np.full(len(x), np.nan)
    for xp, value in points.items():
        out[int(np.argmin(np.abs(np.asarray(x) - xp)))] = value
    return out
