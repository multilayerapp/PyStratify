"""What more than one page script uses (each imports _page first, which puts src/ and benchmarks/ on the path)."""

from __future__ import annotations

import numpy as np
from numpy import inf

# the films plane-wave pages' thick incoherent stack: (n, d [µm], coherent "c" or incoherent "i")
INCOHERENT = ([1.0, 2.08 + 0.03j, 1.47 + 0.001j, 1.82 + 0.02j, 1.56], [inf, 0.09, 12.0, 0.11, inf],
              ["i", "c", "i", "c", "i"])


def guide_modes(reference):
    """The guided modes of a symmetric slab (n = 2 in air, 500 nm) from 100 to 600 nm thick: PyStratify's poles
    against PyMoosh's mode search ("pymoosh") or the slab dispersion relation ("analytic"), highest index first, NaN
    where a mode is not guided. Returns (thickness, ours, theirs, the largest difference, the number of modes)."""
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

    def padded(rows):
        return np.array([v + [np.nan] * (width - len(v)) for v in rows])

    ours, theirs = padded(ours), padded(theirs)
    return thickness, ours, theirs, float(np.nanmax(np.abs(ours - theirs))), width
