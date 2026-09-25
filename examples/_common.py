"""Shared helpers for the examples: an analytic gold model and optional plotting."""

from pathlib import Path

import numpy as np


def gold(wavelength_nm):
    """Refractive index of gold, analytic Drude + two critical points model.

    Etchegoin, Le Ru & Meyer, J. Chem. Phys. 125, 164705 (2006), fitted to
    Johnson & Christy; good to a few per cent over 400-1000 nm.  For real work
    take tabulated n, k (e.g. from refractiveindex.info) instead.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    eps = 1.53 - 1 / (145.0**2 * (1 / lam**2 + 1j / (17000.0 * lam)))
    for amplitude, phase, lam_i, gamma_i in ((0.94, -np.pi / 4, 468.0, 2300.0), (1.36, -np.pi / 4, 331.0, 940.0)):
        eps = eps + amplitude / lam_i * (
            np.exp(1j * phase) / (1 / lam_i - 1 / lam - 1j / gamma_i)
            + np.exp(-1j * phase) / (1 / lam_i + 1 / lam + 1j / gamma_i)
        )
    return np.sqrt(eps)


def figure(name, size=(9, 4)):
    """A matplotlib figure and the PNG path beside this file, or (None, None) without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None
    return plt.figure(figsize=size), Path(__file__).with_name(f"{name}.png")
