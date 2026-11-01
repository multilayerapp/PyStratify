"""Shared helpers for the examples: an analytic gold model and optional plotting."""

from pathlib import Path

import numpy as np


def gold(wavelength_nm):
    """Refractive index of gold, analytic Drude + two critical points model (``au-elm-2007e``).

    Eq. 3 of Etchegoin, Le Ru & Meyer, J. Chem. Phys. 125, 164705 (2006), with the
    parameters of their erratum, J. Chem. Phys. 127, 189901 (2007), fitted to Johnson &
    Christy for the visible and near ultraviolet (the 2006 Table I set had been fitted to a
    slightly different equation; their book, Le Ru & Etchegoin 2009, Eq. E.2, uses these).
    Against Johnson & Christy over 300-1000 nm the rms error is 2 % in n + ik.  For real work
    take tabulated n, k (e.g. from refractiveindex.info) instead.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    eps = 1.54 - 1 / (143.0**2 * (1 / lam**2 + 1j / (14500.0 * lam)))
    for amplitude, phase, lam_i, gamma_i in ((1.27, -np.pi / 4, 470.0, 1900.0), (1.1, -np.pi / 4, 325.0, 1060.0)):
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
