"""Films against meent's RCWA with zero patterning (benchmarks/MATRIX.md, films).

    pip install meent==0.13.2               # MIT; the calculator's grating engine (multilayer/backend/rcwa_solver.py)
    python benchmarks/rcwa_meent.py         # prints the worst |R - R_RCWA|, |T - T_RCWA| (~5 s)

A uniform layer is a grating with one Fourier order, so rigorous coupled-wave analysis must reduce to the transfer
matrix. The call is the one the calculator's backend makes (``meent.call_mee``, backend 0, complex128, one order
``fto = (0, 0)``, a 1 um period that drops out); the reflected and transmitted diffraction efficiencies summed are R
and T. ``tests/test_rcwa_meent.py`` asserts the stack below to 1e-9, skipped unless meent is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pystratify.planar import coh_tmm  # noqa: E402

# a three-layer interference filter on glass: n top, layer indices, thicknesses in um, n bottom
STACK = (1.0, [2.35, 1.46, 2.35], [0.095, 0.110, 0.095], 1.5)


def meent_rt(wavelength, angle_deg=0.0, polarization="s", stack=STACK):
    """meent's (R, T) for the stack as an unpatterned grating."""
    import meent

    n_top, indices, thickness, n_bottom = stack
    solver = meent.call_mee(backend=0, n_top=n_top, n_bot=n_bottom, theta=float(np.deg2rad(angle_deg)), phi=None,
                            pol=0 if polarization == "s" else 1, period=(1.0, 1.0), wavelength=wavelength,
                            ucell=np.asarray(indices, dtype=np.complex128).reshape(-1, 1, 1),
                            thickness=np.asarray(thickness, dtype=float), fto=(0, 0), type_complex=np.complex128)
    result = solver.conv_solve()
    return float(np.real(np.sum(result.res.de_ri))), float(np.real(np.sum(result.res.de_ti)))


def stratify_rt(wavelength, angle_deg=0.0, polarization="s", stack=STACK):
    n_top, indices, thickness, n_bottom = stack
    out = coh_tmm(polarization, [n_top, *indices, n_bottom], [np.inf, *thickness, np.inf], np.deg2rad(angle_deg), wavelength)
    return out["R"], out["T"]


def worst(wavelengths=(0.45, 0.55, 0.65), angles=(0.0, 30.0), polarizations=("s", "p")):
    diffs = [np.abs(np.subtract(meent_rt(w, a, p), stratify_rt(w, a, p))) for w in wavelengths for a in angles for p in polarizations]
    return np.max(diffs, axis=0)


if __name__ == "__main__":
    R, T = worst()
    print(f"meent RCWA, one order, against coh_tmm: worst |R - R_RCWA| = {R:.2e}, |T - T_RCWA| = {T:.2e}")
