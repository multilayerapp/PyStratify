"""Shared inputs for the classic-paper recomputations (roadmap WP1.9, `showcase.md`).

Each script in this folder recomputes the headline figure of one classic paper with the
inputs the multilayer.app calculator receives from the paper's share link, and writes a
CSV, a PNG and the checks against numbers printed in (or read off) the paper.

Optical constants come from the refractiveindex.info database at the commit the
calculator pins (`multilayer/backend/app.py`, `RI_DATABASE_SHA`), copied into `data/`
(public domain, CC0), and are interpolated linearly in wavelength as the backend does.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
DATA = HERE / "data"
RI_DATABASE_SHA = "a66ef8805cdb200973fc7ae9181587e1d89d14eb"


def _yaml_data(path):
    """The DATA block of a refractiveindex.info file: ('tabulated nk', rows) or ('formula 1', coefficients)."""
    text = path.read_text().splitlines()
    kind, rows, coeffs, inside = None, [], None, False
    for line in text:
        s = line.strip()
        if s.startswith("- type:"):
            kind = s.split(":", 1)[1].strip()
        elif s.startswith("coefficients:"):
            coeffs = [float(v) for v in s.split(":", 1)[1].split()]
        elif s.startswith("data: |"):
            inside = True
        elif inside:
            try:
                row = [float(p) for p in s.split()]
            except ValueError:
                row = []
            if len(row) == 3:
                rows.append(row)
            else:
                inside = False
    return kind, np.array(rows) if rows else coeffs


def material(name, wavelength_um):
    """n + ik of `name` ('Au_Johnson', 'Ag_Johnson', 'Si_Green-2008', 'SiO2_Malitson') at the wavelengths (µm)."""
    kind, data = _yaml_data(DATA / f"{name}.yml")
    lam = np.asarray(wavelength_um, dtype=float)
    if kind == "tabulated nk":
        if lam.min() < data[0, 0] - 1e-12 or lam.max() > data[-1, 0] + 1e-12:
            raise ValueError(f"{name} covers {data[0, 0]}-{data[-1, 0]} µm only")
        return np.interp(lam, data[:, 0], data[:, 1]) + 1j * np.interp(lam, data[:, 0], data[:, 2])
    if kind == "formula 1":  # Sellmeier: n² = 1 + c0 + Σ B λ² / (λ² − C²)
        c = data
        n2 = 1 + c[0] + sum(c[i] * lam**2 / (lam**2 - c[i + 1] ** 2) for i in range(1, len(c), 2))
        return np.sqrt(n2) + 0j
    raise ValueError(f"unsupported refractiveindex.info type {kind!r}")


def drude_nk(wavelength_um, lambda_p_um, gamma=0.0, eps_inf=1.0):
    """n + ik of ε = ε∞ − ωp² / (ω² + iγω), γ in units of ωp, with Im n ≥ 0."""
    x = lambda_p_um / np.asarray(wavelength_um, dtype=float)  # ω / ωp
    eps = eps_inf - 1.0 / (x**2 + 1j * gamma * x)
    n = np.sqrt(eps.astype(complex))
    return np.where(n.imag < 0, -n, n)


def lorentz_nk(wavenumber_cm, eps_inf, delta_eps, omega_to, gamma):
    """n + ik of a single lattice oscillator ε = ε∞ + Δε ω_TO² / (ω_TO² − ω² − iγω) (wavenumbers in cm⁻¹)."""
    w = np.asarray(wavenumber_cm, dtype=float)
    eps = eps_inf + delta_eps * omega_to**2 / (omega_to**2 - w**2 - 1j * gamma * w)
    n = np.sqrt(eps.astype(complex))
    return np.where(n.imag < 0, -n, n)


def nk_rows(wavelength_um, nk):
    """Rows [λ µm, n, k] for a calculator upload (the share token's `nkTable.rows`)."""
    return [[round(float(w), 6), round(float(v.real), 6), round(float(v.imag), 6)] for w, v in zip(wavelength_um, nk)]


def save_csv(name, header, columns):
    np.savetxt(RESULTS / f"{name}.csv", np.column_stack(columns), delimiter=",", header=header, comments="")


def report(name, checks):
    """checks: list of (what, paper value, computed value, tolerance or None, source). Writes
    results/<name>.json and prints one line per check; returns True if every toleranced check passes."""
    out, ok = [], True
    for what, paper, ours, tol, source in checks:
        passed = None if tol is None else bool(abs(ours - paper) <= tol)
        ok = ok and passed is not False
        out.append({"check": what, "paper": paper, "computed": round(float(ours), 4), "tolerance": tol, "pass": passed, "source": source})
        flag = "  " if passed is None else ("ok" if passed else "!!")
        print(f"{flag} {what}: paper {paper}, computed {float(ours):.4g}" + ("" if tol is None else f" (±{tol})") + f"  [{source}]")
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return ok


def peak(x, y, lo=None, hi=None):
    """(x, y) of the maximum of y within [lo, hi]."""
    x, y = np.asarray(x), np.asarray(y)
    m = np.ones_like(x, dtype=bool)
    if lo is not None:
        m &= x >= lo
    if hi is not None:
        m &= x <= hi
    i = np.flatnonzero(m)[np.argmax(y[m])]
    return x[i], y[i]
