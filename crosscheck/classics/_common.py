"""Shared inputs for the classic-paper recomputations (roadmap WP1.9, `showcase.md`).

Each script in this folder recomputes the headline figure of one classic paper with the
inputs the multilayer.app calculator receives from the paper's share link, and writes a
CSV, a PNG, the checks against numbers printed in (or read off) the paper (`report`), and
the reproduction figure its Learn card draws (`figure`): the computed curves at plotting
resolution, with the paper's values marked where its checks put them.

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


def png(name):
    """results/<name>.png (built from RESULTS, not from __file__, which may sit under any folder name)."""
    return RESULTS / f"{name}.png"


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
    REPORTED[name] = [c["check"] for c in out]
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


# --- the reproduction figure (R-CLASSICS-CONTENT, Ilia 2026-11-05) ---------------------------------------------------
# What a classic's card on multilayer.app draws: PyStratify's curves at plotting resolution and, as marks, the values
# the paper prints or plots, each mark naming the check that compares it. `python -m pystratify.docs build` copies
# results/<name>.figure.json to docs/classics/figures/<id>.json and places every mark at its check's paper value.

REPORTED = {}  # name -> the check names report() wrote, so a mark can only name a check that exists
FIGURE_POINTS = 240  # a panel keeps about this many points of its grid, plus every curve's extremes between them


def _rounded(values, digits):
    return [float(f"{v:.{digits}g}") if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


def _plotting_indices(n, curves, target):
    """Indices of an even subset of the grid, about `target` points, plus each curve's maximum and minimum between
    two kept points: a decimated curve keeps every peak and dip (a needle resonance included) at its true height."""
    if n <= 1.25 * target:
        return np.arange(n)
    step = int(np.ceil((n - 1) / target))
    edges = list(range(0, n - 1, step)) + [n - 1]
    keep = set(edges)
    for y in curves:
        y = np.asarray(y, dtype=float)
        for a, b in zip(edges[:-1], edges[1:]):
            seg = y[a:b + 1]
            if np.isfinite(seg).any():
                keep.update((a + int(np.nanargmax(seg)), a + int(np.nanargmin(seg))))
    return np.array(sorted(keep))


def curve(key, label, y, quantity=None, group=None, dash=False, guide=False):
    """One plotted curve. `quantity` names the calculator quantity it is (q_ext, md_sca, R, gamma_tot, ...), so the
    card draws it in the calculator's colour and stroke; curves of one `group` share a colour; `dash` dashes it;
    `guide` draws a reference line (an asymptote, a level) in the neutral colour, dotted."""
    out = {"key": key, "label": label, "y": y}
    out.update({k: v for k, v in (("quantity", quantity), ("group", group)) if v})
    out.update({k: True for k, v in (("dash", dash), ("guide", guide)) if v})
    return out


def mark(check, series=None, at=None, label=None, log10=False):
    """Where the card draws the paper's value of a check (named as report() names it). Without `at`, a vertical rule
    at that value (the position of a peak, a dip or a crossing); `at=x`, a point at (x, paper value); `at=(lo, hi)`,
    the paper's value as a level across that range (a band's maximum or minimum). `series` ties it to a curve (its
    colour); `label` is a short name drawn beside a rule; `log10` when the check is the log of the plotted quantity."""
    out = {"check": check}
    if series:
        out["series"] = series
    if at is not None:
        out["at"] = [float(at[0]), float(at[1])] if isinstance(at, (tuple, list)) else float(at)
    if label:
        out["label"] = label
    if log10:
        out["log10"] = True
    return out


def panel(id, title, x, x_label, y_label, curves, marks=(), log=False, points=FIGURE_POINTS):
    """One panel of a reproduction figure: curves on a common x grid, decimated to plotting resolution (x to seven
    significant digits, y to five). Axis labels follow the calculator's: a symbol, then the unit in brackets."""
    x = np.asarray(x, dtype=float)
    for c in curves:
        if np.shape(c["y"]) != x.shape:
            raise ValueError(f"panel {id}: curve {c['key']} has {np.shape(c['y'])} values for {x.size} x")
    keep = _plotting_indices(x.size, [c["y"] for c in curves], points)
    return {"id": id, "title": title, "x_label": x_label, "y_label": y_label, "log": bool(log), "x": _rounded(x[keep], 7),
            "series": [dict(c, y=_rounded(np.asarray(c["y"], dtype=float)[keep], 5)) for c in curves], "marks": list(marks)}


def figure(name, title, panels, note="", default=0):
    """results/<name>.figure.json, written after report(name, ...): every mark must name one of its checks."""
    names = REPORTED.get(name)
    if names is None:
        raise ValueError(f"{name}: call report() before figure()")
    for p in panels:
        for m in p["marks"]:
            if names.count(m["check"]) != 1:
                raise ValueError(f"{name}: a mark names {m['check']!r}, which is not exactly one reported check")
    out = {"script": f"{name}.py", "title": title, "note": note, "default": default, "panels": panels}
    (RESULTS / f"{name}.figure.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
