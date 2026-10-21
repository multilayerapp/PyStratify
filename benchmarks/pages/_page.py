"""What every page script of benchmarks/pages/ shares: its figures' registry, the curve format multilayer.app draws,
and the run that prints each figure's RMSE and writes <page>.json and <page>.png beside the script.

A page script imports this module first (it puts the clone's src/ and benchmarks/ on the path), declares its figures
with `@figure(name, title, x_label, y_label, log=False, env="bench")`, returns each one as `result(x, curves, ...)` and
ends with `run(__file__)`; benchmarks/figures.py imports the same scripts and writes each record to
docs/figures/<name>.json, so the site draws exactly what a reader's run computes.

PyStratify is drawn as `line`s on the dense grid x. Another code, or a closed form, is drawn as `dots`, one circle per
value it computed: on a sparse grid x[start::every] of at most MAX_DOTS points, never thinned for display, so every
circle on the site is a value the reference returned.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "benchmarks"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

MAX_DOTS = 41
FIGURES = {}  # module name -> [(name, title, x label, y label, log, environment, function)]


def plot_name(title):
    """The name of the function computing a figure: plot_ and its title on the page, lower case, "_" between words,
    Greek letters spelled out ("Bare substrates: Ψ" -> plot_bare_substrates_psi)."""
    text = title.replace("'", "").replace("’", "")
    text = "".join(f" {unicodedata.name(c).split()[-1]} " if unicodedata.name(c, "").startswith("GREEK") else c
                   for c in text)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return "plot_" + re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


def figure(name, title, x_label, y_label, log=False, env="bench"):
    def register(fn):
        if fn.__name__ != plot_name(title):
            raise NameError(f"{fn.__name__} computes {title!r}: name it {plot_name(title)}")
        FIGURES.setdefault(fn.__module__, []).append((name, title, x_label, y_label, log, env, fn))
        return fn
    return register


def series(key, label, y, role):
    """A curve; NaN (no value at that point) is written as null."""
    return {"key": key, "label": label, "y": [float(v) if np.isfinite(v) else None for v in np.asarray(y, float)],
            "role": role}


def line(key, label, y):
    """PyStratify's curve, at every point of the figure's grid."""
    return series(key, label, y, "pystratify")


def dots(key, label, y, every=1, start=0, size=None):
    """A reference's values, one circle each. With `every` > 1 (or `start` > 0), y holds the values at x[start::every]
    of an x of `size` points, and the points between are null."""
    return series(key, label, spread(y, every, start, size) if every > 1 or start else y, "reference")


def spread(values, every, start=0, size=None):
    """values placed at [start::every] of a grid of `size` points, NaN between."""
    out = np.full(size, np.nan)
    out[start::every] = values
    return out


def largest(ours, theirs, relative=False):
    """The largest |ours − theirs| over the points both have, relative to the largest |theirs| if asked."""
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    diff = np.nanmax(np.abs(ours - theirs))
    return float(diff / np.nanmax(np.abs(theirs))) if relative else float(diff)


def sig(values, digits=7):
    return [None if v is None else float(f"{v:.{digits}g}") for v in values]


def result(x, curves, worst, worst_label, note):
    """A figure's data: its grid, its curves (`line`s and `dots`), the largest difference and what it measures, and a
    note for the report."""
    return dict(x=sig(x), series=curves, worst=float(worst), worst_label=worst_label, note=note)


def rmse(series_list):
    """The root-mean-square difference between each curve and its reference (`<key>_ref`), over every point both have;
    None for a figure without such pairs."""
    by_key = {s["key"]: s["y"] for s in series_list}
    diffs = [a - b for s in series_list if s["role"] != "reference" and s["key"] + "_ref" in by_key
             for a, b in zip(s["y"], by_key[s["key"] + "_ref"]) if a is not None and b is not None]
    return float(np.sqrt(np.mean(np.square(diffs)))) if diffs else None


def record(name, title, x_label, y_label, fn, log=False, script=None):
    """One figure as the page stores it: every value in full double precision, and the RMSE between the codes."""
    out = {"id": name, "title": title, "x_label": x_label, "y_label": y_label, "log": bool(log), **fn(),
           "script": script, "function": fn.__name__}
    out["x"] = [float(v) for v in out["x"]]
    for s in out["series"]:
        if len(s["y"]) != len(out["x"]):
            raise ValueError(f"{name}: {s['key']} has {len(s['y'])} points on a grid of {len(out['x'])}")
        dots_drawn = sum(v is not None for v in s["y"])
        if s["role"] == "reference" and dots_drawn > MAX_DOTS:
            raise ValueError(f"{name}: {s['key']} draws {dots_drawn} circles, more than {MAX_DOTS}; sample it")
    out["rmse"] = rmse(out["series"])
    return out


def plot(records, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(3, len(records))
    rows = -(-len(records) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 4.6 * rows), constrained_layout=True,
                             squeeze=False)
    for ax in axes.flat[len(records):]:
        ax.axis("off")
    for ax, r in zip(axes.flat, records):
        lines = [s for s in r["series"] if s["role"] != "reference"]
        for i, s in enumerate(lines):
            color = f"C{i}"
            ax.plot(r["x"], [np.nan if v is None else v for v in s["y"]], color=color, lw=1.5, label=s["label"])
            for ref in (t for t in r["series"] if t["role"] == "reference" and t["key"].startswith(s["key"] + "_")):
                y = np.array([np.nan if v is None else v for v in ref["y"]])
                ax.plot(np.asarray(r["x"])[np.isfinite(y)], y[np.isfinite(y)], "o", mfc="none", color=color, ms=4,
                        label=ref["label"])
        if r["log"]:
            ax.set_yscale("log")
        ax.set(xlabel=r["x_label"], ylabel=r["y_label"])
        ax.set_title(r["title"] + (f"\nRMSE {r['rmse']:.2e}" if r["rmse"] is not None else ""), fontsize=9)
        ax.legend(fontsize=6, ncol=2)
    fig.savefig(path, dpi=120)


def run(script_file):
    """Compute every figure the script declares, print each RMSE, write <page>.json and <page>.png beside it."""
    path = Path(script_file).resolve()
    script = f"benchmarks/pages/{path.name}"
    records = []
    for name, title, x_label, y_label, log, _, fn in FIGURES["__main__"]:
        r = record(name, title, x_label, y_label, fn, log, script)
        records.append(r)
        print(f"{name}: {len(r['x'])} points" + (f", RMSE {r['rmse']:.2e}" if r["rmse"] is not None else ""),
              flush=True)
    (path.parent / f"{path.stem}.json").write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {path.stem}.json")
    try:
        plot(records, path.parent / f"{path.stem}.png")
        print(f"wrote {path.stem}.png")
    except ImportError:
        print("matplotlib is not installed: no plot (pip install matplotlib)", file=sys.stderr)
