"""What every page script of benchmarks/pages/ shares: its figures' registry, the curve format multilayer.app draws,
and the run that prints each figure's RMSE and writes <page>.json and <page>.png beside the script.

A page script declares its figures with `@figure(name, title, x_label, y_label, log=False, env="bench")` and ends with
`run(__file__)`; benchmarks/figures.py imports the same scripts and writes each record to docs/figures/<name>.json, so
the site draws exactly what a reader's run computes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FIGURES = {}  # module name -> [(name, title, x label, y label, log, environment, function)]


def figure(name, title, x_label, y_label, log=False, env="bench"):
    def register(fn):
        FIGURES.setdefault(fn.__module__, []).append((name, title, x_label, y_label, log, env, fn))
        return fn
    return register


def series(key, label, y, role):
    """A curve; NaN (a reference computed at only some of the points) is written as null."""
    return {"key": key, "label": label, "y": [float(v) if np.isfinite(v) else None for v in np.asarray(y, float)], "role": role}


def worst(ours, theirs, relative=False):
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    diff = np.max(np.abs(ours - theirs))
    return float(diff / np.max(np.abs(theirs))) if relative else float(diff)


def sig(values, digits=7):
    return [None if v is None else float(f"{v:.{digits}g}") for v in values]


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
    out["rmse"] = rmse(out["series"])
    return out


def plot(records, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(3, len(records))
    rows = -(-len(records) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 4.6 * rows), constrained_layout=True, squeeze=False)
    for ax in axes.flat[len(records):]:
        ax.axis("off")
    for ax, r in zip(axes.flat, records):
        lines = [s for s in r["series"] if s["role"] != "reference"]
        for i, s in enumerate(lines):
            color = f"C{i}"
            ax.plot(r["x"], [np.nan if v is None else v for v in s["y"]], color=color, lw=1.5, label=s["label"])
            ref = next((t for t in r["series"] if t["key"] == s["key"] + "_ref"), None)
            if ref:
                ax.plot(r["x"], [np.nan if v is None else v for v in ref["y"]], "o", mfc="none", color=color, ms=4, label=ref["label"])
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
        print(f"{name}: {len(r['x'])} points" + (f", RMSE {r['rmse']:.2e}" if r["rmse"] is not None else ""), flush=True)
    (path.parent / f"{path.stem}.json").write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {path.stem}.json")
    try:
        plot(records, path.parent / f"{path.stem}.png")
        print(f"wrote {path.stem}.png")
    except ImportError:
        print("matplotlib is not installed: no plot (pip install matplotlib)", file=sys.stderr)
