"""Shared plumbing for the check scripts: every number a check produces is a ``Row`` with its limit.

A check function returns rows; its script prints them as a Markdown report, and its test asserts that
every row holds, so the observed value and the tolerance live in one place (benchmarks/MATRIX.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Row:
    check: str
    quantity: str
    observed: float
    limit: float
    relation: str = "<="  # "<=", ">=", "==" or "<" (strict)

    @property
    def holds(self) -> bool:
        value = float(self.observed)
        if not np.isfinite(value):
            return False
        return {"<=": value <= self.limit, "<": value < self.limit, ">=": value >= self.limit,
                ">": value > self.limit, "==": value == self.limit}[self.relation]


def row(check, quantity, observed, limit, relation="<="):
    return Row(check, quantity, float(np.max(observed)) if np.ndim(observed) else float(observed), float(limit), relation)


def close(check, what, actual, desired, rtol=0.0, atol=0.0):
    """numpy's assert_allclose as a row: max |actual - desired| / (atol + rtol |desired|) <= 1."""
    actual, desired = np.asarray(actual), np.asarray(desired)
    ratio = np.abs(actual - desired) / (atol + rtol * np.abs(desired))
    bound = " + ".join(t for t in (f"{atol:g}" if atol else "", f"{rtol:g} |b|" if rtol else "") if t)
    return Row(check, f"{what}: max |a - b| / ({bound})", float(np.max(ratio)), 1.0)


def flag(check, quantity, condition):
    """A yes/no requirement as a row (1 = holds)."""
    return Row(check, quantity, 1.0 if bool(condition) else 0.0, 1.0, "==")


def failing(rows):
    return [r for r in rows if not r.holds]


def assert_rows(rows):
    bad = failing(rows)
    assert not bad, "\n".join(f"{r.check}: {r.quantity} = {r.observed!r}, needs {r.relation} {r.limit!r}" for r in bad)


def _format(value, relation):
    if relation == "==" and value in (0.0, 1.0):
        return "yes" if value == 1.0 else "no"
    if value == 0 or 1e-3 <= abs(value) < 1e4:
        return f"{value:.6g}"
    return f"{value:.2e}"


def report(title, intro, rows):
    print(f"# {title}\n")
    print(intro.strip() + "\n")
    print("| check | quantity | observed | asserted | holds |")
    print("|---|---|---|---|---|")
    for r in rows:
        limit = "yes" if r.relation == "==" and r.limit == 1.0 and r.observed in (0.0, 1.0) else f"{r.relation} {_format(r.limit, '')}"
        cells = (r.check, r.quantity, _format(r.observed, r.relation), limit, "yes" if r.holds else "**NO**")
        print("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
    bad = failing(rows)
    print(f"\n{len(rows)} rows, {len(rows) - len(bad)} hold" + (f"; **{len(bad)} fail**." if bad else "."))
