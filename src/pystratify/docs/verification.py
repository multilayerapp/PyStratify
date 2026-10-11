"""benchmarks/MATRIX.md as manifest records: one `benchmark` record per row of its geometry tables.

MATRIX.md stays the one place the benchmark matrix is written; this module only reads it. Its legends (kinds of
reference, statuses), the coverage grid and the closing findings travel with the records, and docs/verification.md
adds the page's introduction and which rows open a worked example.
"""

import re

GEOMETRY_HEADINGS = {"Films": "films", "Cylinders": "cylinders", "Spheres": "spheres"}
SOURCE_HEADINGS = {"Plane wave": "plane_wave", "Point dipole": "dipole", "Focused beam": "focused"}
ROW_FIELDS = ("reference", "kind", "quantity", "script", "tolerance", "status")
# The record's own `kind` is "benchmark"; the matrix's kind column (analytic, code, ...) becomes `reference_kind`.
RECORD_FIELDS = ("reference", "reference_kind", "quantity", "script", "tolerance", "status")
PIPE = re.compile(r"(?<!\\)\|")


class MatrixError(ValueError):
    """benchmarks/MATRIX.md is not in the shape the page is built from."""


def _cells(line):
    parts = PIPE.split(line.strip())
    if len(parts) < 3 or parts[0] or parts[-1]:
        raise MatrixError(f"not a table row: {line!r}")
    return [cell.strip().replace("\\|", "|") for cell in parts[1:-1]]


def _tables(lines):
    """[(heading path, header cells, rows)] for every Markdown table, under the headings that precede it."""
    tables, heading, i = [], {}, 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^(#{2,3}) (.+)$", line)
        if match:
            level = len(match.group(1))
            heading = {k: v for k, v in heading.items() if k < level}
            heading[level] = match.group(2).strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s|:-]+\|$", lines[i + 1].strip()):
            header = _cells(line)
            rows, i = [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            tables.append((dict(heading), header, rows))
            continue
        i += 1
    return tables


def _section(lines, title):
    """The Markdown under `## title`, up to the next level-2 heading."""
    out, inside = [], False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip() == title
            continue
        if inside:
            out.append(line)
    return "\n".join(out).strip()


def parse_matrix(text):
    """The matrix as {records, kinds, statuses, coverage, findings}; records in reading order."""
    lines = text.splitlines()
    records, kinds, statuses, coverage = [], None, None, None
    for heading, header, rows in _tables(lines):
        top = heading.get(2)
        if top == "Kinds of reference":
            kinds = [{"kind": r[0], "proves": r[1]} for r in rows]
        elif top == "Status":
            statuses = [{"status": r[0], "meaning": r[1]} for r in rows]
        elif top == "Coverage at a glance":
            coverage = {"columns": header, "rows": rows}
        elif top in GEOMETRY_HEADINGS:
            if tuple(h.lower() for h in header) != ROW_FIELDS:
                raise MatrixError(f"{top} / {heading.get(3)}: columns {header}, expected {ROW_FIELDS}")
            geometry, source = GEOMETRY_HEADINGS[top], SOURCE_HEADINGS.get(heading.get(3))
            if source is None:
                raise MatrixError(f"{top}: unknown source heading {heading.get(3)!r}")
            for n, row in enumerate(rows, start=1):
                if len(row) != len(ROW_FIELDS):
                    raise MatrixError(f"{top} / {heading.get(3)} row {n}: {len(row)} cells")
                records.append({"id": f"{geometry}-{source.replace('_', '-')}-{n}", "kind": "benchmark",
                                "geometry": geometry, "source": source, **dict(zip(RECORD_FIELDS, row))})
    if kinds is None or statuses is None or coverage is None:
        raise MatrixError("MATRIX.md lost its kinds, status or coverage table")
    return {"records": records, "kinds": kinds, "statuses": statuses, "coverage": coverage,
            "findings": _section(lines, "What the matrix shows")}


def table_rows(text):
    """How many rows the geometry tables of MATRIX.md hold, counted without parsing them (for the tests)."""
    count, inside = 0, False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line[3:].strip() in GEOMETRY_HEADINGS
        elif inside and line.startswith("| ") and not line.startswith("| reference |") and not line.startswith("|---"):
            count += 1
    return count
