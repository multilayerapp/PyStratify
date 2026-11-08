"""The timing tables of README.md and docs/NUMERICS.md quote a recorded run: benchmarks/TIMINGS.md.

benchmarks/timings.py writes the report (date, machine, versions, the median of each task); every row of both tables
must be one of its rows, quoting its measured value, under an intro that names the report and the date of the run.
Nothing is timed here: a rerun that moves a number fails this test until the tables quote the new report.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = {"README": ROOT / "README.md", "NUMERICS": ROOT / "docs" / "NUMERICS.md"}


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _report():
    text = (ROOT / "benchmarks" / "TIMINGS.md").read_text()
    date = re.search(r"on (\d{4}-\d{2}-\d{2})", text).group(1)
    measured = {}
    for line in text.split("## The tables' rows", 1)[1].splitlines():
        if line.startswith(tuple(f"| {doc} |" for doc in TABLES)):
            document, row, _stated, value, _ratio = _cells(line)
            measured[(document, row)] = value
    return date, measured


def _table(path):
    """The intro line and the {row: time} of the table that follows the line naming TIMINGS.md."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "benchmarks/TIMINGS.md" in line and not line.startswith("|"))
    rows, i = {}, start + 1
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith("|")):
        if lines[i].startswith("|") and not set(lines[i]) <= set("|-"):
            task, value = _cells(lines[i])
            rows[task] = value
        if rows and not lines[i].strip():
            break
        i += 1
    rows.pop("task", None)  # the header
    return lines[start], rows


def test_the_report_is_not_a_quick_run():
    assert "--quick" not in (ROOT / "benchmarks" / "TIMINGS.md").read_text()


def test_the_tables_quote_the_timings_report():
    date, measured = _report()
    assert measured, "benchmarks/TIMINGS.md lists no table rows"
    for document, path in TABLES.items():
        intro, rows = _table(path)
        assert date in intro, f"{path.name}: the intro names no run of {date}"
        quoted = {(document, task): value for task, value in rows.items()}
        expected = {key for key in measured if key[0] == document}
        assert set(quoted) == expected, f"{path.name}: rows differ from the report"
        for key, value in quoted.items():
            assert value == measured[key], f"{path.name}: '{key[1]}' quotes {value}, the report has {measured[key]}"
