"""docs/API.md places every exported name in exactly one list: frozen, legacy or internal.

A name added to ``__all__`` without a place in the contract, or listed twice, fails here, so the
1.0 surface cannot grow by accident.
"""

import re
from pathlib import Path

import pystratify

API = Path(__file__).resolve().parents[1] / "docs" / "API.md"
LISTS = ("Frozen", "Legacy", "Internal")


def contract():
    sections, current = {name: [] for name in LISTS}, None
    for line in API.read_text().splitlines():
        heading = re.match(r"## (\w+)", line)
        if heading:
            current = heading.group(1) if heading.group(1) in LISTS else None
            continue
        row = re.match(r"\| `([A-Za-z_][A-Za-z0-9_]*)` \|", line)
        if current and row:
            sections[current].append(row.group(1))
    return sections


def test_every_export_has_exactly_one_place():
    sections = contract()
    placed = [name for names in sections.values() for name in names]
    assert sorted(set(placed)) == sorted(placed), "a name is listed twice"
    exported = set(pystratify.__all__)
    assert set(placed) - exported == set(), "API.md lists a name that is not exported"
    assert exported - set(placed) == set(), "an exported name has no place in API.md"

