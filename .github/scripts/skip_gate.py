#!/usr/bin/env python3
"""Explicit-skip gate (assurance WP09, task 7): a skipped test passes CI only when a reviewed list names it.

Shared by the multilayerapp repositories as .github/scripts/skip_gate.py (assurance WP09, applied 2026-11-05);
each reads its own repository's .github/expected-skips*.txt.

    python -m pytest -q -rs --junitxml=report.xml ...      # pytest exits 0 with skips; this step decides
    python .github/scripts/skip_gate.py report.xml .github/expected-skips.txt

expected-skips.txt, one rule per line, three TAB-separated fields (lines starting with # are comments):

    <test id glob>  <TAB>  <reason regex>  <TAB>  <where the skipped check runs instead>

The test id is matched in each of the forms pytest's JUnit XML can give it: "classname::name",
"classname.name" and "name" (a module skipped at collection is reported under its module name).
The third field is mandatory: a skip is explicit only when it says which tier carries the check
(the CI benchmarks job, the local reference environments, the release machine).

Fails (exit 1) when
  * a skip matches no rule (a reference environment went missing silently);
  * the report holds no test case at all (a collection failure that pytest's exit code hid);
  * pytest recorded an error or failure (belt and braces; the pytest step fails first).
Expected failures (xfail) are listed in the summary, not judged: they are declared in the code.
Rules no skip used are listed as stale, without failing (an environment that appeared is good news).
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def load_rules(path: Path) -> list[tuple[str, re.Pattern[str], str, int]]:
    rules = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not all(f.strip() for f in fields):
            sys.exit(f"{path}:{number}: a rule needs three TAB-separated fields: glob, reason regex, where it runs instead")
        rules.append((fields[0].strip(), re.compile(fields[1].strip()), fields[2].strip(), number))
    return rules


def cases(report: Path):
    root = ET.parse(report).getroot()
    for case in root.iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        ids = [f"{classname}::{name}" if classname else name, f"{classname}.{name}".lstrip("."), name]
        outcome, message = "passed", ""
        for child in case:
            if child.tag in ("failure", "error"):
                outcome, message = child.tag, child.get("message", "")
            elif child.tag == "skipped":
                kind = child.get("type", "")
                outcome = "xfail" if "xfail" in kind else "skipped"
                # A skip at collection (an `importorskip` at module top) carries the generic message "collection
                # skipped" and its reason in the element text; read both, or no rule ever matches it (WP13 task 8).
                message = " ".join(part for part in (child.get("message", ""), (child.text or "").strip()) if part)
        yield ids, outcome, message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--summary", type=Path, help="append a Markdown table here (e.g. $GITHUB_STEP_SUMMARY)")
    args = parser.parse_args()

    rules = load_rules(args.expected)
    used: set[int] = set()
    total, unexplained, explicit, xfails, broken = 0, [], [], [], []
    for ids, outcome, message in cases(args.report):
        total += 1
        if outcome in ("failure", "error"):
            broken.append((ids[0], message))
        elif outcome == "xfail":
            xfails.append((ids[0], message))
        elif outcome == "skipped":
            for glob, reason, where, number in rules:
                if any(fnmatch.fnmatchcase(i, glob) for i in ids) and reason.search(message):
                    used.add(number)
                    explicit.append((ids[0], message, where))
                    break
            else:
                unexplained.append((ids[0], message))

    lines = [f"### Skips ({args.report.name}): {total} test cases, {len(explicit)} explicit skips, "
             f"{len(unexplained)} unexplained, {len(xfails)} xfail", "",
             "| test | reason | runs instead |", "|---|---|---|"]
    lines += [f"| `{i}` | {m[:120]} | {w} |" for i, m, w in explicit]
    lines += [f"| `{i}` | {m[:120]} | **UNEXPLAINED: no rule names it** |" for i, m in unexplained]
    stale = [f"line {n}: `{g}`" for g, _, _, n in rules if n not in used]
    if stale:
        lines += ["", "Rules no skip used (stale, not a failure): " + "; ".join(stale)]
    if xfails:
        lines += ["", "Expected failures: " + "; ".join(f"`{i}` ({m[:80]})" for i, m in xfails)]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(text)

    if total == 0:
        print("FAIL: the report holds no test case", file=sys.stderr)
        return 1
    if broken:
        print(f"FAIL: {len(broken)} failed or errored test case(s)", file=sys.stderr)
        return 1
    if unexplained:
        print(f"FAIL: {len(unexplained)} skip(s) no rule in {args.expected} names; a missing reference "
              "environment is not a pass. Install it, or add a rule saying where the check runs instead.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
