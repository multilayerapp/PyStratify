"""python -m pystratify.docs build [--root .] [--run-classics] [--check]
python -m pystratify.docs review SECTION-ID [--date YYYY-MM-DD] [--root .]"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

from .build import ContentError, _front_matter, build_manifest, review_fingerprint, stale, write_manifest


def stamp_review(root, section_id, date):
    """Write Ilia's review into a Learn section's front matter: `reviewed: <date>` and the fingerprint of what he read
    (`review_fingerprint`), replacing any earlier stamp. The page says it was reviewed until it is next edited."""
    for path in sorted((root / "docs" / "learn").glob("*/*/*.md")):
        meta, _ = _front_matter(path.read_text(), str(path))
        if meta.get("id") != section_id:
            continue
        text = path.read_text()
        if not re.search(r"^reviewed:.*$", text, flags=re.M):
            raise ContentError(f"{path}: no `reviewed:` line in the front matter to stamp")
        text = re.sub(r"^reviewed_fingerprint:.*\n", "", text, flags=re.M)
        fingerprint = review_fingerprint(root, path)
        text = re.sub(r"^reviewed:.*$", f'reviewed: {date}\nreviewed_fingerprint: "{fingerprint}"', text, count=1, flags=re.M)
        path.write_text(text)
        return path, fingerprint
    raise ContentError(f"no Learn section {section_id!r} under docs/learn/")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pystratify.docs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="compute every example and write docs/manifest.json")
    build.add_argument("--root", type=Path, default=Path.cwd(), help="the repository root (holds docs/ and crosscheck/)")
    build.add_argument("--run-classics", action="store_true",
                       help="re-run every crosscheck/classics script first (minutes); otherwise their results/ must exist")
    build.add_argument("--check", action="store_true", help="exit 1 if docs/manifest.json is not what a build writes")
    review = sub.add_parser("review", help="stamp a Learn section as reviewed: the date and a fingerprint of its text")
    review.add_argument("section", help="the section's id, e.g. fresnel-equations")
    review.add_argument("--date", default=datetime.date.today().isoformat(), help="the review's date (default: today)")
    review.add_argument("--root", type=Path, default=Path.cwd(), help="the repository root (holds docs/)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if args.command == "review":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
            parser.error("--date is YYYY-MM-DD")
        path, fingerprint = stamp_review(root, args.section, args.date)
        print(f"stamped {path.relative_to(root)}: reviewed {args.date}, fingerprint {fingerprint}; "
              "now run `python -m pystratify.docs build`")
        return 0
    target = root / "docs" / "manifest.json"
    # A fresh clone has no crosscheck/classics/results/ (git-ignored): the committed manifest's checks stand in for a
    # classic whose results file is absent, as they do in the tests. Without this a clean checkout could not build.
    previous = json.loads(target.read_text()) if target.exists() else None
    manifest = build_manifest(root, run_classics=args.run_classics, previous=previous)
    if args.check:
        changed = stale(manifest, target)
        if changed:
            print("stale: run `python -m pystratify.docs build`: " + ", ".join(changed), file=sys.stderr)
            return 1
        print(f"{target} and docs/sweeps/ are current")
        return 0
    write_manifest(manifest, target)
    counts = {kind: sum(r["kind"] == kind for r in manifest["records"]) for kind in ("learn", "classic", "benchmark")}
    sweeps = sum(1 for e in manifest["examples"].values() if e.get("sweep"))
    print(f"wrote {target}: {counts['learn']} learn sections, {counts['classic']} classics, {counts['benchmark']} benchmarks, "
          f"{len(manifest['examples'])} examples, {sweeps} precomputed sweeps in docs/sweeps/")
    learn = [r for r in manifest["records"] if r["kind"] == "learn"]
    edited = [r["id"] for r in learn if r["reviewed"] and not r["review_current"]]
    print(f"learn: {sum(1 for r in learn if r['review_current'])} of {len(learn)} sections reviewed as they read now"
          + (f"; dated but edited since, or never stamped (`python -m pystratify.docs review <id>`): {', '.join(edited)}"
             if edited else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
