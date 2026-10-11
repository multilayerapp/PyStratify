"""python -m pystratify.docs build [--root .] [--run-classics] [--check]"""

import argparse
import sys
from pathlib import Path

from .build import build_manifest, dumps, write_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pystratify.docs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="compute every example and write docs/manifest.json")
    build.add_argument("--root", type=Path, default=Path.cwd(), help="the repository root (holds docs/ and crosscheck/)")
    build.add_argument("--run-classics", action="store_true",
                       help="re-run every crosscheck/classics script first (minutes); otherwise their results/ must exist")
    build.add_argument("--check", action="store_true", help="exit 1 if docs/manifest.json is not what a build writes")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    manifest = build_manifest(root, run_classics=args.run_classics)
    target = root / "docs" / "manifest.json"
    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != dumps(manifest):
            print(f"{target} is stale: run `python -m pystratify.docs build`", file=sys.stderr)
            return 1
        print(f"{target} is current")
        return 0
    write_manifest(manifest, target)
    counts = {kind: sum(r["kind"] == kind for r in manifest["records"]) for kind in ("learn", "classic", "benchmark")}
    print(f"wrote {target}: {counts['learn']} learn sections, {counts['classic']} classics, {counts['benchmark']} benchmarks, "
          f"{len(manifest['examples'])} examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
