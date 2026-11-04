"""The refractiveindex.info page evaluator against its two predecessors, on EVERY page of a database.

    pip install -e ".[benchmarks]"          # PyYAML and refractiveindex==1.0.2 (a pinned reference only)
    python benchmarks/refractiveindex_pages.py [--database DIR] [--expect-sha256 HEX] > REFRACTIVEINDEX_PAGES.md

`pystratify.rii` evaluates a refractiveindex.info page. Two other evaluators had been doing that job, and both are kept
as references after multilayer.app and the replication-corpus harvester call PyStratify (assurance WP07-D2):

* **the multilayer.app API's evaluator**, frozen verbatim in `tests/oracles/api_rii.py` (repository `multilayer`,
  commit ef9c946): what `rii` was ported from, so the agreement is a regression check of the port, over all pages;
* **the pip `refractiveindex` package, 1.0.2**, which the harvester used: independent code that reads the page its
  own way (the last entry of each kind wins; no coverage test; a dropped k reads as 0), so it is the real check, and
  it differs on three pages the report names.

Method (that of the assurance programme's `wp07_rii_evaluators.py`): every page of the catalogue is evaluated at
POINTS wavelengths, log-spaced inside the range of its n data inset 0.5 % from each end, by the three evaluators.
A page where the API refuses a wavelength (no n, or k entries that do not cover it) is a *refusal*, and `rii` must
refuse the same wavelengths. The database is identified first (`rii.database_identity`) and, with
``--expect-sha256``, refused if it is not the one the report is about.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from checks import report, row  # noqa: E402
from pystratify import rii  # noqa: E402

POINTS = 25
INSET = 0.005
TOL_API = 1e-12  # rii vs the frozen API evaluator, max |dn|, |dk| (relative to n, to 1 for k)
TOL_PACKAGE = 1e-6  # a page counts as disagreeing with the package above this
PINNED_SHA256 = "f546d53a0da54a7bac9fb09a9afa17f719e0267d090bb1ebe106f2967a3f2c36"  # the API's snapshot of a66ef88
#: pages where the package is known to differ from the API (and `rii`), with the reason
KNOWN_PACKAGE_DIFFERENCES = {
    "main/CuO/Brimhall": "tabulated nk repeats three wavelengths with different values (rows unsorted): the sort "
    "order within ties, and so the interpolation through them, differs",
    "organic/polyvinylpyrrolidone/Konig": "a formula 5 and a tabulated nk on one page: the API takes n from the "
    "first covering entry (the formula), the package from the last (the table)",
    "3d/plastics/pvp": "the same page file as organic/polyvinylpyrrolidone/Konig",
    "main/AgGaSe2/Boyd-o": "formula 2 with an even number of coefficients (a trailing unpaired one): the package's "
    "loop reads one past the end and raises IndexError (the harvester's fallback then reads tabulated nk only and "
    "refuses the page), the API's loop ignores the trailing coefficient",
}
for _page in rii.CATALOG_MISROUTES:  # the pinned catalogue sends these pages to another book's file; the package follows it
    KNOWN_PACKAGE_DIFFERENCES[_page] = "the pinned catalogue misroutes this page to another file; the package follows " \
        "the catalogue, the API (and `rii`) read the page's own file (`rii.CATALOG_MISROUTES`)"


def _oracle():
    from tests.oracles import api_rii

    return api_rii


def _package(database_root):
    from refractiveindex import RefractiveIndexMaterial

    def evaluate(shelf, book, page, wl_um):
        material = RefractiveIndexMaterial(shelf=shelf, book=book, page=page, db_path=database_root, auto_download=False)
        n = np.array([float(material.get_refractive_index(float(w), unit="um")) for w in wl_um])
        try:
            k = np.array([float(material.get_extinction_coefficient(float(w), unit="um")) for w in wl_um])
        except Exception:  # noqa: BLE001 - the package raises NoExtinctionCoefficient; the harvester read that as 0
            k = np.zeros(len(wl_um))
        return n + 1j * k

    return evaluate


def compare(database, expect_sha256=None, points=POINTS, with_package=True):
    """The per-page comparison; returns a dict of results (see ``rows`` and ``sections``)."""
    import yaml

    db = rii.RIIDatabase(database, expected_sha256=expect_sha256)
    api = _oracle()
    package = _package(db.root) if with_package else None
    pages, bad = [], []
    for page_id, data_path in db.catalog().items():
        entry = {"id": page_id, "data": data_path}
        pages.append(entry)
        try:
            record = yaml.safe_load((db.root / "data" / data_path).read_text(encoding="utf-8"))
            prepared = api.prepare_material_record(yaml.safe_load((db.root / "data" / data_path).read_text("utf-8")))
            page = rii.RIIPage.from_record(record, page_id)
            ranges = api.get_material_ranges(prepared)["n"]
        except Exception as error:  # noqa: BLE001 - recorded
            entry["load_error"] = f"{type(error).__name__}: {error}"[:160]
            bad.append(entry)
            continue
        entry["kinds"] = sorted({str(e.get("type")) for e in prepared.get("DATA", [])})
        if not ranges:
            entry["no_n"] = True
            for evaluator in (lambda w: api.material_index_at(prepared, w, page_id), lambda w: page.nk(w)):
                try:
                    evaluator(0.5)
                    entry["refusal_mismatch"] = True
                except Exception:  # noqa: BLE001
                    pass
            continue
        lo, hi = min(r[0] for r in ranges) * (1 + INSET), max(r[1] for r in ranges) * (1 - INSET)
        wl = np.geomspace(lo, hi, points) if lo > 0 else np.linspace(lo, hi, points)
        reference, refused = np.full(points, np.nan, dtype=complex), np.zeros(points, dtype=bool)
        for i, w in enumerate(wl):
            try:
                reference[i] = complex(*api.material_index_at(prepared, float(w), page_id))
            except Exception:  # noqa: BLE001 - the API refuses it
                refused[i] = True
        ours_refused = np.zeros(points, dtype=bool)
        for i, w in enumerate(wl):
            try:
                page.nk(float(w))
            except ValueError:
                ours_refused[i] = True
        entry["api_refused"] = int(refused.sum())
        if (refused != ours_refused).any():
            entry["refusal_mismatch"] = True
        if refused.all():
            continue
        if not refused.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ours = page.nk(wl)
            entry["dn"] = float(np.max(np.abs(ours.real - reference.real) / reference.real))
            entry["dk"] = float(np.max(np.abs(ours.imag - reference.imag)))
            if package is not None:
                try:
                    theirs = package(*_ids(page_id), wl)
                    entry["pkg_dn"] = float(np.max(np.abs(theirs.real - reference.real)))
                    entry["pkg_dk"] = float(np.max(np.abs(theirs.imag - reference.imag)))
                    if not np.isfinite(theirs).all():
                        entry["pkg_refused"] = "non-finite"
                except Exception as error:  # noqa: BLE001 - the package refuses it
                    entry["pkg_refused"] = f"{type(error).__name__}: {error}"[:100]
    return {"database": db, "pages": pages, "load_errors": bad, "points": points, "package": package is not None}


def _ids(page_id):
    shelf, book, page = page_id.split("/", 2)
    return shelf, book, page


def _documented(entry):
    return entry["id"] in KNOWN_PACKAGE_DIFFERENCES


def rows(results):
    pages = results["pages"]
    compared = [p for p in pages if "dn" in p]
    out = [
        row(f"rii vs the frozen API evaluator, {len(compared)} pages at {results['points']} points", "max relative |dn|, absolute |dk|",
            max([max(p["dn"], p["dk"]) for p in compared], default=0.0), TOL_API),
        row("rii vs the frozen API evaluator: wavelengths one refuses and the other does not (pages)",
            "pages with a different refusal set", sum(bool(p.get("refusal_mismatch")) for p in pages), 0),
        row("pages the frozen API evaluator cannot load", "pages that fail to parse", len(results["load_errors"]), 0),
    ]
    if results["package"]:
        measured = [p for p in compared if "pkg_dn" in p]
        unexplained = [p for p in measured if max(p["pkg_dn"], p["pkg_dk"]) > TOL_PACKAGE and not _documented(p)]
        refused = [p for p in compared if "pkg_refused" in p and not _documented(p)]
        out += [
            row(f"pip refractiveindex 1.0.2 vs rii, {len(measured)} pages", "pages differing by > 1e-6 and not documented",
                len(unexplained), 0),
            row("pip refractiveindex 1.0.2 refuses a page rii evaluates", "pages refused and not documented", len(refused), 0),
        ]
    return out


def sections(results):
    pages = results["pages"]
    compared = [p for p in pages if "dn" in p]
    identity = results["database"].identity
    lines = ["\n## What was compared\n",
             f"Database: `{results['database'].root.name}`, SHA-256 `{identity.sha256}`, {identity.files} files, "
             f"{identity.bytes:,} bytes" + (f", git `{identity.git_commit}`" if identity.git_commit else "") + ".\n",
             f"- catalogue page ids: {len(pages)} (3560 entries in catalog-nk.yml for the pinned database; 17 ids "
             "are listed twice with the same file)",
             f"- pages with no n data (API and `rii` both refuse): {sum(bool(p.get('no_n')) for p in pages)}",
             f"- pages the API refuses at some sampled wavelength (k entries that do not cover n's range): "
             f"{sum(p.get('api_refused', 0) > 0 for p in pages)}",
             f"- pages evaluated in full by the API and by `rii`: {len(compared)}"]
    worst = sorted(compared, key=lambda p: -max(p["dn"], p["dk"]))[:10]
    lines += ["\n### `rii` against the frozen API evaluator: worst pages\n", "| page | entries | max relative dn | max dk |", "|---|---|---|---|"]
    lines += [f"| {p['id']} | {', '.join(p['kinds'])} | {p['dn']:.2e} | {p['dk']:.2e} |" for p in worst]
    if results["package"]:
        measured = [p for p in compared if "pkg_dn" in p]
        lines.append(f"\n### The pip package against `rii` (and the API): {len(measured)} pages measured\n")
        for threshold in (1e-12, 1e-9, 1e-6, 1e-3, 1e-2):
            hit = [p for p in measured if max(p["pkg_dn"], p["pkg_dk"]) > threshold]
            lines.append(f"- pages with max |dn| or |dk| > {threshold:g}: {len(hit)}")
        refused = [p for p in compared if "pkg_refused" in p]
        lines.append(f"- pages the package refused or returned non-finite for: {len(refused)}")
        worst = sorted(measured, key=lambda p: -max(p["pkg_dn"], p["pkg_dk"]))[:12]
        lines += ["\n| page | entries | max abs dn | max abs dk | known reason |", "|---|---|---|---|---|"]
        lines += [f"| {p['id']} | {', '.join(p['kinds'])} | {p['pkg_dn']:.2e} | {p['pkg_dk']:.2e} | "
                  f"{KNOWN_PACKAGE_DIFFERENCES.get(p['id'], '')} |" for p in worst]
        if refused:
            lines += ["\nPages the package refused:\n", "| page | entries | error | known reason |", "|---|---|---|---|"]
            lines += [f"| {p['id']} | {', '.join(p['kinds'])} | {p['pkg_refused']} | {KNOWN_PACKAGE_DIFFERENCES.get(p['id'], '')} |"
                      for p in refused[:12]]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database", default=str(Path("~/.refractiveindex.info-database").expanduser()))
    parser.add_argument("--expect-sha256", default=PINNED_SHA256,
                        help="refuse a database with another SHA-256 (default: the API's pinned a66ef88 snapshot)")
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()
    try:
        from importlib.metadata import version

        package_version = version("refractiveindex")
    except Exception:  # noqa: BLE001
        package_version = "not installed"
    results = compare(args.database, args.expect_sha256, with_package=not args.no_package)
    report("Films: refractiveindex.info pages, `pystratify.rii` against the API's evaluator and the pip package",
           "Generated by `python benchmarks/refractiveindex_pages.py` "
           f"(pip `refractiveindex` {package_version}, a reference only). The API's evaluator is frozen in "
           "`tests/oracles/api_rii.py` (multilayer ef9c946); `rii` follows it. Every catalogue page, 25 log-spaced "
           "wavelengths inside its n range. Where the package and the API disagree `rii` follows the API.",
           rows(results))
    print(sections(results))
