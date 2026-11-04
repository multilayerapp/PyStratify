"""refractiveindex.info database pages as dispersion models, with the database's identity checked.

A page of the `refractiveindex.info database <https://github.com/polyanskiy/refractiveindex.info-database>`_
(CC0) is a YAML record whose ``DATA`` list holds formula entries (``formula 1`` ... ``formula 9``,
wavelengths in micrometres) and tabulated ones (``tabulated n``, ``tabulated k``, ``tabulated nk``).
:class:`RIIPage` evaluates one as a :class:`~pystratify.dispersion.Dispersion`::

    from pystratify import rii, dispersion as dm, planar

    db = rii.RIIDatabase("~/.refractiveindex.info-database", expected_sha256="f546d53a...")
    glass = db.page("main/SiO2/Malitson")           # shelf/book/page of the catalogue
    n = glass.index(633.0)                          # nm in, n + ik out, like every dispersion model
    gold = rii.RIIPage.from_file(".../data/main/Au/nk/Johnson.yml")

Semantics (the multilayer.app API's evaluator, which this module replaces, ported unchanged):

* each wavelength is served by the **first** ``DATA`` entry, in file order, that is an n source
  (a formula, ``tabulated n`` or ``tabulated nk``) and covers it, and k by the first ``tabulated k`` or
  ``tabulated nk`` entry that covers it; a formula entry without ``wavelength_range`` covers every
  wavelength; coverage includes 1e-12 um at each end;
* a wavelength no entry covers raises ``ValueError``: **there is no extrapolation**.  A page with no k
  entry at all is lossless (k = 0), but a page that has k entries and none covering the wavelength is
  refused rather than read as lossless;
* a table is sorted by wavelength before interpolation (``numpy.argsort`` default order, so rows repeated at
  one wavelength are interpolated through in whatever order that sort leaves them: ambiguous data, kept
  as the API does) and interpolated linearly in wavelength;
* a formula that yields a non-finite or non-positive index, or a table whose k is negative, is refused.

Where the pip ``refractiveindex`` package differs from this (``main/CuO/Brimhall``: unsorted repeated rows;
``organic/polyvinylpyrrolidone/Konig``: a formula 5 beside a tabulated nk, whose n the package takes from
the table and this module from the formula that comes first; ``main/AgGaSe2/Boyd-o``: formula 2, which the
package's fallback does not read), this module follows the API; ``benchmarks/REFRACTIVEINDEX_PAGES.md``
measures both over every page.

**Database identity.**  :func:`database_identity` returns the SHA-256 of the database (the scheme of the API's
``tools/ri_snapshot.py``: one ``"<file sha256>  <relative path>"`` line per file of ``catalog-nk.yml`` and
``data/``, paths sorted, then hashed) and, if the root is the top of a git checkout, its commit.
:class:`RIIDatabase` and :func:`verify_database` refuse a database whose identity differs from the one
expected, so a calculation that depends on a pinned database cannot silently read another.

Reading a page needs PyYAML (``pip install pystratify[rii]``).
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .dispersion import Dispersion

__all__ = [
    "RIIPage",
    "RIIDatabase",
    "DatabaseIdentity",
    "DatabaseIdentityError",
    "database_identity",
    "verify_database",
    "evaluate_formula",
    "interpolate_table",
    "CATALOG_MISROUTES",
]

#: Pages whose entry in the pinned catalogue (database commit a66ef88) points at another book's file; the
#: page's own file is read instead (the multilayer.app API's table, ruled 2026-10-02).
CATALOG_MISROUTES = {
    "main/MgO/Synowicki": "main/MgO/nk/Synowicki.yml",
    "main/ZrO2/Synowicki": "main/ZrO2/nk/Synowicki.yml",
}

CATALOG = "catalog-nk.yml"
_RANGE_TOLERANCE_UM = 1e-12
_FORMULAS = range(1, 10)


# --------------------------------------------------------------------------- database identity


class DatabaseIdentityError(ValueError):
    """The database is not the one expected."""


@dataclass(frozen=True)
class DatabaseIdentity:
    """What a database is: ``sha256`` over ``catalog-nk.yml`` and ``data/`` (see :func:`database_identity`),
    the number of ``files`` and ``bytes`` it covers, and ``git_commit`` when the root is a git checkout's top."""

    sha256: str
    files: int
    bytes: int
    git_commit: Optional[str] = None


def _database_root(root) -> Path:
    path = Path(root).expanduser().resolve()
    if (path / "data").is_dir():
        return path
    if path.name == "data" and path.is_dir():  # the data directory itself was given
        return path.parent
    raise FileNotFoundError(f"{path} is not a refractiveindex.info database (no data/ directory)")


def _git_commit(root: Path) -> Optional[str]:
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=20
        )
        if top.returncode != 0:
            return None
        top_path = Path(top.stdout.strip()).resolve()
        # the checkout's own top, or its database/ directory; not a repository that merely contains a copy
        if top_path != root and not (top_path == root.parent and root.name == "database"):
            return None
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        return head.stdout.strip() if head.returncode == 0 and head.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def database_identity(root) -> DatabaseIdentity:
    """Identity of the database at ``root`` (its top directory, or its ``data`` directory).

    ``sha256`` is the SHA-256 of one line ``"<sha256 of file>  <path relative to root>\\n"`` per file of
    ``catalog-nk.yml`` and everything under ``data/``, in sorted path order: the digest the multilayer.app
    API pins (``RI_DATABASE_SNAPSHOT_SHA256``, from ``tools/ri_snapshot.py``), reproducible with
    ``find catalog-nk.yml data -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256``.
    Files outside those two (the upstream ``doc/``, ``tools/``, ``catalog-n2.yml``) do not count, nor do
    hidden files (a name starting with ``.``, such as the ``.DS_Store`` Finder leaves: the harvester's copy
    has one, and it must not make a page-identical database look different).
    """
    base = _database_root(root)
    names = [CATALOG] if (base / CATALOG).is_file() else []
    names += [
        p.relative_to(base).as_posix()
        for p in (base / "data").rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(base).parts)
    ]
    listing, total = hashlib.sha256(), 0
    for name in sorted(names):
        content = (base / name).read_bytes()
        total += len(content)
        listing.update(f"{hashlib.sha256(content).hexdigest()}  {name}\n".encode("utf-8"))
    return DatabaseIdentity(listing.hexdigest(), len(names), total, _git_commit(base))


def verify_database(root, expected_sha256: Optional[str] = None, expected_git_commit: Optional[str] = None):
    """The identity of the database at ``root``, or ``DatabaseIdentityError`` if it differs from what was
    expected.  At least one expectation is required: a check that expects nothing is not a check.
    ``expected_git_commit`` may be a prefix (7 or more characters) and needs ``root`` to be a git checkout."""
    if expected_sha256 is None and expected_git_commit is None:
        raise ValueError("give expected_sha256 and/or expected_git_commit; a check that expects nothing checks nothing")
    identity = database_identity(root)
    if expected_sha256 is not None and identity.sha256 != expected_sha256.strip().lower():
        raise DatabaseIdentityError(
            f"refractiveindex.info database at {root} has SHA-256 {identity.sha256}, expected {expected_sha256}"
        )
    if expected_git_commit is not None:
        want = expected_git_commit.strip().lower()
        if len(want) < 7 or identity.git_commit is None or not identity.git_commit.startswith(want):
            raise DatabaseIdentityError(
                f"refractiveindex.info database at {root} is at git commit {identity.git_commit}, expected {want}"
            )
    return identity


# --------------------------------------------------------------------------- page parsing


def _tokens(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(entry) for entry in value]
    return [float(entry) for entry in str(value).split()]


def _table(text, columns: int) -> np.ndarray:
    rows = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        values = _tokens(stripped)
        if len(values) < columns:
            raise ValueError("A refractiveindex.info data table has too few columns.")
        rows.append(values[:columns])
    if not rows:
        raise ValueError("A refractiveindex.info data table is empty.")
    return np.array(rows, dtype=float)


def evaluate_formula(entry: dict, wavelength_um) -> np.ndarray:
    """Refractive index n from a ``formula 1..9`` entry (``{"type": "formula 2", "coefficients": "0 1.1 0.01"}``)
    at wavelengths in um (scalar or array).  The coefficient list is read from ``entry["_coefficients"]`` when
    prepared, else from the page's ``coefficients`` text.  Raises ``ValueError`` for an unsupported formula or
    an index that is not finite and positive."""
    formula_type = str(entry.get("type", "")).split()
    if len(formula_type) != 2 or formula_type[0] != "formula":
        raise ValueError("Unsupported refractiveindex.info formula entry.")
    formula_number = int(formula_type[1])
    coefficients = entry.get("_coefficients") or _tokens(entry.get("coefficients"))
    wavelength = np.asarray(wavelength_um, dtype=float)

    def coefficient(index: int, default: float = 0.0) -> float:
        return coefficients[index] if index < len(coefficients) else default

    with np.errstate(all="ignore"):
        if formula_number == 1:
            n_squared = 1.0 + coefficient(0)
            for index in range(1, len(coefficients) - 1, 2):
                n_squared = n_squared + (
                    coefficients[index] * wavelength**2 / (wavelength**2 - coefficients[index + 1] ** 2)
                )
            value = np.sqrt(n_squared)
        elif formula_number == 2:
            n_squared = 1.0 + coefficient(0)
            for index in range(1, len(coefficients) - 1, 2):
                n_squared = n_squared + (
                    coefficients[index] * wavelength**2 / (wavelength**2 - coefficients[index + 1])
                )
            value = np.sqrt(n_squared)
        elif formula_number == 3:
            n_squared = coefficient(0)
            for index in range(1, len(coefficients) - 1, 2):
                n_squared = n_squared + coefficients[index] * wavelength ** coefficients[index + 1]
            value = np.sqrt(n_squared)
        elif formula_number == 4:
            n_squared = coefficient(0)
            if len(coefficients) >= 5:
                n_squared = n_squared + (
                    coefficient(1) * wavelength ** coefficient(2) / (wavelength**2 - coefficient(3) ** coefficient(4))
                )
            if len(coefficients) >= 9:
                n_squared = n_squared + (
                    coefficient(5) * wavelength ** coefficient(6) / (wavelength**2 - coefficient(7) ** coefficient(8))
                )
            for index in range(9, len(coefficients) - 1, 2):
                n_squared = n_squared + coefficients[index] * wavelength ** coefficients[index + 1]
            value = np.sqrt(n_squared)
        elif formula_number == 5:
            value = coefficient(0) + 0.0 * wavelength
            for index in range(1, len(coefficients) - 1, 2):
                value = value + coefficients[index] * wavelength ** coefficients[index + 1]
        elif formula_number == 6:
            value = 1.0 + coefficient(0) + 0.0 * wavelength
            inverse_wavelength_squared = wavelength**-2
            for index in range(1, len(coefficients) - 1, 2):
                value = value + coefficients[index] / (coefficients[index + 1] - inverse_wavelength_squared)
        elif formula_number == 7:
            inverse_offset = 1.0 / (wavelength**2 - 0.028)
            value = (
                coefficient(0)
                + coefficient(1) * inverse_offset
                + coefficient(2) * inverse_offset**2
                + coefficient(3) * wavelength**2
                + coefficient(4) * wavelength**4
                + coefficient(5) * wavelength**6
            )
        elif formula_number == 8:
            ratio = (
                coefficient(0)
                + coefficient(1) * wavelength**2 / (wavelength**2 - coefficient(2))
                + coefficient(3) * wavelength**2
            )
            value = np.sqrt((1.0 + 2.0 * ratio) / (1.0 - ratio))
        elif formula_number == 9:
            n_squared = (
                coefficient(0)
                + coefficient(1) / (wavelength**2 - coefficient(2))
                + coefficient(3) * (wavelength - coefficient(4)) / ((wavelength - coefficient(4)) ** 2 + coefficient(5))
            )
            value = np.sqrt(n_squared)
        else:
            raise ValueError(f"Unsupported refractiveindex.info formula {formula_number}.")

    value = np.asarray(np.real_if_close(value), dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("A refractiveindex.info formula produced an invalid refractive index.")
    return value


def interpolate_table(entry: dict, wavelength_um, column_index: int) -> np.ndarray:
    """Linear interpolation of column ``column_index`` (1 = n or k, 2 = k of a ``tabulated nk``) of a
    ``tabulated`` entry at wavelengths in um.  Raises ``ValueError`` outside the table's range (1e-12 um slack)."""
    data_type = str(entry.get("type", ""))
    columns = 3 if data_type == "tabulated nk" else 2
    rows = entry.get("_table")
    if rows is None:
        rows = _table(entry.get("data", ""), columns)
        rows = rows[np.argsort(rows[:, 0])]
    wavelength = np.asarray(wavelength_um, dtype=float)
    wavelengths, values = rows[:, 0], rows[:, column_index]
    if np.any(wavelength < wavelengths[0] - _RANGE_TOLERANCE_UM) or np.any(
        wavelength > wavelengths[-1] + _RANGE_TOLERANCE_UM
    ):
        raise ValueError("Requested wavelength is outside a refractiveindex.info tabulated data range.")
    return np.interp(wavelength, wavelengths, values)


def _prepare(record: dict) -> tuple:
    """The page's entries as (kind, range_um | None, entry) in file order, tables sorted, ranges read."""
    prepared = []
    for entry in record.get("DATA", []):
        if not isinstance(entry, dict):
            continue
        entry = dict(entry)
        data_type = str(entry.get("type", ""))
        span = None
        if data_type.startswith("formula"):
            entry["_coefficients"] = _tokens(entry.get("coefficients"))
            if entry.get("wavelength_range") is not None:
                values = _tokens(entry.get("wavelength_range"))
                if len(values) >= 2:
                    span = (min(values[0], values[1]), max(values[0], values[1]))
        elif data_type.startswith("tabulated"):
            table = _table(entry.get("data", ""), 3 if data_type == "tabulated nk" else 2)
            table = table[np.argsort(table[:, 0])]
            entry["_table"] = table
            span = (float(table[0, 0]), float(table[-1, 0]))
        prepared.append((data_type, span, entry))
    return tuple(prepared)


def _covers(span, wavelength) -> np.ndarray:
    if span is None:
        return np.ones(wavelength.shape, dtype=bool)
    return (span[0] - _RANGE_TOLERANCE_UM <= wavelength) & (wavelength <= span[1] + _RANGE_TOLERANCE_UM)


def _yaml():
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("reading refractiveindex.info pages needs PyYAML: pip install pystratify[rii]") from error
    return yaml


def _check_data_path(path: str) -> str:
    normalized = str(path or "").strip()
    if (
        not normalized
        or normalized.startswith("/")
        or ".." in normalized.split("/")
        or not normalized.endswith((".yml", ".yaml"))
    ):
        raise ValueError("Invalid refractiveindex.info data path.")
    return normalized


# --------------------------------------------------------------------------- the page as a dispersion model


@dataclass(frozen=True, eq=False)
class RIIPage(Dispersion):
    """One refractiveindex.info page: n and k from its formulas and tables, eps = (n + ik)^2.

    Build it with :meth:`from_file`, :meth:`from_record` or ``RIIDatabase.page``.  :meth:`nk` takes
    wavelengths in um (the page's own unit), ``epsilon``/``index`` nm like every model; both refuse a
    wavelength the page does not cover.  ``source`` names where it came from (a path or page id).
    """

    entries: tuple
    source: str = ""

    @classmethod
    def from_record(cls, record: dict, source: str = "") -> "RIIPage":
        """From a parsed page (``{"DATA": [...]}``, as ``yaml.safe_load`` returns it)."""
        if not isinstance(record, dict) or not isinstance(record.get("DATA"), list):
            raise ValueError("The refractiveindex.info material record is not usable.")
        return cls(_prepare(record), source)

    @classmethod
    def from_file(cls, path) -> "RIIPage":
        """From a page YAML file."""
        path = Path(path).expanduser()
        return cls.from_record(_yaml().safe_load(path.read_text(encoding="utf-8")), str(path))

    def ranges_um(self) -> dict:
        """Wavelength ranges in um where the page gives ``n`` and ``k``: ``{"n": [(lo, hi), ...], "k": [...]}``
        (formula entries without a ``wavelength_range`` have none and are left out)."""
        n_ranges, k_ranges = [], []
        for data_type, span, _ in self.entries:
            if span is None:
                continue
            if data_type.startswith("formula") or data_type in {"tabulated n", "tabulated nk"}:
                n_ranges.append(span)
            if data_type in {"tabulated k", "tabulated nk"}:
                k_ranges.append(span)
        return {"n": n_ranges, "k": k_ranges}

    def nk(self, wavelength_um):
        """Complex n + ik at wavelengths in **um** (scalar or array)."""
        wavelength = np.asarray(wavelength_um, dtype=float)
        flat = np.atleast_1d(wavelength).ravel()
        n = np.full(flat.shape, np.nan)
        k = np.full(flat.shape, np.nan)
        pending_n = np.ones(flat.shape, dtype=bool)
        pending_k = np.ones(flat.shape, dtype=bool)
        k_entries_exist = any(data_type in {"tabulated k", "tabulated nk"} for data_type, _, _ in self.entries)
        for data_type, span, entry in self.entries:
            is_n = data_type.startswith("formula") or data_type in {"tabulated n", "tabulated nk"}
            is_k = data_type in {"tabulated k", "tabulated nk"}
            if is_n:
                take = pending_n & _covers(span, flat)
                if take.any():
                    if data_type.startswith("formula"):
                        n[take] = evaluate_formula(entry, flat[take])
                    else:
                        n[take] = interpolate_table(entry, flat[take], 1)
                    pending_n &= ~take
            if is_k:
                take = pending_k & _covers(span, flat)
                if take.any():
                    k[take] = interpolate_table(entry, flat[take], 2 if data_type == "tabulated nk" else 1)
                    pending_k &= ~take
        if pending_n.any():
            raise ValueError(f"{self._label()} has no refractiveindex.info n data at {_um(flat[pending_n])} microns.")
        if pending_k.any():
            if k_entries_exist:
                raise ValueError(
                    f"{self._label()} has no refractiveindex.info k data at {_um(flat[pending_k])} microns."
                )
            k[pending_k] = 0.0
        if np.any(~np.isfinite(n)) or np.any(n <= 0):
            raise ValueError(f"{self._label()} n must be finite and positive.")
        if np.any(~np.isfinite(k)) or np.any(k < 0):
            raise ValueError(f"{self._label()} k must be finite and zero or positive.")
        value = n + 1j * k
        return complex(value[0]) if wavelength.ndim == 0 else value.reshape(wavelength.shape)

    def _label(self) -> str:
        return self.source or "page"

    def _epsilon(self, energy, wavelength):
        return self.nk(wavelength / 1000.0) ** 2


def _um(values) -> str:
    values = np.asarray(values)
    return f"{values[0]:.6g}" if values.size == 1 else f"{values.min():.6g}..{values.max():.6g}"


# --------------------------------------------------------------------------- the database


class RIIDatabase:
    """A refractiveindex.info database on disk, with its identity checked on opening.

    ``root`` is the directory holding ``catalog-nk.yml`` and ``data/`` (or the ``data`` directory itself).
    Give ``expected_sha256`` and/or ``expected_git_commit`` to refuse any other database
    (``DatabaseIdentityError``); without either, ``identity`` still says which database was read, and a
    caller that cares must record it.  Pages are named ``shelf/book/page`` as in the catalogue
    (``main/SiO2/Malitson``), or by their file under ``data/`` (``main/SiO2/nk/Malitson.yml``).
    """

    def __init__(self, root, expected_sha256: Optional[str] = None, expected_git_commit: Optional[str] = None):
        self.root = _database_root(root)
        if expected_sha256 is not None or expected_git_commit is not None:
            self.identity = verify_database(self.root, expected_sha256, expected_git_commit)
        else:
            self.identity = database_identity(self.root)
        self._cache: dict = {}
        self._catalog: Optional[dict] = None

    def catalog(self) -> dict:
        """``{"shelf/book/page": "relative/data/path.yml"}`` from ``catalog-nk.yml`` (misroutes corrected)."""
        if self._catalog is not None:
            return self._catalog
        entries = _yaml().safe_load((self.root / CATALOG).read_text(encoding="utf-8")) or []
        pages = {}
        for shelf in entries:
            if not isinstance(shelf, dict) or "SHELF" not in shelf:
                continue
            for book in shelf.get("content") or []:
                if not isinstance(book, dict) or "BOOK" not in book:
                    continue
                for page in book.get("content") or []:
                    if isinstance(page, dict) and "PAGE" in page:
                        key = f"{shelf['SHELF']}/{book['BOOK']}/{page['PAGE']}"
                        # the first entry of an id wins (the catalogue lists 17 ids twice, with the same file)
                        pages.setdefault(key, CATALOG_MISROUTES.get(key) or str(page.get("data") or ""))
        self._catalog = pages
        return pages

    def page(self, page_id: str) -> RIIPage:
        """The page named ``shelf/book/page`` (catalogue id) or by its data file path."""
        path = page_id if page_id.endswith((".yml", ".yaml")) else self.catalog().get(page_id)
        if path is None:
            raise KeyError(f"{page_id}: no such page in the refractiveindex.info catalogue")
        path = _check_data_path(path)
        if path not in self._cache:
            file = (self.root / "data" / path).resolve()
            if not file.is_relative_to((self.root / "data").resolve()) or not file.is_file():
                raise FileNotFoundError(f"{path} is not in the refractiveindex.info database at {self.root}")
            self._cache[path] = RIIPage.from_file(file)
            object.__setattr__(self._cache[path], "source", page_id)
        return self._cache[path]
