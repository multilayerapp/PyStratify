"""refractiveindex.info page evaluator and database identity (assurance WP07-D2), against the frozen API code
(tests/oracles/api_rii.py, multilayer ef9c946) on a handful of pages copied from the database (CC0, commit a66ef88)."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

from pystratify import dispersion as dm  # noqa: E402
from pystratify import rii  # noqa: E402
from tests.oracles import api_rii as api  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "rii"
API_DATABASE = Path("/Users/iliarasskazov/Projects/multilayerapp/multilayer/backend/data/refractiveindex")
API_DIGEST = "f546d53a0da54a7bac9fb09a9afa17f719e0267d090bb1ebe106f2967a3f2c36"  # README of that snapshot

PAGES = {  # page id -> what it exercises
    "main/TlBr/Palik": "formula 1",
    "main/CdTe/Marple": "formula 2",
    "main/BeAl6O10/Pestryakov-β": "formula 3",
    "main/CuCl/Feldman": "formula 4",
    "main/Ar/Peck-15C": "formula 6",
    "main/Si/Edwards": "formula 7",
    "main/TlCl/Schroter": "formula 8",
    "main/K/Ives": "tabulated nk",
    "main/MoS2/Yim-3nm": "tabulated n + tabulated k",
    "main/CdF2/Bosomworth-300K": "formula 4 + tabulated k",
    "main/AgGaSe2/Boyd-o": "formula 2 (the pip package refuses it)",
    "main/CuO/Brimhall": "tabulated nk, repeated unsorted wavelengths",
    "organic/polyvinylpyrrolidone/Konig": "formula 5 + tabulated nk (the pip package takes n from the table)",
}
FILES = {  # not in the catalogue fixture above, by file
    "organic/CH4N2O - urea/nk/Rosker-e.yml": "formula 9",
    "specs/schott/obsolete/LF7.yml": "formula 2 + tabulated k",
    "specs/hikari/optical/E-LLF2.yml": "formula 3 + tabulated k",
    "other/commercial polymers/CR-39/nk/mono.yml": "tabulated n (one wavelength)",
    "specs/ami/infrared/AMTIR-2.yml": "formula 1 + tabulated k",
    "specs/cargille/index-matching liquids/nk/acrylic_matching_liquid.yml": "formula 5 + tabulated k",
}


def oracle_nk(path, wavelengths_um):
    record = api.prepare_material_record(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    return np.array([complex(*api.material_index_at(record, float(w), "x")) for w in wavelengths_um])


def span(page):
    """Where n and (if the page has k entries) k are both given."""
    ranges = page.ranges_um()
    lo, hi = min(r[0] for r in ranges["n"]), max(r[1] for r in ranges["n"])
    if ranges["k"]:
        lo, hi = max(lo, min(r[0] for r in ranges["k"])), min(hi, max(r[1] for r in ranges["k"]))
    return lo, hi


def sample(page):
    lo, hi = span(page)
    return np.geomspace(lo * 1.005, hi * 0.995, 25) if hi > lo else np.array([lo])


def check(path, page):
    wl = sample(page)
    new, old = page.nk(wl), oracle_nk(path, wl)
    assert np.max(np.abs(new - old)) <= 1e-13 * np.max(np.abs(old))
    # through the nm interface: eps = (n + ik)^2, then the passive root
    assert np.max(np.abs(page.index(wl * 1000.0) - old)) <= 1e-12 * np.max(np.abs(old))
    # range ends are inclusive
    edge = np.array(span(page))
    assert np.max(np.abs(page.nk(edge) - oracle_nk(path, edge))) <= 1e-13 * np.max(np.abs(old))
    return wl


@pytest.fixture(scope="module")
def database():
    return rii.RIIDatabase(FIXTURE)


@pytest.mark.parametrize("page_id", list(PAGES))
def test_catalogue_pages_match_api(database, page_id):
    page = database.page(page_id)
    path = FIXTURE / "data" / database.catalog()[page_id]
    check(path, page)


@pytest.mark.parametrize("file", list(FILES))
def test_file_pages_match_api(file):
    path = FIXTURE / "data" / file
    check(path, rii.RIIPage.from_file(path))


def test_every_formula_and_table_type_is_covered():
    kinds = set()
    for file in FILES:
        kinds |= {e["type"] for e in yaml.safe_load((FIXTURE / "data" / file).read_text(encoding="utf-8"))["DATA"]}
    for page_id in PAGES:
        record = yaml.safe_load((FIXTURE / "data" / rii.RIIDatabase(FIXTURE).catalog()[page_id]).read_text("utf-8"))
        kinds |= {e["type"] for e in record["DATA"]}
    assert {f"formula {i}" for i in range(1, 10)} | {"tabulated n", "tabulated k", "tabulated nk"} <= kinds


def test_no_extrapolation_and_k_rules(database):
    page = database.page("main/CdTe/Marple")  # formula 2, 0.86-2.5 um, no k entry
    low, high = page.ranges_um()["n"][0]
    assert page.nk(0.5 * (low + high)).imag == 0.0
    for outside in (low * 0.99, high * 1.01):
        with pytest.raises(ValueError):
            page.nk(outside)
        with pytest.raises(ValueError):
            page.index(outside * 1000.0)
        with pytest.raises(api.HTTPException):
            oracle_nk(FIXTURE / "data/main/CdTe/nk/Marple.yml", [outside])
    # a page with k entries and a wavelength none covers is refused, not read as lossless
    glass = rii.RIIPage.from_file(FIXTURE / "data/specs/schott/obsolete/LF7.yml")
    n_range = glass.ranges_um()["n"][0]
    k_range = glass.ranges_um()["k"][0]
    if n_range[1] > k_range[1]:
        wl = 0.5 * (k_range[1] + n_range[1])
        with pytest.raises(ValueError):
            glass.nk(wl)
        with pytest.raises(api.HTTPException):
            oracle_nk(FIXTURE / "data/specs/schott/obsolete/LF7.yml", [wl])


def test_dispersion_interface(database):
    page = database.page("main/TlBr/Palik")
    assert isinstance(page, dm.Dispersion)
    wl = np.array([700.0, 1000.0, 2000.0])
    assert page.epsilon(wl).shape == (3,)
    assert isinstance(page.index(1000.0), complex)
    both = dm.EpsilonInf(0.1) + page
    assert np.allclose(both.epsilon(wl), page.epsilon(wl) + 0.1)
    with pytest.raises(ValueError):
        page.index(100.0)


def test_formula_refusals():
    with pytest.raises(ValueError):
        rii.evaluate_formula({"type": "formula 10", "coefficients": "1 1"}, 1.0)
    with pytest.raises(ValueError):
        rii.evaluate_formula({"type": "formula 1", "coefficients": "-5 0 0"}, 1.0)  # n^2 < 0
    with pytest.raises(ValueError):
        rii.RIIPage.from_record({"DATA": "nope"})


# --------------------------------------------------------------------------- identity


def test_identity_is_stable_and_counts_the_catalog_and_data():
    first = rii.database_identity(FIXTURE)
    assert first == rii.database_identity(FIXTURE / "data")  # the data directory names the same database
    assert len(first.sha256) == 64 and first.files == 1 + sum(1 for p in (FIXTURE / "data").rglob("*") if p.is_file())


def copy(tmp_path):
    target = tmp_path / "db"
    shutil.copytree(FIXTURE, target)
    return target


def test_identity_changes_with_content_name_and_catalog(tmp_path):
    base = rii.database_identity(FIXTURE).sha256
    target = copy(tmp_path)
    assert rii.database_identity(target).sha256 == base  # location does not matter
    (target / "data/main/K/nk/Ives.yml").write_bytes((target / "data/main/K/nk/Ives.yml").read_bytes() + b"\n")
    assert rii.database_identity(target).sha256 != base
    renamed = tmp_path / "renamed"
    shutil.copytree(FIXTURE, renamed)
    (renamed / "data/main/K/nk/Ives.yml").rename(renamed / "data/main/K/nk/Ives2.yml")
    assert rii.database_identity(renamed).sha256 != base
    catalog = tmp_path / "cat"
    shutil.copytree(FIXTURE, catalog)
    (catalog / "catalog-nk.yml").write_text((catalog / "catalog-nk.yml").read_text() + "# x\n")
    assert rii.database_identity(catalog).sha256 != base


def test_hidden_files_do_not_change_identity(tmp_path):
    base = rii.database_identity(FIXTURE).sha256
    target = copy(tmp_path)
    (target / "data/.DS_Store").write_bytes(b"\x00junk")
    (target / "data/main/.hidden.yml").write_text("x: 1\n")
    assert rii.database_identity(target).sha256 == base


def test_database_refuses_a_different_identity(tmp_path):
    expected = rii.database_identity(FIXTURE).sha256
    assert rii.RIIDatabase(FIXTURE, expected_sha256=expected).identity.sha256 == expected
    assert rii.verify_database(FIXTURE, expected_sha256=expected.upper()).sha256 == expected
    target = copy(tmp_path)
    (target / "data/main/K/nk/Ives.yml").write_text("DATA: []\n")
    with pytest.raises(rii.DatabaseIdentityError):
        rii.RIIDatabase(target, expected_sha256=expected)
    with pytest.raises(rii.DatabaseIdentityError):
        rii.verify_database(target, expected_sha256=expected)
    assert issubclass(rii.DatabaseIdentityError, ValueError)
    with pytest.raises(ValueError):
        rii.verify_database(FIXTURE)  # expecting nothing is not a check
    with pytest.raises(FileNotFoundError):
        rii.database_identity(tmp_path)


def test_git_commit_identity(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    target = copy(tmp_path)
    assert rii.database_identity(target).git_commit is None  # not a checkout
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}
    for command in (["init", "-q"], ["add", "-A"], ["-c", "commit.gpgsign=false", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(target), *command], check=True, env=env, capture_output=True)
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert rii.database_identity(target).git_commit == head
    assert rii.verify_database(target, expected_git_commit=head[:7]).git_commit == head
    with pytest.raises(rii.DatabaseIdentityError):
        rii.verify_database(target, expected_git_commit="0" * 40)
    with pytest.raises(rii.DatabaseIdentityError):
        rii.verify_database(FIXTURE.parent / "rii", expected_git_commit=head)  # a copy inside this repo is not a checkout top
    # a subdirectory of someone else's repository does not borrow that repository's commit
    assert rii.database_identity(FIXTURE).git_commit is None


@pytest.mark.skipif(not (API_DATABASE / "catalog-nk.yml").is_file(), reason="the API's shipped database is not here")
def test_digest_reproduces_the_apis_pinned_snapshot():
    identity = rii.database_identity(API_DATABASE)
    assert identity.sha256 == API_DIGEST and identity.files == 4165 and identity.bytes == 46372213
    assert rii.database_identity(API_DATABASE / "data").sha256 == API_DIGEST
    copy_in_home = Path("~/.refractiveindex.info-database").expanduser()
    if (copy_in_home / "catalog-nk.yml").is_file():  # the harvester's copy: page-identical (a .DS_Store aside)
        assert rii.database_identity(copy_in_home).sha256 == API_DIGEST
    db = rii.RIIDatabase(API_DATABASE, expected_sha256=API_DIGEST)
    assert db.catalog()["main/MgO/Synowicki"] == "main/MgO/nk/Synowicki.yml"
    path = API_DATABASE / "data" / db.catalog()["main/SiO2/Malitson"]
    page = db.page("main/SiO2/Malitson")
    check(path, page)
    assert page.index(589.3) == pytest.approx(1.4585, abs=1e-4)
