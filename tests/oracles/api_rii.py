"""FROZEN REFERENCE -- do not edit, do not import from production code.

The multilayer.app API's refractiveindex.info page evaluator, copied verbatim (code only; the
``HTTPException`` stub below stands in for FastAPI's) from repository ``multilayer``, commit
``ef9c946``, ``backend/app.py``:
  * lines 660-667   ``parse_float_tokens``
  * lines 1033-1055 ``parse_tabulated_rows``
  * lines 1075-1097 ``prepare_material_record``
  * lines 1246-1291 ``get_entry_range``, ``wavelength_in_range``, ``get_material_ranges``
  * lines 1294-1475 ``evaluate_formula``, ``interpolate_table``, ``find_covering_entry``,
                    ``material_index_at``
  * lines 1821-1826 ``ensure_finite`` and 1828-1844 ``validate_index_parts``
This is the independent check kept after the API calls ``pystratify.rii`` (assurance WP07-D2);
tests/test_rii.py and benchmarks/refractiveindex_pages.py compare against it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def round_float(value: float) -> float:  # backend/app.py lines 1817-1818
    return round(float(value), 6)


def parse_float_tokens(value) -> list[float]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [float(entry) for entry in value]

    return [float(entry) for entry in str(value).split()]


def parse_tabulated_rows(value: str, expected_columns: int) -> np.ndarray:
    rows = []

    for line in str(value or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        columns = parse_float_tokens(stripped)
        if len(columns) < expected_columns:
            raise HTTPException(
                status_code=422,
                detail="A refractiveindex.info data table has too few columns.",
            )
        rows.append(columns[:expected_columns])

    if not rows:
        raise HTTPException(
            status_code=422,
            detail="A refractiveindex.info data table is empty.",
        )

    return np.array(rows, dtype=float)


def prepare_material_record(record: dict) -> dict:
    for entry in record.get("DATA", []):
        if not isinstance(entry, dict):
            continue

        data_type = str(entry.get("type", ""))

        if data_type.startswith("formula"):
            entry["_coefficients"] = parse_float_tokens(entry.get("coefficients"))
            if entry.get("wavelength_range") is not None:
                values = parse_float_tokens(entry.get("wavelength_range"))
                if len(values) >= 2:
                    entry["_range"] = (min(values[0], values[1]), max(values[0], values[1]))

        if data_type.startswith("tabulated"):
            column_count = 3 if data_type == "tabulated nk" else 2
            table = parse_tabulated_rows(entry.get("data", ""), column_count)
            order = np.argsort(table[:, 0])
            table = table[order]
            entry["_table"] = table
            entry["_range"] = (float(table[0, 0]), float(table[-1, 0]))

    return record


def get_entry_range(entry: dict) -> Optional[tuple[float, float]]:
    if entry.get("_range") is not None:
        return entry["_range"]

    if entry.get("wavelength_range") is not None:
        values = parse_float_tokens(entry.get("wavelength_range"))
        if len(values) >= 2:
            return min(values[0], values[1]), max(values[0], values[1])

    data_type = str(entry.get("type", ""))
    if data_type.startswith("tabulated"):
        column_count = 3 if data_type == "tabulated nk" else 2
        rows = parse_tabulated_rows(entry.get("data", ""), column_count)
        wavelengths = rows[:, 0]
        return float(np.min(wavelengths)), float(np.max(wavelengths))

    return None


def wavelength_in_range(wavelength_um: float, entry_range: Optional[tuple[float, float]]) -> bool:
    if entry_range is None:
        return True

    return entry_range[0] - 1e-12 <= wavelength_um <= entry_range[1] + 1e-12


def get_material_ranges(record: dict) -> dict:
    n_ranges = []
    k_ranges = []

    for entry in record.get("DATA", []):
        data_type = str(entry.get("type", ""))
        entry_range = get_entry_range(entry)
        if entry_range is None:
            continue

        if data_type.startswith("formula") or data_type in {"tabulated n", "tabulated nk"}:
            n_ranges.append(entry_range)

        if data_type in {"tabulated k", "tabulated nk"}:
            k_ranges.append(entry_range)

    return {
        "n": n_ranges,
        "k": k_ranges,
    }


def evaluate_formula(entry: dict, wavelength_um: float) -> float:
    formula_type = str(entry.get("type", "")).split()
    if len(formula_type) != 2 or formula_type[0] != "formula":
        raise HTTPException(
            status_code=422,
            detail="Unsupported refractiveindex.info formula entry.",
        )

    formula_number = int(formula_type[1])
    coefficients = entry.get("_coefficients") or parse_float_tokens(entry.get("coefficients"))
    wavelength = float(wavelength_um)

    def coefficient(index: int, default: float = 0.0) -> float:
        return coefficients[index] if index < len(coefficients) else default

    if formula_number == 1:
        n_squared = 1.0 + coefficient(0)
        for index in range(1, len(coefficients) - 1, 2):
            n_squared += (
                coefficients[index] * wavelength**2
                / (wavelength**2 - coefficients[index + 1] ** 2)
            )
        value = np.sqrt(n_squared)
    elif formula_number == 2:
        n_squared = 1.0 + coefficient(0)
        for index in range(1, len(coefficients) - 1, 2):
            n_squared += (
                coefficients[index] * wavelength**2
                / (wavelength**2 - coefficients[index + 1])
            )
        value = np.sqrt(n_squared)
    elif formula_number == 3:
        n_squared = coefficient(0)
        for index in range(1, len(coefficients) - 1, 2):
            n_squared += coefficients[index] * wavelength ** coefficients[index + 1]
        value = np.sqrt(n_squared)
    elif formula_number == 4:
        n_squared = coefficient(0)
        if len(coefficients) >= 5:
            n_squared += (
                coefficient(1) * wavelength ** coefficient(2)
                / (wavelength**2 - coefficient(3) ** coefficient(4))
            )
        if len(coefficients) >= 9:
            n_squared += (
                coefficient(5) * wavelength ** coefficient(6)
                / (wavelength**2 - coefficient(7) ** coefficient(8))
            )
        for index in range(9, len(coefficients) - 1, 2):
            n_squared += coefficients[index] * wavelength ** coefficients[index + 1]
        value = np.sqrt(n_squared)
    elif formula_number == 5:
        value = coefficient(0)
        for index in range(1, len(coefficients) - 1, 2):
            value += coefficients[index] * wavelength ** coefficients[index + 1]
    elif formula_number == 6:
        value = 1.0 + coefficient(0)
        inverse_wavelength_squared = wavelength ** -2
        for index in range(1, len(coefficients) - 1, 2):
            value += coefficients[index] / (coefficients[index + 1] - inverse_wavelength_squared)
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
        n_squared = (1.0 + 2.0 * ratio) / (1.0 - ratio)
        value = np.sqrt(n_squared)
    elif formula_number == 9:
        n_squared = (
            coefficient(0)
            + coefficient(1) / (wavelength**2 - coefficient(2))
            + coefficient(3) * (wavelength - coefficient(4))
            / ((wavelength - coefficient(4)) ** 2 + coefficient(5))
        )
        value = np.sqrt(n_squared)
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported refractiveindex.info formula {formula_number}.",
        )

    value = float(np.real_if_close(value))
    if not np.isfinite(value) or value <= 0:
        raise HTTPException(
            status_code=422,
            detail="A refractiveindex.info formula produced an invalid refractive index.",
        )

    return value


def interpolate_table(entry: dict, wavelength_um: float, column_index: int) -> float:
    data_type = str(entry.get("type", ""))
    column_count = 3 if data_type == "tabulated nk" else 2
    rows = entry.get("_table")
    if rows is None:
        rows = parse_tabulated_rows(entry.get("data", ""), column_count)
        order = np.argsort(rows[:, 0])
        rows = rows[order]

    wavelengths = rows[:, 0]
    values = rows[:, column_index]

    if wavelength_um < wavelengths[0] - 1e-12 or wavelength_um > wavelengths[-1] + 1e-12:
        raise HTTPException(
            status_code=422,
            detail="Requested wavelength is outside a refractiveindex.info tabulated data range.",
        )

    return float(np.interp(wavelength_um, wavelengths, values))


def find_covering_entry(record: dict, wavelength_um: float, kinds: set[str]) -> Optional[dict]:
    for entry in record.get("DATA", []):
        data_type = str(entry.get("type", ""))
        if data_type not in kinds and not (
            "formula" in kinds and data_type.startswith("formula")
        ):
            continue

        if wavelength_in_range(wavelength_um, get_entry_range(entry)):
            return entry

    return None


def material_index_at(record: dict, wavelength_um: float, label: str) -> tuple[float, float]:
    wavelength_um = float(wavelength_um)
    n_entry = find_covering_entry(
        record,
        wavelength_um,
        {"formula", "tabulated n", "tabulated nk"},
    )

    if n_entry is None:
        raise HTTPException(
            status_code=422,
            detail=f"{label} has no refractiveindex.info n data at {round_float(wavelength_um)} microns.",
        )

    n_type = str(n_entry.get("type", ""))
    if n_type.startswith("formula"):
        index_real = evaluate_formula(n_entry, wavelength_um)
    elif n_type == "tabulated nk":
        index_real = interpolate_table(n_entry, wavelength_um, 1)
    else:
        index_real = interpolate_table(n_entry, wavelength_um, 1)

    k_entries_exist = any(
        str(entry.get("type", "")) in {"tabulated k", "tabulated nk"}
        for entry in record.get("DATA", [])
    )
    k_entry = find_covering_entry(
        record,
        wavelength_um,
        {"tabulated k", "tabulated nk"},
    )

    if k_entry is None:
        if k_entries_exist:
            raise HTTPException(
                status_code=422,
                detail=f"{label} has no refractiveindex.info k data at {round_float(wavelength_um)} microns.",
            )
        index_imag = 0.0
    elif str(k_entry.get("type", "")) == "tabulated nk":
        index_imag = interpolate_table(k_entry, wavelength_um, 2)
    else:
        index_imag = interpolate_table(k_entry, wavelength_um, 1)

    return validate_index_parts(index_real, index_imag, label)


def ensure_finite(value: float, label: str) -> float:
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise HTTPException(status_code=422, detail=f"{label} must be finite.")
    return numeric_value


def validate_index_parts(real: float, imag: float, label: str):
    """A NON-MAGNETIC optical constant: n > 0, k >= 0.

    Every derived source -- refractiveindex.info, Palik, an upload, an oscillator
    model -- is a measurement or model of a non-magnetic medium, so this is the
    right rule for all of them and it is unchanged. Only a typed constant can
    carry a permeability; see `validate_magnetic_index_parts`.
    """
    real_value = ensure_finite(real, f"{label} n")
    imag_value = ensure_finite(imag, f"{label} k")

    if real_value <= 0:
        raise HTTPException(status_code=422, detail=f"{label} n must be positive.")

    if imag_value < 0:
        raise HTTPException(status_code=422, detail=f"{label} k must be zero or positive.")

    return real_value, imag_value

