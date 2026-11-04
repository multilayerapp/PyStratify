"""multilayer.app API code displaced by PyStratify (assurance WP07-D2, ruled by Ilia 2026-11-03), frozen as oracles.

Copied VERBATIM from the multilayer.app API: repository `multilayer`, commit `ef9c946`, file `backend/app.py`:

* `round_float`, lines 1817-1818 (the API's serialization rounding, which the next function calls);
* `serialize_ellipsometry_amplitudes`, lines 4305-4324, replaced by `pystratify.planar.psi_delta` (the API keeps
  only its degree / [0, 360) / null formatting);
* `magnetic_field_components` and `magnetic_field_intensity`, lines 5412-5444, replaced by `Hx`, `Hy`, `Hz` in
  `pystratify.planar.position_resolved`.

Only the imports above the copied code are added. Do not edit the functions below; the tests in
`tests/test_planar_api_oracles.py` compare PyStratify against them.
"""
import numpy as np


def round_float(value: float) -> float:
    return round(float(value), 6)


def serialize_ellipsometry_amplitudes(r_s: complex, r_p: complex) -> dict:
    """Return standard ellipsometric angles in degrees for rho = r_p / r_s.

    The TMM core's p basis changes sign at normal incidence, so the established
    project convention uses -r_p/r_s. Delta is wrapped to [0, 360). A phase is
    undefined when either reflected amplitude vanishes and is returned as null.
    """
    s_magnitude = abs(r_s)
    p_magnitude = abs(r_p)
    scale = max(s_magnitude, p_magnitude, 1.0)
    amplitude_floor = 64 * np.finfo(float).eps * scale
    psi = None
    delta = None
    if s_magnitude > amplitude_floor or p_magnitude > amplitude_floor:
        psi = round_float(np.degrees(np.arctan2(p_magnitude, s_magnitude)))
    if s_magnitude > amplitude_floor and p_magnitude > amplitude_floor:
        delta = round_float(np.degrees(np.angle(-r_p * np.conj(r_s))) % 360.0)
        if delta >= 360.0:
            delta = 0.0
    return {"psi_deg": psi, "delta_deg": delta}


def magnetic_field_components(layer_index: int, local_depth: float, tmm_data: dict):
    """H at a point, in units of the incident wave's |H₀|.

    Uses the forward/backward amplitudes position_resolved uses for E, and
    H = y k̂ × E / Z₀ with the admittance index y = n/μ: s gives Hx ∝ −y cosθ (Ef − Eb)
    and Hz ∝ y sinθ (Ef + Eb); p gives Hy ∝ y (Ef + Eb). The incident wave carries
    |H₀| = |y₀| |E₀| with |E₀| = 1.
    """
    if layer_index > 0:
        v, w = tmm_data["vw_list"][layer_index]
    else:
        v, w = 1, tmm_data["r"]
    kz = tmm_data["kz_list"][layer_index]
    th = tmm_data["th_list"][layer_index]
    y_all = tmm_data.get("y_list")
    if y_all is None:
        y_all = tmm_data["n_list"]
    y, y_0 = y_all[layer_index], y_all[0]
    forward = v * np.exp(1j * kz * local_depth)
    d_list = tmm_data["d_list"]
    if 0 < layer_index < len(d_list) - 1 and "backward_bottom" in tmm_data:
        backward = tmm_data["backward_bottom"][layer_index] * np.exp(1j * kz * (d_list[layer_index] - local_depth))
    else:
        backward = w * np.exp(-1j * kz * local_depth)
    scale = y / abs(y_0)
    if tmm_data["pol"] == "s":
        return (-scale * np.cos(th) * (forward - backward), 0j, scale * np.sin(th) * (forward + backward))
    return (0j, scale * (forward + backward), 0j)


def magnetic_field_intensity(layer_index: int, local_depth: float, tmm_data: dict) -> float:
    return float(sum(abs(component) ** 2 for component in magnetic_field_components(layer_index, local_depth, tmm_data)))

