"""Shared source normalization, efficiencies and observable assembly."""

import numpy as np

from .problem import PlaneWave, PointDipole


def orientation_average(values, geometry):
    values = np.asarray(values)
    isotropic = values.mean(axis=-1) if geometry == "cylinders" else (values[..., 0] + 2 * values[..., 1]) / 3
    return np.concatenate((values, isotropic[..., None]), axis=-1)


def rate_result(total, escape, guided, absorbed, geometry, q0, **diagnostics):
    total, escape, guided, absorbed = (orientation_average(a, geometry) for a in (total, escape, guided, absorbed))
    denominator = 1 - q0 + q0 * total
    with np.errstate(all="ignore"):
        escape_efficiency = np.divide(q0 * escape, denominator, out=np.zeros_like(total), where=denominator != 0)
        photonic_efficiency = np.divide(q0 * (escape + guided), denominator, out=np.zeros_like(total), where=denominator != 0)
        extraction = np.divide(escape, total, out=np.zeros_like(total), where=total != 0)
    names = {"films": ["perpendicular", "parallel", "isotropic"], "spheres": ["radial", "tangential", "isotropic"], "cylinders": ["radial", "azimuthal", "axial", "isotropic"]}[geometry]
    return dict(total=total, escape=escape, guided=guided, absorbed=absorbed,
                eta_escape=escape_efficiency, eta_photonic=photonic_efficiency,
                extraction_fraction=extraction, lifetime_factor=denominator,
                orientations=names, normalization="local homogeneous source medium", diagnostics=diagnostics)


def sphere_center(radii, n, wavelength, dipole):
    from .solver import solve, TM, TE, _side
    from .riccati import log_riccati
    sol = solve(radii, n, wavelength, l_max=1)
    p = TM if dipole == "electric" else TE
    total = 1 + np.exp(sol.log_s[p, 0, 0, 0]).real
    reference = (sol.n[0, 0] * sol.mu[0, 0] / (sol.n[0, -1] * sol.mu[0, -1])).real
    radiative = reference * abs(np.exp(-sol.log_b_out[p, 0, 0, 0])) ** 2
    absorbed = 0.0
    free_flux = (sol.n[0, 0] / sol.mu[0, 0]).real / abs(sol.k[0, 0]) ** 2
    for j in range(sol.n_shells):
        if sol.n[0, j].imag == 0:
            continue
        fluxes = []
        for radius in (0 if j == 0 else radii[j - 1], radii[j]):
            if radius == 0:
                fluxes.append(0.0)
                continue
            x = np.array([sol.k[0, j] * radius])
            lp, lx, _, _, d1, d3 = _side(x, *log_riccati(x, 2), np.array([1]))
            lb = sol.log_b_out[p, j, 0, 0] - sol.log_b_out[p, 0, 0, 0]
            la = lb + sol.log_s[p, j, 0, 0]
            regular, outgoing = np.exp(la + lp[0, 0]), np.exp(lb + lx[0, 0])
            value, derivative = regular + outgoing, regular * d1[0, 0] + outgoing * d3[0, 0]
            y = sol.n[0, j] / sol.mu[0, j]
            flux = np.real(-1j * np.conj(y) * derivative * np.conj(value)) if p == TM else np.real(1j * np.conj(y) * value * np.conj(derivative))
            fluxes.append(flux / abs(sol.k[0, j]) ** 2 / free_flux)
        absorbed += fluxes[0] - fluxes[1]
    return np.full(2, total), np.full(2, radiative), np.full(2, absorbed)


def solve_problem(problem, outputs=("rates",), *, theta=None, phi=0):
    """Solve one wavelength of a resolved problem using the owned response core.

    Plane-wave ``solution`` retains the geometry's native field reconstruction;
    point-source rates share normalization and channel/efficiency names.
    Angles are radians, all lengths share the caller's chosen unit.
    """
    geometry, source, n, d, wavelength = problem.geometry, problem.source, problem.n, problem.dimensions, problem.wavelength
    if isinstance(source, PlaneWave):
        if geometry == "films":
            from .planar import coh_tmm
            angle = 0 if source.angle is None else source.angle
            pols = ("s", "p") if source.polarization == "unpolarized" else (source.polarization,)
            solutions = {p:coh_tmm(p, n, d, angle, wavelength) for p in pols}
            R, T = np.mean([s["R"] for s in solutions.values()]), np.mean([s["T"] for s in solutions.values()])
            return dict(solution=solutions, reflectance=R, transmittance=T, absorptance=1-R-T)
        if geometry == "spheres":
            from .solver import solve
            from .farfield import cross_sections
            solution = solve(d, n, wavelength, l_max=problem.order)
            return dict(solution=solution, cross_sections=cross_sections(solution))
        from .cylindrical import solve_cylinder, cross_widths, cylinder_pattern
        angle = np.pi / 2 if source.angle is None else source.angle
        solution = solve_cylinder(d, n, wavelength, beta=2*np.pi*n[-1].real/wavelength*np.cos(angle), m_max=problem.order)
        result = dict(solution=solution, **cross_widths(solution, source.polarization))
        if theta is not None:
            result["pattern"] = cylinder_pattern(solution, theta, source.polarization)
        return result
    if not isinstance(source, PointDipole):
        raise ValueError("source must be PlaneWave or PointDipole")
    model, upper, lower = None, None, None
    if geometry == "films":
        from .planar_emission import FilmSource
        if source.layer is None:
            raise ValueError("planar source needs its region index")
        model = FilmSource(n, d, wavelength, source.layer, source.position, source.dipole_type)
        r = model.rates(problem.tolerance, problem.max_evaluations)
        total, escape, guided, absorbed = r.total, r.escape, r.guided, r.absorbed
        upper, lower = r.upper, r.lower
        diagnostics = dict(converged=r.converged, integration_error=r.error, evaluations=r.evaluations, balance_error=r.balance_error, poles=r.poles)
    elif geometry == "cylinders":
        from .cylinder_emission import CylinderSource
        model = CylinderSource(d, n, wavelength, source.position, source.dipole_type, problem.order, problem.tolerance)
        r = model.rates(problem.tolerance, problem.max_evaluations)
        total, escape, guided, absorbed = r.total, r.escape, r.guided, r.absorbed
        diagnostics = dict(converged=r.converged, integration_error=r.error, evaluations=r.evaluations, orders=r.orders, balance_error=r.balance_error, poles=r.poles)
    else:
        from .decay import decay_rates
        layer = int(np.searchsorted(d, source.position))
        if source.position < 0 or any(n[j].imag != 0 or n[j].real <= 0 for j in (layer, len(n)-1)):
            raise ValueError("source radius is nonnegative and source/exterior must be lossless")
        if source.position == 0:
            total, escape, absorbed = sphere_center(d, n, wavelength, source.dipole_type)
            diagnostics = dict(converged=True, orders=1, balance_error=abs(total-escape-absorbed)/np.maximum(1,abs(total)))
        else:
            r = decay_rates(d, n, wavelength, source.position, dipole=source.dipole_type, normalization="shell", tol=problem.tolerance, l_max=problem.order, l_cap=1500, warn=False)
            total, escape, absorbed = r.total[0], r.radiative[0], r.nonradiative[0]
            diagnostics = dict(converged=bool(r.converged[0]), orders=int(r.orders_used[0]), balance_error=r.balance_error[0])
        guided = np.zeros(2)
        diagnostics["converged"] = diagnostics["converged"] and np.max(diagnostics["balance_error"]) <= problem.tolerance
    result = rate_result(total, escape, guided, absorbed, geometry, source.intrinsic_quantum_yield, **diagnostics)
    if upper is not None:
        result.update(upper=orientation_average(upper, geometry), lower=orientation_average(lower, geometry))
    if "pattern" in outputs and theta is not None:
        if model is not None:
            raw = model.pattern(theta, phi)
            result["pattern"] = orientation_average(raw, geometry)
            if geometry == "films":
                perpendicular_y = model.pattern(theta, np.asarray(phi) + np.pi / 2)[..., 1]
                result["pattern"][..., -1] = (raw[..., 0] + raw[..., 1] + perpendicular_y) / 3
        elif source.position == 0:
            theta, phi = np.broadcast_arrays(theta, phi)
            raw = np.stack((np.sin(theta)**2, 1-np.sin(theta)**2*np.cos(phi)**2), axis=-1) * (3/(8*np.pi)) * escape
            result["pattern"] = orientation_average(raw, geometry)
            result["pattern"][..., -1] = result["escape"][-1] / (4 * np.pi)
        else:
            from .emission import dipole_far_field
            raw = []
            for moment in ([0,0,1], [1,0,0]):
                p = dipole_far_field(d, n, wavelength, [0,0,source.position], moment, theta, phi, dipole=source.dipole_type, l_max=problem.order, tol=problem.tolerance)
                if source.dipole_type == "electric":
                    reference_ratio = n[-1].real/n[layer].real
                else:
                    reference_ratio = (n[-1].real/n[layer].real)**3
                raw.append(p.power_density * reference_ratio)
            # The isotropic pattern needs both tangential axes before averaging.
            p_y = dipole_far_field(d, n, wavelength, [0,0,source.position], [0,1,0], theta, phi, dipole=source.dipole_type, l_max=problem.order, tol=problem.tolerance)
            isotropic = (raw[0] + raw[1] + p_y.power_density * reference_ratio) / 3
            result["pattern"] = np.stack((raw[0], raw[1], isotropic), axis=-1)
    return result
