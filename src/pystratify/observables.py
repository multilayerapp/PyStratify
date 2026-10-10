"""Shared source normalization, efficiencies and observable assembly."""

import numpy as np

from .problem import FocusedBeam, PlaneWave, PointDipole


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


def solve_focused(problem, outputs=("rates",), *, points=None, reference=None):
    """One wavelength of a focused beam (:mod:`pystratify.focused`): the powers as fractions of
    the beam's power and, with ``"field"`` in ``outputs``, the near field at ``points``.
    Films take their reference stack as a :class:`Problem` ``reference``."""
    from . import focused

    geometry, beam, n, d, wavelength = problem.geometry, problem.source, problem.n, problem.dimensions, problem.wavelength
    if geometry == "films":
        stack = None
        if reference is not None:
            if reference.geometry != "films":
                raise ValueError("a film's reference is a film problem")
            stack = (reference.n, reference.dimensions, reference.coherence)
        result = focused.focused_films(n, d, wavelength, beam, problem.coherence, stack, tolerance=problem.tolerance,
                                       max_evaluations=problem.max_evaluations)
    elif geometry == "spheres":
        result = focused.focused_spheres(d, n, wavelength, beam, l_max=problem.order)
    else:
        result = focused.focused_cylinders(d, n, wavelength, beam, m_max=problem.order, tolerance=problem.tolerance,
                                           max_evaluations=problem.max_evaluations)
    def scalar(v):
        return v[0] if np.ndim(v) else v

    out = dict(detected=scalar(result.detected), reference=scalar(result.reference),
               apparent_absorbance=scalar(result.apparent_absorbance),
               **{k: v[0] for k, v in result.values.items()},
               multipoles={k: (v if k == "orders" else v[0]) for k, v in result.multipoles.items()},
               by_polarization={p: {k: v[0] for k, v in values.items()} for p, values in result.by_polarization.items()},
               diagnostics={k: v[0] for k, v in result.diagnostics.items()})
    if "field" in outputs:
        if points is None:
            raise ValueError("a near field needs points")
        function = dict(films=focused.focused_field_films, spheres=focused.focused_field_spheres,
                        cylinders=focused.focused_field_cylinders)[geometry]
        out["field"] = function(*((n, d) if geometry == "films" else (d, n)), wavelength, beam, points)
    return out


def solve_problem(problem, outputs=("rates",), *, theta=None, phi=0, points=None, reference=None):
    """Solve one wavelength of a resolved problem using the owned response core.

    Plane-wave ``solution`` retains the geometry's native field reconstruction;
    point-source rates share normalization and channel/efficiency names; a
    :class:`FocusedBeam` goes to :func:`solve_focused` (``points``, ``reference``).
    Angles are radians, all lengths share the caller's chosen unit.
    """
    geometry, source, n, d, wavelength = problem.geometry, problem.source, problem.n, problem.dimensions, problem.wavelength
    if getattr(problem, "hydrodynamic", None):
        return _solve_nonlocal(problem, outputs, theta=theta, phi=phi)
    if isinstance(source, FocusedBeam):
        return solve_focused(problem, outputs, points=points, reference=reference)
    if isinstance(source, PlaneWave):
        if geometry == "films":
            from .planar import coh_tmm, layered_response
            angle = 0 if source.angle is None else source.angle
            pols = ("s", "p") if source.polarization == "unpolarized" else (source.polarization,)
            if problem.coherence is not None and "i" in problem.coherence[1:-1]:
                solutions = {p: layered_response(p, n, d, problem.coherence, angle, wavelength) for p in pols}
            else:
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
        diagnostics = dict(converged=r.converged, integration_error=r.error, evaluations=r.evaluations, orders=r.orders, balance_error=r.balance_error, poles=r.poles, grazing_error=r.grazing_error)
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


def _solve_nonlocal(problem, outputs=("rates",), *, theta=None, phi=0):
    """:func:`solve_problem` with hydrodynamic regions (``problem.hydrodynamic``): the same outputs
    from the nonlocal solvers (:mod:`pystratify.nonlocal_sweep`)."""
    geometry, source, n, d, wavelength = problem.geometry, problem.source, problem.n, problem.dimensions, problem.wavelength
    hydro = problem.hydrodynamic
    contact = problem.contact
    if isinstance(source, PlaneWave):
        if geometry == "films":
            from .nonlocal_film import solve_nonlocal_film
            angle = 0 if source.angle is None else source.angle
            result = solve_nonlocal_film(n, d, wavelength, hydro, angle, contact)
            if source.polarization == "p":
                R, T = result["R_p"], result["T_p"]
            elif source.polarization == "s":
                R, T = result["R_s"], result["T_s"]
            else:
                R, T = result["reflectance"], result["transmittance"]
            return dict(solution=result, reflectance=R, transmittance=T, absorptance=1 - R - T)
        if geometry == "spheres":
            from .farfield import cross_sections
            from .nonlocal_sphere import solve_nonlocal_sphere
            solution = solve_nonlocal_sphere(d, n, wavelength, hydro, l_max=problem.order, contact=contact).solution
            return dict(solution=solution, cross_sections=cross_sections(solution))
        from .cylindrical import cross_widths, cylinder_pattern
        from .nonlocal_cylinder import solve_nonlocal_cylinder
        angle = np.pi / 2 if source.angle is None else source.angle
        solution = solve_nonlocal_cylinder(d, n, wavelength, hydro, beta=2 * np.pi * n[-1].real / wavelength * np.cos(angle),
                                           m_max=problem.order, contact=contact)
        result = dict(solution=solution, **cross_widths(solution, source.polarization))
        if theta is not None:
            result["pattern"] = cylinder_pattern(solution, theta, source.polarization)
        return result
    if not isinstance(source, PointDipole):
        raise ValueError("source must be PlaneWave or PointDipole")
    model, upper, lower = None, None, None
    if geometry == "films":
        from .nonlocal_sources import NonlocalFilmSource
        if source.layer is None:
            raise ValueError("planar source needs its region index")
        model = NonlocalFilmSource(n, d, wavelength, source.layer, source.position, source.dipole_type, hydro, contact)
        r = model.rates(problem.tolerance, problem.max_evaluations)
        total, escape, guided, absorbed = r.total, r.escape, r.guided, r.absorbed
        upper, lower = r.upper, r.lower
        diagnostics = dict(converged=r.converged, integration_error=r.error, evaluations=r.evaluations,
                           balance_error=r.balance_error, poles=r.poles)
    elif geometry == "cylinders":
        from .nonlocal_sources import NonlocalCylinderSource
        model = NonlocalCylinderSource(d, n, wavelength, source.position, source.dipole_type, problem.order,
                                       problem.tolerance, hydro, contact)
        r = model.rates(problem.tolerance, problem.max_evaluations)
        total, escape, guided, absorbed = r.total, r.escape, r.guided, r.absorbed
        diagnostics = dict(converged=r.converged, integration_error=r.error, evaluations=r.evaluations, orders=r.orders,
                           balance_error=r.balance_error, poles=r.poles, grazing_error=r.grazing_error)
    else:
        from .nonlocal_sphere import nonlocal_sphere_rates
        layer = int(np.searchsorted(d, source.position))
        if source.position < 0 or any(n[j].imag != 0 or n[j].real <= 0 for j in (layer, len(n) - 1)):
            raise ValueError("source radius is nonnegative and source/exterior must be lossless")
        r = nonlocal_sphere_rates(d, n, wavelength, hydro, source.position, dipole=source.dipole_type,
                                  l_max=problem.order, tol=problem.tolerance, contact=contact)
        total, escape, absorbed = r.total, r.radiative, r.absorbed
        guided = np.zeros(2)
        diagnostics = dict(converged=bool(r.converged and np.max(r.balance_error) <= problem.tolerance), orders=r.orders,
                           balance_error=r.balance_error, absorbed_by_region=r.absorbed_by_region)
    result = rate_result(total, escape, guided, absorbed, geometry, source.intrinsic_quantum_yield, **diagnostics)
    if upper is not None:
        result.update(upper=orientation_average(upper, geometry), lower=orientation_average(lower, geometry))
    if "pattern" in outputs and theta is not None:
        if model is None:
            raise NotImplementedError("emission patterns of spheres with hydrodynamic shells are not available yet")
        raw = model.pattern(theta, phi)
        result["pattern"] = orientation_average(raw, geometry)
        if geometry == "films":
            perpendicular_y = model.pattern(theta, np.asarray(phi) + np.pi / 2)[..., 1]
            result["pattern"][..., -1] = (raw[..., 0] + raw[..., 1] + perpendicular_y) / 3
    return result
