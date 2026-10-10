"""Extended-precision global-matrix references for hydrodynamic layered structures.

Independent of the recursion in pystratify.nonlocal_sweep: every amplitude of every region is an
unknown of one linear system built directly from the field expressions and the boundary
conditions (E_t, H_t continuous; n.J = 0 on a hydrodynamic side facing a region without an electron
gas; n.J and (beta^2/omega_p^2) div J continuous between two electron gases).  Columns are scaled
by each basis function at the radius where it is largest, so LU sees O(1) columns; the outgoing
Hankel functions are evaluated through K_nu (DLMF 10.27.8), since H = J + iY cancels at large
imaginary argument even in mpmath.
"""

import mpmath as mp
import numpy as np

I = mp.mpc(0, 1)


def _mp(z):
    return mp.mpc(complex(z))


def _params(model, wavelength, n):
    """(eps_T, eps_b, k_L) in mpmath from a Hydrodynamic model."""
    lam = mp.mpf(wavelength)
    w = mp.mpf(model.plasma_wavelength) / lam
    g = mp.mpf(model.plasma_wavelength) / mp.mpf(model.damping_wavelength) if np.isfinite(model.damping_wavelength) else mp.mpf(0)
    eps_T = _mp(n) ** 2
    eps_b = eps_T + 1 / (w * (w + I * g))
    v2 = mp.mpf(model.fermi_velocity) ** 2
    if model.model == "high-frequency":
        b2 = mp.mpf(3) / 5 * v2
    elif model.model == "thomas-fermi":
        b2 = v2 / 3
    else:
        b2 = v2 * (mp.mpf(3) / 5 * w + I * g / 3) / (w + I * g)
    k0 = 2 * mp.pi / lam
    if model.diffusion:
        b2 = b2 + mp.mpf(model.diffusion) * k0 * (g / w - I)
    kL = mp.sqrt(k0**2 * (1 + I * g / w) * eps_T / (eps_b * b2))
    if mp.im(kL) < 0 or (mp.im(kL) == 0 and mp.re(kL) < 0):
        kL = -kL
    return eps_T, eps_b, kL


def _root(v):
    q = mp.sqrt(v)
    return -q if (mp.im(q) < 0 or (mp.im(q) == 0 and mp.re(q) < 0)) else q


# ----------------------------------------------------------------------------- special functions
def _cyl(kind, m, x):
    """(Z_m(x), Z_m'(x)) for kind 'J' or 'H' (Hankel of the first kind through K)."""
    if kind == "J":
        return mp.besselj(m, x), (mp.besselj(m - 1, x) - mp.besselj(m + 1, x)) / 2
    h = lambda v: 2 / mp.pi * mp.power(I, -v - 1) * mp.besselk(v, -I * x)
    return h(m), (h(m - 1) - h(m + 1)) / 2


def _sph(kind, l, x):
    """(z_l(x), z_l'(x)) spherical Bessel j or Hankel h^(1)."""
    J, dJ = _cyl("J" if kind == "j" else "H", l + mp.mpf(1) / 2, x)
    pre = mp.sqrt(mp.pi / (2 * x))
    return pre * J, pre * (dJ - J / (2 * x))


def _ric(kind, l, x):
    """(u, u') Riccati psi = x j_l or xi = x h_l."""
    z, dz = _sph("j" if kind == "psi" else "h", l, x)
    return x * z, z + x * dz


# ----------------------------------------------------------------------------- generic global solve
def _weights(models, contact):
    """Metal/metal row weights (w_J, w_M) per region: 1 (electrochemical) or Boardman's
    n.v = J_n / (n_0 e) and pressure ~ omega_p^2 M, written independently of the sweep's."""
    if contact == "electrochemical":
        return [(1, 1)] * len(models)
    assert contact == "boardman"
    return [(1, 1) if m is None else (mp.mpf(m.plasma_wavelength) ** 2, 1 / mp.mpf(m.plasma_wavelength) ** 2)
            for m in models]


def _global(regions, radii, n_t, columns, rhs_cols, unknown_index, weights=None):
    """Assemble and solve.  ``columns`` lists (region, trace(r) -> vector, scale) per unknown;
    ``rhs_cols`` the incident-wave columns (region, trace).  Traces are [E_t.., H_t.., Jn, M];
    ``weights[region]`` = (w_J, w_M) multiplies its J_n and M in metal/metal rows."""
    weights = weights or [(1, 1)] * len(regions)
    rows = []
    rhs = []
    for j, r in enumerate(radii):
        hin, hout = regions[j], regions[j + 1]
        sel = list(range(2 * n_t))
        both = hin and hout
        for comp in sel + ([2 * n_t, 2 * n_t + 1] if both else []):
            row = [mp.mpc(0)] * len(columns)
            wt = lambda reg: weights[reg][comp - 2 * n_t] if comp >= 2 * n_t else 1
            for c, (reg, trace, scale) in enumerate(columns):
                if reg == j + 1:
                    row[c] += wt(reg) * trace(r)[comp] / scale
                elif reg == j:
                    row[c] -= wt(reg) * trace(r)[comp] / scale
            b = mp.mpc(0)
            for reg, trace in rhs_cols:
                if reg == j + 1:
                    b -= wt(reg) * trace(r)[comp]
                elif reg == j:
                    b += wt(reg) * trace(r)[comp]
            rows.append(row)
            rhs.append(b)
        if (hin or hout) and not both:  # hard wall on the hydrodynamic side
            side = j if hin else j + 1
            row = [mp.mpc(0)] * len(columns)
            for c, (reg, trace, scale) in enumerate(columns):
                if reg == side:
                    row[c] += trace(r)[2 * n_t] / scale
            b = mp.mpc(0)
            for reg, trace in rhs_cols:
                if reg == side:
                    b -= trace(r)[2 * n_t]
            rows.append(row)
            rhs.append(b)
    M = mp.matrix(rows)
    for i in range(M.rows):  # row equilibration
        s = max(abs(M[i, c]) for c in range(M.cols))
        if s:
            for c in range(M.cols):
                M[i, c] /= s
            rhs[i] /= s
    u = mp.lu_solve(M, mp.matrix(rhs))
    return [u[i] / columns[i][2] for i in range(len(columns))]


# ----------------------------------------------------------------------------- spheres
def sphere_t(radii, n, wavelength, hydrodynamic, l, polarization="TM", dps=60, contact="electrochemical"):
    """T = B/A of the host for order l (TM or TE), hydrodynamic: {region: Hydrodynamic}."""
    with mp.workdps(dps):
        N = len(radii)
        R = [mp.mpf(r) for r in radii]
        k0 = 2 * mp.pi / mp.mpf(wavelength)
        nn = [_mp(v) for v in n]
        hydro = [hydrodynamic.get(j) if polarization == "TM" else None for j in range(N + 1)]
        par = {j: _params(m, wavelength, n[j]) for j, m in enumerate(hydro) if m is not None}
        s = mp.sqrt(l * (l + 1))

        def transverse(reg, kind):
            k, y = k0 * nn[reg], nn[reg]

            def trace(r):
                u, du = _ric(kind, l, k * r)
                x = k * r
                if polarization == "TM":
                    out = [du / x, -I * y * u / x, 0, 0]
                    if reg in par:
                        eT, eb, _ = par[reg]
                        out[2] = (eT - eb) * I * s * u / x**2
                else:
                    out = [u / x, I * y * du / x, 0, 0]
                return out
            return trace

        def longitudinal(reg, kind):
            eT, eb, kL = par[reg]

            def trace(r):
                z, dz = _sph(kind, l, kL * r)
                return [-I * s * z / r, 0, -eb * kL * dz, eT / (eb - eT) * z]
            return trace

        columns = []
        for reg in range(N + 1):
            if reg < N:
                ref = R[reg]
                columns.append((reg, transverse(reg, "psi"), _ric("psi", l, k0 * nn[reg] * ref)[0]))
            if reg > 0:
                ref = R[reg - 1]
                columns.append((reg, transverse(reg, "xi"), _ric("xi", l, k0 * nn[reg] * ref)[0]))
            if reg in par:
                kL = par[reg][2]
                columns.append((reg, longitudinal(reg, "j"), _sph("j", l, kL * R[reg])[0]))
                if reg > 0:
                    columns.append((reg, longitudinal(reg, "h"), _sph("h", l, kL * R[reg - 1])[0]))
        u = _global([m is not None for m in hydro], R, 1, columns, [(N, transverse(N, "psi"))], None,
                    _weights(hydro, contact))
        return complex(u[-1])


# ----------------------------------------------------------------------------- cylinders
def cylinder_t(radii, n, wavelength, hydrodynamic, m, beta, dps=50, contact="electrochemical"):
    """2x2 T block (rows: scattered N, M; columns: incident N, M) for azimuthal order m and
    axial wavenumber beta (PyStratify's vector basis, see pystratify.cylindrical.vectors)."""
    with mp.workdps(dps):
        N = len(radii)
        R = [mp.mpf(r) for r in radii]
        k0 = 2 * mp.pi / mp.mpf(wavelength)
        nn = [_mp(v) for v in n]
        bb = _mp(beta)
        par = {j: _params(mod, wavelength, n[j]) for j, mod in hydrodynamic.items()}

        def transverse(reg, kind, channel):
            k = k0 * nn[reg]
            q = _root(k**2 - bb**2)
            y = nn[reg]

            def trace(r):
                f, df = _cyl(kind, m, q * r)
                if channel == "N":
                    Ez, Ep, Hz, Hp, Er = q * f / k, -bb * m * f / (k * q * r), 0, I * y * df, I * bb * df / k
                else:
                    Ez, Ep, Hz, Hp, Er = 0, -df, -I * y * q * f / k, I * y * bb * m * f / (k * q * r), I * m * f / (q * r)
                out = [Ez, Ep, Hz, Hp, 0, 0]
                if reg in par:
                    eT, eb, _ = par[reg]
                    out[4] = (eT - eb) * Er
                return out
            return trace, q

        def longitudinal(reg, kind):
            eT, eb, kL = par[reg]
            qL = _root(kL**2 - bb**2)

            def trace(r):
                f, df = _cyl(kind, m, qL * r)
                return [I * bb * f, I * m * f / r, 0, 0, -eb * qL * df, eT / (eb - eT) * f]
            return trace, qL

        columns = []
        for reg in range(N + 1):
            for ch in ("N", "M"):
                if reg < N:
                    tr, q = transverse(reg, "J", ch)
                    columns.append((reg, tr, _cyl("J", m, q * R[reg])[0]))
                if reg > 0:
                    tr, q = transverse(reg, "H", ch)
                    columns.append((reg, tr, _cyl("H", m, q * R[reg - 1])[0]))
            if reg in par:
                tr, qL = longitudinal(reg, "J")
                columns.append((reg, tr, _cyl("J", m, qL * R[reg])[0]))
                if reg > 0:
                    tr, qL = longitudinal(reg, "H")
                    columns.append((reg, tr, _cyl("H", m, qL * R[reg - 1])[0]))
        out = np.zeros((2, 2), complex)
        hydro = [j in par for j in range(N + 1)]
        for c, ch in enumerate(("N", "M")):
            u = _global(hydro, R, 2, columns, [(N, transverse(N, "J", ch)[0])], None,
                        _weights([hydrodynamic.get(j) for j in range(N + 1)], contact))
            out[0, c], out[1, c] = complex(u[-2]), complex(u[-1])  # host H columns: N then M
        return out


# ----------------------------------------------------------------------------- films
def film_rp(n, thickness, wavelength, hydrodynamic, K, dps=50, contact="electrochemical"):
    """p-polarised reflection amplitude (H_y convention: r = H_refl / H_inc at the first
    interface) of a stack n[0] (incident) .. n[-1] (exit), thicknesses of n[1:-1], in-plane
    wavenumber K; hydrodynamic: {region: Hydrodynamic} (the exit half-space may be one)."""
    with mp.workdps(dps):
        M = len(n)
        k0 = 2 * mp.pi / mp.mpf(wavelength)
        KK = _mp(K)
        nn = [_mp(v) for v in n]
        z = [mp.mpf(0)]
        for d in thickness:
            z.append(z[-1] + mp.mpf(d))
        par = {j: _params(mod, wavelength, n[j]) for j, mod in hydrodynamic.items()}
        # regions in sweep order: 0 = exit half-space ... M-1 = incident; interface positions zeta
        order = list(range(M - 1, -1, -1))
        zeta = [z[-1] - zz for zz in z[::-1]]  # interface j (sweep order) at zeta_j, increasing
        zeta_of = lambda zz: z[-1] - zz

        def transverse(reg, direction):  # direction +1: forward (+z), -1 backward
            eps = nn[reg] ** 2
            kz = _root((k0 * nn[reg]) ** 2 - KK**2)

            def trace(zt):
                zz = z[-1] - zt
                ph = mp.exp(direction * I * kz * zz)
                out = [direction * kz / (k0 * eps) * ph, ph, 0, 0]
                if reg in par:
                    eT, eb, _ = par[reg]
                    out[2] = (eT - eb) * (-KK / (k0 * eps)) * ph
                return out
            return trace, kz

        def longitudinal(reg, direction):
            eT, eb, kL = par[reg]
            kzL = _root(kL**2 - KK**2)

            def trace(zt):
                zz = z[-1] - zt
                ph = mp.exp(direction * I * kzL * zz)
                return [I * KK * ph, 0, -eb * direction * I * kzL * ph, eT / (eb - eT) * ph]
            return trace, kzL

        regs = order  # sweep region s corresponds to medium order[s]
        columns = []
        for s, reg in enumerate(regs):
            # forward (regular) referred at the interface nearer the incident side, backward at the exit side
            for kind, maker in (("T", transverse), ("L", longitudinal)):
                if kind == "L" and reg not in par:
                    continue
                if s < M - 1:
                    tr, _ = maker(reg, +1)
                    columns.append((s, tr, tr(zeta[s])[1 if kind == "T" else 3]))
                if s > 0:
                    tr, _ = maker(reg, -1)
                    columns.append((s, tr, tr(zeta[s - 1])[1 if kind == "T" else 3]))
        hydro = [reg in par for reg in regs]
        inc, _ = transverse(0, +1)
        u = _global(hydro, [zeta[s] for s in range(M - 1)], 1, columns, [(M - 1, inc)], None,
                    _weights([hydrodynamic.get(reg) for reg in regs], contact))
        # last column: backward wave in the incident medium (reflected), referred at zeta_{M-2}
        refl = transverse(0, -1)[0]
        scale = refl(zeta[M - 2])[1]
        return complex(u[-1] * scale / inc(zeta[M - 2])[1])
