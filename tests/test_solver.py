"""The overflow-free RTMM solver against exact references."""

import mpmath as mp
import numpy as np
import pytest

import pystratify as ps
from pystratify.legacy import ric_h, ric_j, t_mat
from pystratify.riccati import log_riccati

from .mp_reference import coeffs

RAD = np.array([10.0, 13.0, 36.0, 48.0])
REF = np.array([1.45, 0.2 + 3.8j, 1.45, 0.25 + 3.5j, 1.33])  # OSAC Fig. 2(b) matryoshka
MU = np.array([1, 1.3, 1, 1, 1.0])
LAM = 690.0


def _plane_coeffs_legacy(T, pol):
    N = T.n_shells
    tN = T.T(N + 1, pol)
    r = tN[:, 1, 0] / tN[:, 0, 0]
    A = (
        [1 / tN[:, 0, 0]]
        + [T.M(n, pol)[:, 0, 0] + T.M(n, pol)[:, 0, 1] * r for n in range(2, N + 1)]
        + [np.ones_like(r)]
    )
    B = [np.zeros_like(r)] + [T.M(n, pol)[:, 1, 0] + T.M(n, pol)[:, 1, 1] * r for n in range(2, N + 1)] + [r]
    return np.array(A), np.array(B)


def test_matches_direct_products_at_low_order():
    L = 8
    T = t_mat(RAD, REF, MU, LAM, np.arange(1, L + 1))
    S = ps.solve(RAD, REF, MU, LAM, L)
    assert np.allclose(S.a[0], T.a, rtol=1e-11) and np.allclose(S.b[0], T.b, rtol=1e-11)
    for p in "em":
        A, B = _plane_coeffs_legacy(T, p)
        assert np.allclose(np.exp(S.logA[p][:, 0]), A, rtol=1e-9)
        assert np.allclose(np.exp(S.logB[p][1:, 0]), B[1:], rtol=1e-9)
        for n in range(1, 5):
            m = T.M(n, p)
            assert np.allclose(np.exp(S.logBo[p][n - 1, 0]), m[:, 1, 1], rtol=1e-11)


@pytest.mark.parametrize(
    "rad, ref",
    [
        ([50, 70, 90], [1.45 + 0.2j, 2.0, 1.6 + 0.05j, 1.0]),
        (list(RAD), list(REF)),
        ([30, 31, 60], [2.5 + 0.01j, 0.05 + 4.0j, 1.5, 1.33]),  # 1-nm Ag-like film
    ],
)
def test_amplitudes_against_60_digit_reference(rad, ref):
    L = 160
    mu = [1] * len(ref)
    S = ps.solve(rad, ref, mu, LAM, L)

    def err(logv, v):
        if v == 0:
            return 0.0
        d = complex(mp.log(v)) - logv
        return abs(d.real) + abs(np.exp(1j * d.imag) - 1)

    for p in "em":
        for n in (1, 7, 40, 100, 160):
            reg, out = coeffs(rad, ref, mu, LAM, n, p)
            for s in range(len(ref)):
                assert err(S.logA[p][s, 0, n - 1], reg[s][0]) < 1e-10, (p, n, s)
                assert err(S.logB[p][s, 0, n - 1], reg[s][1]) < 1e-7, (p, n, s)
                assert err(S.logBo[p][s, 0, n - 1], out[s][1]) < 1e-10, (p, n, s)


def test_boundary_conditions_hold_at_every_order():
    """The direct products violate them by ~50 % at l = 14 in this particle."""
    L = 200
    S = ps.solve(RAD, REF, MU, LAM, L)
    l = S.l
    k = S.k[0]
    for p in "em":
        for j in range(RAD.size):
            x, xt = k[j] * RAD[j], k[j + 1] * RAD[j]
            lp, lx = log_riccati(np.array([x, xt]), L)
            eta, mur = REF[j] / REF[j + 1], MU[j] / MU[j + 1]
            cf, cd = (eta, mur) if p == "m" else (mur, eta)

            def f(s, i, deriv):
                la, lb = S.logA[p][s, 0], S.logB[p][s, 0]
                xx = (x, xt)[i]
                shift = np.maximum(la.real + lp[i, l].real, lb.real + lx[i, l].real)
                v = np.exp(la + lp[i, l] - shift) + np.exp(lb + lx[i, l] - shift)
                if deriv:
                    v = np.exp(la + lp[i, l - 1] - shift) + np.exp(lb + lx[i, l - 1] - shift) - l / xx * v
                return v, shift

            for deriv, c in ((False, cf), (True, cd)):
                (vi, si), (vo, so) = f(j, 0, deriv), f(j + 1, 1, deriv)
                lhs = np.log(vi) + si
                rhs = np.log(c * vo) + so
                d = lhs - rhs
                assert np.max(np.abs(d.real) + np.abs(np.exp(1j * d.imag) - 1)) < 1e-9, (p, j, deriv)


def test_batch_equals_single():
    lam = np.array([500.0, 600.0, 700.0])
    ref = np.array([[1.45, 0.3 + 2.0j + 0.1j * i, 1.33] for i in range(3)])
    Sb = ps.solve([40, 50], ref, [1, 1, 1], lam, 12)
    for i in range(3):
        Si = ps.solve([40, 50], ref[i], [1, 1, 1], lam[i], 12)
        assert np.allclose(Sb.a[i], Si.a[0]) and np.allclose(Sb.b[i], Si.b[0])
        for p in "em":
            assert np.allclose(Sb.logA[p][:, i], Si.logA[p][:, 0])


def test_large_size_parameter_and_extreme_absorption():
    # x ~ 1000 dielectric sphere and a strongly absorbing thin shell: no overflow, Q_ext -> ~2
    lam = 500.0
    r = 1000 * lam / (2 * np.pi)
    S = ps.solve([r], [1.33 + 1e-4j, 1.0], [1, 1], lam, ps.l_max(r, 1.0, lam))
    cs = ps.cross_sections(S)
    assert np.isfinite(cs.q_ext).all() and 1.9 < cs.q_ext[0] < 2.3
    S = ps.solve([400, 402], [1.5, 0.1 + 40j, 1.0], [1, 1, 1], lam, 60)
    assert all(np.all(np.isfinite(S.logA[p][:, :, :].real) | (S.logA[p].real == -np.inf)) for p in "em")


def test_input_validation():
    with pytest.raises(ValueError):
        ps.solve([50, 40], [1.5, 1.4, 1.0], [1, 1, 1], 500, 5)
    with pytest.raises(ValueError):
        ps.solve([50], [1.5, 1.4, 1.0], [1, 1, 1], 500, 5)
    with pytest.raises(ValueError):
        ps.solve([50], [0, 1.0], [1, 1], 500, 5)


def test_direct_products_lose_the_boundary_conditions():
    """AUDIT.md M9: STRATIFY's composite-matrix products (legacy t_mat) already
    violate the interface conditions at l ~ 14 inside the 3-nm Au shell of the
    OSAC Fig. 2(b) matryoshka; the scaled solver does not (test above)."""
    l = np.arange(1, 15)
    T = t_mat(RAD, REF, MU, LAM, l)
    A, B = _plane_coeffs_legacy(T, "e")
    k = 2 * np.pi * REF / LAM
    x, xt = k[1] * RAD[1], k[2] * RAD[1]
    v_in = A[1] * ric_j(l, x) + B[1] * ric_h(l, x)
    v_out = A[2] * ric_j(l, xt) + B[2] * ric_h(l, xt)
    mismatch = np.abs(v_in - MU[1] / MU[2] * v_out) / np.abs(v_in)
    assert mismatch[:4].max() < 1e-10 and mismatch[-1] > 0.1
