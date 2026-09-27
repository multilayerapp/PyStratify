"""Conventions shared by the cross-check scripts: literature parameters -> PyStratify inputs."""

import numpy as np


def dbf_to_pasteur(eps, mu, chi):
    """Drude-Born-Fedorov D = eps (E + eta curl E), B = mu (H + eta curl H), chi = k0 eta
    -> (n, kappa, mu) of the Pasteur form used here (validated against the published
    polarisabilities in tests/test_chiral.py).  The branch of n follows passivity
    (Im n >= 0), so double-negative media need a little loss to pick n < 0."""
    eps, mu = complex(eps), complex(mu)
    n2 = eps * mu
    s = 1 - n2 * chi**2
    n = np.sqrt(n2) / s
    if n.imag < 0 or (n.imag == 0 and eps.real < 0 and mu.real < 0):
        n = -n
    return n, n2 * chi / s, mu / s


def klimov_molecule(d0, xi):
    """Transition moments <e|d|g> = d0, <e|m|g> = -i m0 with m0 = xi d0 (Guzatov & Klimov: xi > 0
    'right', d0 parallel to m0; xi < 0 'left') -> the (p, m) of PyStratify (vacuum host: dual =
    current-loop moment).  Normalisation of their rates: (4 k0^3 / 3 hbar)(|d0|^2 + |m0|^2), our
    host normalisation."""
    d0 = np.asarray(d0, dtype=complex)
    return d0, -1j * xi * d0


def klimov_polarisabilities(eps, mu, chi, a):
    """Quasi-static polarisabilities of a small DBF sphere (Guzatov & Klimov, New J. Phys. 14, 123009
    (2012), Eq. 48)."""
    den = (eps + 2) * (mu + 2) - 4 * chi**2 * eps * mu
    a_ee = a**3 * ((eps - 1) * (mu + 2) + 2 * chi**2 * eps * mu) / den
    a_hh = a**3 * ((mu - 1) * (eps + 2) + 2 * chi**2 * eps * mu) / den
    a_eh = a**3 * 3j * chi * eps * mu / den
    return a_ee, a_hh, a_eh, -a_eh
