"""Guzatov & Klimov, "The influence of chiral spherical particles on the radiation of optically
active molecules", New J. Phys. 14, 123009 (2012) (arXiv:1203.5393): Figs. 3-6, recomputed.

Their model: a Drude-Born-Fedorov sphere (radius a) in vacuum, D = eps (E + eta curl E),
B = mu (H + eta curl H), chi = k0 eta; a molecule with <e|d|g> = d0, <e|m|g> = -i m0,
m0 = +0.1 d0 ('right') or -0.1 d0 ('left'), at the surface (r0 -> a); radiative decay
rates over (4 k0^3/3 hbar)(|d0|^2 + |m0|^2).  Figure parameters as stated in the text:

  Fig. 3  eps = 6, mu = 1, rate against k0 a, d0 tangential (x) or normal (z), m0 = 0.1 d0
          along the same axis; chi = 0 and 0.1 (the values of their Fig. 2).
  Fig. 4  k0 a = 1, rate against chi: dielectric (eps 4, mu 1), metal (eps -4, mu 1),
          double-negative (eps -4, mu -1.11; here with a 1e-3 loss to fix the n < 0 branch).
  Fig. 5  k0 a = 0.1, eps = eps' + 0.1i against eps', mu = -1.6, chi = 0.2, left and right.
  Fig. 6  Gamma_left / Gamma_right over (eps', mu'), losses 0.01i, chi = 0.2 (k0 a = 0.1 assumed).

Orientation: normal and tangential as in Fig. 3, isotropic averages elsewhere (the text
does not say for Figs. 4-6 - compare the curve that matches their caption).  The quasi-
static formula (their Eq. 46, orientation-averaged) is overlaid where k0 a << 1.  Writes
CSV files and a PNG beside this script; run from the repository root or this folder.
"""

from pathlib import Path

import numpy as np
from _papers import dbf_to_pasteur, klimov_molecule, klimov_polarisabilities

import pystratify as ps

HERE = Path(__file__).parent / "results"
HERE.mkdir(exist_ok=True)
WAVELENGTH = 1000.0  # arbitrary: only k0 a matters
K0 = 2 * np.pi / WAVELENGTH
AXES = {"normal": np.array([0, 0, 1.0]), "tangential": np.array([1.0, 0, 0])}


def rate(eps, mu, chi, ka, xi, orientation):
    """Radiative rate at contact; orientation 'normal', 'tangential' or 'isotropic'."""
    n, kappa, mu_p = dbf_to_pasteur(eps, mu, chi)
    a = ka / K0
    axis = AXES.get(orientation, AXES["normal"])
    p, m = klimov_molecule(axis, xi)
    f = ps.dipole_far_field(
        [a], [n, 1.0], WAVELENGTH, [0, 0, a], p, 0.0, mu=[mu_p, 1.0], kappa=[kappa, 0.0], magnetic_moment=m,
        orientation="isotropic" if orientation == "isotropic" else "fixed", tol=1e-10, warn=False,
    )  # fmt: skip
    return f.power


def eq46(eps, mu, chi, xi):
    """Their quasi-static, orientation-averaged rate at r0 = a (lengths in units of a)."""
    a_ee, a_hh, a_eh, a_he = klimov_polarisabilities(eps, mu, chi, 1.0)
    num = 1 + xi**2 + 2 * (abs(a_ee - 1j * xi * a_eh) ** 2 + abs(a_he - 1j * xi * a_hh) ** 2)
    return num / (1 + xi**2)


def save(name, header, columns):
    np.savetxt(HERE / f"{name}.csv", np.column_stack(columns), delimiter=",", header=header, comments="")


# Fig. 3
ka3 = np.geomspace(0.02, 3.0, 120)
fig3 = {}
for chi in (0.0, 0.1):
    for orientation in ("normal", "tangential"):
        for xi in (0.1, -0.1):
            fig3[chi, orientation, xi] = np.array([rate(6.0, 1.0, chi, x, xi, orientation) for x in ka3])
save("fig3", "k0a," + ",".join(f"chi={c} {o} xi={x:+}" for c, o, x in fig3), [ka3, *fig3.values()])

# Fig. 4
chi4 = np.linspace(0.0, 0.3, 121)
materials = {"dielectric": (4.0, 1.0), "metal": (-4.0, 1.0), "double-negative": (-4.0 + 1e-3j, -1.11 + 1e-3j)}
fig4 = {name: np.array([rate(e, m_, c, 1.0, 0.1, "isotropic") for c in chi4]) for name, (e, m_) in materials.items()}
save("fig4", "chi," + ",".join(fig4), [chi4, *fig4.values()])

# Fig. 5
eps5 = np.linspace(-8.0, 4.0, 241)
fig5 = {}
for xi, label in ((0.1, "right"), (-0.1, "left")):
    fig5[label] = np.array([rate(e + 0.1j, -1.6, 0.2, 0.1, xi, "isotropic") for e in eps5])
    fig5[label + " Eq.46"] = np.array([eq46(e + 0.1j, -1.6, 0.2, xi) for e in eps5])
save("fig5", "eps_real," + ",".join(fig5), [eps5, *fig5.values()])

# Fig. 6
eps6, mu6 = np.linspace(-6.0, 2.0, 81), np.linspace(-3.0, 1.0, 41)
ratio = np.array([[rate(e + 0.01j, m_ + 0.01j, 0.2, 0.1, -0.1, "isotropic") / rate(e + 0.01j, m_ + 0.01j, 0.2, 0.1, 0.1, "isotropic")
                   for e in eps6] for m_ in mu6])  # fmt: skip
np.savetxt(
    HERE / "fig6_left_over_right.csv",
    ratio,
    delimiter=",",
    header=f"rows mu' {mu6[0]}..{mu6[-1]} ({mu6.size}), columns eps' {eps6[0]}..{eps6[-1]} ({eps6.size})",
)
i, j = np.unravel_index(np.argmax(ratio), ratio.shape)
print(f"Fig. 6: largest Gamma_L/Gamma_R = {ratio[i, j]:.1f} at eps' = {eps6[j]:.2f}, mu' = {mu6[i]:.2f}")
k, m_ = np.unravel_index(np.argmin(ratio), ratio.shape)
print(f"        smallest = {ratio[k, m_]:.3f} at eps' = {eps6[m_]:.2f}, mu' = {mu6[k]:.2f}")
for name, v in fig4.items():
    print(f"Fig. 4 {name}: rate from {v.min():.3g} to {v.max():.3g} over chi in [0, 0.3]")
# Eq. 46 is a dipole approximation of the sphere: it has the l = 1 resonance (near eps' = -1.2 here)
# but not the magnetic quadrupole one of a mu ~ -3/2 sphere, which the exact rates show near eps' = 3
low = eps5 < 0.5
for label in ("right", "left"):
    exact, dipolar = fig5[label], fig5[label + " Eq.46"]
    print(
        f"Fig. 5 {label}: l = 1 peak at eps' = {eps5[np.argmax(exact * low)]:.2f} (exact), "
        f"{eps5[np.argmax(dipolar * low)]:.2f} (Eq. 46); for eps' < 0.5 they differ by up to "
        f"{np.max(np.abs(exact[low] / dipolar[low] - 1)):.0%}; second peak at eps' = {eps5[np.argmax(exact * ~low)]:.2f}"
    )

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for (chi, orientation, xi), v in fig3.items():
    axes[0].loglog(ka3, v, ls="-" if xi > 0 else "--", c={(0.0, "normal"): "C0", (0.0, "tangential"): "C1",
                   (0.1, "normal"): "C2", (0.1, "tangential"): "C3"}[chi, orientation],
                   label=f"chi {chi} {orientation} {'R' if xi > 0 else 'L'}")  # fmt: skip
axes[0].set(xlabel="k0 a", ylabel="radiative rate / free", title="Fig. 3: eps 6, contact"), axes[0].legend(fontsize=6)
for name, v in fig4.items():
    axes[1].semilogy(chi4, v, label=name)
axes[1].set(xlabel="chi", title="Fig. 4: k0 a = 1, right, isotropic"), axes[1].legend(fontsize=7)
for name, v in fig5.items():
    axes[2].semilogy(eps5, v, ls=":" if "Eq" in name else "-", label=name)
axes[2].set(xlabel="eps'", title="Fig. 5: mu -1.6, chi 0.2, k0 a 0.1"), axes[2].legend(fontsize=7)
im = axes[3].pcolormesh(eps6, mu6, np.log10(ratio), shading="auto", cmap="RdBu_r")
fig.colorbar(im, ax=axes[3], label="log10 Gamma_L / Gamma_R")
axes[3].set(xlabel="eps'", ylabel="mu'", title="Fig. 6: chi 0.2, losses 0.01i")
fig.tight_layout()
fig.savefig(HERE / "guzatov_klimov_2012.png", dpi=110)
