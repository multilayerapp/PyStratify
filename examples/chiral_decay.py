"""Enantioselective decay of chiral molecules near chiral particles.

A randomly oriented chiral molecule - transition moments (d0, m = -i xi d0), xi = +0.1
('right') and -0.1 ('left'), the model of Guzatov & Klimov, New J. Phys. 14, 123009
(2012) - near
1. a lossy chiral (Pasteur) sphere, R = 50 nm, n = 1.6 + 0.02i, kappa = 0.03 + 0.003i;
2. a gold core (R = 40 nm) under a 10 nm chiral shell (n = 1.5 + 0.005i, kappa = 0.02 + 0.002i).
Total, radiative and nonradiative rates of both enantiomers against the gap, and the
discrimination (Gamma_R - Gamma_L) / (Gamma_R + Gamma_L).  The free rate itself does not
depend on handedness in an achiral host; beside a chiral particle both the radiative
and the Ohmic channels do.  Every point satisfies total = radiative + absorbed to <1e-9.

Circular dichroism of the layers (Im kappa) needs magnetic loss as well: a Pasteur
medium is passive only if eps'' mu'' >= kappa''^2, so the chiral layers get mu = 1 + 0.001i.
With mu real, one enantiomer sees gain near the layer (``emission_rates`` notes it).
"""

import numpy as np
from _common import figure, gold

import pystratify as ps

wavelength = 600.0
gaps = np.geomspace(1, 60, 36)
particles = {  # radii, n, kappa, mu
    "chiral sphere": ([50.0], [1.6 + 0.02j, 1.0], [0.03 + 0.003j, 0.0], [1 + 1e-3j, 1.0]),
    "Au core, chiral shell": (
        [40.0, 50.0],
        [gold(wavelength), 1.5 + 0.005j, 1.0],
        [0, 0.02 + 0.002j, 0],
        [1, 1 + 1e-3j, 1],
    ),
}
results = {}
for name, (radii, n, kappa, mu) in particles.items():
    positions = np.column_stack([np.zeros_like(gaps), np.zeros_like(gaps), radii[-1] + gaps])
    runs = {}
    for label, xi in (("right", 0.1), ("left", -0.1)):
        runs[label] = ps.emission_rates(
            radii, n, wavelength, positions, [0, 0, 1.0], magnetic_moment=[0, 0, -1j * xi], kappa=kappa, mu=mu,
            orientation="isotropic", tol=1e-8,
        )  # fmt: skip
        assert np.all(runs[label].balance_error < 1e-9) and not runs[label].notes
    right, left = runs["right"], runs["left"]
    discrimination = (right.total - left.total) / (right.total + left.total)
    results[name] = (runs, discrimination)
    i = np.argmin(np.abs(gaps - 5))
    print(
        f"{name}: at a 5 nm gap Gamma/Gamma_0 = {right.total[i]:.3f} (right), {left.total[i]:.3f} (left); "
        f"discrimination {discrimination[i]:+.2e}, radiated g_lum {right.dissymmetry[i]:+.3f} / {left.dissymmetry[i]:+.3f}"
    )

fig, path = figure("chiral_decay", (12, 4))
if fig:
    for k, (name, (runs, discrimination)) in enumerate(results.items()):
        ax = fig.add_subplot(1, 3, k + 1)
        for label, style in (("right", "-"), ("left", "--")):
            r = runs[label]
            ax.loglog(gaps, r.total, "k" + style, label=f"total ({label})")
            ax.loglog(gaps, r.radiative, "C0" + style, label=f"radiative ({label})")
            ax.loglog(gaps, r.nonradiative, "C3" + style, label=f"nonradiative ({label})")
        ax.set_title(name), ax.set_xlabel("gap, nm"), ax.set_ylabel("rate / free rate")
        if k == 0:
            ax.legend(fontsize=7)
    ax = fig.add_subplot(1, 3, 3)
    for name, (_, discrimination) in results.items():
        ax.semilogx(gaps, discrimination, label=name)
    ax.axhline(0, c="k", ls=":"), ax.set_xlabel("gap, nm"), ax.set_ylabel("(G_R - G_L) / (G_R + G_L)"), ax.legend()
    fig.tight_layout(), fig.savefig(path)
