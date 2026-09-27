"""Electric-quadrupole (E2) versus electric-dipole (E1) emitters near a gold sphere.

A radial dipole p = z and a radial quadrupole Q = q diag(-1, -1, 2), each normalised
to its own free rate in water, beside a gold sphere (R = 40 nm, 600 nm): the
quadrupole's radiative rate grows by orders of magnitude more than the dipole's -
the sphere's induced dipole turns an E2 transition into E1 radiation, the mechanism
behind 'allowed forbidden transitions' near plasmonic particles - while its quenching
climbs as gap^-5 against the dipole's gap^-3.  The far-field pattern of the quadrupole
changes from the four-lobed E2 pattern to a dipole-like one.
"""

import numpy as np
from _common import figure, gold

import pystratify as ps

wavelength, radius, n_h = 600.0, 40.0, 1.33
n = [gold(wavelength), n_h]
gaps = np.geomspace(1, 200, 40)
positions = np.column_stack([np.zeros_like(gaps), np.zeros_like(gaps), radius + gaps])
dipole = ps.emission_rates([radius], n, wavelength, positions, [0, 0, 1.0], tol=1e-8)
quad = ps.emission_rates([radius], n, wavelength, positions, [0, 0, 0], quadrupole=np.diag([-1.0, -1.0, 2.0]), tol=1e-8)
assert np.all(dipole.balance_error < 1e-9) and np.all(quad.balance_error < 1e-9)
for gap in (2.0, 10.0, 50.0):
    i = np.argmin(np.abs(gaps - gap))
    print(
        f"gap {gaps[i]:5.1f} nm: radiative E1 x{dipole.radiative[i]:8.2f}, E2 x{quad.radiative[i]:8.1f}; "
        f"total E1 x{dipole.total[i]:9.1f}, E2 x{quad.total[i]:10.1f}"
    )
small = gaps < 3
for name, r in (("E1", dipole), ("E2", quad)):
    slope = np.polyfit(np.log(gaps[small]), np.log(r.nonradiative[small]), 1)[0]
    print(f"{name} quenching ~ gap^{slope:.2f} below 3 nm")

theta = np.linspace(0, np.pi, 181)
free = ps.dipole_far_field([radius], [n_h, n_h], wavelength, [0, 0, radius + 10], [0, 0, 0], theta, 0.0,
                           quadrupole=np.diag([-1.0, -1.0, 2.0]))  # fmt: skip
near = ps.dipole_far_field([radius], n, wavelength, [0, 0, radius + 10], [0, 0, 0], theta, 0.0,
                           quadrupole=np.diag([-1.0, -1.0, 2.0]))  # fmt: skip

fig, path = figure("quadrupole_emitter", (12, 4))
if fig:
    ax = fig.add_subplot(1, 3, 1)
    for name, r, color in (("E1", dipole, "C0"), ("E2", quad, "C3")):
        ax.loglog(gaps, r.radiative, color, label=f"{name} radiative")
        ax.loglog(gaps, r.nonradiative, color + "--", label=f"{name} nonradiative")
    ax.set_xlabel("gap, nm"), ax.set_ylabel("rate / free rate"), ax.legend(fontsize=8)
    ax = fig.add_subplot(1, 3, 2)
    ax.semilogx(gaps, quad.radiative / dipole.radiative)
    ax.set_xlabel("gap, nm"), ax.set_ylabel("radiative enhancement E2 / E1")
    ax = fig.add_subplot(1, 3, 3, projection="polar")
    ax.set_theta_zero_location("N"), ax.set_theta_direction(-1)  # z (the emitter's axis) up
    for pattern, style, label in ((free, "k:", "E2 alone"), (near, "C3", "E2 10 nm from Au")):
        mirrored = np.concatenate([pattern.directivity, pattern.directivity[::-1]])  # phi = 0 and pi halves
        ax.plot(np.concatenate([theta, 2 * np.pi - theta[::-1]]), mirrored, style, label=label)
    ax.set_yticklabels([]), ax.set_title("directivity, xz plane"), ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(), fig.savefig(path)
