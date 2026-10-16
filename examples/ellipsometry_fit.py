"""Fit a measured ellipsometry file: the native oxide on a silicon wafer, then a TiO2 film on SiO2/Si.

Both measurements are pyElli's published examples (data/ellipsometry/README.md): a J.A. Woollam RC2
scan of "2 nm SiO2 on Si" at 50/60/70 deg, and a SENTECH scan of an ALD TiO2 film at 70.06 deg.
Silicon is Green 2008 and the oxide Malitson's Sellmeier, both from refractiveindex.info.
"""

from pathlib import Path

import numpy as np
from _common import figure

from pystratify import dispersion as dm
from pystratify import ellipsometry as el

here = Path(__file__).resolve().parent
silicon = dm.Table.from_refractiveindex(here.parent / "crosscheck/classics/data/Si_Green-2008.yml")
oxide = dm.Sellmeier(B=(0.6961663, 0.4079426, 0.8974794), C=(0.0684043**2, 0.1162414**2, 9.896161**2))

# 1. Native oxide: one free thickness, three angles, the file's own standard deviations as weights
wafer = el.read_woollam(here / "data/ellipsometry/complete_ease_example.dat").select(wavelength=(250, 1000))
params = el.Parameters().add("d_oxide", 5.0, min=0, max=50)
native = el.fit(lambda p: el.Sample([el.Layer(oxide, p["d_oxide"])], substrate=silicon), params, wafer,
                grid={"d_oxide": np.arange(0.0, 20.0, 1.0)})
print("native oxide (Woollam RC2):\n" + native.report())

# 2. TiO2 on 276 nm of thermal oxide: a Cauchy film whose thickness and dispersion are fitted together
film = el.read_spectraray(here / "data/ellipsometry/TiO2_400cycles.txt").select(wavelength=(400, 800))
params = el.Parameters().add("d", 20, min=0, max=500).add("A", 2.2, min=1.5, max=3).add("B", 0.05).add("C", 0.0)


def tio2_sample(p):
    tio2 = dm.Cauchy(A=p["A"], B=p["B"], C=p["C"])
    return el.Sample([el.Layer(tio2, p["d"]), el.Layer(dm.Cauchy(1.452, 0.0036), 276.36)], substrate=silicon)


tio2 = el.fit(tio2_sample, params, film, grid={"d": np.arange(5, 200, 5.0)})
print("\nTiO2 film (SENTECH):\n" + tio2.report())

fig, path = figure("ellipsometry_fit")
if fig:
    for k, (result, title) in enumerate(((native, "native oxide on Si"), (tio2, "TiO2 / SiO2 / Si"))):
        model = result.model_spectra()
        ax = fig.add_subplot(1, 2, k + 1)
        for angle in result.data.angles:
            on = result.data.angle == angle
            ax.plot(result.data.wavelength[on], result.data.psi[on], ".", ms=2, color="C0")
            ax.plot(result.data.wavelength[on], result.data.delta[on], ".", ms=2, color="C1")
            ax.plot(model.wavelength[on], model.psi[on], "-", color="k", lw=0.8)
            wrapped = np.where(np.abs(np.diff(model.delta[on], prepend=model.delta[on][0])) > 180, np.nan, 1.0)
            ax.plot(model.wavelength[on], model.delta[on] * wrapped, "-", color="k", lw=0.8)
        ax.set_title(f"{title}: MSE {result.mse:.2f}"), ax.set_xlabel("wavelength, nm"), ax.set_ylabel("Ψ, Δ (deg)")
    fig.tight_layout(), fig.savefig(path)
