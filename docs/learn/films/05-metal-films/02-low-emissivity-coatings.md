---
id: low-emissivity-coatings
title: "Low-emissivity glazing: dielectric / silver / dielectric"
geometry: films
chapter: Metal films
topic: low-e
sources: [plane_wave]
summary: >-
  An ultrathin silver film between two dielectric layers lets daylight through and reflects heat: the Drude metal
  below its plasma frequency, the induced-transmission design that antireflects the silver in the visible, Kirchhoff's
  law and the Hagen–Rubens emissivity of a conductor, the sheet-resistance rule of thumb, the glazing figures of
  merit, and why double- and triple-silver stacks steepen the edge.
examples: [low-e-coating]
references:
- cite: R. J. Martín-Palma et al., Sol. Energy Mater. Sol. Cells 53, 55 (1998)
  doi: 10.1016/S0927-0248(98)00007-5
- cite: H. A. Macleod, Thin-Film Optical Filters, 3rd ed. (Institute of Physics Publishing, 2001), ch. 7 and 9
- cite: C. G. Granqvist, Appl. Phys. A 52, 83 (1991)
  doi: 10.1007/BF00323721
- cite: E. Hagen and H. Rubens, Ann. Phys. 316, 873 (1903)
  doi: 10.1002/andp.19033160811
- cite: Ciesielski et al. (2017), silver n,k to 20.9 µm, refractiveindex.info main/Ag/nk/Ciesielski.yml
  url: https://refractiveindex.info/?shelf=main&book=Ag&page=Ciesielski
---

A window should let daylight in and keep heat where it is. The two wavelength ranges are well separated: sunlight
carries its energy between 0.3 and 2.5 µm, the thermal radiation of a room at 20 °C peaks near 10 µm. Silver does
both jobs, if it is thin enough and dressed in the right dielectrics, and the physics is the same Drude response as
in the [beam-splitter section](learn/films/beam-splitters.html), read at two very different frequencies.

## A metal at two frequencies

Below the plasma frequency a Drude metal has $\varepsilon \approx -\omega_p^2/\omega^2$: at 550 nm silver's
$|\varepsilon|$ is about 13, at 10 µm it is about $4\,000$ with an imaginary part of similar size. The reflectance of
the bulk metal is 97 % in the visible and 99.5 % in the thermal infrared, and the skin depth shrinks from 12 nm to a
few nanometres. A film of 10–15 nm is therefore thin compared with the visible skin depth, so much of the visible
leaks through, yet is already optically thick in the infrared, where it reflects as if it were bulk. That asymmetry
is the whole principle; the dielectric layers only tidy up the visible.

## Antireflecting a metal: induced transmission

Bare, a 12 nm silver film on glass still reflects about half of the visible, because its admittance (large and
imaginary) is far from that of glass or air. Two dielectric layers of index around 2 (ZnO, TiO₂, SnO₂, Si₃N₄) on
either side act as an **induced-transmission filter** (Macleod ch. 7): each is roughly a quarter wave at the design
wavelength, and the pair rotates the stack's admittance locus so that the silver, seen from the glass, is matched to
the air. Reflections from the stack then cancel across the visible, and transmittance rises from about 50 % to 80 %,
with the remainder absorbed rather than reflected. The potential transmittance of a 12 nm silver film in the
visible, the most any matching layers can extract, is about 85 %, and commercial coatings get within a few percent of
it. In the infrared the silver's admittance is far too large for any dielectric of index 2 to match, the quarter-wave
layers are a few hundredths of a wave thick anyway, and the stack reflects as bare silver would.

## Emissivity is absorptance

By Kirchhoff's law a surface's spectral emissivity equals its spectral absorptance at the same wavelength and
direction. For an opaque coated pane $T \approx 0$ in the thermal infrared, so

$$
\varepsilon(\lambda) = A(\lambda) = 1 - R(\lambda) ,
$$

and the quantity glazing engineers quote, the **hemispherical emissivity**, is this absorptance averaged over the
room-temperature black-body spectrum and over angle. Bare glass, which absorbs strongly through its Si–O vibrations
across the whole window, has $\varepsilon \approx 0.84$; a single-silver low-e coating has 0.04–0.10, a triple-silver
stack about 0.02. A pane that emits a twentieth of what bare glass does loses a correspondingly smaller share of a
room's heat by radiation, which is about half of the loss through a window in winter.

For a good conductor the emissivity has a closed form. Hagen and Rubens (1903) showed that far below the plasma
frequency, where $\varepsilon \approx i\sigma/(\varepsilon_0\omega)$ with $\sigma$ the dc conductivity, the normal
emissivity of a thick metal is

$$
\varepsilon_\mathrm{n} \approx 2\sqrt{\frac{2\,\varepsilon_0\,\omega}{\sigma}} ,
$$

about 0.01 for silver at 10 µm, falling as $\sqrt{\omega}$ toward longer wavelengths. For a thin film the same physics
is expressed through its **sheet resistance** $R_\square = 1/(\sigma d)$: a film on a transparent substrate has, to a
good approximation, $\varepsilon \approx 4 R_\square / Z_0$ with $Z_0 = 377\,\Omega$ the impedance of free space, that
is about 0.01 per ohm per square. A 12 nm silver film has $R_\square \approx 3\,\Omega/\square$ and an emissivity near
0.03, and the rule explains at a glance why emissivity and electrical conductivity go together, why a thicker film
is always a better emitter-suppressor, and why transparent conductive oxides (ITO at 20–30 Ω/□) make hard-coat low-e
glass of emissivity 0.15–0.2 but can never reach silver's.

::: example low-e-coating

Thicker silver raises the infrared reflectance and lowers the visible transmittance; the trade-off is the one
number a glazing designer chooses. The silver data here extend to 20 µm (Ciesielski et al., 2017); the classic study
of this design is Martín-Palma et al. (1998), and Granqvist's review (1991) is the standard account of spectrally
selective coatings.

## Figures of merit, and the edge

Three numbers describe a glazing: the visible transmittance $T_\mathrm{vis}$ (weighted by the eye's sensitivity), the
solar transmittance or solar heat gain coefficient $g$ (weighted by the solar spectrum, including the fraction of
absorbed heat re-emitted inward), and the emissivity. A winter window wants high $g$ and low $\varepsilon$; a summer
window in a hot climate wants low $g$ with high $T_\mathrm{vis}$, that is a sharp cut-off at the end of the visible, near
700 nm, with the near infrared of sunlight (half its energy) reflected. The selectivity $T_\mathrm{vis}/g$ of a single
silver film is limited, because its transmittance falls off gently through the near infrared. Repeating the
dielectric/silver/dielectric unit two or three times makes the stack an induced-transmission filter with two or
three coupled cavities, which, exactly as for the [multi-cavity bandpass filter](learn/films/bandpass-filters.html),
squares the top and steepens the edge: triple-silver coatings reach $T_\mathrm{vis}/g \approx 2$, close to the
theoretical limit for the solar spectrum. The price is a lower $T_\mathrm{vis}$ and a slight colour, and the design
problem is to keep the reflected colour neutral, which the dielectric thicknesses control.
