# Measured ellipsometry data

Two real measurements, copied unchanged from the pyElli repository
([PyEllips/pyElli](https://github.com/PyEllips/pyElli), commit `3715c13`, GPL-3.0 like PyStratify).
`examples/ellipsometry_fit.py` fits both; `benchmarks/ellipsometry_codes.py` fits the TiO₂ scan
with pyElli as well and compares the answers.

| file | instrument and format | sample | pyElli path |
|---|---|---|---|
| `complete_ease_example.dat` | J.A. Woollam RC2, CompleteEASE 6.37 export (Ψ/Δ rows `E` with standard deviations, depolarisation rows `dPolE` in %, wavelengths in Å) | "2nm SiO2 on Si", 193–1700 nm at 50°, 60°, 70° | `tests/test_wollam/` |
| `TiO2_400cycles.txt` | SENTECH, SpectraRay export (`; WAVELENGTH` header, Ψ/Δ column pairs per angle) | ALD TiO₂ (400 cycles) on 276 nm SiO₂ on Si, 320–1000 nm at 70.06° | `examples/TiO2 Fit/` |
