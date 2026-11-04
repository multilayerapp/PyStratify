"""Physical constants, written out so that no result depends on the installed SciPy.

The SI defines h, c and e exactly (2019), so those agree in every SciPy. The vacuum permittivity and the
electron mass are measured, and SciPy carries whichever CODATA edition it was released with: 1.15 moved
from CODATA 2018 (epsilon_0 = 8.8541878128e-12 F/m, m_e = 9.1093837015e-31 kg) to CODATA 2022 (below),
and PyStratify allows SciPy back to 1.10. Fixing CODATA 2022 here keeps a Drude model's numbers the same
in every environment (they moved by ~7e-10 between editions); multilayer.app's browser uses the same values.
"""

import math

#: Planck constant, J s (exact).
H = 6.62607015e-34
#: Reduced Planck constant, J s (exact).
HBAR = H / (2 * math.pi)
#: Speed of light in vacuum, m/s (exact).
C = 299_792_458.0
#: Elementary charge, C (exact).
E = 1.602176634e-19
#: Vacuum permittivity, F/m (CODATA 2022).
EPSILON_0 = 8.8541878188e-12
#: Electron mass, kg (CODATA 2022).
M_E = 9.1093837139e-31
