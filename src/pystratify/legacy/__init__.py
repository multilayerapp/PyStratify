"""Direct transcription of STRATIFY's transfer-matrix products (``util/t_mat.m``).

Kept only as the bridge to the original MATLAB code in the tests: the
composite matrices are products of Riccati-Bessel functions and lose accuracy
or overflow once l exceeds the size parameter (AUDIT.md M9).  Use
:func:`pystratify.solve` for real work.
"""

from .bessel import ric_h, ric_h_d, ric_j, ric_j_d, sph_hn, sph_hn_d, sph_jn, sph_jn_d
from .tmatrix import TransferMatrices, interface_matrices, t_mat

__all__ = [
    "t_mat",
    "interface_matrices",
    "TransferMatrices",
    "sph_jn",
    "sph_hn",
    "sph_jn_d",
    "sph_hn_d",
    "ric_j",
    "ric_h",
    "ric_j_d",
    "ric_h_d",
]
