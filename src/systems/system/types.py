from typing import Any

import numpy as np
import numpy.typing as npt
import sympy as sp

VarDict = dict[int, Any]
MatDict = dict[str, sp.Matrix]

Vec = Mat = npt.NDArray[np.float_]
