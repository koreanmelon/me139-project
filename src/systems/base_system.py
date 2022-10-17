from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import sympy as sp


class BaseSystem(ABC):
    """
    Base class used to define a rigid body system.
    """
    
    # Constants
    g = 9.81  # gravitational acceleration (m/s^2)
    
    g_sym = sp.Symbol('g') # gravitational acceleration (m/s^2)
    t_sym = sp.Symbol('t') # time (s)

    @abstractmethod
    def deriv(self, t: npt.NDArray[np.float64], Q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ...
    
    @staticmethod
    def create_T(alpha, b, theta, d) -> sp.Matrix:
        """
        Creates a transformation matrix for given values of alpha, b, theta, and d.
        """

        return sp.simplify(sp.Matrix([
            [sp.cos(theta), -sp.sin(theta), 0, b],  # type: ignore
            [sp.cos(alpha) * sp.sin(theta), sp.cos(alpha) * sp.cos(theta), -sp.sin(alpha), -d * sp.sin(alpha)],  # type: ignore
            [sp.sin(alpha) * sp.sin(theta), sp.sin(alpha) * sp.cos(theta), sp.cos(alpha), d * sp.cos(alpha)],  # type: ignore
            [0, 0, 0, 1]
        ]))

    @staticmethod
    def create_I_c(m, l, shape="rod_center") -> sp.Expr:
        """
        Computes the moment of inertia of a link about its center of mass
        given its mass and length.
        """
        
        if shape == "rod_center":
            return sp.Rational(1, 12) * m * l**2
        elif shape == "rod_hollow":
            return m * l**2
        elif shape == "disk_axis":
            return sp.Rational(1, 2) * m * l**2
        elif shape == "sphere":
            return sp.Rational(2, 5) * m * l**2
        else:
            raise ValueError(f"Invalid shape: {shape}")
    
    @staticmethod
    def create_V(M: sp.Matrix, Q: sp.Matrix, theta_list: list[Any]) -> sp.Matrix:
        """
        Computes the Coriolis and centrifugal matrix.
        """
        
        M_d: sp.Matrix = M.diff(BaseSystem.t_sym)  # type: ignore
        Q_d: sp.Matrix = Q.diff(BaseSystem.t_sym)  # type: ignore
        
        temp_mat_components = [Q_d.T @ M.diff(theta) @ Q_d for theta in theta_list]
        
        return sp.simplify(M_d @ Q_d - sp.Rational(1, 2) * sp.Matrix.vstack(*temp_mat_components))
    
    @staticmethod
    def extract_R(T: sp.Matrix) -> sp.Matrix:
        """
        Extracts the rotation matrix from a transformation matrix.
        """

        return T.extract([0, 1, 2], [0, 1, 2])

    @staticmethod
    def extract_D(T: sp.Matrix) -> sp.Matrix:
        """
        Extracts the translation vector from a transformation matrix.
        """

        return T.extract([0, 1, 2], [3])

