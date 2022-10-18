from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import sympy as sp

VarDict = dict[int, Any]
MatDict = dict[str, sp.Matrix]

Vec = npt.NDArray[np.float64]

g = 9.81  # gravitational acceleration (m/s^2)

g_sym = sp.Symbol('g')  # gravity symbol
t = sp.Symbol("t")  # time symbol


@dataclass
class TCoordinate:
    x: Vec
    y: Vec


@dataclass
class TLine:
    start: TCoordinate
    end: TCoordinate


@dataclass
class Joint(TCoordinate):
    edgecolor: Optional[str] = None
    facecolor: Optional[str] = None
    color: str = 'k'
    radius: float = 0.05
    zorder: int = 1


@dataclass
class Link(TLine):
    linewidth: float = 1
    color: str = "black"
    zorder: int = 1


class RoboticSystem(ABC):
    """
    Base class used to define a robotic system. Note that all computation
    related to the system is done in this class.
    """

    @property
    @abstractmethod
    def n(self) -> int:
        """
        Number of links in the system.
        """
        ...

    @abstractmethod
    def links(self, theta_t_vec: list[Vec]) -> list[Link]:
        ...

    @abstractmethod
    def joints(self, theta_t_vec: list[Vec]) -> list[Joint]:
        ...

    @abstractmethod
    def solve_system(self) -> None:
        ...

    @abstractmethod
    def deriv(self, t: Vec, Q: Vec) -> Vec:
        ...

    @staticmethod
    @abstractmethod
    def compute_T(alpha: VarDict, b: VarDict, theta: VarDict, d: VarDict) -> MatDict:
        ...

    @staticmethod
    @abstractmethod
    def compute_R(T: MatDict) -> MatDict:
        ...

    @staticmethod
    @abstractmethod
    def compute_J(R: MatDict, D: MatDict) -> tuple[MatDict, MatDict]:
        ...

    @staticmethod
    @abstractmethod
    def compute_M(m: VarDict, I_c: MatDict, J_v: MatDict, J_omega: MatDict) -> sp.Matrix:
        ...

    @staticmethod
    @abstractmethod
    def compute_V(M: sp.Matrix, Q: sp.Matrix, theta: VarDict) -> sp.Matrix:
        ...

    @staticmethod
    @abstractmethod
    def compute_G(P: sp.Expr, theta: VarDict) -> sp.Matrix:
        ...

    @staticmethod
    @abstractmethod
    def compute_torque(M: sp.Matrix, Q: sp.Matrix, V: sp.Matrix, G: sp.Matrix) -> sp.Matrix:
        ...

    @staticmethod
    def construct_T(alpha, b, theta, d) -> sp.Matrix:
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
    def construct_I(m, l, shape="rod_center") -> sp.Matrix:
        """
        Computes the moment of inertia of a link about its center of mass
        given its mass and length.
        """

        if shape == "rod_center":
            val = sp.Rational(1, 12) * m * l**2
        elif shape == "rod_hollow":
            val = m * l**2
        elif shape == "disk_axis":
            val = sp.Rational(1, 2) * m * l**2
        elif shape == "sphere":
            val = sp.Rational(2, 5) * m * l**2
        else:
            raise ValueError(f"Invalid shape: {shape}")

        return sp.diag(val, val, val)

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
