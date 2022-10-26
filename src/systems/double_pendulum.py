from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from systems.system.system import Link, LinkType
from systems.system.system import RoboticSystem as RS
from systems.system.system import StyledJointT, StyledLinkT, CoordinateT, Vec


@dataclass
class DPParams:
    l_1: float = 1.0
    l_c1: float = 1.0
    l_2: float = 1.0
    l_c2: float = 1.0
    m_1: float = 1.0
    m_2: float = 1.0


class DoublePendulum(RS):
    """
    A double pendulum.
    """

    def __init__(self, params: DPParams) -> None:
        super().__init__(Link(1, 1, 0.5, LinkType.ROD), Link(1, 1, 0.5, LinkType.ROD))

        # System Parameters
        self.l_1 = params.l_1       # rod 1 length (m)
        self.l_c1 = params.l_c1     # rod 1 center of mass (m)
        self.l_2 = params.l_2       # rod 2 length (m)
        self.l_c2 = params.l_c2     # rod 2 center of mass (m)

    def solve_system(self):
        system = [sp.Eq(self.torque[0], 0), sp.Eq(self.torque[1], 0)]
        sol = sp.solve(system, [self.theta_dd[1], self.theta_dd[2]])

        self.sol_theta_1dd: Callable[[float, float, float, float], float] = sp.lambdify(
            (self.theta[1], self.theta[2], self.theta_d[1], self.theta_d[2]),
            sol[self.theta_dd[1]]
        )

        self.sol_theta_2dd: Callable[[float, float, float, float], float] = sp.lambdify(
            (self.theta[1], self.theta[2], self.theta_d[1], self.theta_d[2]),
            sol[self.theta_dd[2]]
        )

    def deriv(self, t: Vec, Q: Vec) -> Vec:
        theta_1 = Q[0]
        theta_2 = Q[1]
        theta_1d = Q[2]
        theta_2d = Q[3]

        theta_1dd: float = self.sol_theta_1dd(theta_1, theta_2, theta_1d, theta_2d)
        theta_2dd: float = self.sol_theta_2dd(theta_1, theta_2, theta_1d, theta_2d)

        return np.array([theta_1d, theta_2d, theta_1dd, theta_2dd])

    def link_positions(self, theta_t_vec: list[Vec]) -> list[StyledLinkT]:
        assert len(theta_t_vec) == 2, "This system only has two links."

        theta_1, theta_2 = theta_t_vec

        t_len = theta_1.size

        l1_start = CoordinateT(np.zeros(t_len), np.zeros(t_len))
        l1_end = CoordinateT(self.l_1 * np.cos(theta_1), self.l_1 * np.sin(theta_1))
        l1 = StyledLinkT(l1_start, l1_end, 2, 'b')

        l2_start = l1_end
        l2_end = CoordinateT(l1_end.x + self.l_2 * np.cos(theta_1 + theta_2),
                             l1_end.y + self.l_2 * np.sin(theta_1 + theta_2))
        l2 = StyledLinkT(l2_start, l2_end, 2, 'r', zorder=11)

        return [l1, l2]

    def joint_positions(self, theta_t_vec: list[Vec]) -> list[StyledJointT]:
        assert len(theta_t_vec) == 2, "This system only has two links."

        theta_1, theta_2 = theta_t_vec

        t_len = theta_1.size

        j1 = StyledJointT(
            np.zeros(t_len),
            np.zeros(t_len),
            color='k',
            zorder=10
        )
        j2 = StyledJointT(
            self.l_1 * np.cos(theta_1),
            self.l_1 * np.sin(theta_1),
            color='k',
            zorder=10
        )
        j3 = StyledJointT(
            self.l_1 * np.cos(theta_1) + self.l_2 * np.cos(theta_1 + theta_2),
            self.l_1 * np.sin(theta_1) + self.l_2 * np.sin(theta_1 + theta_2),
            color='k',
            zorder=10
        )

        return [j1, j2, j3]
