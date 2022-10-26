from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp
from scipy.integrate import trapezoid

from systems.system.link import CoordinateT, LinkType
from systems.system.system import Link
from systems.system.system import RoboticSystem as RS
from systems.system.system import StyledJointT, StyledLinkT, Vec, g


@dataclass
class RWParams:
    l_1: float = 1.0
    l_c1: float = 1.0
    r: float = 1.0
    m_1: float = 1.0
    m_2: float = 1.0


class ReactionWheel(RS):
    """
    A system with a single link and a reaction wheel attached to the end of the link.
    """

    def __init__(self, params: RWParams) -> None:
        super().__init__(
            Link(
                m=params.m_1,
                l=params.l_1,
                l_c=params.l_c1,
                type=LinkType.ROD
            ),
            Link(
                m=params.m_2,
                l=params.r,
                l_c=0,
                type=LinkType.DISK
            )
        )

        # Controls
        self.err_vec = np.zeros(10)

        # System Parameters
        self.l_1 = params.l_1       # rod length (m)
        self.l_c1 = params.l_c1     # rod center of mass (m)
        self.r = params.r           # reaction wheel radius (m)

        self.m_1 = params.m_1
        self.m_2 = params.m_2

    def solve_system(self):
        tau = sp.Symbol("tau")

        system = [sp.Eq(self.torque[0], 0), sp.Eq(self.torque[1], tau)]
        sol = sp.solve(system, [self.theta_dd[1], self.theta_dd[2]])

        self.sol_theta_1dd: Callable[[float, float, float, float, float], float] = sp.lambdify(
            (self.theta[1], self.theta[2], self.theta_d[1], self.theta_d[2], tau),
            sol[self.theta_dd[1]],
            "numpy"
        )

        self.sol_theta_2dd: Callable[[float, float, float, float, float], float] = sp.lambdify(
            (self.theta[1], self.theta[2], self.theta_d[1], self.theta_d[2], tau),
            sol[self.theta_dd[2]],
            "numpy"
        )

    def deriv(self, t: Vec, Q: Vec) -> Vec:
        theta_1 = Q[0]
        theta_2 = Q[1]
        theta_1d = Q[2]
        theta_2d = Q[3]

        tau = self.torque_func(Q)

        theta_1dd: float = self.sol_theta_1dd(theta_1, theta_2, theta_1d, theta_2d, tau)
        theta_2dd: float = self.sol_theta_2dd(theta_1, theta_2, theta_1d, theta_2d, tau)

        return np.array([theta_1d, theta_2d, theta_1dd, theta_2dd])

    def torque_func(self, Q):
        theta_1 = Q[0]
        theta_2 = Q[1]
        theta_1d = Q[2]
        theta_2d = Q[3]

        # Arbitrarily chosen values
        K_p = 5
        K_i = 0.9
        K_d = 1

        err = theta_1 - np.pi/2
        self.err_vec = np.append(self.err_vec, err)
        self.err_vec = np.delete(self.err_vec, 0)

        integ = trapezoid(self.err_vec, dx=0.01)

        t_added = (K_p * err) + (K_i * integ) + (K_d * theta_1d)

        t_const = -g * (self.l_c1 * self.m_1 + self.l_1 * self.m_2) * np.cos(theta_1)

        return t_added + t_const

    def link_positions(self, theta_t_vec: list[Vec]) -> list[StyledLinkT]:
        assert len(theta_t_vec) == 2, "This system only has two links."

        theta_1, theta_2 = theta_t_vec

        t_len = theta_1.size

        l1_start = CoordinateT(np.zeros(t_len), np.zeros(t_len))
        l1_end = CoordinateT(self.l_1 * np.cos(theta_1), self.l_1 * np.sin(theta_1))
        l1 = StyledLinkT(l1_start, l1_end, 2, 'b')

        l2_start = l1_end
        l2_end = CoordinateT(l1_end.x + self.r * np.cos(theta_1 + theta_2),
                             l1_end.y + self.r * np.sin(theta_1 + theta_2))
        l2 = StyledLinkT(l2_start, l2_end, 2, 'r', zorder=11)

        return [l1, l2]

    def joint_positions(self, theta_t_vec: list[Vec]) -> list[StyledJointT]:
        assert len(theta_t_vec) == 2, "This system only has two links."

        theta_1, theta_2 = theta_t_vec

        t_len = theta_1.size

        j1 = StyledJointT(
            np.zeros(t_len),
            np.zeros(t_len),
            radius=0.02,
            zorder=10
        )
        j2 = StyledJointT(
            self.l_1 * np.cos(theta_1),
            self.l_1 * np.sin(theta_1),
            radius=self.r,
            edgecolor='k',
            facecolor='w',
            zorder=10
        )

        return [j1, j2]
