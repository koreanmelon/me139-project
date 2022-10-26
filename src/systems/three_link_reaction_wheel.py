from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp
from scipy.integrate import trapezoid

from .system.link import CoordinateT, Link, LinkType, StyledJointT, StyledLinkT
from .system.system import RoboticSystem as RS
from .system.system import g
from .system.types import Vec

SolFunc = Callable[
    [float, float, float, float, float, float, float, float, float, float, float],
    float
]


@dataclass
class TLRWParams:
    l_1: float = 0.1524
    l_c1: float = 0.1524 / 2
    l_2: float = 0.1778
    l_c2: float = 0.1778 / 2
    l_3: float = 0.1524 / 2
    l_c3: float = 0.125
    r: float = 0.05
    m_1: float = 0.5
    m_2: float = 0.5
    m_3: float = 0.5
    m_w: float = 0.5


class TLRW(RS):

    def __init__(self, params: TLRWParams) -> None:
        super().__init__(
            Link(params.m_1, params.l_1, params.l_c1, LinkType.ROD),
            Link(params.m_2, params.l_2, params.l_c2, LinkType.ROD),
            Link(params.m_3, params.l_3, params.l_c3, LinkType.ROD),
            Link(params.m_w, params.r, 0, LinkType.RING)
        )

        self.err_vec = np.zeros(25)

        self.l_1 = params.l_1
        self.l_c1 = params.l_c1
        self.l_2 = params.l_2
        self.l_c2 = params.l_c2
        self.l_3 = params.l_3
        self.l_c3 = params.l_c3
        self.r = params.r
        self.m_1 = params.m_1
        self.m_2 = params.m_2
        self.m_3 = params.m_3
        self.m_w = params.m_w

    def solve_system(self):
        tau_1, tau_2, tau_3 = sp.symbols("tau_1 tau_2 tau_3")

        system = [
            sp.Eq(self.torque[0], 0),
            sp.Eq(self.torque[1], tau_1),
            sp.Eq(self.torque[2], tau_2),
            sp.Eq(self.torque[3], tau_3)
        ]
        self.sol = sp.solve(system, [
            self.theta_dd[1],
            self.theta_dd[2],
            self.theta_dd[3],
            self.theta_dd[4]
        ])

        self.sol_theta_1dd: SolFunc = sp.lambdify(
            (
                self.theta[1], self.theta[2], self.theta[3], self.theta[4],
                self.theta_d[1], self.theta_d[2], self.theta_d[3], self.theta_d[4],
                tau_1, tau_2, tau_3
            ),
            sp.trigsimp(self.sol[self.theta_dd[1]]),
            "numpy"
        )

        self.sol_theta_2dd: SolFunc = sp.lambdify(
            (
                self.theta[1], self.theta[2], self.theta[3], self.theta[4],
                self.theta_d[1], self.theta_d[2], self.theta_d[3], self.theta_d[4],
                tau_1, tau_2, tau_3
            ),
            sp.trigsimp(self.sol[self.theta_dd[2]]),
            "numpy"
        )

        self.sol_theta_3dd: SolFunc = sp.lambdify(
            (
                self.theta[1], self.theta[2], self.theta[3], self.theta[4],
                self.theta_d[1], self.theta_d[2], self.theta_d[3], self.theta_d[4],
                tau_1, tau_2, tau_3
            ),
            sp.trigsimp(self.sol[self.theta_dd[3]]),
            "numpy"
        )

        self.sol_theta_4dd: SolFunc = sp.lambdify(
            (
                self.theta[1], self.theta[2], self.theta[3], self.theta[4],
                self.theta_d[1], self.theta_d[2], self.theta_d[3], self.theta_d[4],
                tau_1, tau_2, tau_3
            ),
            sp.trigsimp(self.sol[self.theta_dd[4]]),
            "numpy"
        )

    def deriv(self, t: Vec, Q: Vec) -> Vec:
        theta_1 = Q[0]
        theta_2 = Q[1]
        theta_3 = Q[2]
        theta_4 = Q[3]
        theta_1d = Q[4]
        theta_2d = Q[5]
        theta_3d = Q[6]
        theta_4d = Q[7]

        torque_1 = 0
        torque_2 = 0
        torque_3 = 0
        # torque_1 = self.torque1_func(Q)
        # torque_2 = self.torque2_func(Q)
        # torque_3 = self.torque3_func(Q)

        theta_1dd: float = self.sol_theta_1dd(
            theta_1, theta_2, theta_3, theta_4,
            theta_1d, theta_2d, theta_3d, theta_4d,
            torque_1, torque_2, torque_3
        )
        theta_2dd: float = self.sol_theta_2dd(
            theta_1, theta_2, theta_3, theta_4,
            theta_1d, theta_2d, theta_3d, theta_4d,
            torque_1, torque_2, torque_3
        )
        theta_3dd: float = self.sol_theta_3dd(
            theta_1, theta_2, theta_3, theta_4,
            theta_1d, theta_2d, theta_3d, theta_4d,
            torque_1, torque_2, torque_3
        )
        theta_4dd: float = self.sol_theta_4dd(
            theta_1, theta_2, theta_3, theta_4,
            theta_1d, theta_2d, theta_3d, theta_4d,
            torque_1, torque_2, torque_3
        )

        return np.array([theta_1d, theta_2d, theta_3d, theta_4d, theta_1dd, theta_2dd, theta_3dd, theta_4dd])

    def torque1_func(self, Q):
        theta_1 = Q[0]
        theta_2 = Q[1]
        theta_3 = Q[2]
        theta_4 = Q[3]
        theta_1d = Q[4]
        theta_2d = Q[5]
        theta_3d = Q[6]
        theta_4d = Q[7]

        larm_2 = self.l_c2 * np.cos(theta_1 + theta_2)
        t_l2 = self.m_2 * g * larm_2

        larm_3 = self.l_2 * np.cos(theta_1 + theta_2) + self.l_c3 * np.cos(theta_1 + theta_2 + theta_3)
        t_l3 = self.m_3 * g * larm_3

        larm_4 = self.l_2 * np.cos(theta_1 + theta_2) + self.l_3 * np.cos(theta_1 + theta_2 + theta_3)
        t_l4 = self.m_w * g * larm_4

        t_const = t_l2 + t_l3 + t_l4

        return -t_const
        # return 0

    def torque2_func(self, Q):
        theta_1 = Q[0]
        theta_2 = Q[1]
        theta_3 = Q[2]
        theta_4 = Q[3]
        theta_1d = Q[4]
        theta_2d = Q[5]
        theta_3d = Q[6]
        theta_4d = Q[7]

        larm_3 = self.l_c3 * np.cos(theta_1 + theta_2 + theta_3)
        t_l3 = self.m_3 * g * larm_3

        larm_4 = self.l_3 * np.cos(theta_1 + theta_2 + theta_3)
        t_l4 = self.m_w * g * larm_4

        t_const = t_l3 + t_l4

        return t_const
        # return 0

    def torque3_func(self, Q):
        theta_1 = Q[0]
        theta_2 = Q[1]
        theta_3 = Q[2]
        theta_4 = Q[3]
        theta_1d = Q[4]
        theta_2d = Q[5]
        theta_3d = Q[6]
        theta_4d = Q[7]

        # Arbitrarily chosen values
        K_p = 5
        K_i = 0.9
        K_d = 1

        vec = self.l_1 * np.array([np.cos(theta_1), np.sin(theta_1)]) + \
            self.l_2 * np.array([np.cos(theta_1 + theta_2), np.sin(theta_1 + theta_2)]) + \
            self.l_3 * np.array([np.cos(theta_1 + theta_2 + theta_3), np.sin(theta_1 + theta_2 + theta_3)])

        err = vec[0]
        self.err_vec = np.append(self.err_vec, err)
        self.err_vec = np.delete(self.err_vec, 0)

        integ = trapezoid(self.err_vec, dx=0.01)

        t_added = (K_p * err) + (K_i * integ) + (K_d * theta_1d)

        larm_1 = self.l_c1 * np.cos(theta_1)
        t_l1 = self.m_1 * g * larm_1

        larm_2 = larm_1 + self.l_c2 * np.cos(theta_1 + theta_2)
        t_l2 = self.m_2 * g * larm_2

        larm_3 = larm_2 + self.l_c3 * np.cos(theta_1 + theta_2 + theta_3)
        t_l3 = self.m_3 * g * larm_3

        larm_4 = larm_2 + self.l_3 * np.cos(theta_1 + theta_2 + theta_3)
        t_l4 = self.m_w * g * larm_4

        t_const = self.m_w * g * vec[0] + (self.m_1 + self.m_2 + self.m_3) * g * vec[0] / 2
        # t_const = t_l1 + t_l2 + t_l3 + t_l4

        # return t_added - t_const
        return 0

    def link_positions(self, theta_t_vec: list[Vec]) -> list[StyledLinkT]:
        assert len(theta_t_vec) == 4, "A system with four links requires four angles."

        theta_1, theta_2, theta_3, theta_4 = theta_t_vec

        t_len = theta_1.size

        l1_start = CoordinateT(np.zeros(t_len), np.zeros(t_len))
        l1_end = CoordinateT(self.l_1 * np.cos(theta_1), self.l_1 * np.sin(theta_1))
        l1 = StyledLinkT(l1_start, l1_end, 2, 'b')

        l2_start = l1_end
        l2_end = CoordinateT(l2_start.x + self.l_2 * np.cos(theta_1 + theta_2),
                             l2_start.y + self.l_2 * np.sin(theta_1 + theta_2))
        l2 = StyledLinkT(l2_start, l2_end, 2, 'b')

        l3_start = l2_end
        l3_end = CoordinateT(l3_start.x + self.l_3 * np.cos(theta_1 + theta_2 + theta_3),
                             l3_start.y + self.l_3 * np.sin(theta_1 + theta_2 + theta_3))
        l3 = StyledLinkT(l3_start, l3_end, 2, 'b')

        l4_start = l3_end
        l4_end = CoordinateT(l3_end.x + self.r * np.cos(theta_1 + theta_2 + theta_3 + theta_4),
                             l3_end.y + self.r * np.sin(theta_1 + theta_2 + theta_3 + theta_4))
        l4 = StyledLinkT(l4_start, l4_end, 2, 'r', zorder=11)

        return [l1, l2, l3, l4]

    def joint_positions(self, theta_t_vec: list[Vec]) -> list[StyledJointT]:
        assert len(theta_t_vec) == 4, "A system with four links requires four angles."

        theta_1, theta_2, theta_3, theta_4 = theta_t_vec

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
            radius=0.02,
            zorder=10
        )
        j3 = StyledJointT(
            j2.x + self.l_2 * np.cos(theta_1 + theta_2),
            j2.y + self.l_2 * np.sin(theta_1 + theta_2),
            radius=0.02,
            zorder=10
        )
        j4 = StyledJointT(
            j3.x + self.l_3 * np.cos(theta_1 + theta_2 + theta_3),
            j3.y + self.l_3 * np.sin(theta_1 + theta_2 + theta_3),
            radius=self.r,
            edgecolor='k',
            facecolor='w',
            zorder=10
        )

        return [j1, j2, j3, j4]
