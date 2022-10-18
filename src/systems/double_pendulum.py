from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from systems.system import Joint, Link, MatDict
from systems.system import RoboticSystem as RS
from systems.system import TCoordinate, VarDict, Vec, g, t


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
        super().__init__()

        # System Parameters
        self.l_1 = params.l_1       # rod 1 length (m)
        self.l_c1 = params.l_c1     # rod 1 center of mass (m)
        self.l_2 = params.l_2       # rod 2 length (m)
        self.l_c2 = params.l_c2     # rod 2 center of mass (m)

        self.m = {1: params.m_1, 2: params.m_2}

        self.I_c = {
            "1": RS.construct_I(self.m[1], self.l_1),
            "2": RS.construct_I(self.m[2], self.l_2)
        }

        self.alpha = {1: 0, 2: 0}
        self.b = {1: 0, 2: self.l_1}
        self.d = {1: 0, 2: 0}

        self.theta = {
            1: dynamicsymbols("theta_1"),
            2: dynamicsymbols("theta_2")
        }
        self.theta_d = {
            1: sp.diff(self.theta[1], t),
            2: sp.diff(self.theta[2], t)
        }
        self.theta_dd = {
            1: sp.diff(self.theta_d[1], t),
            2: sp.diff(self.theta_d[2], t)
        }

        self.Q = sp.Matrix([self.theta[1], self.theta[2]]).reshape(2, 1)

        self.T: MatDict = self.compute_T(self.alpha, self.b, self.theta, self.d)

        self.R: MatDict = self.compute_R(self.T)

        self.D: MatDict = {
            "1->0": RS.extract_D(self.T["1->0"]),
            "2->1": RS.extract_D(self.T["2->1"]),
            "2->0": RS.extract_D(self.T["2->0"])
        }
        self.D |= {
            "c1->1": sp.Matrix([self.l_c1, 0, 0]).reshape(3, 1),
            "c2->2": sp.Matrix([self.l_c2, 0, 0]).reshape(3, 1)
        }
        self.D |= {
            "c1->0": self.D["1->0"] + self.R["1->0"] * self.D["c1->1"],
            "c2->0": self.D["2->0"] + self.R["2->0"] * self.D["c2->2"],
            "c2->1": self.D["2->1"] + self.R["2->1"] * self.D["c2->2"]
        }

        self.P = sp.simplify(
            (self.m[1] * g * self.D["c1->0"][1]) +  # type: ignore
            (self.m[2] * g * self.D["c2->0"][1])  # type: ignore
        )

        self.J_v, self.J_omega = self.compute_J(self.R, self.D)

        self.M = self.compute_M(self.m, self.I_c, self.J_v, self.J_omega)
        self.V = self.compute_V(self.M, self.Q, self.theta)
        self.G = self.compute_G(self.P, self.theta)

        self.torque = self.compute_torque(self.M, self.Q, self.V, self.G)

        self.solve_system()

    @property
    def n(self) -> int:
        return 2

    @staticmethod
    def compute_T(alpha: VarDict, b: VarDict, theta: VarDict, d: VarDict) -> MatDict:
        T: dict[str, sp.Matrix] = {
            "1->0": RS.construct_T(alpha[1], b[1], theta[1], d[1]),
            "2->1": RS.construct_T(alpha[2], b[2], theta[2], d[2])
        }
        T |= {
            "2->0": sp.simplify(T["1->0"] * T["2->1"])
        }

        return T

    @staticmethod
    def compute_R(T: MatDict) -> MatDict:
        R: dict[str, sp.Matrix] = {
            "1->0": RS.extract_R(T["1->0"]),
            "2->1": RS.extract_R(T["2->1"]),
            "2->0": RS.extract_R(T["2->0"]),
        }

        return R

    @staticmethod
    def compute_J(R: MatDict, D: MatDict) -> tuple[MatDict, MatDict]:
        J_v: dict[str, sp.Matrix] = {
            "1": sp.simplify(sp.Matrix.hstack(
                R["1->0"].extract([0, 1, 2], [2]).cross(R["1->0"] * D["c1->1"]),
                sp.zeros(3, 1)
            )),
            "2": sp.simplify(sp.Matrix.hstack(
                R["1->0"].extract([0, 1, 2], [2]).cross(R["1->0"] * D["c2->1"]),
                R["2->0"].extract([0, 1, 2], [2]).cross(R["2->0"] * D["c2->2"])
            ))
        }

        J_omega: dict[str, sp.Matrix] = {
            "1": sp.simplify(sp.Matrix.hstack(
                R["1->0"][:, 2],
                sp.zeros(3, 1)
            )),
            "2": sp.simplify(sp.Matrix.hstack(
                R["1->0"][:, 2],
                R["2->0"][:, 2]
            ))
        }

        return J_v, J_omega

    @staticmethod
    def compute_M(m: VarDict, I_c: MatDict, J_v: MatDict, J_omega: MatDict) -> sp.Matrix:
        M = sp.simplify(
            (m[1] * J_v["1"].T * J_v["1"] + J_omega["1"].T * I_c["1"] * J_omega["1"]) +
            (m[2] * J_v["2"].T * J_v["2"] + J_omega["2"].T * I_c["2"] * J_omega["2"])
        )

        return M

    @staticmethod
    def compute_V(M: sp.Matrix, Q: sp.Matrix, theta: VarDict) -> sp.Matrix:
        M_d: sp.Matrix = M.diff(t)  # type: ignore
        Q_d: sp.Matrix = Q.diff(t)  # type: ignore

        temp_mat_components = [Q_d.T @ M.diff(dtheta) @ Q_d for dtheta in theta.values()]

        return sp.simplify(M_d @ Q_d - sp.Rational(1, 2) * sp.Matrix.vstack(*temp_mat_components))

    @staticmethod
    def compute_G(P: sp.Expr, theta: VarDict) -> sp.Matrix:
        G = sp.simplify(sp.Matrix([
            P.diff(theta[1]),
            P.diff(theta[2])
        ]).reshape(2, 1))

        return G

    @staticmethod
    def compute_torque(M: sp.Matrix, Q: sp.Matrix, V: sp.Matrix, G: sp.Matrix) -> sp.Matrix:
        Q_dd = Q.diff(t, 2)  # type: ignore

        torque = sp.simplify(M * Q_dd + V + G)

        return torque

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

    def links(self, theta_t_vec: list[Vec]) -> list[Link]:
        assert len(theta_t_vec) == 2, "This system only has two links."

        theta_1, theta_2 = theta_t_vec

        t_len = theta_1.size

        l1_start = TCoordinate(np.zeros(t_len), np.zeros(t_len))
        l1_end = TCoordinate(self.l_1 * np.cos(theta_1), self.l_1 * np.sin(theta_1))
        l1 = Link(l1_start, l1_end, 2, 'b')

        l2_start = l1_end
        l2_end = TCoordinate(l1_end.x + self.l_2 * np.cos(theta_1 + theta_2),
                             l1_end.y + self.l_2 * np.sin(theta_1 + theta_2))
        l2 = Link(l2_start, l2_end, 2, 'r', zorder=11)

        return [l1, l2]

    def joints(self, theta_t_vec: list[Vec]) -> list[Joint]:
        assert len(theta_t_vec) == 2, "This system only has two links."

        theta_1, theta_2 = theta_t_vec

        t_len = theta_1.size

        j1 = Joint(
            np.zeros(t_len),
            np.zeros(t_len),
            color='k',
            zorder=10
        )
        j2 = Joint(
            self.l_1 * np.cos(theta_1),
            self.l_1 * np.sin(theta_1),
            color='k',
            zorder=10
        )
        j3 = Joint(
            self.l_1 * np.cos(theta_1) + self.l_2 * np.cos(theta_1 + theta_2),
            self.l_1 * np.sin(theta_1) + self.l_2 * np.sin(theta_1 + theta_2),
            color='k',
            zorder=10
        )

        return [j1, j2, j3]
