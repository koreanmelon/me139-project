from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt


@dataclass
class NRParams:
    n: int
    n: int
    m_list: list[float]
    l_list: list[tuple[float, float]]
    alpha_list: list[float]
    b_list: list[float]
    d_list: list[float]


class NumRobot:
    g = 9.81  # gravity (m/s^2)

    def __init__(self, params: NRParams) -> None:
        self.m: list[float] = [-1] + params.m_list
        self.l: list[float] = [-1] + [l_tup[0] for l_tup in params.l_list]
        self.l_c: list[float] = [-1] + [l_tup[1] for l_tup in params.l_list]

        self.alpha: list[float] = [-1] + params.alpha_list
        self.b: list[float] = [-1] + params.b_list
        self.d: list[float] = [-1] + params.d_list

        # Transformation Matrices
        self.T: dict[tuple, Callable[[list[float]], npt.NDArray]] = {
            (0, 1): lambda theta_list: NumRobot.create_T(self.alpha[1], self.b[1], theta_list[0], self.d[1]),
            (1, 2): lambda theta_list: NumRobot.create_T(self.alpha[2], self.b[2], theta_list[1], self.d[2]),
            (2, 3): lambda theta_list: NumRobot.create_T(self.alpha[3], self.b[3], theta_list[2], self.d[3]),
        }
        self.T |= {
            (0, 2): lambda theta_list: self.T[(0, 1)](theta_list) @ self.T[(1, 2)](theta_list),
            (0, 3): lambda theta_list: self.T[(0, 1)](theta_list) @ self.T[(1, 2)](theta_list) @ self.T[(2, 3)](theta_list),
            (1, 3): lambda theta_list: self.T[(1, 2)](theta_list) @ self.T[(2, 3)](theta_list)
        }

        # Rotation Matrices (extracted from transformation matrices)
        self.R: dict[tuple, Callable[[list[float]], npt.NDArray]] = {
            (0, 1): lambda theta_list: NumRobot.extract_R(self.T[(0, 1)](theta_list)),
            (1, 2): lambda theta_list: NumRobot.extract_R(self.T[(1, 2)](theta_list)),
            (2, 3): lambda theta_list: NumRobot.extract_R(self.T[(2, 3)](theta_list)),
            (0, 2): lambda theta_list: NumRobot.extract_R(self.T[(0, 2)](theta_list)),
            (0, 3): lambda theta_list: NumRobot.extract_R(self.T[(0, 3)](theta_list)),
            (1, 3): lambda theta_list: NumRobot.extract_R(self.T[(1, 3)](theta_list))
        }

        # Displacement Vectors (extracted from transformation matrices)
        self.D: dict[tuple, Callable[[list[float]], npt.NDArray]] = {
            (0, 1): lambda theta_list: NumRobot.extract_D(self.T[(0, 1)](theta_list)),
            (1, 2): lambda theta_list: NumRobot.extract_D(self.T[(1, 2)](theta_list)),
            (2, 3): lambda theta_list: NumRobot.extract_D(self.T[(2, 3)](theta_list)),
            (0, 2): lambda theta_list: NumRobot.extract_D(self.T[(0, 2)](theta_list)),
            (0, 3): lambda theta_list: NumRobot.extract_D(self.T[(0, 3)](theta_list)),
            (1, 3): lambda theta_list: NumRobot.extract_D(self.T[(1, 3)](theta_list))
        }

        # Jacobians
        self.J_v2: Callable[[list[float]], npt.NDArray] = lambda theta_list: np.hstack([
            np.cross(self.R[(0, 1)](theta_list)[:, 2], self.R[(0, 1)](theta_list) @ self.D[(1, 2)](theta_list)),
            np.zeros((3, 1)),
            np.zeros((3, 1))
        ])

        self.J_v3: Callable[[list[float]], npt.NDArray] = lambda theta_list: np.hstack([
            np.cross(self.R[(0, 1)](theta_list)[:, 2], self.R[(0, 1)](theta_list) @ self.D[(1, 3)](theta_list)),
            np.cross(self.R[(0, 2)](theta_list)[:, 2], self.R[(0, 2)](theta_list) @ self.D[(2, 3)](theta_list)),
            np.zeros((3, 1))
        ])

        self.J_omega2: Callable[[list[float]], npt.NDArray] = lambda theta_list: np.hstack([
            self.R[(0, 1)](theta_list)[:, 2],
            np.zeros((3, 1)),
            np.zeros((3, 1))
        ])

        self.J_omega3: Callable[[list[float]], npt.NDArray] = lambda theta_list: np.hstack([
            self.R[(0, 1)](theta_list)[:, 2],
            self.R[(0, 2)](theta_list)[:, 2],
            np.zeros((3, 1))
        ])

        # Moment of Inertias
        self.I_c1 = NumRobot.create_I_c(self.m[1], self.l[1])
        self.I_c2 = NumRobot.create_I_c(self.m[2], self.l[2])
        self.I_c3 = NumRobot.create_I_c(self.m[3], self.l[3])

        # M_term_2: Callable[[list[float]], npt.NDArray] = lambda theta_list: self.m[2] * self.J_v2(
        #     theta_list).T @ self.J_v2(theta_list) + self.I_c2 * self.J_omega2(theta_list).T @ self.J_omega2(theta_list)
        # M_term_3: Callable[[list[float]], npt.NDArray] = lambda theta_list: self.m[3] * self.J_v3(
        #     theta_list).T @ self.J_v3(theta_list) + self.I_c3 * self.J_omega3(theta_list).T @ self.J_omega3(theta_list)

        # self.M: Callable[[list[float]], npt.NDArray] = lambda theta_list: M_term_2(theta_list) + M_term_3(theta_list)
        # self.M_dot: Callable[[list[float]], npt.NDArray] = lambda theta_d_list: self.M(theta_d_list)

        # self.Q: Callable[[list[float]], npt.NDArray] = lambda theta_list: np.array(theta_list)
        # self.Q_dot: Callable[[list[float]], npt.NDArray] = lambda theta_d_list: np.array(theta_d_list)
        # self.Q_ddot: Callable[[list[float]], npt.NDArray] = lambda theta_dd_list: np.array(theta_dd_list)

        # self.V: Callable[[list[float], list[float]], npt.NDArray] = lambda theta_list, theta_d_list: (self.M_dot(theta_d_list) @ self.Q_dot(theta_d_list) - 0.5 * np.vstack(
        #     self.Q_dot(theta_d_list).T @ self.M.diff(self.theta_1) @ self.Q_dot(theta_d_list),
        #     self.Q_dot(theta_d_list).T @ self.M.diff(self.theta_2) @ self.Q_dot(theta_d_list),
        #     self.Q_dot(theta_d_list).T @ self.M.diff(self.theta_3) @ self.Q_dot(theta_d_list)
        # ))

        # h_1: sp.Expr = (self.T[(0, 1)] @ sp.Matrix([self.l_c1, 0, 0, 1]).reshape(4, 1))[1]
        # h_2: sp.Expr = (self.T[(0, 2)] @ sp.Matrix([self.l_c2, 0, 0, 1]).reshape(4, 1))[1]
        # h_3: sp.Expr = (self.T[(0, 3)] @ sp.Matrix([self.l_c3, 0, 0, 1]).reshape(4, 1))[1]

        # self.P = sp.simplify(SymRobot.g * (self.M_1 * h_1 + self.M_2 * h_2 + self.M_3 * h_3))

        # self.G = sp.Matrix([
        #     self.P.diff(self.theta_1),
        #     self.P.diff(self.theta_2),
        #     self.P.diff(self.theta_3)
        # ]).reshape(3, 1)

        # self.Torque: sp.Matrix = sp.simplify(self.M @ self.Q_ddot + self.V + self.G)

    @staticmethod
    def create_T(alpha_n, b_n, theta_n, d_n) -> npt.NDArray[np.float64]:
        """
        Creates a transformation matrix for given values of alpha, b, theta, and d.
        """

        return np.array([
            [np.cos(theta_n), -np.sin(theta_n), 0, b_n],
            [np.cos(alpha_n) * np.sin(theta_n), np.cos(alpha_n) *
             np.cos(theta_n), -np.sin(alpha_n), -d_n * np.sin(alpha_n)],
            [np.sin(alpha_n) * np.sin(theta_n), np.sin(alpha_n) * np.cos(theta_n),
             np.cos(alpha_n), d_n * np.cos(alpha_n)], [0, 0, 0, 1]
        ])

    @staticmethod
    def create_I_c(M_n, l_n) -> float:
        """
        Computes the moment of inertia of a link (rod) about its center of mass
        given its mass and length.
        """

        return (1 / 12) * M_n * l_n ** 2

    @staticmethod
    def extract_R(T: npt.NDArray) -> npt.NDArray:
        """
        Extracts the rotation matrix from a transformation matrix.
        """

        return T[:3, :3]

    @staticmethod
    def extract_D(T: npt.NDArray) -> npt.NDArray:
        """
        Extracts the translation vector from a transformation matrix.
        """

        return T[:3, 3]
