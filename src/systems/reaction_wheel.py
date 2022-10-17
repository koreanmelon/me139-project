from typing import Callable

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from systems.system import Joint, Link, MatDict
from systems.system import RoboticSystem as RS
from systems.system import TCoordinate, VarDict, Vec


class ReactionWheel(RS):
    """
    A system with a single link and a reaction wheel attached to the end of the link.
    """
    
    def __init__(self, l_1=1.0, l_c1=1.0, r=1.0, m_1=1.0, m_2=1.0) -> None:
        super().__init__()
        
        # Controls
        self.tau = lambda Q: 0
        
        # System Parameters
        self.l_1 = l_1 # rod length (m)
        self.l_c1 = l_c1 # rod center of mass (m)
        self.r = r # reaction wheel radius (m)
        
        self.m = {
            1: m_1,
            2: m_2
        }
        
        self.I_c = {
            "1": sp.diag(
                1/12 * self.m[1] * self.l_1**2,
                1/12 * self.m[1] * self.l_1**2,
                1/12 * self.m[1] * self.l_1**2
            ),
            "2": sp.diag(
                1/2 * self.m[2] * self.r**2,
                1/2 * self.m[2] * self.r**2,
                1/2 * self.m[2] * self.r**2
            )
        }
        
        self.alpha = { 1: 0, 2: 0 }
        self.b = { 1: 0, 2: self.l_1 }
        self.d = { 1: 0, 2: 0 }
        
        self.theta = {
            1: dynamicsymbols("theta_1"),
            2: dynamicsymbols("theta_2")
        }
        self.theta_d = {
            1: sp.diff(self.theta[1], RS.t_sym),
            2: sp.diff(self.theta[2], RS.t_sym)
        }
        self.theta_dd = {
            1: sp.diff(self.theta_d[1], RS.t_sym),
            2: sp.diff(self.theta_d[2], RS.t_sym)
        }
        
        self.Q = sp.Matrix([self.theta[1], self.theta[2]]).reshape(2, 1)
        
        self.P = sp.simplify((self.m[1] * RS.g_sym * l_c1 * sp.sin(self.theta[1])) + (self.m[2] * RS.g_sym * l_1 * sp.sin(self.theta[1])))
        
        self.T = self.compute_T(self.alpha, self.b, self.theta, self.d)
        
        self.R = self.compute_R(self.T)

        self.D: dict[str, sp.Matrix] = {
            "1->0": RS.extract_D(self.T["1->0"]),
            "2->1": RS.extract_D(self.T["2->1"])
        }
        self.D |= {
            "c1->1": sp.Matrix([self.l_c1, 0, 0]).reshape(3, 1),
            "c2->1": self.D["2->1"],
            "c2->2": sp.zeros(3, 1),
        }
        
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
        M_d: sp.Matrix = M.diff(RS.t_sym)  # type: ignore
        Q_d: sp.Matrix = Q.diff(RS.t_sym)  # type: ignore
        
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
        Q_dd = Q.diff(RS.t_sym, 2)  # type: ignore
        
        torque = sp.simplify(M * Q_dd + V + G)
        
        return torque

    def solve_system(self):
        tau_1, tau_2 = sp.symbols('tau_1 tau_2')
        system = [sp.Eq(self.torque[0], tau_1), sp.Eq(self.torque[1], tau_2)]
        sol = sp.solve(system, [self.theta_dd[1], self.theta_dd[2]])
        
        self.sol_theta_1dd: Callable[[float, float, float, float, float, float], float] = sp.lambdify(
            (self.theta[1], self.theta_d[1], self.theta[2], self.theta_d[2], tau_1, tau_2),
            sol[self.theta_dd[1]].subs(RS.g_sym, 9.81)
        )
        
        self.sol_theta_2dd: Callable[[float, float, float, float, float, float], float] = sp.lambdify(
            (self.theta[1], self.theta[2], self.theta_d[1], self.theta_d[2], tau_1, tau_2),
            sol[self.theta_dd[2]].subs(RS.g_sym, 9.81)
        )

    def deriv(self, t: Vec, Q: Vec) -> Vec:
        theta_1 = Q[0]
        theta_2 = Q[1]
        theta_1d = Q[2]
        theta_2d = Q[3]

        theta_1dd: float = self.sol_theta_1dd(theta_1, theta_2, theta_1d, theta_2d, 0, self.tau(Q))
        theta_2dd: float = self.sol_theta_2dd(theta_1, theta_2, theta_1d, theta_2d, 0, self.tau(Q))
        
        return np.array([theta_1d, theta_2d, theta_1dd, theta_2dd])

    
    def links(self, theta_t_vec: list[Vec]) -> list[Link]:
        assert len(theta_t_vec) == 2, "This system only has two links."
        
        theta_1, theta_2 = theta_t_vec
        
        t_len = theta_1.size
        
        l1_start = TCoordinate(np.zeros(t_len), np.zeros(t_len))
        l1_end = TCoordinate(self.l_1 * np.cos(theta_1), self.l_1 * np.sin(theta_1))
        l1 = Link(l1_start, l1_end, 2, 'b')
        
        l2_start = l1_end
        l2_end = TCoordinate(l1_end.x + self.r * np.cos(theta_1 + theta_2), l1_end.y + self.r * np.sin(theta_1 + theta_2))
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
            radius=self.r,
            edgecolor='k',
            facecolor='w',
            zorder=10
        )
        
        return [j1, j2]
