import numpy as np
import numpy.typing as npt
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from systems.base_system import BaseSystem


class ReactionWheelSystem(BaseSystem):
    
    def __init__(self) -> None:
        super().__init__()
        
        # System Parameters
        self.l_1 = 0.45 # rod length (m)
        self.l_c1 = 0.225 # rod center of mass (m)
        self.m_1 = 0.5 # rod mass (kg)
        self.I_c1 = sp.diag(
            1/12 * self.m_1 * self.l_1**2,
            1/12 * self.m_1 * self.l_1**2,
            1/12 * self.m_1 * self.l_1**2
        )

        self.r = 0.3 # reaction wheel radius (m)
        self.m_2 = 10 # reaction wheel mass (kg)
        self.I_c2 = sp.diag(
            1/2 * self.m_2 * self.r**2,
            1/2 * self.m_2 * self.r**2,
            1/2 * self.m_2 * self.r**2
        )

        self.alpha = [-1] + [0, 0]
        self.b = [-1] + [0, self.l_1]
        self.theta = [-1] + dynamicsymbols("theta_1 theta_2")
        self.d = [-1] + [0, 0]
        
        self.theta_d = [-1] + [sp.diff(self.theta[1], BaseSystem.t_sym), sp.diff(self.theta[2], BaseSystem.t_sym)]
        self.theta_dd = [-1] + [sp.diff(self.theta_d[1], BaseSystem.t_sym), sp.diff(self.theta_d[2], BaseSystem.t_sym)]

        self.T: dict[str, sp.Matrix] = {
            "1->0": BaseSystem.create_T(self.alpha[1], self.b[1], self.theta[1], self.d[1]),
            "2->1": BaseSystem.create_T(self.alpha[2], self.b[2], self.theta[2], self.d[2])
        }
        self.T |= {
            "2->0": sp.simplify(self.T["1->0"] @ self.T["2->1"])
        }
        
        self.R: dict[str, sp.Matrix] = {
            "1->0": BaseSystem.extract_R(self.T["1->0"]),
            "2->1": BaseSystem.extract_R(self.T["2->1"]),
            "2->0": BaseSystem.extract_R(self.T["2->0"]),
        }
        
        self.D: dict[str, sp.Matrix] = {
            "1->0": BaseSystem.extract_D(self.T["1->0"]),
            "2->1": BaseSystem.extract_D(self.T["2->1"])
        }
        self.D |= {
            "c1->1": sp.Matrix([self.l_c1, 0, 0]).reshape(3, 1),
            "c2->1": self.D["2->1"],
            "c2->2": sp.zeros(3, 1),
        }

        self.J_v: dict[str, sp.Matrix] = {
            "1": sp.simplify(sp.Matrix.hstack(
                self.R["1->0"].extract([0, 1, 2], [2]).cross(self.R["1->0"] @ self.D["c1->1"]),
                sp.zeros(3, 1)
            )),
            "2": sp.simplify(sp.Matrix.hstack(
                self.R["1->0"].extract([0, 1, 2], [2]).cross(self.R["1->0"] @ self.D["c2->1"]),
                self.R["2->0"].extract([0, 1, 2], [2]).cross(self.R["2->0"] @ self.D["c2->2"])
            ))
        }
        
        self.J_omega: dict[str, sp.Matrix] = {
            "1": sp.simplify(sp.Matrix.hstack(
                self.R["1->0"][:, 2],
                sp.zeros(3, 1)
            )),
            "2": sp.simplify(sp.Matrix.hstack(
                self.R["1->0"][:, 2],
                self.R["2->0"][:, 2]
            ))
        }
        
        self.M = sp.simplify(
            (self.m_1 * self.J_v["1"].T @ self.J_v["1"] + self.J_omega["1"].T @ self.I_c1 @ self.J_omega["1"]) +
            (self.m_2 * self.J_v["2"].T @ self.J_v["2"] + self.J_omega["2"].T @ self.I_c2 @ self.J_omega["2"])
        )
        self.M_d = sp.simplify(self.M.diff(BaseSystem.t_sym))

        self.Q = sp.Matrix([self.theta[1], self.theta[2]]).reshape(2, 1)
        self.Q_d = self.Q.diff(BaseSystem.t_sym)
        self.Q_dd = self.Q_d.diff(BaseSystem.t_sym)

        self.V = BaseSystem.create_V(self.M, self.Q, [self.theta[1], self.theta[2]])

        self.P = sp.simplify((self.m_1 * BaseSystem.g_sym * self.l_c1 * sp.sin(self.theta[1])) + (self.m_2 * BaseSystem.g_sym * self.l_1 * sp.sin(self.theta[1])))

        self.G = sp.simplify(sp.Matrix([
            self.P.diff(self.theta[1]),
            self.P.diff(self.theta[2])
        ]).reshape(2, 1))

        self.T = sp.simplify(self.M * self.Q_dd + self.V + self.G)

        tau_1, tau_2 = sp.symbols('tau_1 tau_2')
        system = [sp.Eq(self.T[0], tau_1), sp.Eq(self.T[1], tau_2)]
        sol = sp.solve(system, [self.theta_dd[1], self.theta_dd[2]])
        
        self.sol_theta_1dd = sp.lambdify((self.theta[1], self.theta_d[1], self.theta[2], self.theta_d[2], tau_1, tau_2), sol[self.theta_dd[1]].subs(BaseSystem.g_sym, 9.81))
        self.sol_theta_2dd = sp.lambdify((self.theta[1], self.theta_d[1], self.theta[2], self.theta_d[2], tau_1, tau_2), sol[self.theta_dd[2]].subs(BaseSystem.g_sym, 9.81))

    def deriv(self, t: npt.NDArray[np.float64], Q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        theta_1 = Q[0]
        theta_1d = Q[1]
        theta_2 = Q[2]
        theta_2d = Q[3]
        
        # tau_2 = -RSystem.g * (self.l_1 * self.m_2 + self.l_c1 * self.m_1) * np.cos(theta_1)

        theta_1dd: float = self.sol_theta_1dd(theta_1, theta_1d, theta_2, theta_2d, 0, 0)
        theta_2dd: float = self.sol_theta_2dd(theta_1, theta_1d, theta_2, theta_2d, 0, 0)
        
        return np.array([theta_1d, theta_1dd, theta_2d, theta_2dd])
