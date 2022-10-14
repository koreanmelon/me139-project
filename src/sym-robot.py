#####################################################
##  This is taken from code used for Homework 3.   ##
##  Do not use this as is. It likely doesn't even  ##
##  work for HW 3 properly.                        ##
#####################################################

import sympy as sp
from sympy.physics.mechanics import dynamicsymbols


# ! DONT USE THIS CODE
# ! THIS IS ONLY HERE FOR REFERENCE IF NEEDED
# ! MOST OF THIS MATH IS WRONG, SEE `case-1.ipynb` INSTEAD
class SymRobot:
    g = sp.Symbol('g')  # gravity
    t = sp.Symbol('t')  # time

    def __init__(self) -> None:
        # Symbolic Variables
        self.m_1 = sp.symbols("M_1")  # mass of tibia
        self.m_2 = sp.symbols("M_2")  # mass of femur
        self.m_3 = sp.symbols("M_3")  # mass of rest of human
        self.l_1 = sp.symbols("l_1")  # tibia length
        self.l_2 = sp.symbols("l_2")  # femur length
        self.l_3 = sp.symbols("l_3")  # body length
        self.l_c1 = sp.symbols("l_c1")  # tibia length
        self.l_c2 = sp.symbols("l_c2")  # femur length
        self.l_c3 = sp.symbols("l_c3")  # body length

        self.alpha_1 = 0
        self.alpha_2 = 0
        self.alpha_3 = 0
        self.b_1 = 0
        self.b_2 = self.l_1
        self.b_3 = self.l_2
        self.theta_1 = dynamicsymbols("theta_1")
        self.theta_2 = dynamicsymbols("theta_2")
        self.theta_3 = dynamicsymbols("theta_3")
        self.d_1 = 0
        self.d_2 = 0
        self.d_3 = 0

        # Transformation Matrices
        self.T: dict[tuple, sp.Matrix] = {
            (0, 1): SymRobot.create_T(self.alpha_1, self.b_1, self.theta_1, self.d_1),
            (1, 2): SymRobot.create_T(self.alpha_2, self.b_2, self.theta_2, self.d_2),
            (2, 3): SymRobot.create_T(self.alpha_3, self.b_3, self.theta_3, self.d_3),
        }
        self.T |= {
            (0, 2): self.T[(0, 1)] @ self.T[(1, 2)],
            (0, 3): self.T[(0, 1)] @ self.T[(1, 2)] @ self.T[(2, 3)],
            (1, 3): self.T[(1, 2)] @ self.T[(2, 3)]
        }

        # Rotation Matrices (extracted from transformation matrices)
        self.R: dict[tuple, sp.Matrix] = {
            (0, 1): SymRobot.extract_R(self.T[(0, 1)]),
            (1, 2): SymRobot.extract_R(self.T[(1, 2)]),
            (2, 3): SymRobot.extract_R(self.T[(2, 3)]),
            (0, 2): SymRobot.extract_R(self.T[(0, 2)]),
            (0, 3): SymRobot.extract_R(self.T[(0, 3)]),
            (1, 3): SymRobot.extract_R(self.T[(1, 3)])
        }

        # Displacement Vectors (extracted from transformation matrices)
        self.D: dict[tuple, sp.Matrix] = {
            (0, 1): SymRobot.extract_D(self.T[(0, 1)]),
            (1, 2): SymRobot.extract_D(self.T[(1, 2)]),
            (2, 3): SymRobot.extract_D(self.T[(2, 3)]),
            (0, 2): SymRobot.extract_D(self.T[(0, 2)]),
            (0, 3): SymRobot.extract_D(self.T[(0, 3)]),
            (1, 3): SymRobot.extract_D(self.T[(1, 3)])
        }

        # Jacobians
        self.J_v2: sp.Matrix = sp.simplify(
            sp.Matrix.hstack(
                self.R[(0, 1)].extract([0, 1, 2], [2]).cross(self.R[(0, 1)] @ self.D[(1, 2)]),
                sp.zeros(3, 1),
                sp.zeros(3, 1)
            )
        )

        self.J_v3: sp.Matrix = sp.simplify(
            sp.Matrix.hstack(
                self.R[(0, 1)].extract([0, 1, 2], [2]).cross(self.R[(0, 1)] @ self.D[(1, 3)]),
                self.R[(0, 2)].extract([0, 1, 2], [2]).cross(self.R[(0, 2)] @ self.D[(2, 3)]),
                sp.zeros(3, 1)
            )
        )

        self.J_omega2: sp.Matrix = sp.simplify(
            sp.Matrix.hstack(
                self.R[(0, 1)].extract([0, 1, 2], [2]),
                sp.zeros(3, 1),
                sp.zeros(3, 1)
            )
        )

        self.J_omega3: sp.Matrix = sp.simplify(
            sp.Matrix.hstack(
                self.R[(0, 1)].extract([0, 1, 2], [2]),
                self.R[(0, 2)].extract([0, 1, 2], [2]),
                sp.zeros(3, 1)
            )
        )

        # Moment of Inertias
        self.I_c1 = SymRobot.create_I_c(self.m_1, self.l_1)
        self.I_c2 = SymRobot.create_I_c(self.m_2, self.l_2)
        self.I_c3 = SymRobot.create_I_c(self.m_3, self.l_3)

        M_term_2 = sp.simplify(self.m_2 * self.J_v2.T @ self.J_v2 + self.I_c2 * self.J_omega2.T @ self.J_omega2)
        M_term_3 = sp.simplify(self.m_3 * self.J_v3.T @ self.J_v3 + self.I_c3 * self.J_omega3.T @ self.J_omega3)

        self.M: sp.Matrix = sp.simplify(M_term_2 + M_term_3)
        self.M_dot = self.M.diff(SymRobot.t)

        self.Q = sp.Matrix([self.theta_1, self.theta_2, self.theta_3]).reshape(3, 1)
        self.Q_dot = self.Q.diff(SymRobot.t)
        self.Q_ddot = self.Q_dot.diff(SymRobot.t)

        self.V: sp.Matrix = sp.simplify(self.M_dot @ self.Q_dot - sp.Rational(1, 2) * sp.Matrix.vstack(
            self.Q_dot.T @ self.M.diff(self.theta_1) @ self.Q_dot,
            self.Q_dot.T @ self.M.diff(self.theta_2) @ self.Q_dot,
            self.Q_dot.T @ self.M.diff(self.theta_3) @ self.Q_dot
        ))

        h_1: sp.Expr = (self.T[(0, 1)] @ sp.Matrix([self.l_c1, 0, 0, 1]).reshape(4, 1))[1]
        h_2: sp.Expr = (self.T[(0, 2)] @ sp.Matrix([self.l_c2, 0, 0, 1]).reshape(4, 1))[1]
        h_3: sp.Expr = (self.T[(0, 3)] @ sp.Matrix([self.l_c3, 0, 0, 1]).reshape(4, 1))[1]

        self.P = sp.simplify(SymRobot.g * (self.m_1 * h_1 + self.m_2 * h_2 + self.m_3 * h_3))

        self.G = sp.Matrix([
            self.P.diff(self.theta_1),
            self.P.diff(self.theta_2),
            self.P.diff(self.theta_3)
        ]).reshape(3, 1)

        self.Torque: sp.Matrix = sp.simplify(self.M @ self.Q_ddot + self.V + self.G)

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
    def extract_R(T: sp.Matrix) -> sp.Matrix:
        """
        Extracts the rotation matrix from a transformation matrix.
        """

        return T[:3, :3]  # type: ignore

    @staticmethod
    def extract_D(T: sp.Matrix) -> sp.Matrix:
        """
        Extracts the translation vector from a transformation matrix.
        """

        return T[:3, 3]  # type: ignore

    @staticmethod
    def create_I_c(m, l) -> sp.Expr:
        """
        Computes the moment of inertia of a link (rod) about its center of mass
        given its mass and length.
        """

        return sp.Rational(1, 12) * m * l ** 2

    def evalute(self, theta_1, theta_2, theta_3):
        M_ = self.M.subs([(self.theta_1, theta_1), (self.theta_2, theta_2), (self.theta_3, theta_3)])
        V_ = self.V.subs([(self.theta_1, theta_1), (self.theta_2, theta_2), (self.theta_3, theta_3)])
        G_ = self.G.subs([(self.theta_1, theta_1), (self.theta_2, theta_2), (self.theta_3, theta_3)])
        Q_ddot_ = self.Q_ddot.subs([(self.theta_1, theta_1), (self.theta_2, theta_2), (self.theta_3, theta_3)])

        return sp.simplify(M_ @ Q_ddot_ + V_ + G_)


bot = SymRobot()
sp.pprint(sp.simplify(bot.Torque.evalf(subs={
    bot.theta_1: sp.pi / 2,
    bot.theta_2: 0,
    bot.theta_3: 0,
})))
# sp.pprint(bot.evalute(sp.pi / 2, 0, 0))
