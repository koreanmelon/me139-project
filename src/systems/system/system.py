from abc import ABC, abstractmethod
from time import perf_counter

import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from .link import Link, StyledJointT, StyledLinkT
from .types import MatDict, VarDict, Vec

g = 9.81  # gravitational acceleration (m/s^2)

g_sym = sp.Symbol('g')  # gravity symbol
t = sp.Symbol("t")  # time symbol


class RoboticSystem(ABC):
    """
    Base class used to define a robotic system.
    """

    def __init__(self, *links: Link) -> None:
        """
        Args:
            n (int): the number of links in the system
        """

        self.links = links
        self.n = len(links)

        self.alpha: VarDict = {}
        self.b: VarDict = {}
        self.d: VarDict = {}
        self.theta: VarDict = {}
        self.theta_d: VarDict = {}
        self.theta_dd: VarDict = {}

        for i in range(1, self.n + 1):
            self.alpha[i] = 0
            self.b[i] = 0 if i == 1 else self.links[i - 2].l
            self.d[i] = 0
            self.theta[i] = dynamicsymbols(f"theta_{i}")
            self.theta_d[i] = dynamicsymbols(f"theta_{i}", 1)
            self.theta_dd[i] = dynamicsymbols(f"theta_{i}", 2)

        self.Q = sp.Matrix([self.theta[i] for i in range(1, self.n + 1)])
        self.Q_d = sp.Matrix([self.theta_d[i] for i in range(1, self.n + 1)])
        self.Q_dd = sp.Matrix([self.theta_dd[i] for i in range(1, self.n + 1)])

        self.T: MatDict = {}
        self.R: MatDict = {}
        self.D: MatDict = {}
        self.J_v: MatDict = {}
        self.J_w: MatDict = {}

        self.P = None

        self.M: sp.Matrix = sp.zeros(self.n, self.n)
        self.M_d: sp.Matrix = sp.zeros(self.n, self.n)
        self.V: sp.Matrix = sp.zeros(self.n, 1)
        self.G: sp.Matrix = sp.zeros(self.n, 1)
        self.torque: sp.Matrix = sp.zeros(self.n, 1)

        self.compute_T()
        self.compute_R()
        self.compute_D()
        self.compute_J()

        self.compute_M()
        self.compute_V()
        self.compute_G()
        self.compute_torque()

        self.solve_system()

    @abstractmethod
    def solve_system(self) -> None:
        """
        Solve the system of equations.
        """
        ...

    @abstractmethod
    def deriv(self, t: Vec, Q: Vec) -> Vec:
        """
        Compute the derivative of the state vector.
        """
        ...

    @abstractmethod
    def link_positions(self, theta_t_vec: list[Vec]) -> list[StyledLinkT]:
        """
        Given a list of angle vectors, returns a list of link positions as a function of time.
        """
        ...

    @abstractmethod
    def joint_positions(self, theta_t_vec: list[Vec]) -> list[StyledJointT]:
        """
        Given a list of angle vectors, returns a list of joint positions as a function of time.
        """
        ...

    def compute_T(self) -> None:
        """
        Compute the transformation matrices for each link.
        """

        for i in range(1, self.n + 1):
            self.T[f"{i}->{i - 1}"] = self.construct_T(self.alpha[i], self.b[i], self.theta[i], self.d[i])

        for i in range(2, self.n + 1):
            for j in range(0, i - 1):
                self.T[f"{i}->{j}"] = sp.simplify(self.T[f"{i - 1}->{j}"] * self.T[f"{i}->{i - 1}"])

    def compute_R(self) -> None:
        """
        Computes the rotation matrices for each link.
        """

        for key in self.T.keys():
            self.R[key] = self.extract_R(self.T[key])

    def compute_D(self) -> None:
        """
        Compute the distance vectors.
        """
        for key in self.T.keys():
            self.D[key] = self.extract_D(self.T[key])

        for i, link in enumerate(self.links):
            vec = sp.Matrix([link.l_c, 0, 0])

            self.D[f"c{i + 1}->{i + 1}"] = vec

            for j in range(i + 1):
                self.D[f"c{i + 1}->{j}"] = sp.simplify(self.D[f"{i + 1}->{j}"] + self.R[f"{i + 1}->{j}"] * vec)

    def compute_J(self) -> None:
        """
        Compute the Jacobian matrices for the system.
        """

        for i in range(1, self.n + 1):
            cols = []

            for j in range(1, self.n + 1):
                if j > i:
                    cols.append(sp.zeros(3, 1))
                else:
                    cols.append(
                        sp.simplify(
                            self.R[f"{j}->0"].extract([0, 1, 2], [2]).cross(self.R[f"{j}->0"] * self.D[f"c{i}->{j}"])
                        )
                    )

            self.J_v[f"{i}"] = sp.simplify(sp.Matrix.hstack(*cols))

        for i in range(1, self.n + 1):
            cols = []

            for j in range(1, self.n + 1):
                if j > i:
                    cols.append(sp.zeros(3, 1))
                else:
                    cols.append(self.R[f"{j}->0"].extract([0, 1, 2], [2]))

            self.J_w[f"{i}"] = sp.simplify(sp.Matrix.hstack(*cols))

    def compute_M(self) -> None:
        """
        Compute the inertia matrix.
        """

        for i, link in enumerate(self.links):
            linear = link.m * self.J_v[f"{i + 1}"].T * self.J_v[f"{i + 1}"]
            angular = self.J_w[f"{i + 1}"].T * link.I * self.J_w[f"{i + 1}"]
            self.M += linear + angular

        self.M = sp.simplify(self.M)
        self.M_d = self.M.diff(t)  # type: ignore

    def compute_V(self) -> None:
        """
        Compute the Coriolis and centrifugal matrix.
        """

        self.V = sp.simplify(
            self.M_d * self.Q_d - sp.Rational(1, 2) *
            sp.Matrix.vstack(*[self.Q_d.T * self.M.diff(theta) * self.Q_d for theta in self.theta.values()])
        )

    def compute_G(self) -> None:
        """
        Compute the gravity matrix after computing the potential energy.
        """

        expr = 0

        for i, link in enumerate(self.links):
            expr += link.m * g * self.D[f"c{i + 1}->{0}"][1]

        self.P = sp.simplify(expr)

        self.G = sp.simplify(sp.Matrix([[self.P.diff(theta)] for theta in self.theta.values()]))

    def compute_torque(self) -> None:
        """
        Compute the torque matrix.
        """

        self.torque = sp.simplify(self.M * self.Q_dd + self.V + self.G)

    @staticmethod
    def construct_T(alpha, b, theta, d) -> sp.Matrix:
        """
        Creates a transformation matrix for given values of alpha, b, theta, and d.
        """

        return sp.simplify(sp.Matrix([
            [sp.cos(theta), sp.Mul(-1, sp.sin(theta)), 0, b],
            [sp.Mul(sp.cos(alpha), sp.sin(theta)), sp.Mul(sp.cos(alpha), sp.cos(theta)),
             sp.Mul(-1, sp.sin(alpha)), -d * sp.sin(alpha)],
            [sp.Mul(sp.sin(alpha), sp.sin(theta)), sp.Mul(sp.sin(alpha), sp.cos(theta)),
             sp.cos(alpha), d * sp.cos(alpha)],
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
