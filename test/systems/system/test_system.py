import unittest

import numpy as np
import sympy as sp
from src.systems.inv_pend_1l_rw import IP1LRW, IP1LRWParams
from src.systems.system.system import g


class TestSystem(unittest.TestCase):

    def setUp(self) -> None:
        self.system = IP1LRW(IP1LRWParams())

    def test_T(self):
        self.assertEqual(self.system.T["1->0"], sp.Matrix([
            [sp.cos(self.system.theta[1]), sp.Mul(-1, sp.sin(self.system.theta[1])), 0, 0],
            [sp.sin(self.system.theta[1]), sp.cos(self.system.theta[1]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]))

        self.assertEqual(self.system.T["2->1"], sp.Matrix([
            [sp.cos(self.system.theta[2]), sp.Mul(-1, sp.sin(self.system.theta[2])), 0, 1.0],
            [sp.sin(self.system.theta[2]), sp.cos(self.system.theta[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]))

        self.assertEqual(self.system.T["2->0"], sp.Matrix([
            [sp.cos(self.system.theta[1] + self.system.theta[2]), sp.Mul(-1, sp.sin(self.system.theta[1] + self.system.theta[2])),
             0, sp.Mul(1.0, sp.cos(self.system.theta[1]))],
            [sp.sin(self.system.theta[1] + self.system.theta[2]), sp.cos(self.system.theta[1] + self.system.theta[2]),
             0, sp.Mul(1.0, sp.sin(self.system.theta[1]))],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]))

    def test_D(self):
        self.assertEqual(self.system.D["c1->1"], sp.Matrix([[0.5], [0], [0]]))
        self.assertEqual(self.system.D["c2->1"], sp.Matrix([[1.0], [0], [0]]))
        self.assertEqual(self.system.D["c2->2"], sp.Matrix([[0], [0], [0]]))

    def test_J_v(self):
        self.assertEqual(self.system.J_v["1"], sp.Matrix([
            [sp.Mul(-0.5, sp.sin(self.system.theta[1])), 0],
            [sp.Mul(0.5, sp.cos(self.system.theta[1])), 0],
            [0, 0]
        ]))
        self.assertEqual(self.system.J_v["2"], sp.Matrix([
            [sp.Mul(-1.0, sp.sin(self.system.theta[1])), 0],
            [sp.Mul(1.0, sp.cos(self.system.theta[1])), 0],
            [0, 0]
        ]))

    def test_M(self):
        actual_M = self.system.M
        expected_M = np.array([
            [1.45833333333333, 0.125],
            [0.125, 0.125]
        ])

        self.assertAlmostEqual(actual_M[0, 0], expected_M[0, 0], places=5)
        self.assertAlmostEqual(actual_M[0, 1], expected_M[0, 1], places=5)
        self.assertAlmostEqual(actual_M[1, 0], expected_M[1, 0], places=5)
        self.assertAlmostEqual(actual_M[1, 1], expected_M[1, 1], places=5)

    def test_G(self):
        actual_G = self.system.G
        expected_G = sp.Matrix([
            [sp.Mul(1.5 * g, sp.cos(self.system.theta[1]))],
            [0]
        ])

        self.assertEqual(actual_G, expected_G)


if __name__ == '__main__':
    unittest.main()
