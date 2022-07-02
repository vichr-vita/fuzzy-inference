from random import uniform
import unittest

from fuzzy_inference.fuzzy_set import FuzzySet, piecewise_linear, trapezoidal, triangular
import numpy as np

class FuzzySetTestCase(unittest.TestCase):

    def test_fuzzy_set_piecewise_linear(self):
        a = FuzzySet.piecewise_linear(vertices=(
            (0, 0),
            (1, 1),
            (2, 0)
        ))

        self.assertEqual(a.mu(0.5), 0.5)
        self.assertEqual(a.mu(1.5), 0.5)
        self.assertEqual(a.mu(-3), 0)
        self.assertEqual(a.mu(3), 0)

    def test_fuzzy_set_triangular(self):
        t = FuzzySet.triangular(0, 1, 2)

        self.assertEqual(t.mu(0.5), 0.5)
        self.assertEqual(t.mu(1.5), 0.5)

    def test_fuzzy_set_trapezoidal(self):
        t = FuzzySet.trapezoidal(0, 1, 2, 3)

        self.assertEqual(t.mu(0.5), 0.5)
        self.assertEqual(t.mu(1.5), 1)
        self.assertEqual(t.mu(2.5), 0.5)

    def test_discrete(self):
        t = FuzzySet.trapezoidal(0, 1, 2, 3)
        self.assertEqual(t.discrete(-1, 3, 5).shape, (5, 2))

    def test_intersection(self):
        t1 = FuzzySet.trapezoidal(0, 1, 2, 3)
        t2 = FuzzySet.triangular(0.4, 0.5, 0.6)
        intersect = t1.intersection(t2, 0.4, 0.6, resolution=3)
        self.assertEqual(intersect[1, 1], 0.5)

    def test_union(self):
        t1 = FuzzySet.trapezoidal(0, 1, 2, 3)
        t2 = FuzzySet.triangular(0.4, 0.5, 0.6)
        intersect = t1.union(t2, 0.4, 0.6, resolution=3)
        self.assertEqual(intersect[1, 1], 1)

    def test_l_ramp(self):
        lr = FuzzySet.l_ramp(0, 1)

        self.assertEqual(lr.mu(-10), 1)
        self.assertEqual(lr.mu(0.5), 0.5)
        self.assertEqual(lr.mu(2), 0)
        self.assertEqual(lr.mu(10), 0)

    def test_r_ramp(self):
        lr = FuzzySet.r_ramp(0, 1)

        self.assertEqual(lr.mu(-10), 0)
        self.assertEqual(lr.mu(0.5), 0.5)
        self.assertEqual(lr.mu(2), 1)
        self.assertEqual(lr.mu(10), 1)

    def test_ceil(self):
        lr = FuzzySet.r_ramp(0, 1)
        ceil = FuzzySet.uniform(0.5)
        self.assertEqual(lr.intersection(ceil, 0, 1, 11)[-1][1], 0.5)
