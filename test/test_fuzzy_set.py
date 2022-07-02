import unittest

from fuzzy_inference.set.fuzzy_set import FuzzySet, piecewise_linear, trapezoidal, triangular


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
