from unittest import TestCase
from main import *


class TestSigmoid(TestCase):
    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(-100), 3.72008e-44, places=5)
        self.assertAlmostEqual(sigmoid(-2), 0.119203, places=5)
        self.assertAlmostEqual(sigmoid(-1), 0.268941, places=5)
        self.assertAlmostEqual(sigmoid(-0.1), 0.475021, places=5)
        self.assertAlmostEqual(sigmoid(0), 0.5, places=5)
        self.assertAlmostEqual(sigmoid(0.1), 0.524979, places=5)
        self.assertAlmostEqual(sigmoid(1), 0.731059, places=5)
        self.assertAlmostEqual(sigmoid(2), 0.880797, places=5)
        self.assertAlmostEqual(sigmoid(100), 1.0, places=5)
        self.assertAlmostEqual(sigmoid(80000), 1.0, places=5)

