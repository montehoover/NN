from unittest import TestCase
import numpy as np
from main import *


class TestSigmoid(TestCase):
    def test_sigmoid(self):
        # self.assertAlmostEqual(sigmoid(-100), 3.72008e-44, places=5)
        # self.assertAlmostEqual(sigmoid(-2), 0.119203, places=5)
        # self.assertAlmostEqual(sigmoid(-1), 0.268941, places=5)
        # self.assertAlmostEqual(sigmoid(-0.1), 0.475021, places=5)
        # self.assertAlmostEqual(sigmoid(0), 0.5, places=5)
        # self.assertAlmostEqual(sigmoid(0.1), 0.524979, places=5)
        # self.assertAlmostEqual(sigmoid(1), 0.731059, places=5)
        # self.assertAlmostEqual(sigmoid(2), 0.880797, places=5)
        # self.assertAlmostEqual(sigmoid(100), 1.0, places=5)
        # self.assertAlmostEqual(sigmoid(80000), 1.0, places=5)
        pass


class TestSingle_squared_loss(TestCase):
    def test_single_squared_loss(self):
        a = np.array([3, 2])
        b = np.array([1, 1])
        z = np.array([3 - 1, 2 - 1])
        np.testing.assert_array_equal(a - b, z)
        y = z
        z = np.array([2 ** 2, 1 ** 1])
        np.testing.assert_array_equal(np.square(y), z)
        y = z
        z = 2 ** 2 + 1 ** 1
        self.assertEqual(np.sum(y), z)
        self.assertEqual(single_squared_loss(a, b), z, 5)

        a = np.arange(100)
        b = np.arange(100)
        a += 3
        self.assertEqual(single_squared_loss(a, b), 100 * (3 ** 2))


class TestSquared_loss(TestCase):
    def test_squared_loss(self):
        A = np.arange(1000).reshape(10,100)
        B = np.arange(1000).reshape(10,100)
        A += 3
        self.assertEqual(squared_loss_over_dataset(A, B), (100 * (3 ** 2) * 10) / 2)

