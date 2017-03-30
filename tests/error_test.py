#!/usr/bin/env python3

import unittest
import numpy as np

import pycpreg.pointcloud_generator as gen
import pycpreg.error_function

class GenericErrorTester():
    def test_error(self):
        plane = gen.Plane(np.array([0.0, 0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0]), )

        reference = plane.sample_uniformly(100)
        reading = reference.copy()

        n_points = reference.shape[1]
        associations = np.empty((n_points, 2), dtype=int)
        associations[:,0] = np.arange(n_points)
        associations[:,1] = np.arange(n_points)

        self.error_function.reference = reference
        error = self.error_function.compute(reading, associations)

        self.assertAlmostEqual(0.0, error)

    def test_minimization(self):
        plane = gen.Plane(np.array([0.0, 0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0]), )

        reference = plane.sample_uniformly(100)
        reading = reference.copy()

        n_points = reference.shape[1]
        associations = np.empty((n_points, 2), dtype=int)
        associations[:,0] = np.arange(n_points)
        associations[:,1] = np.arange(n_points)

        self.error_function.reference = reference
        error = self.error_function.compute(reading, associations)

        T = self.error_function.minimize(reading, associations)
        self.assertTrue(np.all(np.isclose(T, np.identity(4))))

    def test_non_degenerate_case(self):
        plane = gen.Plane(np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0]), )

        reference = gen.add_noise(plane.sample_uniformly(100))
        reading = gen.add_noise(reference.copy())
        reading += np.array([0.0, 0.0, -1.0, 1.0]).reshape((4,1))

        n_points = reference.shape[1]
        associations = np.empty((n_points, 2), dtype=int)
        associations[:,0] = np.arange(n_points)
        associations[:,1] = np.arange(n_points)

        self.error_function.reference = reference
        error = self.error_function.compute(reading, associations)

        T = self.error_function.minimize(reading, associations)

        self.assertTrue(0.0 < T[2,3])


class PointToPointErrorTest(unittest.TestCase, GenericErrorTester):
    def setUp(self):
        self.error_function = pycpreg.error_function.PointToPointICPErrorFunction()


class PointToPlaneErrorTest(unittest.TestCase, GenericErrorTester):
    def setUp(self):
        self.error_function = pycpreg.error_function.PointToPlaneICPErrorFunction()

    def testFullyConstrainedCase(self):
        plane1 = gen.Plane(np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0]))
        plane2 = gen.Plane(np.array([20.0, 20.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0, 1.0]))
        plane3 = gen.Plane(np.array([-20.0, -20.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0, 1.0]))

        reference = np.hstack((plane1.sample_uniformly(100), plane2.sample_uniformly(100), plane3.sample_uniformly(100)))
        reading = gen.add_noise(reference.copy() + np.array([[0.0], [0.0], [1.0], [1.0]]))

        n_points = reference.shape[1]
        associations = np.empty((n_points, 2), dtype=int)
        associations[:,0] = np.arange(n_points)
        associations[:,1] = np.arange(n_points)

        self.error_function.reference = reference
        T = self.error_function.minimize(reading, associations)

    def testTranslationInvariance(self):
        plane = gen.Plane(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 5.0, 1.0]), )

        reference = plane.sample_uniformly(100)
        reading = reference.copy()

        # Translate the reading but leave it on the same plane.
        reading += np.array([10.0, 10.0, 0.0, 1.0]).reshape((4,1))

        n_points = reference.shape[1]
        associations = np.empty((n_points, 2), dtype=int)
        associations[:,0] = np.arange(n_points)
        associations[:,1] = np.arange(n_points)

        self.error_function.reference = reference
        error = self.error_function.compute(reading, associations)

        self.assertAlmostEqual(0.0, error)


if __name__ == '__main__':
    unittest.main()
