#!/usr/bin/env python3

import unittest
import numpy as np

import pycpreg

class GenericAssociationTester():
    def test_association(self):
        reading = np.empty((4, 10))
        reading[0,:] = np.linspace(0, 10, num=10, endpoint=False)
        reading[1,:] = reading[0,:]
        reading[2,:] = reading[0,:]
        reading[3,:] = 1.0

        reference = reading.copy()

        associations, distances = self.assoc_algo.associate(reading, reference)

        print(associations)

        self.assertTrue(np.all(associations[:,0] == associations[:,1]))
        self.assertFalse(np.any(distances)) # All distances are 0.0

    def test_distances(self):
        reading = np.empty((4,5))
        reading[0,:] = np.linspace(0, 5, num=5, endpoint=False)
        reading[1,:] = reading[0,:]
        reading[2,:] = reading[0,:]
        reading[3,:] = 1.0

        reference = reading.copy()
        reference[0:3, :] += 0.2

        associations, distances = self.assoc_algo.associate(reading, reference)

        self.assertTrue(np.allclose(distances, np.sqrt(3 * 0.2 * 0.2)))
        self.assertTrue(np.all(np.equal(np.array([5,1]), distances.shape)))


class TestKdTreeAssociation(unittest.TestCase,GenericAssociationTester):
    def setUp(self):
        self.assoc_algo = pycpreg.association.KdTreePointAssociationAlgorithm()


class TestBruteForceAssociation(unittest.TestCase, GenericAssociationTester):
    def setUp(self):
        self.assoc_algo = pycpreg.association.BruteForcePointAssociationAlgorithm()

if __name__ == '__main__':
    unittest.main()
