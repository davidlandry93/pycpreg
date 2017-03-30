#!/usr/bin/env python3

import numpy as np
import unittest

import pycpreg

class GenericOutlierTest():
    def test_filtering(self):
        associations = np.random.randint(0, 100, (5,2))
        distances = np.random.rand(5,1)

        filtered = self.filter.filter(associations, distances)

        self.assertTrue(5 >= filtered.shape[0])
        self.assertTrue(2, filtered.shape[1])

class QuantileOutlierFilterTest(unittest.TestCase, GenericOutlierTest):
    def setUp(self):
        self.filter = pycpreg.outlier.QuantileOutlierFilterAlgorithm()

if __name__ == '__main__':
    unittest.main()
