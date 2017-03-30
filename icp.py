#!/usr/bin/env python3

import math
import numpy as np

from pycpreg.association import BruteForcePointAssociationAlgorithm
from pycpreg.outlier import QuantileOutlierFilterAlgorithm
from pycpreg.error_function import PointToPointICPErrorFunction


DEFAULT_CONVERGENCE_THRESHOLD = 0.01
DEFAULT_MAX_ITERATIONS = 500

class ICPAlgorithm:
    def __init__(self,
                 convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
                 max_iterations=DEFAULT_MAX_ITERATIONS):
        self.outlier_filter = QuantileOutlierFilterAlgorithm()
        self.point_association_algorithm = BruteForcePointAssociationAlgorithm()
        self.error_function = PointToPointICPErrorFunction()
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

    def run(self,
            reading,
            reference,
            initial_estimate=np.identity(4),
            print_verbose=False):
        delta = math.inf
        latest_error = math.inf
        n_iterations = 0

        self.error_function.reference = reference

        T = initial_estimate
        while (abs(delta) > self.convergence_threshold and
               n_iterations < self.max_iterations):
            reading_copy = np.dot(T, reading)

            associations, errors = self.point_association_algorithm.associate(
                reading_copy, reference)
            filtered_associations = self.outlier_filter.filter(associations,
                                                               errors)

            error_before = self.error_function.compute(
                reading_copy, filtered_associations)

            correction_T = self.error_function.minimize(
                reading_copy, filtered_associations)
            T = np.dot(correction_T, T)

            error_after = self.error_function.compute(
                np.dot(correction_T, reading_copy),
                filtered_associations)

            delta = error_after - error_before
            latest_error = error_after
            if print_verbose:
                print('Error: {} (Δ {})'.format(error_after, delta))

            n_iterations = n_iterations + 1

        if print_verbose:
            print('N iterations: {}'.format(n_iterations))

        return T, latest_error
