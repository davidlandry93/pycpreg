#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod
import numpy as np

DEFAULT_PERCENTAGE_ASSOCIATIONS_TO_KEEP = 0.85

class OutlierFilterAlgorithm(metaclass=ABCMeta):
    @abstractmethod
    def filter(self, associations, errors):
        pass


class QuantileOutlierFilterAlgorithm(OutlierFilterAlgorithm):
    def __init__(self,
                 percentage_to_keep=DEFAULT_PERCENTAGE_ASSOCIATIONS_TO_KEEP):
        self.percentage_to_keep = percentage_to_keep

    def filter(self, associations, errors):
        permutation = np.argsort(errors)
        associations = associations[permutation]

        n_points_to_keep = int(associations.shape[0] * self.percentage_to_keep)
        associations = associations[0:n_points_to_keep]
        return associations
