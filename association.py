#!/usr/bin/env python3

from importlib import import_module
import numpy as np

from pycpreg.kd_tree import ScipyKdTree, LibnaboKdTree

class PointAssociationAlgorithm():
    def associate(self, reading, reference):
        """
        reference: The reference point cloud
        reading: The reading to match against the reference

        returns: A tuple containing a numpy array describing the pairs of points and
        a numpy array containing the distance between each pair.
        """
        raise NotImplemented('Point association algorithms must override associate')


class BruteForcePointAssociationAlgorithm(PointAssociationAlgorithm):
    def associate(self, reading, reference):
        associations = np.empty(reading.shape[1])

        # Create a matrix of the distances from every reading point to every reference point
        distances = np.empty((reading.shape[1], reference.shape[1]))
        for i, point in enumerate(reading.T):
            distances[i] = np.linalg.norm(
                reference[:] - point.reshape(4, 1), axis=0)

        # Get the index of the minimum of each row.
        indices_of_reading = np.arange(reading.shape[1])
        indices_of_reference = np.argmin(distances, axis=1)

        associations = np.vstack((indices_of_reading, indices_of_reference)).T

        distances = np.min(distances, axis=1)

        return associations, distances


class KdTreePointAssociationAlgorithm(PointAssociationAlgorithm):
    """
    Builds a kd-tree of reading, and queries it to find the nearest neighbors in reference.

    Be careful not to mutate reading when using this class, as the kd-trees are cached.
    """
    def __init__(self, kdtree=None):
        if kdtree is None:
            kdtree = ScipyKdTree()
            try:
                kdtree = LibnaboKdTree()
            except ImportError:
                pass

        self.kdtree = kdtree


    def associate(self, reading, reference):
        self.kdtree.build(reference)

        return self.kdtree.query(reading)



class LibnaboPointAssociationAlgorithm(KdTreePointAssociationAlgorithm):
    """
    Uses linbano to make the associations.
    Requires the pynabo module to be installed.
    Checks for the existence of the pynabo package before doing anything.
    """
    def __init__(self):
        super().__init__(LibnaboKdTree())
