#!/usr/bin/env python3

import ctypes
from importlib import import_module
import numpy as np
from scipy.spatial import kdtree

class KdTree():
    def build(pointcloud):
        raise NotImplemented('KdTrees must implement build.')

    def query(query_points, k=1, self_match=True):
        '''
        Returns a tuple (associations, distances)
        associations is a N x (k+1) matrix where the first column is the index
        of the associated point in the reading, and the remaining columns are
        the associated indices in the reference.

        distances is a Nxk matrix of the distance between the reference point and the
        point in the reading.
        '''
        raise NotImplemented('KdTrees must implement queries.')

class KdTreeFactory():
    @classmethod
    def create(cls):
        kdtree = None
        try:
            kdtree = LibnaboKdTree()
        except ImportError as e:
            kdtree = ScipyKdTree()

        return kdtree


class ScipyKdTree(KdTree):
    def build(self, pointcloud):
        self.pointcloud = pointcloud.copy()
        self.kdtree = kdtree.KDTree(pointcloud.T)

    def query(self, query_points, n=1, self_match=True):
        associations = np.empty((query_points.shape[1], 2))
        associations[:,0] = np.arange(query_points.shape[1])

        distances, associations[:,1] = self.kdtree.query(query_points.T)

        return associations.astype(int), distances


class LibnaboKdTree(KdTree):
    def __init__(self):
        try:
            self.pynabo = import_module('pynabo')
        except ImportError as e:
            raise ImportError('Could not instantiate LibnaboKdTree: pynabo not found') from e

    def build(self, pointcloud):
        self.pointcloud = pointcloud.T.copy()
        self.kdtree = self.pynabo.NearestNeighbourSearch(self.pointcloud)

    def query(self, query_points, n=1, self_match=True):
        associations = np.empty((query_points.shape[1], n+1))
        associations[:,0] = np.arange(query_points.shape[1])

        matches = None
        if self_match:
          matches = self.kdtree.knn(query_points.T.copy(), n, 0, optionFlags=self.pynabo.SearchOptionFlags.ALLOW_SELF_MATCH)
        else:
          matches = self.kdtree.knn(query_points.T.copy(), n, 0)

        associations[:,1:n+1] = matches[0]
        distances = np.sqrt(matches[1])

        return associations.astype(int), distances
