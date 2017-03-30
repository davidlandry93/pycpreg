#!/usr/bin/env python3

import numpy as np
import numpy.linalg as linalg
from scipy.linalg import cho_solve, cho_factor

import pycpreg.kd_tree as kd_tree


class ICPErrorFunction():
    def __init__(self):
        self._reference = None

    def compute(self, reading, reference, associations):
        raise NotImplemented('ICP Error functions must override compute.')

    def minimize(self, reading, reference, associations):
        raise NotImplemented('ICP Error functions must override minimize.')

class PointToPointICPErrorFunction(ICPErrorFunction):
    def compute(self, reading, associations):
        vectors = (
            reading[:, associations[:, 0]] - self.reference[:, associations[:, 1]])
        distances = np.linalg.norm(vectors, axis=0)

        return np.sum(distances)

    def minimize(self, reading, associations):
        associated_reading = homogeneous2euclidian(
            reading[:, associations[:, 0]])
        associated_reference = homogeneous2euclidian(
            self.reference[:, associations[:, 1]])

        reading_centroid = np.average(associated_reading, axis=1)
        reference_centroid = np.average(associated_reference, axis=1)

        reading_offsets = associated_reading - reading_centroid.reshape(3, 1)
        reference_offsets = associated_reference - reference_centroid.reshape(
            3, 1)

        covariance_matrix = np.dot(reading_offsets, reference_offsets.T)

        U, lambdas, Vt = linalg.svd(covariance_matrix)

        R = np.dot(Vt.transpose(), U.transpose())
        t = reference_centroid - np.dot(R, reading_centroid)

        T = np.identity(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t

        displaced_reading = np.dot(T, reading)

        return T

class PointToPlaneICPErrorFunction(ICPErrorFunction):
    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, pointcloud):
        self._reference = pointcloud
        self.kdtree = kd_tree.KdTreeFactory.create()
        self.kdtree.build(pointcloud)
        self.normals = self.compute_normals(pointcloud)

    def set_reference(self, pointcloud, kdtree=None, normals=None):
        self._reference = pointcloud

        if kdtree is None:
            self.kdtree = kd_tree.KdTreeFactory.create()
            self.kdtree.build(pointcloud)
        else:
            self.kdtree = kdtree

        self.normals = self.compute_normals(pointcloud) if normals is None else normals

    def compute(self, reading, associations):
        vecs = (reading[0:3,associations[:,0]] - self.reference[0:3,associations[:,1]]) * self.normals[0:3,associations[:,1]]
        return np.sum(np.linalg.norm(vecs, axis=0))

    def minimize(self, reading, associations):
        n_points = associations.shape[0]

        G = np.empty((6, n_points))
        G[0:3,:] = np.cross(reading[0:3,associations[:,0]], self.normals[0:3,associations[:,1]], axisa=0, axisb=0).T
        G[3:6,:] = self.normals[0:3,associations[:,1]]

        h = np.einsum('ij,ij->j', self.reference[:,associations[:,1]][0:3,:] - reading[0:3,associations[:,0]], self.normals[0:3, associations[:,0]])

        A = np.dot(G, G.T)
        b = np.dot(G, h)

        x = None
        try:
            # Solve Ax = b using cholesky decomposition
            x = cho_solve(cho_factor(A), b)
        except linalg.LinAlgError:
            x, residuals, ranks, _ = np.linalg.lstsq(A, b)
            print('Warning: could not compute the Cholesky decomposition of a singular matrix.'+
                  'Approximated it instead. Residuals: {}'.format(residuals))

        T = np.identity(4)
        T[0:3,3] = x[3:]
        T[0,1] = -x[0]
        T[1,0] = x[0]
        T[0,2] = x[1]
        T[2,0] = -x[1]
        T[1,2] = -x[2]
        T[2,1] = x[2]

        return T

    def compute_normals(self, pointcloud):
        n_neighbours_for_normal = 6
        associations, _ = self.kdtree.query(pointcloud, n_neighbours_for_normal, self_match=False)

        normals = np.empty((pointcloud.shape[1], 4))
        for row in associations:
            point_matrix = pointcloud.T[row[1:]]
            normals[row[0]] = self.normal_of_pointcloud(point_matrix.T)

        return normals.T

    def normal_of_pointcloud(self, pointcloud):
        centroid = np.average(pointcloud[0:3, :], axis=1)
        pointcloud = pointcloud[0:3, :] - centroid.reshape((3,1))

        u, s, v = linalg.svd(pointcloud)

        # The left singular vector of the smallest singular value is the normal.
        normal = np.ones(4)
        normal[0:3] = u[:, 2]
        return normal


def homogeneous2euclidian(matrix):
    matrix = matrix / matrix[3, :]
    return matrix[0:3]
