#!/usr/bin/env python3

import math
import numpy as np

class Plane:
    def __init__(self, point, normal):
        self.point = point
        self.normal = normal

    def sample_uniformly(self, n_samples, xmin=0, xmax=10, ymin=0, ymax=10):
        n_samples_per_dim = math.sqrt(n_samples)
        stepx = (xmax - xmin)/n_samples_per_dim
        stepy = (ymax - ymin)/n_samples_per_dim

        d = np.dot(-self.point[0:3], self.normal[0:3])

        xs, ys = np.mgrid[xmin:xmax:stepx, ymin:ymax:stepy]
        zs = (self.normal[0] * xs + self.normal[1] * ys - d) / self.normal[2]

        actual_n_samples = xs.shape[0] * xs.shape[1]

        pointcloud = np.empty((4, actual_n_samples))
        pointcloud[0, :] = xs.flatten()
        pointcloud[1, :] = ys.flatten()
        pointcloud[2, :] = zs.flatten()
        pointcloud[3, :] = 1.0

        return pointcloud

class LineSegment:
    def __init__(self, origin, destination):
        """
        Accepts origin and destination as 3D homogeneous coordinates.
        """
        self.origin = origin
        self.destination = destination

    def homogeneous_vector(self):
        # TODO: Something that works in 3D.
        origin_2d = self.origin[[0,1,3]]
        destination_2d = self.destination[[0,1,3]]

        print(origin_2d)

        return np.cross(origin_2d, destination_2d)

    def sample_radially(self, origin, delta_theta, max_range=20):
        # Compute the angle of the vector from the lidar to the edges of the line.
        v = self.origin - origin
        min_angle = np.arctan2(v[1], v[0])

        v = self.destination - origin
        max_angle = np.arctan2(v[1], v[0])

        if max_angle < min_angle:
            max_angle, min_angle = min_angle, max_angle

        # Generate the angles that would normally be sampled and filter them.
        all_angles = np.arange(0.0, 2*math.pi, delta_theta)
        large_angles = min_angle < all_angles
        small_angles = all_angles < max_angle

        good_angles = np.all(np.vstack((large_angles, small_angles)), axis=0)
        ray_angles = all_angles[good_angles]

        print(min_angle)
        print(max_angle)
        print(good_angles)
        print(ray_angles)

        # Make homogeneous vectors representing a line for each ray
        ray_generators = np.empty((3, ray_angles.shape[0]))
        ray_generators[0,:] = np.cos(ray_angles)
        ray_generators[1,:] = np.sin(ray_angles)
        ray_generators[2,:] = 1.0

        rays = np.cross(ray_generators, origin[[0,1,3]], axisa=0)

        # Check where the rays intersect
        intersections = np.cross(rays, self.homogeneous_vector(), axisa=1)
        intersections = intersections / np.expand_dims(intersections[:,2], 1)

        return intersections

    def sample_uniformly(self, n_points, endpoint=True):
        '''
        Creates a point cloud shaped like a line, from origin to destination.
        Origin and destination are 4 long numpy arrays representing points in homogeneous coordinates.

        returns: A 4xN homogeneous matrix containing points interpolating the given line.
        '''

        m = np.zeros((4, n_points))

        for i in range(n_points):
            m[:, i] = self.origin

        to_add = np.empty((4, n_points))

        for i in range(4):
            to_add[i, :] = (self.destination - self.origin)[i]

        lambdas = np.linspace(0, 1.0, num=n_points, endpoint=endpoint)
        for i in range(4):
            to_add[i,:] *= lambdas

        return m + to_add


class Rectangle:
    def __init__(self, bottom_left, top_right, z=0.0):
        pt1 = bottom_left
        pt2 = np.array([top_right[0], bottom_left[1], z, 1.0])
        pt3 = top_right
        pt4 = np.array([bottom_left[0], top_right[1], z, 1.0])

        self.lines = [LineSegment(pt1, pt2),
                      LineSegment(pt2, pt3),
                      LineSegment(pt3, pt4),
                      LineSegment(pt4, pt1)]

    def sample_uniformly(self, n_points):
        m = np.empty((4,0))
        for line in self.lines:
            m = np.concatenate((m, line.sample_uniformly(n_points // 4, endpoint=False)), axis=1)

        return m

def line(origin, destination, n_points, endpoint=True):
    line = LineSegment(origin, destination)
    return line.sample_uniformly(n_points, endpoint)


def rectangle(origin, destination, n_points, z=0.0):
    rectangle = Rectangle(origin, destination, z=z)
    return rectangle.sample_uniformly(n_points)


def add_noise(pointcloud, stddev=0.01):
    """
    Adds noise to the pointcloud. For every point n in the point cloud,
    we return a new point sampled from a gaussian centered in n and
    with isotropic covariance stddev.
    """
    result = pointcloud
    result[0:2, :] = np.random.normal(pointcloud[0:2, :], stddev)
    return result

def wrap2pi(angle):
    return angle % 2*math.pi
