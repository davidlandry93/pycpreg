# pycpreg
Library for point cloud registration under the ICP algorithm in python.
Compatible with python 3 only.

## Installation

Clone the repository into your `PYTHONPATH` and `import pycpreg`.

## Usage

```
import numpy as np
import pycpreg
import pycpreg.pointcloud_generator as gen

# Pycpreg uses homogeneous coordinates to encode points.
rectangle = gen.Rectangle(np.array([0.0, 0.0, 0.0, 1.0]), np.array([10.0, 10.0, 0.0, 1.0]))
reference = rectangle.sample_uniformly(500)
reading = gen.add_noise(reference)

# Homogeneous coordinates are also used for the transformation matrices.
T = np.array([[1.0, 0.0, 0.0, 5.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

# Introduce a displacement that ICP will have to recover.
reading = np.dot(T, reading)

icp = pycpreg.ICPAlgorithm()
icp_T, error = icp.run(reading, reference)

# icp_T is a transform form the reading frame to the reference frame.
# That means icp_T should be equal to roughly T^-1
print(icp_T)
```

## Kd Trees

ICP needs to do a nearest neighbour search to associate the points.
This search is much more efficient is we build a kd-tree beforehand.
`pycpreg` can use the `pynabo` bindings of [http://github.com/davidlandry93/libnabo](libnabo) to build a kd-tree. 
To benefit from this, make sure that `pynabo.so` is in your `PYTHONPATH`.
You can get `pynabo.so` by installing `libnabo` from source with special CMake flags.
Then, set the point association algorithm:

```
icp = pycpreg.ICPAlgorithm()
icp.assoc_algo = pycpreg.KdTreePointAssociationAlgorithm()
```

If pynabo is not found at this point, pycpreg will fallback to scikit's kd-tree, which is actually slower than the brute-force nearest-neighbour search.
