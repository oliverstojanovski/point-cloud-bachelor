from plyfile import PlyData
import numpy as np

'''
This function is used to to read the point cloud data of a PLY file (Polygon File Format)

:param: filename (string): the path to the PLY file containing the point cloud data

This function reads a point from a PLY file, extracts its X, Y and Z coordinates and returns them as a NumPy array

The function returns an Nx3 array, representing the 3D coordinates of the point cloud

'''


def read_point_cloud_from_ply(filename):
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex']
    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    return points
