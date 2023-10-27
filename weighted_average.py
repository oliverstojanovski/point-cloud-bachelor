from scipy.spatial import KDTree
import numpy as np
from rigid_transform_3D import rigid_transform

'''
This is a function that performs a point cloud alignment using a weighted average algorithm. It is part of the ICP algorithm
and can also be called Weighted Iterative Closest Point (ICP) algorithm.

:param target_points: this is a array in python which contains the target point cloud.
:param source_points: this is a python array which contains the source point cloud
:param max_iterations: the maximum number of iterations of the ICP algorithm is set to 100
:param tolerance: Convergence tolerance, which is optional. If the alignment error is below this value, the algorithm stops

This function aligns the target and source point clouds using the weighted ICP algorithm. It iteratively finds the optimal
rigid transformation (rotation and translation) that minimizes the distance od corresponding points in the two point clouds

:return:
The function returns a tuple of three elements:
source points: the aligned source points 
R: a 3x3 Rotation matrix which is used for the alignment
t: a 1x3 translation vector also used for the alignment


'''


def weighted_average(target_points, source_points, max_iterations=100, tolerance=1e-6):
    for iteration in range(max_iterations):
        tree = KDTree(target_points)
        closest_points = []

        # This starts a loop that iterates a maximum of max_iteration times. This loop is a must, as it performs the ICP algorithm.
        # With each iteration, a KDTree data structure is constructed from the traget_points. KDTree is a data structure for nearest-neigbor
        # search. So basically, with the help of KDTree, we are trying to find the closest points of target_points for each point in the
        # source_points during each iteration. "closest_points" is an empty list at the moment.

        for point in source_points:
            _, index = tree.query(point)
            closest_points.append(target_points[index])

        # This starts a new loop for each "point" in "source points" with the end goal of finding the closest point in the "target_points"
        # for each point in the "source_points". "tree.query(point)" uses a KDTree structure (tree) created from the "target points"
        # to find the closest point in the "target_points" to the current "point" in the "source_points". "point" is the point processed
        # in the loop. The "query" method returns two values, but we only need the second one, which is the index of the nearest nieghbor in
        # the "target_point". The "-" in "-, index" is a convention that indicates that the first value is not being used. After finding the
        # closest point in the "target_points", that point is appended (added) to the "closest_points" list. So it's building a list of the
        # closest points in "target_points" corresponding to the "source_points"

        closest_points = np.array(closest_points)

        # the "closest_points" list is convertiert in an array

        R, t = rigid_transform(source_points.T, closest_points.T)

        # the function "rigid_transformation" is called to find the rotation matrix and translation vector for the optimal rigid transformation
        # "source_points" and "target_points" are aligned and then transposed to match the expected input of the function

        source_points = (np.dot(R, source_points.T) + t.reshape((3, 1))).T

        # the "source_points" are rotated using the rotation matrix "R" than translated using the translation vector "t", which are
        # reshaped to match their original dimensions and assigned back to "source_points"

        error = np.linalg.norm(source_points - target_points, axis=1)

        # This is a NumPy structure used for the error, which is the average distance between the corresponding points of the aligned
        # "source_points" and "target_points". The structure "mean" calculates the Euclidean norm between two point clouds and than takes
        # the mean (average) along axis 1

        if iteration % 10 == 0:
            print(f"iteration: {iteration + 1} error: {error}")

        # This is a progress tracking line. I prints every 10 iterations, starting from the first (so 1, 11, 21, 31...), and its error.

        if error < tolerance:
            break

        # if the error is below the specified tolerance, the iterations break early. There is no need of the whole iteration process
        # if the error falls below the tolerance.

    return source_points, R, t
