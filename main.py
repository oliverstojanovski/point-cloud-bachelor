import numpy as np
import open3d as o3d
from read_files import read_point_cloud_from_ply
from load_data import TARGET_FILE_PATH, SOURCE_FILE_PATH
from weighted_average import weighted_average
from visualize_the_results import visualize_point_cloud_matching

if __name__ == "__main__":

    target_points = read_point_cloud_from_ply(TARGET_FILE_PATH)
    source_points = read_point_cloud_from_ply(SOURCE_FILE_PATH)

    # Perform ICP matching
    source_points, R, t = weighted_average(target_points, source_points)

    # Transform the source point cloud based on the ICP results
    transformed_source_points = np.dot(R, source_points.T) + t.reshape((3, 1))

    # Visualization using Open3D
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)

    transformed_cloud = o3d.geometry.PointCloud()

    visualize_point_cloud_matching(target_points, source_points, transformed_source_points, R, t)

    matching_objects = []

    for i in range(len(source_points)):
        x, y, z = source_points[i]
        print(f"Coordinates of the {i}-th matching object x = {x}, y = {y}, z = {z}")
        matching_objects.append((x, y, z))

    print(source_cloud)
    print(transformed_cloud)
    print(visualize_point_cloud_matching)
