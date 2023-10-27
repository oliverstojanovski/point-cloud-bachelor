import numpy as np
import open3d as o3d

"""
This function visualizes the results using open3d and sets a coordinate system on the matching objects. 

:param: target_points: The 3D coordinates of the target point cloud.
:param: source_points: The 3D coordinates of the source point cloud.
:param: transformed_source_points: The 3D coordinates of the source point cloud after transformation.
:param: R: The 3x3 rotation matrix obtained from the point cloud matching.
:param: t: The 1x3 translation vector obtained from the point cloud matching. 



"""


def visualize_point_cloud_matching(target_points, source_points, transformed_source_points, R, t):
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    # an empty  Open3D point cloud object is created called "target_cloud"
    # "o3d.utility.Vector3dVector(target_points)" creates an Open3D point cloud from the "target_points" data

    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)

    # same as the previous one, but creates an Open3D point cloud object for the "source_points"

    transformed_cloud = o3d.geometry.PointCloud()
    transformed_cloud.points = o3d.utility.Vector3dVector(transformed_source_points)

    # same as the previous one, but creates an Open3D point cloud object for the "transformed_source_points"

    coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # A coordinate system using Open3D is created. It's a 3D coordinate frame represented as a triangle mesh. Its size is 0.1 and and position is at the orgin [0, 0, 0].

    coordinate_system.transform(np.hstack((R, t.reshape(3, 1))))

    # here the coordinate system is being transformed. Using "np.hstack" it combines the rotation "R" and and translation vector "t". The "transform" method is than called upon the coordinate system
    # to apply the transformation

    o3d.visualization.draw_geometries([target_cloud, source_cloud, transformed_cloud, coordinate_system])

    # This line creates a visualization window using Open3D's visualization functions to draw all the geometries together. The ²target_cloud", ²source_clod", "transformed_cloud" and ²coordinate_system" are displayed.
