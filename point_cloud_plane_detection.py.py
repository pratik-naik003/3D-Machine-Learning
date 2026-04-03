import open3d as o3d

pcd = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd.path)

# plane detection
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)

# plane points
plane = pcd.select_by_index(inliers)

# remaining points
rest = pcd.select_by_index(inliers, invert=True)

# visualize
plane.paint_uniform_color([1, 0, 0])  # red plane
rest.paint_uniform_color([0.5, 0.5, 0.5])

o3d.visualization.draw_geometries([plane, rest])
