import open3d as o3d

# sample point cloud load
pcd = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd.path)

# visualize
o3d.visualization.draw_geometries([pcd])