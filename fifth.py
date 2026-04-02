import open3d as o3d

pcd = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd.path)

# estimate normals
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

# visualize normals
o3d.visualization.draw_geometries([pcd], point_show_normal=True)