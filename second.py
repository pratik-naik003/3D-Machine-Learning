import open3d as o3d

pcd = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd.path)

# noise removal
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# clean point cloud
clean_pcd = pcd.select_by_index(ind)

# visualize
o3d.visualization.draw_geometries([clean_pcd])