import open3d as o3d

pcd = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd.path)

# downsample
down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

o3d.visualization.draw_geometries([down_pcd])