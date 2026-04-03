import open3d as o3d
import numpy as np

pcd = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd.path)

# downsample
pcd = pcd.voxel_down_sample(voxel_size=0.05)

# clustering
labels = np.array(
    pcd.cluster_dbscan(eps=0.02, min_points=10)
)

# number of clusters
max_label = labels.max()
print("Clusters found:", max_label + 1)

# generate colors
colors = np.random.rand(max_label + 1, 3)

# assign colors safely
color_map = np.zeros((labels.shape[0], 3))

for i in range(len(labels)):
    if labels[i] == -1:
        color_map[i] = [0, 0, 0]  # noise → black
    else:
        color_map[i] = colors[labels[i]]

pcd.colors = o3d.utility.Vector3dVector(color_map)

o3d.visualization.draw_geometries([pcd])
