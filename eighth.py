import open3d as o3d
import numpy as np

# load sample data
pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd_data.path)

# estimate normals
pcd.estimate_normals()

# points + normals
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

features = np.concatenate([points, normals], axis=1)

# print("Shape:", features.shape)
# print(features[:5])

import torch

