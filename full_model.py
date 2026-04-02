import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# model
class SimplePointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        x = torch.max(x, dim=1)[0]  # 🔥 global feature
        
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd_data.path)

pcd.estimate_normals()

points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

features = np.concatenate([points, normals], axis=1)

print("Original shape:", features.shape)

idx = np.random.choice(len(features), 1024, replace=False)
sampled = features[idx]

print("Sampled shape:", sampled.shape)

sampled_tensor = torch.tensor(sampled, dtype=torch.float32)

# add batch dimension
sampled_tensor = sampled_tensor.unsqueeze(0)

print("Tensor shape:", sampled_tensor.shape)


model = SimplePointNet()

output = model(sampled_tensor)

print("Raw output:", output)


pred = torch.argmax(output, dim=1)

print("Predicted class:", pred.item())