import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
# batch = 32, points = 1024, features = 6
X = torch.rand(32, 1024, 6)

# labels (0 = chair, 1 = table)
y = torch.randint(0, 2, (32,))

model = SimplePointNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    
    outputs = model(X)        # forward
    loss = criterion(outputs, y)  # loss
    
    loss.backward()           # backprop
    optimizer.step()          # update
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")
