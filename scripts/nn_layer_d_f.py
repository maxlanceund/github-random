import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import numpy as np
import json
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# 小网络：784 -> 32 -> 16 -> 10
# ============================================================
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
    def forward(self, x, return_all=False):
        x = x.view(-1, 784)
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        a3 = self.fc3(a2)
        if return_all:
            return x, a1, a2, a3
        return a3

# ============================================================
# 数据：只用 5000 张，快速
# ============================================================
transform = T.ToTensor()
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_subset = Subset(train_set, range(5000))
test_subset = Subset(test_set, range(1000))

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

# ============================================================
# 训练网络
# ============================================================
model = SmallNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training...")
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}")

# ============================================================
# 收集各层激活
# ============================================================
model.eval()
all_x, all_a1, all_a2, all_a3, all_y = [], [], [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x_flat, a1, a2, a3 = model(x, return_all=True)
        all_x.append(x_flat.cpu().numpy())
        all_a1.append(a1.cpu().numpy())
        all_a2.append(a2.cpu().numpy())
        all_a3.append(a3.cpu().numpy())
        all_y.append(y.cpu().numpy())

X = np.concatenate(all_x, axis=0)
A1 = np.concatenate(all_a1, axis=0)
A2 = np.concatenate(all_a2, axis=0)
A3 = np.concatenate(all_a3, axis=0)
Y = np.concatenate(all_y, axis=0)

print(f"Collected: X={X.shape}, A1={A1.shape}, A2={A2.shape}, A3={A3.shape}")

# ============================================================
# 对每一层激活，训一个小分类器，估计 H(Y|层)
# ============================================================
def estimate_conditional_entropy(features, labels, n_epochs=100):
    """在 features 上训 MLP 预测 labels，返回交叉熵损失（自然对数）"""
    X_t = torch.tensor(features, dtype=torch.float32).to(device)
    y_t = torch.tensor(labels, dtype=torch.long).to(device)
    d = X_t.shape[1]
    clf = nn.Sequential(
        nn.Linear(d, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        clf.train()
        opt.zero_grad()
        out = clf(X_t)
        loss = F.cross_entropy(out, y_t)
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        out = clf(X_t)
        loss = F.cross_entropy(out, y_t)
    return loss.item()

# 所有层的 D_f
log2_10 = math.log(10)
layers = {
    "input (784)": X,
    "layer1 (32)": A1,
    "layer2 (16)": A2,
    "output (10)": A3,
}

print()
print("=" * 60)
print(f"{'Layer':>15} | {'CE (nats)':>10} | {'D_f':>8}")
print("-" * 60)

results = {}
for name, feat in layers.items():
    # 标准化
    feat_std = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)
    ce = estimate_conditional_entropy(feat_std, Y)
    D_f = ce / log2_10
    results[name] = {"CE": ce, "D_f": D_f}
    print(f"{name:>15} | {ce:>10.4f} | {D_f:>8.4f}")

with open("nn_layer_d_f_result.json", "w") as f:
    json.dump(results, f, indent=2)
print()
print("Saved to nn_layer_d_f_result.json")
