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

class FlexNet(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        sizes = [784] + hidden_sizes + [10]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
    def forward(self, x, return_all=False):
        x = x.view(-1, 784)
        activations = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                activations.append(x)
        activations.append(x)
        if return_all:
            return activations
        return x

transform = T.ToTensor()
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_subset = Subset(train_set, range(5000))
test_subset = Subset(test_set, range(1000))
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

def train_network(hidden, n_epochs=10):
    torch.manual_seed(42)
    model = FlexNet(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return model

def collect_h1(model):
    model.eval()
    h1_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            acts = model(x, return_all=True)
            h1_list.append(acts[1].cpu().numpy())
    return np.concatenate(h1_list, axis=0)

def discretize(x, n_bins=10):
    if x.std() < 1e-8:
        return np.zeros_like(x, dtype=np.int32)
    bins = np.linspace(x.min() - 1e-8, x.max() + 1e-8, n_bins + 1)
    return np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)

def entropy_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def mutual_info(x, y, n_bins=10):
    xd = discretize(x, n_bins)
    yd = np.asarray(y, dtype=np.int32)
    n_y = yd.max() + 1
    joint = np.zeros((n_bins, n_y))
    for i in range(len(xd)):
        joint[xd[i], yd[i]] += 1
    Hx = entropy_from_counts(np.bincount(xd, minlength=n_bins))
    Hy = entropy_from_counts(np.bincount(yd, minlength=n_y))
    Hxy = entropy_from_counts(joint.flatten())
    return max(0.0, Hx + Hy - Hxy)

def mi_matrix(h1):
    n = h1.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            mi = mutual_info(h1[:, i], h1[:, j])
            M[i, j] = mi
            M[j, i] = mi
    return M

hidden = [128, 64, 32, 16]
model = train_network(hidden)
h1 = collect_h1(model)
print(f"h1 shape: {h1.shape}")

# 互信息矩阵 + 特征分解
print("Computing MI matrix...")
M = mi_matrix(h1)
eigs, vecs = np.linalg.eigh(M)
order = np.argsort(eigs)[::-1]
eigs = eigs[order]
vecs = vecs[:, order]

pos = eigs[eigs > 1e-10]
p = pos / pos.sum()
r_eff = float(np.exp(-np.sum(p * np.log(p))))
print(f"r_eff = {r_eff:.4f}")

# 投影到前 r_eff 个主成分
k = int(np.round(r_eff))
proj = h1 @ vecs[:, :k]
print(f"Projected to {k} dimensions")

# 对不同的分箱数 K，计算熵
print()
print("=" * 70)
print(f"{'K':>6} | {'Σ H(x_i)':>12} | {'上界 r_eff·log2(K)':>20} | {'是否满足':>10}")
print("-" * 70)

results = []
for K in [5, 10, 20, 50, 128]:
    total_H = 0.0
    for i in range(k):
        xd = discretize(proj[:, i], K)
        H = entropy_from_counts(np.bincount(xd, minlength=K))
        total_H += H
    bound = r_eff * math.log2(K)
    ok = "✅" if total_H <= bound else "❌ 违反"
    print(f"{K:>6} | {total_H:>12.4f} | {bound:>20.4f} | {ok:>10}", flush=True)
    results.append({"K": K, "total_H": total_H, "bound": bound, "ok": total_H <= bound})

# 对比：不做 PCA，直接所有 128 维
print()
print("对照：不做 PCA，所有 128 维")
print(f"{'K':>6} | {'Σ H(x_i)':>12} | {'上界 128·log2(K)':>18}")
print("-" * 50)
for K in [5, 10, 20]:
    total_H = 0.0
    for i in range(128):
        xd = discretize(h1[:, i], K)
        H = entropy_from_counts(np.bincount(xd, minlength=K))
        total_H += H
    bound = 128 * math.log2(K)
    print(f"{K:>6} | {total_H:>12.4f} | {bound:>18.4f}", flush=True)

output = {"r_eff": r_eff, "k_used": k, "results": results}
with open("nn_entropy_bound_result.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to nn_entropy_bound_result.json")
