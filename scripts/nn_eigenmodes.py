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
    labels_list = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            acts = model(x, return_all=True)
            h1_list.append(acts[1].cpu().numpy())
            labels_list.append(y.numpy())
    return np.concatenate(h1_list, axis=0), np.concatenate(labels_list, axis=0)

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
h1, labels = collect_h1(model)
print(f"h1 shape: {h1.shape}, labels shape: {labels.shape}")

# 计算互信息矩阵
M = mi_matrix(h1)
eigs, vecs = np.linalg.eigh(M)
order = np.argsort(eigs)[::-1]
eigs = eigs[order]
vecs = vecs[:, order]

print()
print("=" * 70)
print("特征模式分析")
print("=" * 70)
print(f"{'模式':>5} | {'特征值':>10} | {'与类别的互信息':>15}")
print("-" * 45)

# 对每个主要模式，把 h1 投影到该方向，测与类别标签的互信息
mode_results = []
for k in range(15):
    v = vecs[:, k]
    proj = h1 @ v
    mi = mutual_info(proj, labels)
    mode_results.append({"mode": k, "eig": float(eigs[k]), "mi": float(mi)})
    print(f"{k:>5} | {eigs[k]:>10.4f} | {mi:>15.4f}", flush=True)

# 随机方向对照
print()
print("=" * 70)
print("随机方向对照")
print("=" * 70)
np.random.seed(0)
random_mis = []
for i in range(20):
    v_rand = np.random.randn(h1.shape[1])
    v_rand = v_rand / np.linalg.norm(v_rand)
    proj = h1 @ v_rand
    mi = mutual_info(proj, labels)
    random_mis.append(mi)

print(f"随机方向 MI 均值: {np.mean(random_mis):.4f}")
print(f"随机方向 MI 标准差: {np.std(random_mis):.4f}")
print(f"随机方向 MI 最大: {np.max(random_mis):.4f}")

# 对比
print()
print("=" * 70)
print("对比：前 5 个模式的 MI vs 随机方向")
print("=" * 70)
top5_mi = [r["mi"] for r in mode_results[:5]]
print(f"前 5 个模式 MI: {[round(m, 4) for m in top5_mi]}")
print(f"随机方向平均:  {round(np.mean(random_mis), 4)}")

# 关键判断
best_mode_mi = max(top5_mi)
random_95pct = np.percentile(random_mis, 95)

if best_mode_mi > random_95pct:
    print(f"\n→ 前 5 个模式中，有模式与类别强相关（MI = {best_mode_mi:.4f} > 随机 95% 分位 = {random_95pct:.4f}）")
else:
    print(f"\n→ 前 5 个模式的 MI 都在随机范围内")

# 多模式联合
print()
print("=" * 70)
print("多模式联合的 MI")
print("=" * 70)
for k in [1, 2, 3, 5, 10, 12, 15]:
    proj = h1 @ vecs[:, :k]
    # 多维互信息用分类器近似
    X_t = torch.tensor(proj, dtype=torch.float32).to(device)
    y_t = torch.tensor(labels, dtype=torch.long).to(device)
    clf = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, 10)).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(50):
        clf.train()
        opt.zero_grad()
        loss = F.cross_entropy(clf(X_t), y_t)
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        ce = F.cross_entropy(clf(X_t), y_t).item()
    # 用 CE 近似
    print(f"前 {k:>2} 个模式联合：CE = {ce:.4f}, accuracy = {(clf(X_t).argmax(dim=1) == y_t).float().mean().item()*100:.2f}%")

output = {
    "modes": mode_results,
    "random_mi_mean": float(np.mean(random_mis)),
    "random_mi_std": float(np.std(random_mis)),
    "random_mi_max": float(np.max(random_mis)),
    "random_95pct": float(random_95pct),
}
with open("nn_eigenmodes_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_eigenmodes_result.json")
