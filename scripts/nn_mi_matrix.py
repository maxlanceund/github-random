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

def train_network(hidden_sizes, n_epochs=10):
    torch.manual_seed(42)
    model = FlexNet(hidden_sizes).to(device)
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
    """把连续激活离散化到 n_bins 个桶"""
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
    yd = discretize(y, n_bins)
    n = len(xd)
    # 联合直方图
    joint = np.zeros((n_bins, n_bins))
    for i in range(n):
        joint[xd[i], yd[i]] += 1
    Hx = entropy_from_counts(np.bincount(xd, minlength=n_bins))
    Hy = entropy_from_counts(np.bincount(yd, minlength=n_bins))
    Hxy = entropy_from_counts(joint.flatten())
    return max(0.0, Hx + Hy - Hxy)

hidden = [128, 64, 32, 16]
model = train_network(hidden, n_epochs=10)
h1 = collect_h1(model)
n_neurons = h1.shape[1]
print(f"h1 shape: {h1.shape}")
print(f"Computing {n_neurons}x{n_neurons} mutual information matrix...")

# 计算互信息矩阵
M = np.zeros((n_neurons, n_neurons))
for i in range(n_neurons):
    for j in range(i+1, n_neurons):
        mi = mutual_info(h1[:, i], h1[:, j])
        M[i, j] = mi
        M[j, i] = mi
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{n_neurons} done", flush=True)

print()
print("=" * 70)
print("互信息矩阵统计")
print("=" * 70)

# 去掉对角线
off_diag = M[~np.eye(n_neurons, dtype=bool)]
print(f"矩阵元素数: {len(off_diag)}")
print(f"互信息均值: {off_diag.mean():.4f}")
print(f"互信息标准差: {off_diag.std():.4f}")
print(f"互信息最大值: {off_diag.max():.4f}")
print(f"互信息最小值: {off_diag.min():.4f}")

# 每行的平均互信息（度）
degrees = M.sum(axis=1)
print(f"\n度（每行互信息和）均值: {degrees.mean():.4f}")
print(f"度标准差: {degrees.std():.4f}")
print(f"度最大值: {degrees.max():.4f} (neuron {degrees.argmax()})")
print(f"度最小值: {degrees.min():.4f} (neuron {degrees.argmin()})")

# 特征值分解
print()
print("=" * 70)
print("特征值谱分析")
print("=" * 70)

eigenvalues = np.linalg.eigvalsh(M)
eigenvalues = np.sort(eigenvalues)[::-1]
print(f"最大特征值: {eigenvalues[0]:.4f}")
print(f"最小特征值: {eigenvalues[-1]:.4f}")

# 有效秩
positive_eigs = eigenvalues[eigenvalues > 0]
p = positive_eigs / positive_eigs.sum()
effective_rank = float(np.exp(-np.sum(p * np.log(p))))
print(f"有效秩（熵指数）: {effective_rank:.4f}")

# 前 k 个特征值的累积占比
total_sum = positive_eigs.sum()
for k in [1, 2, 5, 10, 20, 50]:
    if k <= len(positive_eigs):
        ratio = positive_eigs[:k].sum() / total_sum
        print(f"前 {k} 个特征值占比: {ratio:.4f}")

# 特征值间隙
gaps = eigenvalues[:-1] - eigenvalues[1:]
max_gap_idx = np.argmax(gaps)
print(f"最大特征值间隙: {gaps[max_gap_idx]:.4f} (在 {max_gap_idx+1} 和 {max_gap_idx+2} 之间)")

# 和随机矩阵对比
print()
print("=" * 70)
print("对比：随机矩阵的谱")
print("=" * 70)

np.random.seed(0)
M_rand = np.random.rand(n_neurons, n_neurons) * off_diag.mean()
M_rand = (M_rand + M_rand.T) / 2
np.fill_diagonal(M_rand, 0)

eigs_rand = np.linalg.eigvalsh(M_rand)
eigs_rand = np.sort(eigs_rand)[::-1]
pos_rand = eigs_rand[eigs_rand > 0]
p_rand = pos_rand / pos_rand.sum()
eff_rank_rand = float(np.exp(-np.sum(p_rand * np.log(p_rand))))
print(f"随机矩阵最大特征值: {eigs_rand[0]:.4f}")
print(f"随机矩阵有效秩: {eff_rank_rand:.4f}")
print(f"真实矩阵有效秩: {effective_rank:.4f}")

# 保存
output = {
    "n_neurons": int(n_neurons),
    "mi_mean": float(off_diag.mean()),
    "mi_std": float(off_diag.std()),
    "mi_max": float(off_diag.max()),
    "degree_mean": float(degrees.mean()),
    "degree_std": float(degrees.std()),
    "eigenvalues_top20": [float(e) for e in eigenvalues[:20]],
    "eigenvalues_bottom5": [float(e) for e in eigenvalues[-5:]],
    "effective_rank": effective_rank,
    "effective_rank_random": eff_rank_rand,
    "max_gap": float(gaps[max_gap_idx]),
    "max_gap_position": int(max_gap_idx + 1),
    "top_k_ratios": {
        str(k): float(positive_eigs[:k].sum() / total_sum)
        for k in [1, 2, 5, 10, 20, 50] if k <= len(positive_eigs)
    },
}
with open("nn_mi_matrix_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_mi_matrix_result.json")
