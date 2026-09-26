import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np
import json
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

class FlexNet(nn.Module):
    def __init__(self, hidden_sizes, n_classes):
        super().__init__()
        sizes = [784] + hidden_sizes + [n_classes]
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

class SimpleDS(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

transform = T.ToTensor()
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def make_dataset(classes, n_per_class=5000):
    """每类取 n_per_class 张"""
    label_map = {c: i for i, c in enumerate(classes)}
    train_data = []
    counts = {c: 0 for c in classes}
    for i in range(len(mnist_train)):
        label = mnist_train[i][1]
        if label in classes and counts[label] < n_per_class:
            train_data.append((mnist_train[i][0], label_map[label]))
            counts[label] += 1
        if all(counts[c] >= n_per_class for c in classes):
            break
    # 测试集：每类 1000 张
    test_data = []
    test_counts = {c: 0 for c in classes}
    for i in range(len(mnist_test)):
        label = mnist_test[i][1]
        if label in classes and test_counts[label] < 1000:
            test_data.append((mnist_test[i][0], label_map[label]))
            test_counts[label] += 1
        if all(test_counts[c] >= 1000 for c in classes):
            break
    return train_data, test_data

def train_network(hidden, n_classes, train_loader, n_epochs=30):
    torch.manual_seed(42)
    model = FlexNet(hidden, n_classes).to(device)
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

def collect_h1(model, test_loader):
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

def effective_rank(M):
    eigs = np.linalg.eigvalsh(M)
    pos = eigs[eigs > 1e-10]
    if len(pos) == 0:
        return 0.0
    p = pos / pos.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

hidden = [128, 64, 32, 16]
class_list = [2, 3, 4, 5, 6, 8, 10]

print("=" * 75)
print("有效秩 vs 类别数（充分训练版）")
print("=" * 75)
print(f"{'k':>4} | {'acc':>7} | {'r_eff':>10} | {'r_eff - k':>10} | {'log2(k)':>10} | {'r_eff/k':>8}")
print("-" * 75)

results = []
for k in class_list:
    classes = list(range(k))
    train_data, test_data = make_dataset(classes, n_per_class=5000)
    train_loader = DataLoader(SimpleDS(train_data), batch_size=128, shuffle=True)
    test_loader = DataLoader(SimpleDS(test_data), batch_size=128, shuffle=False)

    model = train_network(hidden, k, train_loader, n_epochs=30)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total

    h1 = collect_h1(model, test_loader)
    M = mi_matrix(h1)
    r_eff = effective_rank(M)

    results.append({"k": k, "acc": acc, "r_eff": r_eff})
    print(f"{k:>4} | {acc*100:>6.2f}% | {r_eff:>10.4f} | {r_eff - k:>10.4f} | {math.log2(k):>10.4f} | {r_eff/k:>8.4f}", flush=True)

ks = np.array([r["k"] for r in results])
r_effs = np.array([r["r_eff"] for r in results])
diffs = r_effs - ks
logks = np.log2(ks)

# 拟合 r_eff = k + c * log2(k)
c_fit = np.sum(diffs * logks) / np.sum(logks ** 2)

# 带截距
coef = np.polyfit(logks, diffs, 1)
a_fit, b_fit = coef[0], coef[1]

# 线性拟合 r_eff = a * k + b
coef_lin = np.polyfit(ks, r_effs, 1)
lin_a, lin_b = coef_lin[0], coef_lin[1]

print()
print("=" * 75)
print("拟合结果")
print("=" * 75)
print(f"r_eff - k = {c_fit:.4f} * log2(k)               (过原点)")
print(f"r_eff - k = {a_fit:.4f} * log2(k) + {b_fit:.4f}  (带截距)")
print(f"r_eff = {lin_a:.4f} * k + {lin_b:.4f}            (线性)")

print()
print("对比：")
print(f"{'k':>4} | {'r_eff':>10} | {'k+c*log2k':>10} | {'a*log2k+b':>12} | {'lin':>10}")
print("-" * 65)
for r in results:
    k = r["k"]
    pred_c = k + c_fit * math.log2(k)
    pred_ab = k + a_fit * math.log2(k) + b_fit
    pred_lin = lin_a * k + lin_b
    print(f"{k:>4} | {r['r_eff']:>10.4f} | {pred_c:>10.4f} | {pred_ab:>12.4f} | {pred_lin:>10.4f}")

# 检查饱和
print()
print("=" * 75)
print("饱和检查")
print("=" * 75)
for i in range(1, len(results)):
    r_prev = results[i-1]
    r_curr = results[i]
    delta_r = r_curr["r_eff"] - r_prev["r_eff"]
    delta_k = r_curr["k"] - r_prev["k"]
    ratio = delta_r / delta_k if delta_k > 0 else 0
    print(f"  k: {r_prev['k']} → {r_curr['k']}, Δr_eff = {delta_r:+.3f}, Δr_eff/Δk = {ratio:+.4f}")

output = {"results": results, "c_fit": float(c_fit), "a_fit": float(a_fit), "b_fit": float(b_fit),
          "lin_a": float(lin_a), "lin_b": float(lin_b)}
with open("nn_rank_fit2_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_rank_fit2_result.json")
