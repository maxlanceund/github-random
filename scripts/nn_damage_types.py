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

def estimate_ce_split(features, labels, n_epochs=50):
    n = len(features)
    n_train = int(0.7 * n)
    idx = np.random.RandomState(0).permutation(n)
    X_train = torch.tensor(features[idx[:n_train]], dtype=torch.float32).to(device)
    y_train = torch.tensor(labels[idx[:n_train]], dtype=torch.long).to(device)
    X_test = torch.tensor(features[idx[n_train:]], dtype=torch.float32).to(device)
    y_test = torch.tensor(labels[idx[n_train:]], dtype=torch.long).to(device)
    d = X_train.shape[1]
    clf = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        clf.train()
        opt.zero_grad()
        loss = F.cross_entropy(clf(X_train), y_train)
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        return F.cross_entropy(clf(X_test), y_test).item()

def collect_labels():
    ys = []
    for _, y in test_loader:
        ys.append(y.numpy())
    return np.concatenate(ys, axis=0)

Y = collect_labels()
log2_10 = math.log(10)

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
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return model, correct / total

def collect_h1(model):
    model.eval()
    h1_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            acts = model(x, return_all=True)
            h1_list.append(acts[1].cpu().numpy())
    return np.concatenate(h1_list, axis=0)

def normalize(feat):
    return (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)

def compute_D_f(feat):
    fs = normalize(feat)
    ce = estimate_ce_split(fs, Y)
    return ce / log2_10

def damage_shuffle(feat, fraction, seed=0):
    """打乱：随机选 fraction 的列，重排每列的行顺序"""
    out = feat.copy()
    n_cols = feat.shape[1]
    n_dmg = int(fraction * n_cols)
    if n_dmg == 0:
        return out
    rng = np.random.RandomState(seed)
    cols = rng.choice(n_cols, n_dmg, replace=False)
    for j in cols:
        out[:, j] = feat[rng.permutation(len(feat)), j]
    return out

def damage_erase(feat, fraction, seed=0):
    """擦除：随机选 fraction 的列，全部置零"""
    out = feat.copy()
    n_cols = feat.shape[1]
    n_dmg = int(fraction * n_cols)
    if n_dmg == 0:
        return out
    rng = np.random.RandomState(seed)
    cols = rng.choice(n_cols, n_dmg, replace=False)
    for j in cols:
        out[:, j] = 0.0
    return out

def damage_noise(feat, fraction, seed=0):
    """噪声：随机选 fraction 的列，加高斯噪声（强度=该列标准差）"""
    out = feat.copy()
    n_cols = feat.shape[1]
    n_dmg = int(fraction * n_cols)
    if n_dmg == 0:
        return out
    rng = np.random.RandomState(seed)
    cols = rng.choice(n_cols, n_dmg, replace=False)
    for j in cols:
        sigma = feat[:, j].std()
        out[:, j] = feat[:, j] + rng.randn(len(feat)) * sigma
    return out

hidden = [128, 64, 32, 16]
model, acc = train_network(hidden)
print(f"Config: 128-64-32-16")
print(f"Accuracy: {acc*100:.2f}%")
print()

h1 = collect_h1(model)
print(f"h1 shape: {h1.shape}")
print()

D_0 = compute_D_f(h1)
print(f"基线 D_f (无损坏): {D_0:.4f}")
print()

fractions = [0.1, 0.2, 0.4, 0.6, 0.8]

print(f"{'frac':>6} | {'打乱':>10} | {'擦除':>10} | {'噪声':>10}")
print("-" * 50)

all_results = {"baseline": D_0, "fractions": fractions, "shuffle": [], "erase": [], "noise": []}

for frac in fractions:
    D_shuf = float(np.mean([compute_D_f(damage_shuffle(h1, frac, seed=s)) for s in range(3)]))
    D_erase = float(np.mean([compute_D_f(damage_erase(h1, frac, seed=s)) for s in range(3)]))
    D_noise = float(np.mean([compute_D_f(damage_noise(h1, frac, seed=s)) for s in range(3)]))

    print(f"{frac:>6.2f} | {D_shuf:>10.4f} | {D_erase:>10.4f} | {D_noise:>10.4f}", flush=True)

    all_results["shuffle"].append(D_shuf)
    all_results["erase"].append(D_erase)
    all_results["noise"].append(D_noise)

# 归一化（相对于 baseline 到 1.0）
print()
print("=" * 70)
print("归一化（0 = 基线，1 = 完全损坏）")
print("=" * 70)
print(f"{'frac':>6} | {'打乱':>10} | {'擦除':>10} | {'噪声':>10}")
print("-" * 50)

norm_shuf = [(d - D_0) / (1.0 - D_0) for d in all_results["shuffle"]]
norm_erase = [(d - D_0) / (1.0 - D_0) for d in all_results["erase"]]
norm_noise = [(d - D_0) / (1.0 - D_0) for d in all_results["noise"]]

all_results["norm_shuffle"] = norm_shuf
all_results["norm_erase"] = norm_erase
all_results["norm_noise"] = norm_noise

for i, frac in enumerate(fractions):
    print(f"{frac:>6.2f} | {norm_shuf[i]:>10.4f} | {norm_erase[i]:>10.4f} | {norm_noise[i]:>10.4f}")

print()
print("解读：")
print("  如果 打乱 的 D_f 远低于 擦除 和 噪声，说明网络能容忍重排但不能容忍信息丢失。")
print("  如果三者接近，说明网络对损坏类型不敏感，损伤是'总量'问题。")
print("  如果 擦除 最严重，说明神经元本身携带信息。")
print("  如果 噪声 最严重，说明网络对连续性敏感。")

with open("nn_damage_types_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print()
print("Saved to nn_damage_types_result.json")
