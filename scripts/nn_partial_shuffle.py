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
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    X_train = torch.tensor(features[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(labels[train_idx], dtype=torch.long).to(device)
    X_test = torch.tensor(features[test_idx], dtype=torch.float32).to(device)
    y_test = torch.tensor(labels[test_idx], dtype=torch.long).to(device)
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

def partial_shuffle(feat, fraction, seed=0):
    """打乱 fraction 比例的列"""
    out = feat.copy()
    n_cols = feat.shape[1]
    n_shuffle = int(fraction * n_cols)
    if n_shuffle == 0:
        return out
    rng = np.random.RandomState(seed)
    cols_to_shuffle = rng.choice(n_cols, n_shuffle, replace=False)
    for j in cols_to_shuffle:
        out[:, j] = feat[rng.permutation(len(feat)), j]
    return out

hidden = [128, 64, 32, 16]
model, acc = train_network(hidden)
print(f"Config: 128-64-32-16")
print(f"Accuracy: {acc*100:.2f}%")
print()

h1 = collect_h1(model)
print(f"h1 shape: {h1.shape}")
print()

# 基线：0% 打乱
D_f_0 = compute_D_f(h1)
print(f"基线 D_f (0% shuffled): {D_f_0:.4f}")
print()

# 扫描打乱比例
print(f"{'frac':>8} | {'n_shuffled':>10} | {'D_f':>8} | {'ΔD_f':>8} | {'normalized':>10}")
print("-" * 60)

fractions = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []
n_cols = h1.shape[1]

for frac in fractions:
    if frac == 0.0:
        D = D_f_0
    else:
        D_list = [compute_D_f(partial_shuffle(h1, frac, seed=s)) for s in range(3)]
        D = float(np.mean(D_list))
    delta = D - D_f_0
    norm = delta / (1.0 - D_f_0) if D_f_0 < 1.0 else 0
    n_shuf = int(frac * n_cols)
    print(f"{frac:>8.2f} | {n_shuf:>10} | {D:>8.4f} | {delta:>+8.4f} | {norm:>10.4f}", flush=True)
    results.append({"frac": frac, "n_shuffled": n_shuf, "D_f": D, "delta": delta, "normalized": norm})

print()
print("=" * 60)
print("总结：")
print("=" * 60)

# 检查线性度
fracs_arr = np.array([r["frac"] for r in results])
norm_arr = np.array([r["normalized"] for r in results])
coef = np.polyfit(fracs_arr, norm_arr, 1)
print(f"归一化曲线线性拟合：斜率 = {coef[0]:.4f}, 截距 = {coef[1]:.4f}")

if abs(coef[0] - 1.0) < 0.15:
    print("→ 近似线性：关联均匀分布")
elif coef[0] > 1.15:
    print("→ 上凸：少数关联列携带大部分信息")
else:
    print("→ 下凸：多数关联列共同起作用")

# 检查突变点
diffs = np.diff(norm_arr)
max_diff_idx = np.argmax(diffs)
max_jump = diffs[max_diff_idx]
print(f"最大单步跳变：{max_jump:.4f} 在 frac = {fracs_arr[max_diff_idx+1]:.2f}")

output = {"acc": acc, "D_f_0": float(D_f_0), "results": results, "linear_slope": float(coef[0])}
with open("nn_partial_shuffle_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_partial_shuffle_result.json")
