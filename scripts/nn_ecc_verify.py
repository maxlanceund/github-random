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

def partial_shuffle(feat, fraction, seed=0):
    out = feat.copy()
    n_cols = feat.shape[1]
    n_shuffle = int(fraction * n_cols)
    if n_shuffle == 0:
        return out
    rng = np.random.RandomState(seed)
    cols = rng.choice(n_cols, n_shuffle, replace=False)
    for j in cols:
        out[:, j] = feat[rng.permutation(len(feat)), j]
    return out

# 三种 h1 宽度
configs = {
    "h1=32":  [32, 32, 16],
    "h1=128": [128, 64, 32, 16],
    "h1=512": [512, 128, 64, 32, 16],
}

fractions = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

all_results = {}

for name, hidden in configs.items():
    print(f"\n{'='*70}")
    print(f"Config: {name}  (hidden={hidden})")
    print(f"{'='*70}")

    model, acc = train_network(hidden)
    print(f"Accuracy: {acc*100:.2f}%")

    h1 = collect_h1(model)
    print(f"h1 shape: {h1.shape}")
    print()

    print(f"{'frac':>6} | {'D_f':>8} | {'normalized':>10}")
    print("-" * 35)

    D_0 = compute_D_f(h1)
    D_full = None
    results = []

    for frac in fractions:
        if frac == 0.0:
            D = D_0
        else:
            D_list = [compute_D_f(partial_shuffle(h1, frac, seed=s)) for s in range(2)]
            D = float(np.mean(D_list))

        if frac == 1.0:
            D_full = D

        results.append({"frac": frac, "D_f": D})

    for r in results:
        delta = r["D_f"] - D_0
        norm = delta / (D_full - D_0) if D_full > D_0 else 0
        r["normalized"] = norm
        print(f"{r['frac']:>6.2f} | {r['D_f']:>8.4f} | {norm:>10.4f}", flush=True)

    # 提取半程值：frac=0.5 时的 normalized
    half_norm = next(r["normalized"] for r in results if r["frac"] == 0.5)
    # 提取 0.4 的 normalized
    quarter_norm = next(r["normalized"] for r in results if r["frac"] == 0.4)

    print(f"\n  frac=0.4 时 normalized: {quarter_norm:.4f}")
    print(f"  frac=0.5 时 normalized: {half_norm:.4f}")
    print(f"  线性预测 (frac=0.5):   0.5000")
    print(f"  偏差:                  {half_norm - 0.5:+.4f}")

    all_results[name] = {
        "acc": acc,
        "n_neurons": int(h1.shape[1]),
        "results": results,
        "D_0": float(D_0),
        "D_full": float(D_full),
        "half_norm": float(half_norm),
        "quarter_norm": float(quarter_norm),
    }

# 汇总
print()
print("=" * 85)
print("汇总：不同 h1 宽度下的曲线形状")
print("=" * 85)
print(f"{'Config':>10} | {'h1_dim':>8} | {'acc':>7} | {'D_f(0)':>8} | {'D_f(1)':>8} | {'norm@0.4':>10} | {'norm@0.5':>10}")
print("-" * 85)
for name, r in all_results.items():
    print(f"{name:>10} | {r['n_neurons']:>8} | {r['acc']*100:>6.2f}% | {r['D_0']:>8.4f} | {r['D_full']:>8.4f} | {r['quarter_norm']:>10.4f} | {r['half_norm']:>10.4f}")

print()
print("解读：")
print("  如果所有配置的 norm@0.5 都明显小于 0.5，说明凸曲线是普遍的（纠错码结构）。")
print("  如果只有 h1=128 凸，其他配置线性，说明是局部现象。")

with open("nn_ecc_verify_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print()
print("Saved to nn_ecc_verify_result.json")
