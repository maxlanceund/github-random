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
    """用独立训练/测试划分估计 CE，避免过拟合"""
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

def collect_acts(model):
    model.eval()
    all_acts = None
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            acts = model(x, return_all=True)
            if all_acts is None:
                all_acts = [[] for _ in acts]
            for i, a in enumerate(acts):
                all_acts[i].append(a.cpu().numpy())
    return [np.concatenate(a, axis=0) for a in all_acts]

def normalize(feat):
    return (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)

def shuffle_columns(feat, seed=0):
    out = feat.copy()
    rng = np.random.RandomState(seed)
    for j in range(feat.shape[1]):
        out[:, j] = feat[rng.permutation(len(feat)), j]
    return out

def compute_D_f_split(feat, labels):
    """用 70/30 划分估计 CE"""
    fs = normalize(feat)
    ce = estimate_ce_split(fs, labels)
    return ce / log2_10

hidden = [128, 64, 32, 16]
model, acc = train_network(hidden)
print(f"Config: 128-64-32-16")
print(f"Accuracy: {acc*100:.2f}%")
print()

acts = collect_acts(model)
layer_names = ["input(784)", "h1(128)", "h2(64)", "h3(32)", "h4(16)", "output(10)"]

# ============================================================
# 实验 A：真实标签下，intact vs shuffled
# ============================================================
print("=" * 90)
print("实验 A：真实标签")
print("=" * 90)
print(f"{'Layer':>12} | {'D_f(intact)':>12} | {'D_f(shuffled)':>14} | {'corr_contrib':>13} | {'corr_frac':>10}")
print("-" * 90)

results_A = []
for name, a in zip(layer_names, acts):
    D_intact = compute_D_f_split(a, Y)
    D_shuf_list = [compute_D_f_split(shuffle_columns(a, seed=s), Y) for s in range(3)]
    D_shuf = float(np.mean(D_shuf_list))
    corr_contrib = D_shuf - D_intact
    corr_frac = corr_contrib / D_shuf if D_shuf > 0 else 0.0
    print(f"{name:>12} | {D_intact:>12.4f} | {D_shuf:>14.4f} | {corr_contrib:>+13.4f} | {corr_frac:>10.4f}", flush=True)
    results_A.append({
        "layer": name,
        "D_f_intact": float(D_intact),
        "D_f_shuffled_mean": D_shuf,
        "corr_contrib": float(corr_contrib),
        "corr_frac": float(corr_frac),
    })

# ============================================================
# 实验 B：随机标签下，intact vs shuffled（null test）
# ============================================================
print()
print("=" * 90)
print("实验 B：随机标签（null test）")
print("=" * 90)
Y_rand = np.random.RandomState(0).permutation(Y)
print(f"{'Layer':>12} | {'D_f(intact)':>12} | {'D_f(shuffled)':>14} | {'corr_contrib':>13} | {'corr_frac':>10}")
print("-" * 90)

results_B = []
for name, a in zip(layer_names, acts):
    D_intact = compute_D_f_split(a, Y_rand)
    D_shuf_list = [compute_D_f_split(shuffle_columns(a, seed=s), Y_rand) for s in range(3)]
    D_shuf = float(np.mean(D_shuf_list))
    corr_contrib = D_shuf - D_intact
    corr_frac = corr_contrib / D_shuf if D_shuf > 0 else 0.0
    print(f"{name:>12} | {D_intact:>12.4f} | {D_shuf:>14.4f} | {corr_contrib:>+13.4f} | {corr_frac:>10.4f}", flush=True)
    results_B.append({
        "layer": name,
        "D_f_intact": float(D_intact),
        "D_f_shuffled_mean": D_shuf,
        "corr_contrib": float(corr_contrib),
        "corr_frac": float(corr_frac),
    })

# ============================================================
# 对比：真实 corr_frac vs null corr_frac
# ============================================================
print()
print("=" * 70)
print("关键对比：真实 corr_frac vs null corr_frac")
print("=" * 70)
print(f"{'Layer':>12} | {'corr_frac(true)':>16} | {'corr_frac(null)':>16} | {'差值':>10}")
print("-" * 70)
for ra, rb in zip(results_A, results_B):
    diff = ra["corr_frac"] - rb["corr_frac"]
    print(f"{ra['layer']:>12} | {ra['corr_frac']:>16.4f} | {rb['corr_frac']:>16.4f} | {diff:>+10.4f}")

print()
print("解读：")
print("  如果 null 的 corr_frac 也接近真实 corr_frac，说明 corr_frac 是估计偏差，不是信息关联。")
print("  如果 null 的 corr_frac 明显低于真实值，说明真实关联确实存在。")

output = {"acc": acc, "true_label": results_A, "null_label": results_B}
with open("nn_null_test_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_null_test_result.json")
