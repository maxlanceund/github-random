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

def estimate_ce_split(features, labels, n_epochs=30):
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

def eval_ecc(h1, fractions=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]):
    D_0 = compute_D_f(h1)
    results = []
    D_full = None
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

    return D_0, D_full, results

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

hidden = [128, 64, 32, 16]

# ============================================================
# 对照组 1：随机初始化（不训练）
# ============================================================
print("=" * 70)
print("对照组 1：随机初始化（不训练）")
print("=" * 70)
torch.manual_seed(0)
model_random = FlexNet(hidden).to(device)
h1_random = collect_h1(model_random)
D0_r, Df_r, res_r = eval_ecc(h1_random)

print(f"{'frac':>6} | {'D_f':>8} | {'normalized':>10}")
print("-" * 35)
for r in res_r:
    print(f"{r['frac']:>6.2f} | {r['D_f']:>8.4f} | {r['normalized']:>10.4f}", flush=True)

# ============================================================
# 对照组 2：训练 10 epoch
# ============================================================
print()
print("=" * 70)
print("对照组 2：训练 10 epoch")
print("=" * 70)
model_trained = train_network(hidden, n_epochs=10)

# 准确率
model_trained.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model_trained(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
acc = correct / total
print(f"Accuracy: {acc*100:.2f}%")

h1_trained = collect_h1(model_trained)
D0_t, Df_t, res_t = eval_ecc(h1_trained)

print(f"{'frac':>6} | {'D_f':>8} | {'normalized':>10}")
print("-" * 35)
for r in res_t:
    print(f"{r['frac']:>6.2f} | {r['D_f']:>8.4f} | {r['normalized']:>10.4f}", flush=True)

# ============================================================
# 对比
# ============================================================
print()
print("=" * 70)
print("对比：随机 vs 训练")
print("=" * 70)
print(f"{'frac':>6} | {'norm(random)':>14} | {'norm(trained)':>14} | {'差值':>10}")
print("-" * 60)

for i in range(len(res_r)):
    fr = res_r[i]["frac"]
    nr = res_r[i]["normalized"]
    nt = res_t[i]["normalized"]
    print(f"{fr:>6.2f} | {nr:>14.4f} | {nt:>14.4f} | {nt-nr:>+10.4f}")

# 关键：norm@0.5
norm_r_50 = next(r["normalized"] for r in res_r if r["frac"] == 0.5)
norm_t_50 = next(r["normalized"] for r in res_t if r["frac"] == 0.5)

print()
print(f"norm@0.5 (随机): {norm_r_50:.4f}")
print(f"norm@0.5 (训练): {norm_t_50:.4f}")
print(f"差值:            {norm_t_50 - norm_r_50:+.4f}")

if abs(norm_r_50 - norm_t_50) < 0.05:
    print("→ 纠错码在随机初始化时就存在，与训练无关")
elif norm_t_50 < norm_r_50 - 0.05:
    print("→ 训练增强了纠错码")
else:
    print("→ 训练削弱了纠错码")

output = {
    "random": {"D_0": float(D0_r), "D_full": float(Df_r), "results": res_r, "norm_50": float(norm_r_50)},
    "trained": {"D_0": float(D0_t), "D_full": float(Df_t), "results": res_t, "norm_50": float(norm_t_50), "acc": acc},
}
with open("nn_random_ecc_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_random_ecc_result.json")
