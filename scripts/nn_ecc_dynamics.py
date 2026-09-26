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

def evaluate_ecc(h1):
    """返回 (D_0, D_50, normalized_50)"""
    D_0 = compute_D_f(h1)
    D_50 = float(np.mean([compute_D_f(partial_shuffle(h1, 0.5, seed=s)) for s in range(2)]))
    D_100 = float(np.mean([compute_D_f(partial_shuffle(h1, 1.0, seed=s)) for s in range(2)]))
    norm_50 = (D_50 - D_0) / (D_100 - D_0) if D_100 > D_0 else 0
    return D_0, D_50, D_100, norm_50

hidden = [128, 64, 32, 16]
torch.manual_seed(42)
model = FlexNet(hidden).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Config: 128-64-32-16")
print(f"{'Epoch':>6} | {'TrainLoss':>10} | {'TestAcc':>8} | {'D_f(0)':>8} | {'D_f(50)':>8} | {'norm@50':>8}")
print("-" * 70)

history = []
checkpoints = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]

for epoch in range(1, 31):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    if epoch in checkpoints:
        # 测准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total

        # 测纠错曲线
        h1 = collect_h1(model)
        D_0, D_50, D_100, norm_50 = evaluate_ecc(h1)

        print(f"{epoch:>6} | {avg_loss:>10.4f} | {acc*100:>7.2f}% | {D_0:>8.4f} | {D_50:>8.4f} | {norm_50:>8.4f}", flush=True)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "acc": acc,
            "D_f_0": D_0,
            "D_f_50": D_50,
            "D_f_100": D_100,
            "norm_50": norm_50,
        })

# 分析：三种可能的模式
print()
print("=" * 70)
print("训练动力学分析")
print("=" * 70)

if len(history) >= 3:
    norms = [h["norm_50"] for h in history]
    epochs = [h["epoch"] for h in history]
    accs = [h["acc"] for h in history]

    print(f"{'Epoch':>6} | {'norm_50':>10} | {'acc':>8}")
    print("-" * 30)
    for h in history:
        print(f"{h['epoch']:>6} | {h['norm_50']:>10.4f} | {h['acc']*100:>7.2f}%")

    print()
    print("模式判断：")
    early_norm = np.mean(norms[:2])
    late_norm = np.mean(norms[-2:])
    print(f"  早期 norm@50: {early_norm:.4f}")
    print(f"  晚期 norm@50: {late_norm:.4f}")

    if abs(early_norm - late_norm) < 0.05:
        print("  → 可能一：纠错码一开始就存在（架构固有）")
    elif late_norm < early_norm - 0.05:
        print("  → 可能二：纠错码随训练逐渐增强（动力学涌现）")
    elif late_norm > early_norm + 0.05:
        print("  → 可能三：纠错码先形成后减弱（相变）")
    else:
        print("  → 不明确")

with open("nn_ecc_dynamics_result.json", "w") as f:
    json.dump(history, f, indent=2)
print()
print("Saved to nn_ecc_dynamics_result.json")
