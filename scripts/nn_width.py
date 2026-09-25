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

def estimate_ce(features, labels, n_epochs=50):
    X_t = torch.tensor(features, dtype=torch.float32).to(device)
    y_t = torch.tensor(labels, dtype=torch.long).to(device)
    d = X_t.shape[1]
    clf = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        clf.train()
        opt.zero_grad()
        loss = F.cross_entropy(clf(X_t), y_t)
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        return F.cross_entropy(clf(X_t), y_t).item()

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

def collect_labels():
    ys = []
    for _, y in test_loader:
        ys.append(y.numpy())
    return np.concatenate(ys, axis=0)

Y = collect_labels()

# 固定深度（4层隐层），变化第一层宽度
configs = {
    "narrow-16 (16-32-16-8)": [16, 32, 16, 8],
    "narrow-32 (32-32-16-8)": [32, 32, 16, 8],
    "medium-128 (128-32-16-8)": [128, 32, 16, 8],
    "wide-256 (256-32-16-8)": [256, 32, 16, 8],
    "wide-512 (512-32-16-8)": [512, 32, 16, 8],
    "very-wide-1024 (1024-32-16-8)": [1024, 32, 16, 8],
}

all_results = {}

for name, hidden in configs.items():
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")

    model, acc = train_network(hidden)
    print(f"Test accuracy: {acc*100:.2f}%")

    model.eval()
    all_acts = [[] for _ in range(len(hidden) + 2)]
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            acts = model(x, return_all=True)
            for i, a in enumerate(acts):
                all_acts[i].append(a.cpu().numpy())

    layer_dfs = []
    layer_names = ["input"] + [f"h{i+1}({h})" for i, h in enumerate(hidden)] + ["output"]

    print(f"{'Layer':>20} | {'CE':>8} | {'D_f':>8}")
    print("-" * 45)
    for i, acts in enumerate(all_acts):
        feat = np.concatenate(acts, axis=0)
        feat_std = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)
        ce = estimate_ce(feat_std, Y)
        D_f = ce / log2_10
        layer_dfs.append(D_f)
        print(f"{layer_names[i]:>20} | {ce:>8.4f} | {D_f:>8.4f}", flush=True)

    all_results[name] = {"acc": acc, "D_f": layer_dfs, "layers": layer_names}

# 汇总
print()
print("=" * 75)
print("Summary: first-layer width vs accuracy")
print("=" * 75)
print(f"{'Config':>35} | {'h1_D_f':>8} | {'out_D_f':>8} | {'acc':>7}")
print("-" * 75)
for name, r in all_results.items():
    h1_df = r["D_f"][1]
    out_df = r["D_f"][-1]
    print(f"{name:>35} | {h1_df:>8.4f} | {out_df:>8.4f} | {r['acc']*100:>6.2f}%")

with open("nn_width_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to nn_width_result.json")
