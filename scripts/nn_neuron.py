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

def estimate_ce(features, labels, n_epochs=30):
    X_t = torch.tensor(features, dtype=torch.float32).to(device)
    y_t = torch.tensor(labels, dtype=torch.long).to(device)
    d = X_t.shape[1]
    if d == 0:
        return math.log(10)
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
    if feat.shape[1] == 0:
        return 1.0
    fs = normalize(feat)
    ce = estimate_ce(fs, Y)
    return ce / log2_10

hidden = [128, 64, 32, 16]
model, acc = train_network(hidden)
print(f"Config: 128-64-32-16")
print(f"Accuracy: {acc*100:.2f}%")
print()

h1 = collect_h1(model)
print(f"h1 shape: {h1.shape}")
print()

# 用方差排序神经元（方差大的可能携带更多信息）
variances = h1.var(axis=0)
order = np.argsort(-variances)  # 从大到小

print("按方差排序，逐步加入神经元，看 D_f 如何变化：")
print(f"{'神经元数':>8} | {'D_f':>8}")
print("-" * 25)

cum_indices = []
prev_D = None
records = []
for k in [0, 1, 2, 4, 8, 16, 32, 64, 96, 128]:
    if k == 0:
        D = 1.0
    else:
        idx = order[:k]
        D = compute_D_f(h1[:, idx])
    records.append((k, D))
    print(f"{k:>8} | {D:>8.4f}", flush=True)

print()
print("解读：")
print("  如果 D_f 在前几个神经元就快速下降到很低，说明信息集中在少数神经元。")
print("  如果 D_f 缓慢下降，说明信息分散在所有神经元。")

# 反过来：随机排序，对比
print()
print("对照：随机排序，逐步加入神经元：")
np.random.seed(42)
random_order = np.random.permutation(128)
print(f"{'神经元数':>8} | {'D_f':>8}")
print("-" * 25)
for k in [1, 2, 4, 8, 16, 32, 64, 96, 128]:
    idx = random_order[:k]
    D = compute_D_f(h1[:, idx])
    print(f"{k:>8} | {D:>8.4f}", flush=True)

# 对比：方差排序 vs 随机排序
print()
print("=" * 55)
print(f"{'神经元数':>8} | {'D_f(方差排序)':>15} | {'D_f(随机排序)':>15}")
print("-" * 55)

records_var = []
records_rand = []
for k in [1, 2, 4, 8, 16, 32, 64, 128]:
    d_var = compute_D_f(h1[:, order[:k]])
    d_rand = compute_D_f(h1[:, random_order[:k]])
    records_var.append((k, d_var))
    records_rand.append((k, d_rand))
    print(f"{k:>8} | {d_var:>15.4f} | {d_rand:>15.4f}", flush=True)

output = {
    "acc": acc,
    "n_neurons": int(h1.shape[1]),
    "var_sorted": records_var,
    "rand_sorted": records_rand,
}
with open("nn_neuron_result.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to nn_neuron_result.json")
