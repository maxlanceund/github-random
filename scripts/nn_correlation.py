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
    clf = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
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
    fs = normalize(feat)
    ce = estimate_ce(fs, Y)
    return ce / log2_10

hidden = [128, 64, 32, 16]
model, acc = train_network(hidden)
print(f"Config: 128-64-32-16")
print(f"Accuracy: {acc*100:.2f}%")
print()

h1 = collect_h1(model)
n = h1.shape[1]
print(f"h1 shape: {h1.shape}")
print()

# 基线
D_f_full = compute_D_f(h1)
print(f"D_f(全部 {n} 个神经元): {D_f_full:.4f}")
print()

# 实验一：单神经元平均
print("实验一：单神经元平均 D_f")
print(f"{'神经元':>8} | {'D_f':>8}")
print("-" * 25)
single_Ds = []
sample_neurons = [0, 1, 2, 4, 8, 16, 32, 64, 127]
for i in sample_neurons:
    D = compute_D_f(h1[:, [i]])
    single_Ds.append(D)
    print(f"neuron {i:>3} | {D:>8.4f}", flush=True)

# 实验二：破坏关联（每列独立打乱）
print()
print("实验二：破坏关联（每列独立打乱行顺序）")
print(f"{'打乱次数':>8} | {'D_f':>8}")
print("-" * 25)

shuffle_Ds = []
for seed in range(5):
    h1_shuffled = h1.copy()
    rng = np.random.RandomState(seed)
    for j in range(n):
        h1_shuffled[:, j] = h1[rng.permutation(len(h1)), j]
    D = compute_D_f(h1_shuffled)
    shuffle_Ds.append(D)
    print(f"  trial {seed} | {D:>8.4f}", flush=True)

D_shuffled_mean = np.mean(shuffle_Ds)

# 实验三：部分神经元 vs 全体，破坏关联后
print()
print("实验三：前 64 个神经元，破坏关联前后对比")
print("-" * 40)

subset = h1[:, :64]
D_subset_intact = compute_D_f(subset)

subset_shuffled = subset.copy()
rng = np.random.RandomState(0)
for j in range(subset.shape[1]):
    subset_shuffled[:, j] = subset[rng.permutation(len(subset)), j]
D_subset_shuffled = compute_D_f(subset_shuffled)

print(f"  64 个神经元，保留关联: {D_subset_intact:.4f}")
print(f"  64 个神经元，破坏关联: {D_subset_shuffled:.4f}")
print(f"  差异:                  {D_subset_shuffled - D_subset_intact:+.4f}")

# 汇总
print()
print("=" * 60)
print("汇总：")
print("=" * 60)
print(f"  全部神经元，保留关联:   {D_f_full:.4f}")
print(f"  全部神经元，破坏关联:   {D_shuffled_mean:.4f}")
print(f"  关联贡献:               {D_shuffled_mean - D_f_full:+.4f}")
print()
print(f"  单神经元平均 D_f:       {np.mean(single_Ds):.4f}")

output = {
    "acc": acc,
    "n_neurons": int(n),
    "D_f_intact": float(D_f_full),
    "D_f_shuffled_mean": float(D_shuffled_mean),
    "D_f_shuffled_list": [float(d) for d in shuffle_Ds],
    "correlation_contribution": float(D_shuffled_mean - D_f_full),
    "single_neuron_mean": float(np.mean(single_Ds)),
    "subset_intact": float(D_subset_intact),
    "subset_shuffled": float(D_subset_shuffled),
}
with open("nn_correlation_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_correlation_result.json")

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
    ce = estimate_ce(fs, Y)
    return ce / log2_10

hidden = [128, 64, 32, 16]
model, acc = train_network(hidden)
print(f"Config: 128-64-32-16")
print(f"Accuracy: {acc*100:.2f}%")
print()

h1 = collect_h1(model)
n = h1.shape[1]
print(f"h1 shape: {h1.shape}")
print()

# 基线
D_f_full = compute_D_f(h1)
print(f"D_f(全部 {n} 个神经元): {D_f_full:.4f}")
print()

# 实验一：单神经元平均
print("实验一：单神经元平均 D_f")
print(f"{'神经元':>8} | {'D_f':>8}")
print("-" * 25)
single_Ds = []
sample_neurons = [0, 1, 2, 4, 8, 16, 32, 64, 127]
for i in sample_neurons:
    D = compute_D_f(h1[:, [i]])
    single_Ds.append(D)
    print(f"neuron {i:>3} | {D:>8.4f}", flush=True)

# 实验二：破坏关联（每列独立打乱）
print()
print("实验二：破坏关联（每列独立打乱行顺序）")
print(f"{'打乱次数':>8} | {'D_f':>8}")
print("-" * 25)

shuffle_Ds = []
for seed in range(5):
    h1_shuffled = h1.copy()
    rng = np.random.RandomState(seed)
    for j in range(n):
        h1_shuffled[:, j] = h1[rng.permutation(len(h1)), j]
    D = compute_D_f(h1_shuffled)
    shuffle_Ds.append(D)
    print(f"  trial {seed} | {D:>8.4f}", flush=True)

D_shuffled_mean = np.mean(shuffle_Ds)

# 实验三：部分神经元 vs 全体，破坏关联后
print()
print("实验三：前 64 个神经元，破坏关联前后对比")
print("-" * 40)

subset = h1[:, :64]
D_subset_intact = compute_D_f(subset)

subset_shuffled = subset.copy()
rng = np.random.RandomState(0)
for j in range(subset.shape[1]):
    subset_shuffled[:, j] = subset[rng.permutation(len(subset)), j]
D_subset_shuffled = compute_D_f(subset_shuffled)

print(f"  64 个神经元，保留关联: {D_subset_intact:.4f}")
print(f"  64 个神经元，破坏关联: {D_subset_shuffled:.4f}")
print(f"  差异:                  {D_subset_shuffled - D_subset_intact:+.4f}")

# 汇总
print()
print("=" * 60)
print("汇总：")
print("=" * 60)
print(f"  全部神经元，保留关联:   {D_f_full:.4f}")
print(f"  全部神经元，破坏关联:   {D_shuffled_mean:.4f}")
print(f"  关联贡献:               {D_shuffled_mean - D_f_full:+.4f}")
print()
print(f"  单神经元平均 D_f:       {np.mean(single_Ds):.4f}")

output = {
    "acc": acc,
    "n_neurons": int(n),
    "D_f_intact": float(D_f_full),
    "D_f_shuffled_mean": float(D_shuffled_mean),
    "D_f_shuffled_list": [float(d) for d in shuffle_Ds],
    "correlation_contribution": float(D_shuffled_mean - D_f_full),
    "single_neuron_mean": float(np.mean(single_Ds)),
    "subset_intact": float(D_subset_intact),
    "subset_shuffled": float(D_subset_shuffled),
}
with open("nn_correlation_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_correlation_result.json")
