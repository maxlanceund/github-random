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

def compute_D_f(feat):
    fs = normalize(feat)
    ce = estimate_ce(fs, Y)
    return ce / log2_10

# 深层网络：784 -> 128 -> 64 -> 32 -> 16 -> 10
hidden = [128, 64, 32, 16]
model, acc = train_network(hidden)
print(f"Config: 128-64-32-16")
print(f"Accuracy: {acc*100:.2f}%")
print()

acts = collect_acts(model)
layer_names = ["input(784)", "h1(128)", "h2(64)", "h3(32)", "h4(16)", "output(10)"]

# 单层 D_f
print(f"{'Layer':>15} | {'dim':>5} | {'D_f(单层)':>10}")
print("-" * 40)
single_D = []
for name, a in zip(layer_names, acts):
    D = compute_D_f(a)
    single_D.append(D)
    print(f"{name:>15} | {a.shape[1]:>5} | {D:>10.4f}", flush=True)

# 累积 D_f
print()
print(f"{'累积到':>15} | {'累积维度':>10} | {'D_f(累积)':>10} | {'边际增益':>10}")
print("-" * 55)
cum = acts[0]
prev_D = compute_D_f(cum)
print(f"{layer_names[0]:>15} | {cum.shape[1]:>10} | {prev_D:>10.4f} | {'—':>10}")

for i in range(1, len(acts)):
    cum = np.concatenate([cum, acts[i]], axis=1)
    D = compute_D_f(cum)
    gain = prev_D - D
    print(f"{layer_names[i]:>15} | {cum.shape[1]:>10} | {D:>10.4f} | {gain:>+10.4f}", flush=True)
    prev_D = D

# 每一层单独 vs 去掉某一层
print()
print("去掉某一层的联合 D_f：")
print(f"{'去掉':>15} | {'D_f(其余层联合)':>15}")
print("-" * 40)
for skip_idx in range(1, len(acts) - 1):
    combined = np.concatenate([acts[i] for i in range(len(acts)) if i != skip_idx], axis=1)
    D = compute_D_f(combined)
    print(f"{layer_names[skip_idx]:>15} | {D:>15.4f}", flush=True)

all_combined = np.concatenate(acts[1:-1], axis=1)
D_all = compute_D_f(all_combined)
print(f"{'(全部隐藏层)':>15} | {D_all:>15.4f}")

output = {
    "acc": acc,
    "layer_names": layer_names,
    "single_D": [float(d) for d in single_D],
    "D_all_hidden": float(D_all),
}
with open("nn_multilayer_result.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to nn_multilayer_result.json")
