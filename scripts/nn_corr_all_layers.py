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

def shuffle_columns(feat, seed=0):
    out = feat.copy()
    rng = np.random.RandomState(seed)
    for j in range(feat.shape[1]):
        out[:, j] = feat[rng.permutation(len(feat)), j]
    return out

hidden = [128, 64, 32, 16]
model, acc = train_network(hidden)
print(f"Config: 128-64-32-16")
print(f"Accuracy: {acc*100:.2f}%")
print()

acts = collect_acts(model)
layer_names = ["input(784)", "h1(128)", "h2(64)", "h3(32)", "h4(16)", "output(10)"]

print(f"{'Layer':>12} | {'dim':>5} | {'D_f(intact)':>12} | {'D_f(shuffled)':>14} | {'corr_contrib':>13} | {'corr_frac':>10}")
print("-" * 90)

results = []
for name, a in zip(layer_names, acts):
    D_intact = compute_D_f(a)
    D_shuf_list = []
    for seed in range(5):
        D_s = compute_D_f(shuffle_columns(a, seed=seed))
        D_shuf_list.append(D_s)
    D_shuf = float(np.mean(D_shuf_list))
    corr_contrib = D_shuf - D_intact
    corr_frac = corr_contrib / D_shuf if D_shuf > 0 else 0.0
    print(f"{name:>12} | {a.shape[1]:>5} | {D_intact:>12.4f} | {D_shuf:>14.4f} | {corr_contrib:>+13.4f} | {corr_frac:>10.4f}", flush=True)
    results.append({
        "layer": name,
        "dim": int(a.shape[1]),
        "D_f_intact": float(D_intact),
        "D_f_shuffled_mean": D_shuf,
        "D_f_shuffled_list": [float(x) for x in D_shuf_list],
        "corr_contrib": float(corr_contrib),
        "corr_frac": float(corr_frac),
    })

print()
print("解读：")
print("  corr_contrib = D_f(shuffled) - D_f(intact)：破坏关联后 D_f 升了多少。")
print("  corr_frac    = corr_contrib / D_f(shuffled)：关联贡献占总信息的比例。")
print("  如果所有层的 corr_frac 都接近 0.85，说明'信息在关联里'是普遍规律。")
print("  如果只有 h1 高，说明是局部现象。")

output = {"acc": acc, "results": results}
with open("nn_corr_all_layers_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_corr_all_layers_result.json")
