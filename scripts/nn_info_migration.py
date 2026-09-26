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

def collect_all(model):
    model.eval()
    all_acts = None
    all_out = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            acts = model(x, return_all=True)
            if all_acts is None:
                all_acts = [[] for _ in acts]
            for i, a in enumerate(acts):
                all_acts[i].append(a.cpu().numpy())
            all_out.append(acts[-1].cpu().numpy())

    all_acts = [np.concatenate(a, axis=0) for a in all_acts]
    out = all_acts[-1]
    penult = all_acts[-2]

    # 计算 margin
    out_t = torch.tensor(out)
    y_t = torch.tensor(Y)
    correct_logits = out_t.gather(1, y_t.unsqueeze(1)).squeeze(1)
    masked = out_t.clone()
    masked.scatter_(1, y_t.unsqueeze(1), -1e9)
    max_wrong = masked.max(dim=1).values
    margins = (correct_logits - max_wrong).numpy().reshape(-1, 1)

    return out, penult, margins

def normalize(feat):
    return (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)

def compute_D_f(feat):
    fs = normalize(feat)
    ce = estimate_ce(fs, Y)
    return ce / log2_10

configs = {
    "A (64-32-10)": [64, 32],
    "B (64-32-128-10)": [64, 32, 128],
    "C (64-32-512-10)": [64, 32, 512],
}

all_results = {}

for name, hidden in configs.items():
    print(f"\n{'='*65}")
    print(f"Config: {name}")
    print(f"{'='*65}")

    model, acc = train_network(hidden)
    out, penult, margins = collect_all(model)

    # 四种宏观描述
    D_f_out = compute_D_f(out)
    D_f_out_margin = compute_D_f(np.concatenate([out, margins], axis=1))
    D_f_out_penult = compute_D_f(np.concatenate([out, penult], axis=1))
    D_f_penult = compute_D_f(penult)

    print(f"  accuracy: {acc*100:.2f}%")
    print(f"  D_f(output only):           {D_f_out:.4f}")
    print(f"  D_f(output + margin):       {D_f_out_margin:.4f}")
    print(f"  D_f(output + penult):       {D_f_out_penult:.4f}")
    print(f"  D_f(penult only):           {D_f_penult:.4f}", flush=True)

    all_results[name] = {
        "acc": acc,
        "D_f_out": D_f_out,
        "D_f_out_margin": D_f_out_margin,
        "D_f_out_penult": D_f_out_penult,
        "D_f_penult": D_f_penult,
    }

print()
print("=" * 85)
print("Summary: 信息是否迁移？")
print("=" * 85)
print(f"{'Config':>22} | {'out':>7} | {'out+margin':>11} | {'out+penult':>11} | {'penult':>7}")
print("-" * 85)
for name, r in all_results.items():
    print(f"{name:>22} | {r['D_f_out']:>7.4f} | {r['D_f_out_margin']:>11.4f} | {r['D_f_out_penult']:>11.4f} | {r['D_f_penult']:>7.4f}")

print()
print("解读：")
print("  如果 out+margin 或 out+penult 的 D_f 明显低于 out，说明信息迁移了。")
print("  如果三者一样，说明信息真的丢了。")

with open("nn_info_migration_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to nn_info_migration_result.json")
