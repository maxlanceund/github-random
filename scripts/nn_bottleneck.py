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

# 测试：倒数第二层比输出层还窄
# 输出层10维，倒数第二层分别 2, 4, 8, 16 维
configs = {
    "penult=2":  [64, 32, 2],
    "penult=4":  [64, 32, 4],
    "penult=8":  [64, 32, 8],
    "penult=16": [64, 32, 16],
    "penult=32": [64, 32, 32],
}

all_results = {}

for name, hidden in configs.items():
    print(f"\n{'='*65}")
    print(f"Config: {name}")
    print(f"{'='*65}")

    model, acc = train_network(hidden)
    acts = collect_acts(model)

    out = acts[-1]      # 输出层 10 维
    penult = acts[-2]   # 倒数第二层，维度可变
    combined = np.concatenate([out, penult], axis=1)

    D_f_out = compute_D_f(out)
    D_f_penult = compute_D_f(penult)
    D_f_combined = compute_D_f(combined)

    print(f"  penult dim:    {penult.shape[1]}")
    print(f"  accuracy:      {acc*100:.2f}%")
    print(f"  D_f(out):      {D_f_out:.4f}")
    print(f"  D_f(penult):   {D_f_penult:.4f}")
    print(f"  D_f(combined): {D_f_combined:.4f}", flush=True)

    all_results[name] = {
        "acc": acc,
        "penult_dim": int(penult.shape[1]),
        "D_f_out": D_f_out,
        "D_f_penult": D_f_penult,
        "D_f_combined": D_f_combined,
    }

print()
print("=" * 85)
print("Summary: 瓶颈比输出层窄时，信息能迁移吗？")
print("=" * 85)
print(f"{'Config':>15} | {'penult_dim':>10} | {'acc':>7} | {'D_f_out':>8} | {'D_f_penult':>10} | {'D_f_combined':>12}")
print("-" * 90)
for name, r in all_results.items():
    print(f"{name:>15} | {r['penult_dim']:>10} | {r['acc']*100:>6.2f}% | {r['D_f_out']:>8.4f} | {r['D_f_penult']:>10.4f} | {r['D_f_combined']:>12.4f}")

print()
print("解读：")
print("  如果 D_f_combined < D_f_out，说明信息迁移到了 penult。")
print("  如果 D_f_penult < D_f_out，说明 penult 本身比 out 保留更多信息。")
print("  如果 penult 很窄（2维）时 D_f_combined 仍然明显低于 D_f_out，说明迁移机制依然有效。")

with open("nn_bottleneck_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to nn_bottleneck_result.json")
