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
    def __init__(self, hidden_sizes, n_classes):
        super().__init__()
        sizes = [784] + hidden_sizes + [n_classes]
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
train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def filter_classes(dataset, classes):
    indices = [i for i in range(len(dataset)) if dataset[i][1] in classes]
    return Subset(dataset, indices)

def estimate_ce(features, labels, n_classes, n_epochs=50):
    X_t = torch.tensor(features, dtype=torch.float32).to(device)
    y_t = torch.tensor(labels, dtype=torch.long).to(device)
    d = X_t.shape[1]
    clf = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n_classes)).to(device)
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

def train_network(hidden_sizes, n_classes, train_loader, test_loader, n_epochs=10):
    torch.manual_seed(42)
    model = FlexNet(hidden_sizes, n_classes).to(device)
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

def collect_acts(model, test_loader):
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

def collect_labels(test_loader):
    ys = []
    for _, y in test_loader:
        ys.append(y.numpy())
    return np.concatenate(ys, axis=0)

def normalize(feat):
    return (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)

def compute_D_f(feat, labels, n_classes):
    fs = normalize(feat)
    ce = estimate_ce(fs, labels, n_classes)
    return ce / math.log(n_classes)

# 两种类别数
scenarios = {
    "5 classes": [0, 1, 2, 3, 4],
    "10 classes": list(range(10)),
}

# 瓶颈维度
penult_dims = [4, 8, 16, 32]

all_results = {}

for scenario_name, classes in scenarios.items():
    n_classes = len(classes)
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}, n_classes = {n_classes}")
    print(f"{'='*70}")

    train_subset_full = filter_classes(train_full, classes)
    test_subset_full = filter_classes(test_full, classes)

    # 重映射标签
    label_map = {c: i for i, c in enumerate(classes)}
    train_data = [(train_full[i][0], label_map[train_full[i][1]]) for i in range(len(train_full)) if train_full[i][1] in classes]
    test_data = [(test_full[i][0], label_map[test_full[i][1]]) for i in range(len(test_full)) if test_full[i][1] in classes]

    train_data = train_data[:5000]
    test_data = test_data[:1000]

    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_loader = DataLoader(SimpleDS(train_data), batch_size=128, shuffle=True)
    test_loader = DataLoader(SimpleDS(test_data), batch_size=128, shuffle=False)

    scenario_results = {}

    for pdim in penult_dims:
        hidden = [64, 32, pdim]
        model, acc = train_network(hidden, n_classes, train_loader, test_loader)
        acts = collect_acts(model, test_loader)
        labels = collect_labels(test_loader)

        out = acts[-1]
        penult = acts[-2]
        combined = np.concatenate([out, penult], axis=1)

        D_f_out = compute_D_f(out, labels, n_classes)
        D_f_penult = compute_D_f(penult, labels, n_classes)
        D_f_combined = compute_D_f(combined, labels, n_classes)

        ratio = n_classes / pdim
        gain = D_f_out - D_f_combined

        print(f"  penult={pdim:>2} ({pdim/n_classes:.2f}x classes): acc={acc*100:>5.1f}%, D_f_out={D_f_out:.4f}, D_f_combined={D_f_combined:.4f}, gain={gain:+.4f}", flush=True)

        scenario_results[pdim] = {
            "acc": acc,
            "D_f_out": D_f_out,
            "D_f_penult": D_f_penult,
            "D_f_combined": D_f_combined,
            "gain": gain,
            "ratio": ratio,
        }

    all_results[scenario_name] = scenario_results

print()
print("=" * 95)
print("Summary: 迁移阈值是绝对维度还是成比例？")
print("=" * 95)
print(f"{'Scenario':>14} | {'penult':>6} | {'ratio':>6} | {'acc':>7} | {'D_f_out':>8} | {'D_f_comb':>9} | {'gain':>8}")
print("-" * 95)
for sname, sr in all_results.items():
    for pdim, r in sr.items():
        print(f"{sname:>14} | {pdim:>6} | {r['ratio']:>6.2f} | {r['acc']*100:>6.1f}% | {r['D_f_out']:>8.4f} | {r['D_f_combined']:>9.4f} | {r['gain']:>+8.4f}")

print()
print("解读：")
print("  如果 5 classes 在 penult=8 (1.6x) 就出现明显 gain，而 10 classes 需要 penult=16 (1.6x)，")
print("  则阈值为比例型。")
print("  如果两者都在相同的绝对维度（如 16）出现 gain，则为绝对型。")

with open("nn_threshold_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to nn_threshold_result.json")
