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
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
fm_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fm_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

def make_dataset_a(n_classes):
    """只用 MNIST 的 n_classes 个类别"""
    classes = list(range(n_classes))
    label_map = {c: i for i, c in enumerate(classes)}
    train_data = [(mnist_train[i][0], label_map[mnist_train[i][1]]) for i in range(len(mnist_train)) if mnist_train[i][1] in classes][:5000]
    test_data = [(mnist_test[i][0], label_map[mnist_test[i][1]]) for i in range(len(mnist_test)) if mnist_test[i][1] in classes][:1000]
    return train_data, test_data

def make_dataset_b():
    """MNIST + FashionMNIST，共 20 类"""
    train_data = [(mnist_train[i][0], mnist_train[i][1]) for i in range(5000)]
    test_data = [(mnist_test[i][0], mnist_test[i][1]) for i in range(1000)]
    for i in range(2500):
        train_data.append((fm_train[i][0], fm_train[i][1] + 10))
    for i in range(500):
        test_data.append((fm_test[i][0], fm_test[i][1] + 10))
    return train_data, test_data

class SimpleDS(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def train_network(hidden, n_classes, train_loader, n_epochs=10):
    torch.manual_seed(42)
    model = FlexNet(hidden, n_classes).to(device)
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

def collect_h1(model, test_loader):
    model.eval()
    h1_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            acts = model(x, return_all=True)
            h1_list.append(acts[1].cpu().numpy())
    return np.concatenate(h1_list, axis=0)

def discretize(x, n_bins=10):
    if x.std() < 1e-8:
        return np.zeros_like(x, dtype=np.int32)
    bins = np.linspace(x.min() - 1e-8, x.max() + 1e-8, n_bins + 1)
    return np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)

def entropy_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def mutual_info(x, y, n_bins=10):
    xd = discretize(x, n_bins)
    yd = discretize(y, n_bins)
    joint = np.zeros((n_bins, n_bins))
    for i in range(len(xd)):
        joint[xd[i], yd[i]] += 1
    Hx = entropy_from_counts(np.bincount(xd, minlength=n_bins))
    Hy = entropy_from_counts(np.bincount(yd, minlength=n_bins))
    Hxy = entropy_from_counts(joint.flatten())
    return max(0.0, Hx + Hy - Hxy)

def mi_matrix(h1):
    n = h1.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            mi = mutual_info(h1[:, i], h1[:, j])
            M[i, j] = mi
            M[j, i] = mi
    return M

def effective_rank(M):
    eigs = np.linalg.eigvalsh(M)
    pos = eigs[eigs > 1e-10]
    if len(pos) == 0:
        return 0.0
    p = pos / pos.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

hidden = [128, 64, 32, 16]

scenarios = [
    ("5 classes", "mnist", 5),
    ("10 classes", "mnist", 10),
    ("20 classes", "mixed", 20),
]

all_results = {}

for name, source, n_classes in scenarios:
    print(f"\n{'='*65}")
    print(f"Scenario: {name}")
    print(f"{'='*65}")

    if source == "mnist":
        train_data, test_data = make_dataset_a(n_classes)
    else:
        train_data, test_data = make_dataset_b()

    train_loader = DataLoader(SimpleDS(train_data), batch_size=128, shuffle=True)
    test_loader = DataLoader(SimpleDS(test_data), batch_size=128, shuffle=False)

    model = train_network(hidden, n_classes, train_loader)

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
    print(f"Accuracy: {acc*100:.2f}%")

    h1 = collect_h1(model, test_loader)
    M = mi_matrix(h1)
    eff_rank = effective_rank(M)

    # 最大特征值和间隙
    eigs = np.sort(np.linalg.eigvalsh(M))[::-1]
    max_gap = float(eigs[0] - eigs[1])

    print(f"n_classes:      {n_classes}")
    print(f"effective_rank: {eff_rank:.4f}")
    print(f"max_eig:        {eigs[0]:.4f}")
    print(f"gap(1-2):       {max_gap:.4f}")

    all_results[name] = {
        "n_classes": n_classes,
        "acc": float(acc),
        "effective_rank": eff_rank,
        "max_eig": float(eigs[0]),
        "max_gap": max_gap,
    }

print()
print("=" * 75)
print("核心对比：有效秩 vs 类别数")
print("=" * 75)
print(f"{'Scenario':>15} | {'n_classes':>10} | {'effective_rank':>15} | {'ratio':>8}")
print("-" * 75)
for name, r in all_results.items():
    ratio = r["effective_rank"] / r["n_classes"]
    print(f"{name:>15} | {r['n_classes']:>10} | {r['effective_rank']:>15.4f} | {ratio:>8.4f}")

print()
print("解读：")
print("  如果 effective_rank / n_classes ≈ 常数，说明有效秩是任务驱动的。")
print("  如果 effective_rank 固定不随 n_classes 变化，说明有效秩是架构驱动的。")

with open("nn_rank_vs_classes_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print()
print("Saved to nn_rank_vs_classes_result.json")
