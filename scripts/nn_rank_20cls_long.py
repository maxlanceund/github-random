import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

class SimpleDS(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def make_20cls():
    train_data = [(mnist_train[i][0], mnist_train[i][1]) for i in range(5000)]
    test_data = [(mnist_test[i][0], mnist_test[i][1]) for i in range(1000)]
    for i in range(2500):
        train_data.append((fm_train[i][0], fm_train[i][1] + 10))
    for i in range(500):
        test_data.append((fm_test[i][0], fm_test[i][1] + 10))
    return train_data, test_data

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

def collect_h1(model, test_loader):
    model.eval()
    h1_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            acts = model(x, return_all=True)
            h1_list.append(acts[1].cpu().numpy())
    return np.concatenate(h1_list, axis=0)

def measure(model, test_loader):
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
    h1 = collect_h1(model, test_loader)
    M = mi_matrix(h1)
    er = effective_rank(M)
    return acc, er

hidden = [128, 64, 32, 16]
n_classes = 20

train_data, test_data = make_20cls()
train_loader = DataLoader(SimpleDS(train_data), batch_size=128, shuffle=True)
test_loader = DataLoader(SimpleDS(test_data), batch_size=128, shuffle=False)

torch.manual_seed(42)
model = FlexNet(hidden, n_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Config: 128-64-32-16, n_classes=20")
print(f"{'Epoch':>6} | {'Acc':>8} | {'effective_rank':>15}")
print("-" * 40)

history = []
checkpoints = [5, 10, 15, 20, 25, 30]

for epoch in range(1, 31):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    if epoch in checkpoints:
        acc, er = measure(model, test_loader)
        print(f"{epoch:>6} | {acc*100:>7.2f}% | {er:>15.4f}", flush=True)
        history.append({"epoch": epoch, "acc": acc, "effective_rank": er})

print()
print("=" * 50)
print("对比：20 类任务的有效秩随时间的变化")
print("=" * 50)
for h in history:
    print(f"epoch {h['epoch']:>2}: acc={h['acc']*100:.1f}%, eff_rank={h['effective_rank']:.4f}")

print()
print("对比之前的 10 epoch 结果：")
print("  epoch 10 时 effective_rank = 8.73")
print("  如果 30 epoch 时显著更高 → 之前是训练不足")
print("  如果 30 epoch 时仍在 8-9 → 是真实结构")

with open("nn_rank_20cls_long_result.json", "w") as f:
    json.dump(history, f, indent=2)
print()
print("Saved to nn_rank_20cls_long_result.json")
