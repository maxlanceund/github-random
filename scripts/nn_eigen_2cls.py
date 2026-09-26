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

class SimpleDS(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

transform = T.ToTensor()
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def make_dataset(classes):
    label_map = {c: i for i, c in enumerate(classes)}
    train_data = [(mnist_train[i][0], label_map[mnist_train[i][1]]) for i in range(len(mnist_train)) if mnist_train[i][1] in classes][:5000]
    test_data = [(mnist_test[i][0], label_map[mnist_test[i][1]]) for i in range(len(mnist_test)) if mnist_test[i][1] in classes][:1000]
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
    yd = np.asarray(y, dtype=np.int32)
    n_y = yd.max() + 1
    joint = np.zeros((n_bins, n_y))
    for i in range(len(xd)):
        joint[xd[i], yd[i]] += 1
    Hx = entropy_from_counts(np.bincount(xd, minlength=n_bins))
    Hy = entropy_from_counts(np.bincount(yd, minlength=n_y))
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
    h1_list, labels_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            acts = model(x, return_all=True)
            h1_list.append(acts[1].cpu().numpy())
            labels_list.append(y.numpy())
    return np.concatenate(h1_list, axis=0), np.concatenate(labels_list, axis=0)

def analyze(h1, labels, tag):
    print(f"\n{'='*70}")
    print(f"Config: {tag}")
    print(f"{'='*70}")

    M = mi_matrix(h1)
    eigs, vecs = np.linalg.eigh(M)
    order = np.argsort(eigs)[::-1]
    eigs = eigs[order]
    vecs = vecs[:, order]

    # 有效秩
    pos = eigs[eigs > 1e-10]
    p = pos / pos.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    print(f"effective_rank: {eff_rank:.4f}")
    print(f"max_eig:        {eigs[0]:.4f}")
    print(f"gap(1-2):       {eigs[0] - eigs[1]:.4f}")

    # 前 15 个模式的 MI
    print()
    print(f"{'模式':>5} | {'特征值':>10} | {'MI(与类别)':>12}")
    print("-" * 40)
    mode_results = []
    for k in range(15):
        v = vecs[:, k]
        proj = h1 @ v
        mi = mutual_info(proj, labels)
        mode_results.append({"mode": k, "eig": float(eigs[k]), "mi": float(mi)})
        print(f"{k:>5} | {eigs[k]:>10.4f} | {mi:>12.4f}", flush=True)

    # 随机方向对照
    np.random.seed(0)
    random_mis = []
    for _ in range(20):
        v_rand = np.random.randn(h1.shape[1])
        v_rand = v_rand / np.linalg.norm(v_rand)
        proj = h1 @ v_rand
        random_mis.append(mutual_info(proj, labels))
    random_mean = float(np.mean(random_mis))
    random_95 = float(np.percentile(random_mis, 95))

    print(f"\n随机方向 MI 均值: {random_mean:.4f}")
    print(f"随机方向 95% 分位: {random_95:.4f}")

    return {
        "effective_rank": eff_rank,
        "max_eig": float(eigs[0]),
        "max_gap": float(eigs[0] - eigs[1]),
        "modes": mode_results,
        "random_mean": random_mean,
        "random_95": random_95,
    }

hidden = [128, 64, 32, 16]
all_results = {}

# 2 类任务
print("\n" + "#" * 70)
print("# 2 类任务（MNIST 数字 0 vs 1）")
print("#" * 70)
train_data, test_data = make_dataset([0, 1])
train_loader = DataLoader(SimpleDS(train_data), batch_size=128, shuffle=True)
test_loader = DataLoader(SimpleDS(test_data), batch_size=128, shuffle=False)

model_2 = train_network(hidden, 2, train_loader)
# 准确率
model_2.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model_2(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"Accuracy: {correct/total*100:.2f}%")

h1_2, labels_2 = collect_h1(model_2, test_loader)
res_2 = analyze(h1_2, labels_2, "2 classes")
all_results["2 classes"] = res_2

# 10 类任务
print("\n" + "#" * 70)
print("# 10 类任务（完整 MNIST）")
print("#" * 70)
train_data, test_data = make_dataset(list(range(10)))
train_loader = DataLoader(SimpleDS(train_data), batch_size=128, shuffle=True)
test_loader = DataLoader(SimpleDS(test_data), batch_size=128, shuffle=False)

model_10 = train_network(hidden, 10, train_loader)
model_10.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model_10(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"Accuracy: {correct/total*100:.2f}%")

h1_10, labels_10 = collect_h1(model_10, test_loader)
res_10 = analyze(h1_10, labels_10, "10 classes")
all_results["10 classes"] = res_10

# 汇总
print()
print("=" * 75)
print("汇总：2 类 vs 10 类")
print("=" * 75)
print(f"{'任务':>12} | {'有效秩':>8} | {'最大特征值':>10} | {'最大模式 MI':>12} | {'随机 MI 均值':>13}")
print("-" * 75)
for name, r in all_results.items():
    print(f"{name:>12} | {r['effective_rank']:>8.4f} | {r['max_eig']:>10.4f} | {r['modes'][0]['mi']:>12.4f} | {r['random_mean']:>13.4f}")

with open("nn_eigen_2cls_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print()
print("Saved to nn_eigen_2cls_result.json")
