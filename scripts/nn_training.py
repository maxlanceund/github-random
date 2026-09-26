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
    def forward(self, x):
        x = x.view(-1, 784)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
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

def evaluate(model):
    model.eval()
    all_out = []
    correct = 0
    total = 0
    margins = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_out.append(logits.cpu().numpy())
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            correct_logits = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, y.unsqueeze(1), -1e9)
            max_wrong = masked.max(dim=1).values
            margins.append((correct_logits - max_wrong).cpu().numpy())

    out_np = np.concatenate(all_out, axis=0)
    margin_np = np.concatenate(margins, axis=0)
    acc = correct / total

    feat_std = (out_np - out_np.mean(axis=0)) / (out_np.std(axis=0) + 1e-8)
    ce = estimate_ce(feat_std, Y)
    D_f = ce / log2_10

    return acc, D_f, float(margin_np.mean())

# 使用固定配置
hidden = [256, 128, 64, 32, 16]

torch.manual_seed(42)
model = FlexNet(hidden).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Config: {hidden}")
print()
print(f"{'Epoch':>6} | {'TrainLoss':>10} | {'TestAcc':>8} | {'D_f':>8} | {'Margin':>8}")
print("-" * 60)

history = []

for epoch in range(1, 41):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    if epoch % 2 == 0 or epoch <= 5:
        acc, D_f, margin = evaluate(model)
        history.append({"epoch": epoch, "loss": avg_loss, "acc": acc, "D_f": D_f, "margin": margin})
        print(f"{epoch:>6} | {avg_loss:>10.4f} | {acc*100:>7.2f}% | {D_f:>8.4f} | {margin:>8.4f}", flush=True)

# 汇总
print()
print("=" * 60)
print("趋势分析")
print("=" * 60)

if len(history) >= 3:
    accs = np.array([h["acc"] for h in history])
    dfs = np.array([h["D_f"] for h in history])
    margins = np.array([h["margin"] for h in history])
    epochs = np.array([h["epoch"] for h in history])

    print(f"{'Epoch':>6} | {'Acc':>7} | {'D_f':>7} | {'Margin':>7}")
    print("-" * 40)
    for h in history:
        print(f"{h['epoch']:>6} | {h['acc']*100:>6.2f}% | {h['D_f']:>7.4f} | {h['margin']:>7.4f}")

    # 计算各项的稳定点
    def find_saturation(values, tol=0.01, window=3):
        for i in range(window, len(values)):
            recent = values[i-window:i+1]
            if np.max(recent) - np.min(recent) < tol:
                return history[i]["epoch"]
        return None

    d_f_sat = find_saturation(dfs, tol=0.02)
    margin_sat = find_saturation(margins, tol=0.1)
    acc_sat = find_saturation(accs, tol=0.005)

    print()
    print("饱和点分析：")
    print(f"  D_f 饱和点:    epoch {d_f_sat}")
    print(f"  Margin 饱和点: epoch {margin_sat}")
    print(f"  Acc 饱和点:    epoch {acc_sat}")

with open("nn_training_result.json", "w") as f:
    json.dump(history, f, indent=2)
print("\nSaved to nn_training_result.json")
