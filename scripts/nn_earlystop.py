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

def estimate_ce(features, labels, n_epochs=30):
    X_t = torch.tensor(features, dtype=torch.float32).to(device)
    y_t = torch.tensor(labels, dtype=torch.long).to(device)
    d = X_t.shape[1]
    clf = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 10)).to(device)
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
            cl = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, y.unsqueeze(1), -1e9)
            mw = masked.max(dim=1).values
            margins.append((cl - mw).cpu().numpy())

    out_np = np.concatenate(all_out, axis=0)
    margin_np = np.concatenate(margins, axis=0)
    acc = correct / total

    fs = (out_np - out_np.mean(axis=0)) / (out_np.std(axis=0) + 1e-8)
    ce = estimate_ce(fs, Y)
    D_f = ce / log2_10
    return acc, D_f, float(margin_np.mean())

def find_saturation(values, tol):
    """返回第一个饱和点（连续 3 次变化 < tol）"""
    for i in range(3, len(values)):
        recent = values[i-3:i+1]
        if max(recent) - min(recent) < tol:
            return i
    return None

configs = {
    "A (32-16)": [32, 16],
    "B (128-64-32)": [128, 64, 32],
    "C (256-128-64-32)": [256, 128, 64, 32],
}

all_results = {}

for name, hidden in configs.items():
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = FlexNet(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Acc':>8} | {'D_f':>8} | {'Margin':>8}")
    print("-" * 55)

    n_epochs = 30
    for epoch in range(1, n_epochs + 1):
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

        if epoch % 3 == 0 or epoch <= 3:
            acc, D_f, margin = evaluate(model)
            history.append({"epoch": epoch, "loss": avg_loss, "acc": acc,
                           "D_f": D_f, "margin": margin})
            print(f"{epoch:>6} | {avg_loss:>8.4f} | {acc*100:>7.2f}% | {D_f:>8.4f} | {margin:>8.4f}", flush=True)

    # 饱和点分析
    if len(history) >= 4:
        accs = [h["acc"] for h in history]
        dfs = [h["D_f"] for h in history]
        margins = [h["margin"] for h in history]
        epochs = [h["epoch"] for h in history]

        d_f_idx = find_saturation(dfs, tol=0.02)
        acc_idx = find_saturation(accs, tol=0.005)
        margin_idx = find_saturation(margins, tol=0.3)

        d_f_ep = epochs[d_f_idx] if d_f_idx is not None else "no sat"
        acc_ep = epochs[acc_idx] if acc_idx is not None else "no sat"
        margin_ep = epochs[margin_idx] if margin_idx is not None else "no sat"

        print(f"\n  饱和点：")
        print(f"    D_f:    epoch {d_f_ep}")
        print(f"    Acc:    epoch {acc_ep}")
        print(f"    Margin: epoch {margin_ep}")

        all_results[name] = {
            "history": history,
            "d_f_sat": d_f_ep,
            "acc_sat": acc_ep,
            "margin_sat": margin_ep,
        }

# 汇总
print()
print("=" * 70)
print("Summary: 饱和点对比")
print("=" * 70)
print(f"{'Config':>25} | {'D_f sat':>10} | {'Acc sat':>10} | {'Margin sat':>12}")
print("-" * 70)
for name, r in all_results.items():
    print(f"{name:>25} | {str(r['d_f_sat']):>10} | {str(r['acc_sat']):>10} | {str(r['margin_sat']):>12}")

# 关键结论
n_d_f_earliest = 0
for name, r in all_results.items():
    if isinstance(r['d_f_sat'], int) and isinstance(r['acc_sat'], int):
        if r['d_f_sat'] <= r['acc_sat']:
            n_d_f_earliest += 1

print()
print(f"D_f 饱和早于或等于 Acc 饱和: {n_d_f_earliest} / {len(all_results)}")

with open("nn_earlystop_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to nn_earlystop_result.json")
