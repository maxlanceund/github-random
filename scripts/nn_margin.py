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
    return model

log2_10 = math.log(10)

def collect_labels():
    ys = []
    for _, y in test_loader:
        ys.append(y.numpy())
    return np.concatenate(ys, axis=0)

Y = collect_labels()

def analyze_model(model):
    model.eval()
    all_out = []
    all_pred = []
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
            # 计算 margin
            correct_logits = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, y.unsqueeze(1), -1e9)
            max_wrong = masked.max(dim=1).values
            margins.append((correct_logits - max_wrong).cpu().numpy())

    out_np = np.concatenate(all_out, axis=0)
    margin_np = np.concatenate(margins, axis=0)
    acc = correct / total

    # 输出层 D_f
    feat_std = (out_np - out_np.mean(axis=0)) / (out_np.std(axis=0) + 1e-8)
    ce = estimate_ce(feat_std, Y)
    D_f = ce / log2_10

    # 输出熵（softmax 的熵）
    probs = F.softmax(torch.tensor(out_np), dim=1).numpy()
    probs = np.clip(probs, 1e-12, 1.0)
    out_entropy = -np.sum(probs * np.log(probs), axis=1).mean()

    return {
        "acc": acc,
        "D_f_out": D_f,
        "margin_mean": float(margin_np.mean()),
        "margin_std": float(margin_np.std()),
        "out_entropy": float(out_entropy),
        "logit_scale": float(np.abs(out_np).mean()),
    }

configs = {
    "narrow-16 (16-32-16-8)": [16, 32, 16, 8],
    "narrow-32 (32-32-16-8)": [32, 32, 16, 8],
    "medium-128 (128-32-16-8)": [128, 32, 16, 8],
    "wide-256 (256-32-16-8)": [256, 32, 16, 8],
    "wide-512 (512-32-16-8)": [512, 32, 16, 8],
    "very-wide-1024 (1024-32-16-8)": [1024, 32, 16, 8],
}

all_results = {}
for name, hidden in configs.items():
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")
    model = train_network(hidden)
    r = analyze_model(model)
    all_results[name] = r
    print(f"  acc:          {r['acc']*100:.2f}%")
    print(f"  D_f (output): {r['D_f_out']:.4f}")
    print(f"  margin mean:  {r['margin_mean']:.4f}")
    print(f"  margin std:   {r['margin_std']:.4f}")
    print(f"  out entropy:  {r['out_entropy']:.4f}")
    print(f"  logit scale:  {r['logit_scale']:.4f}", flush=True)

print()
print("=" * 95)
print(f"{'Config':>35} | {'acc':>7} | {'D_f':>7} | {'margin':>8} | {'out_H':>7} | {'logit':>7}")
print("-" * 95)
for name, r in all_results.items():
    print(f"{name:>35} | {r['acc']*100:>6.2f}% | {r['D_f_out']:>7.4f} | {r['margin_mean']:>8.4f} | {r['out_entropy']:>7.4f} | {r['logit_scale']:>7.4f}")

# 相关性
accs = np.array([r['acc'] for r in all_results.values()])
dfs = np.array([r['D_f_out'] for r in all_results.values()])
margins = np.array([r['margin_mean'] for r in all_results.values()])
ents = np.array([r['out_entropy'] for r in all_results.values()])
logits = np.array([r['logit_scale'] for r in all_results.values()])

print()
print("与准确率的相关系数：")
print(f"  D_f_out:     {np.corrcoef(dfs, accs)[0,1]:+.4f}")
print(f"  margin_mean: {np.corrcoef(margins, accs)[0,1]:+.4f}")
print(f"  out_entropy: {np.corrcoef(ents, accs)[0,1]:+.4f}")
print(f"  logit_scale: {np.corrcoef(logits, accs)[0,1]:+.4f}")

with open("nn_margin_result.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to nn_margin_result.json")
