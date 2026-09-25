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

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
    def forward(self, x, return_all=False):
        x = x.view(-1, 784)
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        a3 = self.fc3(a2)
        if return_all:
            return x, a1, a2, a3
        return a3

transform = T.ToTensor()
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_subset = Subset(train_set, range(5000))
test_subset = Subset(test_set, range(1000))
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

def collect_activations(model):
    model.eval()
    all_x, all_a1, all_a2, all_a3, all_y = [], [], [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_flat, a1, a2, a3 = model(x, return_all=True)
            all_x.append(x_flat.cpu().numpy())
            all_a1.append(a1.cpu().numpy())
            all_a2.append(a2.cpu().numpy())
            all_a3.append(a3.cpu().numpy())
            all_y.append(y.cpu().numpy())
    return (np.concatenate(all_x, axis=0),
            np.concatenate(all_a1, axis=0),
            np.concatenate(all_a2, axis=0),
            np.concatenate(all_a3, axis=0),
            np.concatenate(all_y, axis=0))

def estimate_ce(features, labels, n_epochs=100):
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

log2_10 = math.log(10)

torch.manual_seed(42)
model_random = SmallNet().to(device)
X_r, A1_r, A2_r, A3_r, Y = collect_activations(model_random)

print("Random network:")
print(f"{'Layer':>15} | {'CE (nats)':>10} | {'D_f':>8}")
print("-" * 45)
results_random = {}
for name, feat in [("input", X_r), ("layer1", A1_r), ("layer2", A2_r), ("output", A3_r)]:
    feat_std = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)
    ce = estimate_ce(feat_std, Y)
    D_f = ce / log2_10
    results_random[name] = {"CE": ce, "D_f": D_f}
    print(f"{name:>15} | {ce:>10.4f} | {D_f:>8.4f}", flush=True)

model_trained = SmallNet().to(device)
optimizer = torch.optim.Adam(model_trained.parameters(), lr=1e-3)
print()
print("Training...")
for epoch in range(10):
    model_trained.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model_trained(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}", flush=True)

model_trained.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model_trained(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
acc = correct / total * 100
print(f"Test accuracy: {acc:.2f}%")

X_t, A1_t, A2_t, A3_t, _ = collect_activations(model_trained)

print()
print("Trained network:")
print(f"{'Layer':>15} | {'CE (nats)':>10} | {'D_f':>8}")
print("-" * 45)
results_trained = {}
for name, feat in [("input", X_t), ("layer1", A1_t), ("layer2", A2_t), ("output", A3_t)]:
    feat_std = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)
    ce = estimate_ce(feat_std, Y)
    D_f = ce / log2_10
    results_trained[name] = {"CE": ce, "D_f": D_f}
    print(f"{name:>15} | {ce:>10.4f} | {D_f:>8.4f}", flush=True)

print()
print("=" * 60)
print(f"{'Layer':>15} | {'D_f random':>12} | {'D_f trained':>12} | {'diff':>8}")
print("-" * 60)
for name in ["input", "layer1", "layer2", "output"]:
    dr = results_random[name]["D_f"]
    dt = results_trained[name]["D_f"]
    print(f"{name:>15} | {dr:>12.4f} | {dt:>12.4f} | {dt-dr:>+8.4f}")

output = {"random": results_random, "trained": results_trained, "test_accuracy": acc}
with open("nn_control_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_control_result.json")
