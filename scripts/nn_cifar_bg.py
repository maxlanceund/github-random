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
    def __init__(self, hidden_sizes, n_classes, input_dim):
        super().__init__()
        sizes = [input_dim] + hidden_sizes + [n_classes]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
    def forward(self, x, return_all=False):
        x = x.view(x.size(0), -1)
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
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_data = [(cifar_train[i][0], cifar_train[i][1]) for i in range(5000)]
test_data = [(cifar_test[i][0], cifar_test[i][1]) for i in range(1000)]

train_loader = DataLoader(SimpleDS(train_data), batch_size=128, shuffle=True)
test_loader = DataLoader(SimpleDS(test_data), batch_size=128, shuffle=False)

def train_network(hidden, n_classes, input_dim, n_epochs=15):
    torch.manual_seed(42)
    model = FlexNet(hidden, n_classes, input_dim).to(device)
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

def collect_h1_and_images(model):
    model.eval()
    h1_list, img_list, labels_list = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x_dev = x.to(device)
            acts = model(x_dev, return_all=True)
            h1_list.append(acts[1].cpu().numpy())
            img_list.append(x.numpy())
            labels_list.append(y.numpy())
    return (np.concatenate(h1_list, axis=0),
            np.concatenate(img_list, axis=0),
            np.concatenate(labels_list, axis=0))

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

def image_stats(images):
    N = images.shape[0]
    img_flat = images.reshape(N, -1)
    brightness = img_flat.mean(axis=1)
    contrast = img_flat.std(axis=1)
    edge = np.abs(np.diff(img_flat, axis=1)).mean(axis=1)
    if len(images.shape) == 4:
        gray = images.mean(axis=1)
    else:
        gray = images
    freq = np.fft.fft2(gray, axes=(1, 2))
    freq_mag = np.abs(freq)
    h, w = gray.shape[1], gray.shape[2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fy**2 + fx**2)
    low_mask = radius < 0.25
    high_mask = radius >= 0.25
    low_e = (freq_mag**2 * low_mask).sum(axis=(1, 2))
    high_e = (freq_mag**2 * high_mask).sum(axis=(1, 2))
    high_ratio = high_e / (low_e + high_e + 1e-10)
    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge,
        "high_freq_ratio": high_ratio,
    }

def rank_correlation(x, y):
    def rank(a):
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r
    rx = rank(x); ry = rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    num = (rx * ry).sum()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float(num / (den + 1e-10))

# ============================================================
# 主程序
# ============================================================
input_dim = 3 * 32 * 32  # CIFAR-10 输入
hidden = [256, 128, 64]
n_classes = 10

print("=" * 70)
print("CIFAR-10 分析")
print("=" * 70)

model = train_network(hidden, n_classes, input_dim, n_epochs=15)

# 准确率
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
acc = correct / total
print(f"Accuracy: {acc*100:.2f}%")

h1, images, labels = collect_h1_and_images(model)
print(f"h1 shape: {h1.shape}")
print(f"images shape: {images.shape}")

# 互信息矩阵
print("\nComputing MI matrix...")
M = mi_matrix(h1)
eigs, vecs = np.linalg.eigh(M)
order = np.argsort(eigs)[::-1]
eigs = eigs[order]
vecs = vecs[:, order]

# 有效秩
pos = eigs[eigs > 1e-10]
p = pos / pos.sum()
eff_rank = float(np.exp(-np.sum(p * np.log(p))))
print(f"\neffective_rank: {eff_rank:.4f}")
print(f"max_eig:        {eigs[0]:.4f}")
print(f"gap(1-2):       {eigs[0] - eigs[1]:.4f}")

# 特征模式与图像统计量的相关性
stats = image_stats(images)
print()
print("=" * 75)
print("特征模式 vs 图像统计量")
print("=" * 75)
print(f"{'模式':>5} | {'特征值':>9} | {'与类别MI':>9} | {'亮度':>7} | {'对比度':>7} | {'边缘':>7} | {'高频':>7}")
print("-" * 75)

mode_analysis = []
for k in range(5):
    v = vecs[:, k]
    proj = h1 @ v
    mi_cls = mutual_info(proj, labels)
    cb = rank_correlation(proj, stats["brightness"])
    cc = rank_correlation(proj, stats["contrast"])
    ce = rank_correlation(proj, stats["edge_density"])
    cf = rank_correlation(proj, stats["high_freq_ratio"])
    print(f"{k:>5} | {eigs[k]:>9.3f} | {mi_cls:>9.4f} | {cb:>+7.3f} | {cc:>+7.3f} | {ce:>+7.3f} | {cf:>+7.3f}", flush=True)
    mode_analysis.append({
        "mode": k, "eig": float(eigs[k]), "mi_class": float(mi_cls),
        "corr_brightness": cb, "corr_contrast": cc,
        "corr_edge_density": ce, "corr_high_freq_ratio": cf,
    })

# 背景模式
bg = mode_analysis[0]
corrs = {
    "brightness": bg["corr_brightness"],
    "contrast": bg["corr_contrast"],
    "edge_density": bg["corr_edge_density"],
    "high_freq_ratio": bg["corr_high_freq_ratio"],
}
best = max(corrs, key=lambda k: abs(corrs[k]))

print()
print("=" * 70)
print(f"背景模式（模式 0）")
print("=" * 70)
print(f"  与类别 MI:  {bg['mi_class']:.4f}")
print(f"  与亮度:     {bg['corr_brightness']:+.4f}")
print(f"  与对比度:   {bg['corr_contrast']:+.4f}")
print(f"  与边缘:     {bg['corr_edge_density']:+.4f}")
print(f"  与高频:     {bg['corr_high_freq_ratio']:+.4f}")
print(f"\n相关性最强: {best} ({corrs[best]:+.4f})")

# 和 MNIST 对比
print()
print("=" * 70)
print("与 MNIST 结果对比")
print("=" * 70)
print(f"{'量':>20} | {'MNIST':>10} | {'CIFAR':>10}")
print("-" * 50)
print(f"{'有效秩':>20} | {12.30:>10.2f} | {eff_rank:>10.2f}")
print(f"{'背景模式 MI':>20} | {0.433:>10.3f} | {bg['mi_class']:>10.3f}")
print(f"{'背景模式 vs 对比度':>20} | {-0.942:>10.3f} | {bg['corr_contrast']:>10.3f}")
print(f"{'背景模式 vs 亮度':>20} | {-0.941:>10.3f} | {bg['corr_brightness']:>10.3f}")

output = {
    "acc": acc,
    "effective_rank": eff_rank,
    "max_eig": float(eigs[0]),
    "mode_analysis": mode_analysis,
    "best_correlation": {"stat": best, "value": corrs[best]},
}
with open("nn_cifar_bg_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_cifar_bg_result.json")
