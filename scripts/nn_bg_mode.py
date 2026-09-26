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

def train_network(hidden, n_epochs=10):
    torch.manual_seed(42)
    model = FlexNet(hidden).to(device)
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
    if yd.max() > 100:
        yd = discretize(yd, n_bins)
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
    """计算每张图像的统计量"""
    N = images.shape[0]
    img_flat = images.reshape(N, -1)
    brightness = img_flat.mean(axis=1)
    contrast = img_flat.std(axis=1)
    # 边缘密度：用差分近似
    edge = np.abs(np.diff(img_flat, axis=1)).mean(axis=1)
    # 空间频率：傅里叶变换的高频能量占比
    freq = np.fft.fft2(images[:, 0], axes=(1, 2))
    freq_mag = np.abs(freq)
    # 把频率分成低频和高频
    h, w = images.shape[1], images.shape[2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fy**2 + fx**2)
    low_mask = radius < 0.25
    high_mask = radius >= 0.25
    low_energy = (freq_mag**2 * low_mask).sum(axis=(1, 2))
    high_energy = (freq_mag**2 * high_mask).sum(axis=(1, 2))
    high_freq_ratio = high_energy / (low_energy + high_energy + 1e-10)
    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge,
        "high_freq_ratio": high_freq_ratio,
    }

def rank_correlation(x, y):
    """Spearman 秩相关系数"""
    def rank(a):
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r
    rx = rank(x)
    ry = rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    num = (rx * ry).sum()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float(num / (den + 1e-10))

hidden = [128, 64, 32, 16]
model = train_network(hidden)
h1, images, labels = collect_h1_and_images(model)
print(f"h1 shape: {h1.shape}, images shape: {images.shape}")

# 计算互信息矩阵并特征分解
print("Computing MI matrix...")
M = mi_matrix(h1)
eigs, vecs = np.linalg.eigh(M)
order = np.argsort(eigs)[::-1]
eigs = eigs[order]
vecs = vecs[:, order]

# 取前 5 个模式，看它们和图像统计量的相关性
print()
print("=" * 75)
print("特征模式 vs 图像统计量")
print("=" * 75)

stats = image_stats(images)

print(f"{'模式':>5} | {'特征值':>9} | {'与类别MI':>9} | {'亮度':>7} | {'对比度':>7} | {'边缘':>7} | {'高频':>7}")
print("-" * 75)

mode_analysis = []
for k in range(5):
    v = vecs[:, k]
    proj = h1 @ v
    mi_cls = mutual_info(proj, labels)
    corr_bright = rank_correlation(proj, stats["brightness"])
    corr_contrast = rank_correlation(proj, stats["contrast"])
    corr_edge = rank_correlation(proj, stats["edge_density"])
    corr_freq = rank_correlation(proj, stats["high_freq_ratio"])
    print(f"{k:>5} | {eigs[k]:>9.3f} | {mi_cls:>9.4f} | {corr_bright:>+7.3f} | {corr_contrast:>+7.3f} | {corr_edge:>+7.3f} | {corr_freq:>+7.3f}", flush=True)

    mode_analysis.append({
        "mode": k,
        "eig": float(eigs[k]),
        "mi_class": float(mi_cls),
        "corr_brightness": corr_bright,
        "corr_contrast": corr_contrast,
        "corr_edge_density": corr_edge,
        "corr_high_freq_ratio": corr_freq,
    })

# 检查背景模式（模式 0）
print()
print("=" * 75)
print("背景模式（模式 0）分析")
print("=" * 75)

bg = mode_analysis[0]
print(f"特征值:    {bg['eig']:.3f}")
print(f"与类别 MI: {bg['mi_class']:.4f}")
print()
print("与图像统计量的秩相关：")
print(f"  亮度:      {bg['corr_brightness']:+.4f}")
print(f"  对比度:    {bg['corr_contrast']:+.4f}")
print(f"  边缘密度:  {bg['corr_edge_density']:+.4f}")
print(f"  高频占比:  {bg['corr_high_freq_ratio']:+.4f}")

# 找出最强的相关性
corrs = {
    "brightness": bg["corr_brightness"],
    "contrast": bg["corr_contrast"],
    "edge_density": bg["corr_edge_density"],
    "high_freq_ratio": bg["corr_high_freq_ratio"],
}
best_stat = max(corrs, key=lambda k: abs(corrs[k]))
print()
print(f"背景模式与'{best_stat}'相关性最强: {corrs[best_stat]:+.4f}")

output = {
    "mode_analysis": mode_analysis,
    "background_mode": bg,
    "best_correlation": {"stat": best_stat, "value": corrs[best_stat]},
}
with open("nn_bg_mode_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_bg_mode_result.json")
