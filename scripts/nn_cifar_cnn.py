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

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # 3x32x32
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 经过 3 次 pool：32 -> 16 -> 8 -> 4
        # 128 x 4 x 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, n_classes)
    def forward(self, x, return_all=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x_flat = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        out = self.fc2(h1)
        if return_all:
            return x_flat, h1, out
        return out

class SimpleDS(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 用更多数据：20000 训练
train_data = [(cifar_train[i][0], cifar_train[i][1]) for i in range(20000)]
test_data = [(cifar_test[i][0], cifar_test[i][1]) for i in range(2000)]

train_loader = DataLoader(SimpleDS(train_data), batch_size=128, shuffle=True)
test_loader = DataLoader(SimpleDS(test_data), batch_size=128, shuffle=False)

def train_network(n_classes, n_epochs=30):
    torch.manual_seed(42)
    model = SimpleCNN(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}", flush=True)
    return model

def collect_h1_and_images(model):
    model.eval()
    h1_list, img_list, labels_list = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x_dev = x.to(device)
            _, h1, _ = model(x_dev, return_all=True)
            h1_list.append(h1.cpu().numpy())
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
    gray = images.mean(axis=1)
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
n_classes = 10

print("=" * 70)
print("CIFAR-10 CNN 分析")
print("=" * 70)

model = train_network(n_classes, n_epochs=30)

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
print(f"\nTest Accuracy: {acc*100:.2f}%")

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

# 对比
print()
print("=" * 70)
print("三方对比：MNIST vs CIFAR(FC) vs CIFAR(CNN)")
print("=" * 70)
print(f"{'量':>22} | {'MNIST':>10} | {'CIFAR-FC':>10} | {'CIFAR-CNN':>10}")
print("-" * 65)
print(f"{'准确率':>22} | {0.883*100:>9.1f}% | {0.366*100:>9.1f}% | {acc*100:>9.1f}%")
print(f"{'有效秩':>22} | {12.30:>10.2f} | {11.96:>10.2f} | {eff_rank:>10.2f}")
print(f"{'背景模式 MI':>22} | {0.433:>10.3f} | {0.205:>10.3f} | {bg['mi_class']:>10.3f}")
print(f"{'背景 vs 亮度':>22} | {-0.941:>10.3f} | {-0.863:>10.3f} | {bg['corr_brightness']:>10.3f}")

output = {
    "acc": acc,
    "effective_rank": eff_rank,
    "max_eig": float(eigs[0]),
    "mode_analysis": mode_analysis,
    "best_correlation": {"stat": best, "value": corrs[best]},
}
with open("nn_cifar_cnn_result.json", "w") as f:
    json.dump(output, f, indent=2)
print()
print("Saved to nn_cifar_cnn_result.json")
