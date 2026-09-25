import numpy as np
import math
from collections import Counter
import json

def simulate_ising(L, T, n_thermalize, n_samples):
    spins = np.random.choice([-1, 1], size=(L, L))
    for _ in range(n_thermalize):
        for _ in range(L * L):
            i, j = np.random.randint(0, L, 2)
            s = spins[i, j]
            nb = (spins[(i+1)%L, j] + spins[(i-1)%L, j] +
                  spins[i, (j+1)%L] + spins[i, (j-1)%L])
            dE = 2 * s * nb
            if dE <= 0 or np.random.random() < math.exp(-dE / T):
                spins[i, j] = -s
    samples = []
    for _ in range(n_samples):
        for _ in range(L * L):
            i, j = np.random.randint(0, L, 2)
            s = spins[i, j]
            nb = (spins[(i+1)%L, j] + spins[(i-1)%L, j] +
                  spins[i, (j+1)%L] + spins[i, (j-1)%L])
            dE = 2 * s * nb
            if dE <= 0 or np.random.random() < math.exp(-dE / T):
                spins[i, j] = -s
        samples.append(spins.copy())
    return samples

def compute_D_f(L, T, n_bins=20, n_runs=10, n_samples_per_run=50):
    all_bins = []
    for _ in range(n_runs):
        samples = simulate_ising(L, T, 300, n_samples_per_run)
        for s in samples:
            mag = int(s.sum())
            mag_norm = mag / (L * L)
            b = int((mag_norm + 1) / 2 * n_bins)
            b = min(max(b, 0), n_bins - 1)
            all_bins.append(b)
    counts = Counter(all_bins)
    n = len(all_bins)
    H_macro = 0
    for c in counts.values():
        p = c / n
        H_macro -= p * math.log2(p)
    H_micro = L * L
    return (H_micro - H_macro) / H_micro, H_macro

# ============================================================
# 主程序
# ============================================================
Tc = 2.269

Ls = [8, 12, 16, 24, 32]
T_list = [0.5, 1.0, 1.5, 2.0, 2.269, 2.5, 3.0, 4.0]

print("=" * 78)
print("反例检验：D_f 是否在所有温度下都趋近 1？")
print("=" * 78)
print()
print(f"{'T':>6} | " + " | ".join([f"L={L:<2}" for L in Ls]))
print("-" * 78)

results = {}
for T in T_list:
    row = []
    for L in Ls:
        Df, Hm = compute_D_f(L, T)
        row.append(Df)
    results[T] = row
    print(f"{T:>6.3f} | " + " | ".join([f"{d:.4f}" for d in row]), flush=True)

# ============================================================
# 检验：每个温度下，1-D_f 是否随 L 单调下降
# ============================================================
print()
print("=" * 78)
print("检验：1-D_f 是否随 L 单调下降？")
print("=" * 78)
print(f"{'T':>6} | {'1-D_f(L=8)':>12} | {'1-D_f(L=32)':>12} | {'趋势':>10}")
print("-" * 55)

for T in T_list:
    one_minus_8 = 1 - results[T][0]
    one_minus_32 = 1 - results[T][-1]
    if one_minus_32 < one_minus_8 * 0.9:
        trend = "下降 ✓"
    elif one_minus_32 > one_minus_8 * 1.1:
        trend = "上升 ✗"
    else:
        trend = "饱和 ?"
    print(f"{T:>6.3f} | {one_minus_8:>12.6f} | {one_minus_32:>12.6f} | {trend:>10}")

# ============================================================
# 关键检验：有没有温度下的 D_f 不趋近 1？
# ============================================================
print()
print("=" * 78)
print("结论：")
print("=" * 78)
all_approach_1 = True
for T in T_list:
    if results[T][-1] < 0.99:
        all_approach_1 = False
        print(f"⚠️ T={T} 时 D_f(L=32) = {results[T][-1]:.4f} < 0.99，反例！")
if all_approach_1:
    print("所有温度下 D_f(L=32) 均 > 0.99，未发现反例。")

with open("counterexample_gap_result.json", "w") as f:
    json.dump({"Tc": Tc, "Ls": Ls, "T_list": T_list,
               "results": {str(T): results[T] for T in T_list}}, f, indent=2)
print()
print("结果已保存到 counterexample_gap_result.json")
