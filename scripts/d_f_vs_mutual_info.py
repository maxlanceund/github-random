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

def entropy_from_list(items, n_bins):
    counts = Counter(items)
    n = len(items)
    H = 0
    for c in counts.values():
        p = c / n
        H -= p * math.log2(p)
    return H

def compute_metrics(L, T, n_bins=20, n_runs=8, n_samples_per_run=40):
    all_M_total = []
    all_M_A = []
    all_M_B = []

    for _ in range(n_runs):
        samples = simulate_ising(L, T, 300, n_samples_per_run)
        for s in samples:
            M_total = int(s.sum())
            M_A = int(s[:, :L//2].sum())
            M_B = int(s[:, L//2:].sum())
            all_M_total.append(M_total)
            all_M_A.append(M_A)
            all_M_B.append(M_B)

    n = len(all_M_total)

    def bin_idx(M, max_M):
        norm = M / max_M
        b = int((norm + 1) / 2 * n_bins)
        return min(max(b, 0), n_bins - 1)

    # D_f：总磁化
    max_M = L * L
    bins_total = [bin_idx(M, max_M) for M in all_M_total]
    H_M = entropy_from_list(bins_total, n_bins)
    H_micro = L * L
    D_f = (H_micro - H_M) / H_micro

    # I_AB：左半 vs 右半的互信息
    max_M_half = L * L // 2
    bins_A = [bin_idx(M, max_M_half) for M in all_M_A]
    bins_B = [bin_idx(M, max_M_half) for M in all_M_B]

    H_A = entropy_from_list(bins_A, n_bins)
    H_B = entropy_from_list(bins_B, n_bins)
    H_AB = entropy_from_list(list(zip(bins_A, bins_B)), n_bins * n_bins)

    I_AB = H_A + H_B - H_AB

    return D_f, I_AB

# ============================================================
# 主程序
# ============================================================
Tc = 2.269

Ls = [8, 16, 24, 32]
T_range = [1.5, 1.8, 2.0, 2.2, 2.269, 2.4, 2.6, 3.0, 3.5]

print("=" * 78)
print("D_f 与互信息 I_AB 的关系")
print("=" * 78)
print(f"{'L':>4} | {'T':>6} | {'D_f':>10} | {'I_AB':>10} | {'1-D_f':>10}")
print("-" * 60)

all_data = []
for L in Ls:
    for T in T_range:
        D_f, I_AB = compute_metrics(L, T)
        one_minus = 1 - D_f
        all_data.append({"L": L, "T": T, "D_f": D_f, "I_AB": I_AB})
        print(f"{L:>4} | {T:>6.3f} | {D_f:>10.6f} | {I_AB:>10.6f} | {one_minus:>10.6f}", flush=True)

# ============================================================
# 分析：1-D_f 和 I_AB 的关系
# ============================================================
print()
print("=" * 78)
print("分析：1-D_f vs I_AB")
print("=" * 78)

# 按 L 分组，看同一 L 下 1-D_f 和 I_AB 的关系
for L in Ls:
    subset = [d for d in all_data if d["L"] == L]
    print(f"\nL = {L}:")
    for d in subset:
        print(f"  T={d['T']:.3f}: 1-D_f={1-d['D_f']:.6f}, I_AB={d['I_AB']:.6f}")

# 全局相关性
one_minus_d = np.array([1 - d["D_f"] for d in all_data])
I_ab = np.array([d["I_AB"] for d in all_data])

if np.std(I_ab) > 1e-10 and np.std(one_minus_d) > 1e-10:
    corr = np.corrcoef(one_minus_d, I_ab)[0, 1]
    print(f"\n全局相关系数（1-D_f vs I_AB）: {corr:.4f}")

with open("d_f_vs_mi_result.json", "w") as f:
    json.dump(all_data, f, indent=2)
print("\n结果已保存到 d_f_vs_mi_result.json")
