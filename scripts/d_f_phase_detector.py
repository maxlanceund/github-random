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

def compute_D_f_coarse(L, T, n_bins=20, n_runs=8, n_samples_per_run=40):
    """用粗粒化磁化计算 D_f"""
    all_bins = []
    for _ in range(n_runs):
        samples = simulate_ising(L, T, 300, n_samples_per_run)
        for s in samples:
            mag = int(s.sum())
            # 归一化到 [-1, 1]，再分到 n_bins 个 bin
            mag_norm = mag / (L * L)
            # 映射到 [0, n_bins-1]
            bin_idx = int((mag_norm + 1) / 2 * n_bins)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            if bin_idx < 0:
                bin_idx = 0
            all_bins.append(bin_idx)

    counts = Counter(all_bins)
    n = len(all_bins)
    H_macro = 0
    for c in counts.values():
        p = c / n
        H_macro -= p * math.log2(p)

    H_micro = L * L
    D_f = (H_micro - H_macro) / H_micro
    return D_f, H_macro

# ============================================================
# 主程序
# ============================================================
Tc_known = 2.269
Ls = [8, 12, 16, 24, 32]
T_range = np.linspace(1.5, 3.5, 21)
N_BINS = 20

print("=" * 78)
print("D_f 作为相变探测器（粗粒化磁化版）")
print("=" * 78)
print(f"已知 Tc = {Tc_known}")
print(f"宏观描述：磁化分为 {N_BINS} 个 bin")
print(f"最大宏观熵 = log2({N_BINS}) = {math.log2(N_BINS):.4f} bits")
print()

all_results = {}
for L in Ls:
    Df_list = []
    for T in T_range:
        Df, H_macro = compute_D_f_coarse(L, T, n_bins=N_BINS)
        Df_list.append(Df)
    Df_arr = np.array(Df_list)
    T_min = T_range[np.argmin(Df_arr)]
    Df_min = Df_arr.min()
    all_results[L] = {"T_range": T_range.tolist(),
                      "Df_list": Df_arr.tolist(),
                      "T_min": float(T_min),
                      "Df_min": float(Df_min)}
    print(f"L = {L:>3}: T_min = {T_min:.3f}, D_f_min = {Df_min:.6f}", flush=True)

print()
print("=" * 78)
print("T_min(L) vs Tc")
print("=" * 78)
print(f"{'L':>6} | {'T_min(D_f)':>12} | {'Tc_known':>10} | {'偏差':>10}")
print("-" * 55)
for L in Ls:
    T_min = all_results[L]["T_min"]
    diff = abs(T_min - Tc_known)
    print(f"{L:>6} | {T_min:>12.4f} | {Tc_known:>10.4f} | {diff:>10.4f}")

with open("d_f_phase_detector_result.json", "w") as f:
    json.dump({"Tc_known": Tc_known, "n_bins": N_BINS,
               "results": {str(k): v for k, v in all_results.items()}}, f, indent=2)
print()
print("结果已保存到 d_f_phase_detector_result.json")
