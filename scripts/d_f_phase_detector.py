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

def compute_D_f(L, T, n_runs=8, n_samples_per_run=40):
    all_mags = []
    for _ in range(n_runs):
        samples = simulate_ising(L, T, 300, n_samples_per_run)
        all_mags.extend([int(s.sum()) for s in samples])
    counts = Counter(all_mags)
    n = len(all_mags)
    H_macro = 0
    for c in counts.values():
        p = c / n
        H_macro -= p * math.log2(p)
    H_micro = L * L
    return (H_micro - H_macro) / H_micro

# ============================================================
# 主程序
# ============================================================
Tc_known = 2.269

Ls = [8, 12, 16, 24]
T_range = np.linspace(1.5, 3.5, 21)

print("=" * 78)
print("D_f 作为相变探测器")
print("=" * 78)
print(f"已知 Tc = {Tc_known}")
print(f"只用 D_f，不用磁化率/比热/序参量")
print()

all_results = {}

for L in Ls:
    Df_list = []
    for T in T_range:
        Df = compute_D_f(L, T)
        Df_list.append(Df)
    Df_arr = np.array(Df_list)
    T_min = T_range[np.argmin(Df_arr)]
    Df_min = Df_arr.min()
    all_results[L] = {"T_range": T_range.tolist(),
                      "Df_list": Df_arr.tolist(),
                      "T_min": float(T_min),
                      "Df_min": float(Df_min)}
    print(f"L = {L:>3}: D_f 极小值在 T = {T_min:.3f}, D_f_min = {Df_min:.6f}")

print()
print("=" * 78)
print("D_f 极小值位置 vs 已知 Tc")
print("=" * 78)
print(f"{'L':>6} | {'T_min(D_f)':>12} | {'Tc_known':>10} | {'偏差':>10}")
print("-" * 55)
for L in Ls:
    T_min = all_results[L]["T_min"]
    diff = abs(T_min - Tc_known)
    print(f"{L:>6} | {T_min:>12.4f} | {Tc_known:>10.4f} | {diff:>10.4f}")

print()
print("如果 T_min(L) 随 L 增大趋近 Tc，则 D_f 可以探测相变。")

with open("d_f_phase_detector_result.json", "w") as f:
    json.dump({"Tc_known": Tc_known, "results": {str(k): v for k, v in all_results.items()}}, f, indent=2)
print("结果已保存到 d_f_phase_detector_result.json")
