import numpy as np
from numba import njit
from collections import Counter
import math
import json

@njit
def sweep(spins, T):
    L = spins.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        nb = (spins[(i+1)%L, j] + spins[(i-1)%L, j] +
              spins[i, (j+1)%L] + spins[i, (j-1)%L])
        dE = 2 * s * nb
        if dE <= 0 or np.random.random() < np.exp(-dE / T):
            spins[i, j] = -s

@njit
def run_sim(L, T, n_thermalize, n_samples):
    spins = np.ones((L, L), dtype=np.int32)
    for _ in range(n_thermalize):
        sweep(spins, T)
    mags = np.zeros(n_samples, dtype=np.int32)
    for k in range(n_samples):
        sweep(spins, T)
        mags[k] = spins.sum()
    return mags

def compute_D_f(L, T, n_bins=20, n_runs=10, n_samples_per_run=100):
    all_mags = []
    for _ in range(n_runs):
        mags = run_sim(L, T, 300, n_samples_per_run)
        all_mags.extend(mags.tolist())
    max_M = L * L
    bins = []
    for M in all_mags:
        b = int((M / max_M + 1) / 2 * n_bins)
        b = min(max(b, 0), n_bins - 1)
        bins.append(b)
    counts = Counter(bins)
    n = len(bins)
    H_macro = 0
    for c in counts.values():
        p = c / n
        H_macro -= p * math.log2(p)
    return (L * L - H_macro) / (L * L)

if __name__ == "__main__":
    Tc = 2.269
    Ls = [8, 16, 32, 64]
    T_range = np.linspace(1.8, 2.8, 21)

    print("编译中...", flush=True)
    _ = run_sim(8, 2.269, 10, 5)
    print("编译完成\n")

    print("=" * 70)
    print("D_f 相变探测器（numba 加速，L 到 64）")
    print("=" * 70)
    print()

    all_results = {}
    for L in Ls:
        Df_list = []
        for T in T_range:
            Df = compute_D_f(L, T)
            Df_list.append(Df)
        Df_arr = np.array(Df_list)
        T_min = float(T_range[np.argmin(Df_arr)])
        Df_min = float(Df_arr.min())
        all_results[L] = {"T_min": T_min, "Df_min": Df_min,
                          "T_range": T_range.tolist(), "Df_list": Df_arr.tolist()}
        print(f"L = {L:>3}: T_min = {T_min:.4f}, D_f_min = {Df_min:.6f}", flush=True)

    print()
    print("=" * 70)
    print(f"{'L':>6} | {'T_min(D_f)':>12} | {'Tc_known':>10} | {'偏差':>10}")
    print("-" * 55)
    for L in Ls:
        T_min = all_results[L]["T_min"]
        print(f"{L:>6} | {T_min:>12.4f} | {Tc:>10.4f} | {abs(T_min - Tc):>10.4f}")

    with open("fast_phase_detector_result.json", "w") as f:
        json.dump({"Tc": Tc, "Ls": Ls,
                   "results": {str(L): all_results[L] for L in Ls}}, f, indent=2)
    print()
    print("结果已保存到 fast_phase_detector_result.json")
