import numpy as np
from numba import njit
from collections import Counter
import math
import json

@njit
def metropolis_sweep(spins, T):
    """一次完整扫描（L² 次翻转尝试）"""
    L = spins.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        nb = (spins[(i+1) % L, j] + spins[(i-1) % L, j] +
              spins[i, (j+1) % L] + spins[i, (j-1) % L])
        dE = 2 * s * nb
        if dE <= 0 or np.random.random() < np.exp(-dE / T):
            spins[i, j] = -s
    return spins

@njit
def run_simulation(L, T, n_thermalize, n_samples):
    """跑一次模拟，返回所有采样的总磁化"""
    spins = np.ones((L, L), dtype=np.int32)
    # 热化
    for _ in range(n_thermalize):
        metropolis_sweep(spins, T)
    # 采样
    mags = np.zeros(n_samples, dtype=np.int32)
    for k in range(n_samples):
        metropolis_sweep(spins, T)
        mags[k] = spins.sum()
    return mags

def compute_D_f(L, T, n_bins=20, n_runs=10, n_samples_per_run=100):
    all_mags = []
    for _ in range(n_runs):
        mags = run_simulation(L, T, 300, n_samples_per_run)
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

    H_micro = L * L
    return (H_micro - H_macro) / H_micro, H_macro

if __name__ == "__main__":
    import time
    print("=" * 70)
    print("numba 加速版 Ising 模拟")
    print("=" * 70)
    print()

    # 第一次编译
    print("编译中（首次运行需要几秒）...", flush=True)
    t0 = time.time()
    _ = run_simulation(8, 2.269, 10, 5)
    print(f"编译完成，耗时 {time.time() - t0:.1f} 秒")
    print()

    Ls = [8, 16, 24, 32, 48, 64]
    T_list = [1.5, 2.0, 2.269, 2.5, 3.0]

    print(f"{'T':>6} | " + " | ".join([f"L={L:<2}" for L in Ls]))
    print("-" * 70)

    results = {}
    for T in T_list:
        row = []
        for L in Ls:
            Df, Hm = compute_D_f(L, T)
            row.append(Df)
        results[T] = row
        print(f"{T:>6.3f} | " + " | ".join([f"{d:.4f}" for d in row]), flush=True)

    with open("fast_ising_result.json", "w") as f:
        json.dump({"Ls": Ls, "T_list": T_list,
                   "results": {str(T): results[T] for T in T_list}}, f, indent=2)
    print()
    print("结果已保存到 fast_ising_result.json")
