import numpy as np
import math
from collections import Counter

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

L = 32
H_micro = L * L
N_RUNS = 10
N_SAMPLES_PER_RUN = 100

print("=" * 70)
print(f"Ising 模型：D_f 对温度的依赖（L={L}，GitHub Actions 云端）")
print("=" * 70)
print(f"每个温度 {N_RUNS} 次独立 run，每次 {N_SAMPLES_PER_RUN} 个采样")
print()
print(f"{'T':>7} | {'H(M)':>8} | {'D_f':>10} | {'磁化范围':>16}")
print("-" * 65)

for T in [0.5, 1.0, 1.5, 2.0, 2.269, 2.5, 3.0, 4.0, 5.0, 10.0]:
    all_mags = []
    for run in range(N_RUNS):
        samples = simulate_ising(L, T, 500, N_SAMPLES_PER_RUN)
        all_mags.extend([int(s.sum()) for s in samples])

    counts = Counter(all_mags)
    n = len(all_mags)
    H_macro = 0
    for c in counts.values():
        p = c / n
        H_macro -= p * math.log2(p)
    D_f = (H_micro - H_macro) / H_micro
    print(f"{T:>7.3f} | {H_macro:>8.4f} | {D_f:>10.6f} | [{min(all_mags):>5},{max(all_mags):>5}]")

print()
print("完成。D_f 最小值应出现在 T ≈ 2.269 附近。")
