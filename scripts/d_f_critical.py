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

def compute_D_f(L, T, n_runs=10, n_samples_per_run=50):
    """多起点采样，计算 D_f"""
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
    D_f = (H_micro - H_macro) / H_micro
    return D_f, H_macro

# ============================================================
# 主程序：扫描 L，看 D_f 如何收敛到极限
# ============================================================
Tc = 2.269  # 2D Ising 临界温度

Ls = [8, 12, 16, 24, 32]

print("=" * 78)
print("D_f 的临界收敛指数：新预言")
print("=" * 78)
print(f"T = {Tc}（2D Ising 临界点），扫描 L")
print()
print(f"{'L':>6} | {'H(M)':>10} | {'D_f':>12} | {'1-D_f':>12} | {'ln(1-D_f)':>12}")
print("-" * 70)

results = []
for L in Ls:
    D_f, H_macro = compute_D_f(L, Tc, n_runs=15, n_samples_per_run=60)
    one_minus = 1 - D_f
    ln_one_minus = np.log(one_minus) if one_minus > 0 else float('-inf')
    print(f"{L:>6} | {H_macro:>10.4f} | {D_f:>12.6f} | {one_minus:>12.6f} | {ln_one_minus:>12.4f}", flush=True)
    results.append({"L": L, "H_macro": H_macro, "D_f": D_f, "1_minus_D_f": one_minus})

# ============================================================
# 拟合 1-D_f = a * L^(-omega)
# 即 ln(1-D_f) = ln(a) - omega * ln(L)
# ============================================================
print()
print("=" * 78)
print("拟合 1-D_f = a * L^(-omega)")
print("=" * 78)

valid = [r for r in results if r["1_minus_D_f"] > 1e-10]
ln_L = np.array([np.log(r["L"]) for r in valid])
ln_1mD = np.array([np.log(r["1_minus_D_f"]) for r in valid])

coef = np.polyfit(ln_L, ln_1mD, 1)
omega = -coef[0]
a = np.exp(coef[1])

print(f"拟合结果: 1-D_f = {a:.4f} * L^(-{omega:.4f})")
print(f"即 omega = {omega:.4f}")
print()
print("对比已知 2D Ising 临界指数：")
print(f"  nu   = 1.0000（关联长度）")
print(f"  beta = 0.1250（磁化）")
print(f"  gamma= 1.7500（磁化率）")
print(f"  omega = {omega:.4f} ← 这个是新的吗？")
print()

# 检查 omega 是否等于已知指数的组合
known_exponents = {
    "nu": 1.0,
    "1/nu": 1.0,
    "beta/nu": 0.125,
    "gamma/nu": 1.75,
    "2/nu": 2.0,
}
print("omega 与已知指数的比较：")
for name, val in known_exponents.items():
    diff = abs(omega - val)
    marker = " ← 匹配！" if diff < 0.1 else ""
    print(f"  {name:>10} = {val:.4f}，偏差 = {diff:.4f}{marker}")

# 保存
with open("d_f_critical_result.json", "w") as f:
    json.dump({"results": results, "omega": float(omega), "a": float(a)}, f, indent=2)
print()
print("结果已保存到 d_f_critical_result.json")
