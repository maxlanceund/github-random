import numpy as np
import json

def xx_correlation_pbc(L, J=1.0, mu=0.0):
    """周期边界 XX 链 + 化学势"""
    h = np.zeros((L, L))
    for i in range(L):
        j = (i + 1) % L
        h[i, j] = -J/2
        h[j, i] = -J/2
        h[i, i] = -mu  # 化学势
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    occupied = eigenvalues < 0
    C = eigenvectors[:, occupied] @ eigenvectors[:, occupied].conj().T
    return np.real(C)

def entropy_from_corr(C):
    lambdas = np.linalg.eigvalsh(C)
    lambdas = np.clip(lambdas, 1e-15, 1 - 1e-15)
    return float(-np.sum(lambdas * np.log(lambdas) + (1-lambdas) * np.log(1-lambdas)))

def mutual_information(C, i, j):
    n_i = C[i, i]
    n_j = C[j, j]
    S_i = -n_i * np.log(n_i + 1e-15) - (1-n_i) * np.log(1-n_i + 1e-15)
    S_j = -n_j * np.log(n_j + 1e-15) - (1-n_j) * np.log(1-n_j + 1e-15)
    C_ij = np.array([[C[i,i], C[i,j]], [C[j,i], C[j,j]]])
    S_ij = entropy_from_corr(C_ij)
    return float(S_i + S_j - S_ij)

L = 400
# 用化学势让填充约为 0.4（非半填充）
mu = -0.5
C = xx_correlation_pbc(L, J=1.0, mu=mu)

# 报告填充数
filling = np.mean(np.diag(C))
print("=" * 72)
print(f"从纠缠中涌现的几何：周期 XX 链（非半填充）")
print("=" * 72)
print(f"L = {L}, 化学势 mu = {mu}, 平均填充 = {filling:.4f}")
print()

i0 = L // 2
print(f"{'|i-j|':>8} | {'C(i,j)':>12} | {'I(i,j)':>14} | {'d=-lnI':>10} | {'ln|i-j|':>10} | {'d/ln|i-j|':>12}")
print("-" * 78)

results = []
for d in [1, 2, 4, 8, 16, 32, 64, 100]:
    j = (i0 + d) % L
    Cij = C[i0, j]
    I = mutual_information(C, i0, j)
    if I > 1e-10:
        d_geom = -np.log(I)
        ln_d = np.log(d)
        ratio = d_geom / ln_d
    else:
        d_geom = float('inf')
        ln_d = np.log(d)
        ratio = float('inf')
    print(f"{d:>8} | {Cij:>12.6f} | {I:>14.8f} | {d_geom:>10.4f} | {ln_d:>10.4f} | {ratio:>12.4f}")
    results.append({"d": d, "C_ij": Cij, "I": I, "d_geom": d_geom})

# 拟合
valid = [r for r in results if r["I"] > 1e-10]
if len(valid) >= 2:
    ds = np.array([r["d"] for r in valid])
    d_geoms = np.array([r["d_geom"] for r in valid])
    coef = np.polyfit(np.log(ds), d_geoms, 1)
    print()
    print("=" * 72)
    print(f"拟合 d(i,j) = {coef[0]:.4f} * ln|i-j| + {coef[1]:.4f}")
    print(f"理论预言（临界 CFT）：斜率 ≈ 2")
    print(f"偏差 = {abs(coef[0] - 2):.4f}")
    print("=" * 72)

    with open("emergent_geometry_result.json", "w") as f:
        json.dump({"results": results, "slope": float(coef[0]),
                   "intercept": float(coef[1]), "filling": float(filling)}, f, indent=2)
    print("结果已保存到 emergent_geometry_result.json")
