import numpy as np
import json

def xx_correlation(L, J=1.0):
    """临界 XX 链的基态关联矩阵（开边界）"""
    h = np.zeros((L, L))
    for i in range(L-1):
        h[i, i+1] = -J/2
        h[i+1, i] = -J/2
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    occupied = eigenvalues < 0
    C = eigenvectors[:, occupied] @ eigenvectors[:, occupied].conj().T
    return np.real(C)

def entropy_from_corr(C):
    """从关联矩阵计算纠缠熵（自然对数）"""
    lambdas = np.linalg.eigvalsh(C)
    lambdas = np.clip(lambdas, 1e-15, 1-1e-15)
    return float(-np.sum(lambdas * np.log(lambdas) + (1-lambdas) * np.log(1-lambdas)))

def mutual_information(C, i, j):
    """两点互信息 I(i,j) = S(i) + S(j) - S(i∪j)"""
    n_i = C[i, i]
    n_j = C[j, j]
    S_i = -n_i * np.log(n_i + 1e-15) - (1-n_i) * np.log(1-n_i + 1e-15)
    S_j = -n_j * np.log(n_j + 1e-15) - (1-n_j) * np.log(1-n_j + 1e-15)
    C_ij = np.array([[C[i,i], C[i,j]], [C[j,i], C[j,j]]])
    S_ij = entropy_from_corr(C_ij)
    return float(S_i + S_j - S_ij)

# ============================================================
# 主程序
# ============================================================
L = 400
C = xx_correlation(L, J=1.0)

print("=" * 72)
print("从纠缠中涌现的几何：临界 XX 链")
print("=" * 72)
print()
print(f"L = {L}, 中心格点附近测量互信息")
print()
print(f"{'|i-j|':>8} | {'I(i,j)':>14} | {'d(i,j)':>12} | {'ln|i-j|':>10} | {'d/ln|i-j|':>12}")
print("-" * 72)

i0 = L // 2
results = []
for d in [1, 2, 4, 8, 16, 32, 64, 128]:
    j = i0 + d
    I = mutual_information(C, i0, j)
    if I > 1e-15:
        d_geom = -np.log(I)
        ln_d = np.log(d)
        ratio = d_geom / ln_d
    else:
        d_geom = float('inf')
        ln_d = np.log(d)
        ratio = float('inf')
    print(f"{d:>8} | {I:>14.8f} | {d_geom:>12.4f} | {ln_d:>10.4f} | {ratio:>12.4f}")
    results.append({"d": d, "I": I, "d_geom": d_geom})

# 拟合 d(i,j) = a * ln|i-j| + b
ds = np.array([r["d"] for r in results if r["I"] > 1e-15])
d_geoms = np.array([r["d_geom"] for r in results if r["I"] > 1e-15])

if len(ds) >= 2:
    coef = np.polyfit(np.log(ds), d_geoms, 1)
    print()
    print("=" * 72)
    print(f"拟合 d(i,j) = {coef[0]:.4f} * ln|i-j| + {coef[1]:.4f}")
    print(f"理论预言（临界 CFT）：斜率 ≈ 2")
    print(f"偏差 = {abs(coef[0] - 2):.4f}")
    print("=" * 72)

with open("emergent_geometry_result.json", "w") as f:
    json.dump({"results": results, "slope": float(coef[0]), "intercept": float(coef[1])}, f, indent=2)

print("结果已保存到 emergent_geometry_result.json")
