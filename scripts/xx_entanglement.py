import numpy as np
import json

def xx_correlation(L, J=1.0, periodic=False):
    """XX 模型基态的关联矩阵 C_ij = <c†_i c_j>"""
    h = np.zeros((L, L))
    for i in range(L-1):
        h[i, i+1] = -J/2
        h[i+1, i] = -J/2
    if periodic:
        h[0, L-1] = -J/2
        h[L-1, 0] = -J/2

    eigenvalues, eigenvectors = np.linalg.eigh(h)
    occupied = eigenvalues < 0
    C = eigenvectors[:, occupied] @ eigenvectors[:, occupied].conj().T
    return np.real(C)

def entanglement_entropy(L, subsystem_size, J=1.0, periodic=False):
    """计算子区域的纠缠熵（自然对数）"""
    C = xx_correlation(L, J, periodic)
    C_sub = C[:subsystem_size, :subsystem_size]
    lambdas = np.linalg.eigvalsh(C_sub)
    lambdas = np.clip(lambdas, 1e-15, 1-1e-15)
    S = -np.sum(lambdas * np.log(lambdas) + (1-lambdas) * np.log(1-lambdas))
    return float(S)

# ============================================================
# 主程序
# ============================================================
print("=" * 70)
print("一维 XX 模型：临界点纠缠熵验证 Calabrese-Cardy 公式")
print("=" * 70)
print()
print("理论预言（PBC）：S(L/2) = (c/3) ln(L) + const,  c=1")
print("理论预言（OBC）：S(L/2) = (c/6) ln(L) + const,  c=1")
print()

Ls = [20, 40, 80, 160, 320, 640, 1280]

results = []
print(f"{'L':>6} | {'S(PBC)':>10} | {'S/lnL':>8} | {'S(OBC)':>10} | {'S/lnL':>8}")
print("-" * 70)

for L in Ls:
    S_pbc = entanglement_entropy(L, L//2, periodic=True)
    S_obc = entanglement_entropy(L, L//2, periodic=False)
    ratio_pbc = S_pbc / np.log(L)
    ratio_obc = S_obc / np.log(L)
    print(f"{L:>6} | {S_pbc:>10.4f} | {ratio_pbc:>8.4f} | {S_obc:>10.4f} | {ratio_obc:>8.4f}")
    results.append({"L": L, "S_pbc": S_pbc, "S_obc": S_obc})

# ============================================================
# 拟合斜率
# ============================================================
print()
print("=" * 70)
print("拟合 S(L) = a * ln(L) + b")
print("=" * 70)

ln_L = np.log([r["L"] for r in results])
S_pbc = np.array([r["S_pbc"] for r in results])
S_obc = np.array([r["S_obc"] for r in results])

coef_pbc = np.polyfit(ln_L, S_pbc, 1)
coef_obc = np.polyfit(ln_L, S_obc, 1)

print(f"PBC 拟合: S = {coef_pbc[0]:.4f} * ln(L) + {coef_pbc[1]:.4f}")
print(f"         理论斜率 = c/3 = 1/3 ≈ {1/3:.4f}")
print(f"         偏差 = {abs(coef_pbc[0] - 1/3):.4f}")
print()
print(f"OBC 拟合: S = {coef_obc[0]:.4f} * ln(L) + {coef_obc[1]:.4f}")
print(f"         理论斜率 = c/6 = 1/6 ≈ {1/6:.4f}")
print(f"         偏差 = {abs(coef_obc[0] - 1/6):.4f}")
print()

if abs(coef_pbc[0] - 1/3) < 0.02:
    print("✅ PBC 斜率验证成功！c = 1")
else:
    print(f"⚠️ PBC 斜率偏差 {abs(coef_pbc[0] - 1/3):.4f}")

if abs(coef_obc[0] - 1/6) < 0.02:
    print("✅ OBC 斜率验证成功！c = 1")
else:
    print(f"⚠️ OBC 斜率偏差 {abs(coef_obc[0] - 1/6):.4f}")

# 保存
with open("xx_result.json", "w") as f:
    json.dump({"results": results,
               "fit_pbc": {"slope": float(coef_pbc[0]), "intercept": float(coef_pbc[1])},
               "fit_obc": {"slope": float(coef_obc[0]), "intercept": float(coef_obc[1])}},
              f, indent=2)
print()
print("结果已保存到 xx_result.json")
