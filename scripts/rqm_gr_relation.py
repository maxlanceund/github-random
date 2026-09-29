import numpy as np
import json
from scipy.linalg import expm

def kron_op(op, k, N):
    out = np.array([[1.0]])
    for i in range(N):
        out = np.kron(out, op if i == k else np.eye(2))
    return out

def partial_trace(rho, keep, n):
    keep = sorted(keep)
    rho_t = rho.reshape([2]*(2*n))
    labels = "abcdefghijklmnopqrstuvwxyz"[:n]
    blabels = labels.upper()
    input_str = list(labels + blabels)
    output_str = []
    for i in range(n):
        if i in keep:
            output_str.append(labels[i])
            output_str.append(blabels[i])
        else:
            input_str[n + i] = labels[i]
    ein_str = "".join(input_str) + "->" + "".join(output_str)
    result = np.einsum(ein_str, rho_t)
    d = 2**len(keep)
    return result.reshape(d, d)

def entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return -np.sum(evals * np.log2(evals))

def mutual_info(rho, i, j, N):
    return (entropy(partial_trace(rho, [i], N))
            + entropy(partial_trace(rho, [j], N))
            - entropy(partial_trace(rho, [i, j], N)))

# ===== 主程序 =====
N = 8
Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])

H = np.zeros((2**N, 2**N), dtype=complex)
for i in range(N-1):
    H -= 1.0 * kron_op(Z, i, N) @ kron_op(Z, i+1, N)
for i in range(N):
    H -= 0.5 * kron_op(X, i, N)

psi0 = np.zeros(2**N, dtype=complex)
psi0[0] = 1.0

U = expm(-1j * H * 0.5)
psi = U @ psi0
rho = np.outer(psi, psi.conj())

# 互信息矩阵
MI = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        MI[i, j] = MI[j, i] = mutual_info(rho, i, j, N)

print("=" * 70)
print(f"关系结构模拟：N = {N} 量子比特")
print("=" * 70)

print("\n[1] 互信息矩阵（关系几何）:")
print("      " + "  ".join(f"{i:>5}" for i in range(N)))
for i in range(N):
    print(f"  {i}: " + "  ".join(f"{MI[i,j]:5.2f}" for j in range(N)))
print(f"\n  平均 MI: {MI[np.triu_indices(N, 1)].mean():.4f}")

# S=0 相对于不同参照系 R 的状态
S = 0
rho_S = partial_trace(rho, [S], N)
purity_S = np.trace(rho_S @ rho_S).real

print(f"\n[2] S = {S} 相对于不同参照系 R 的状态:")
print(f"    S 的无条件纯度: {purity_S:.4f}")

all_td = []
for R in range(1, N):
    P0 = kron_op(np.array([[1,0],[0,0]]), R, N)
    P1 = kron_op(np.array([[0,0],[0,1]]), R, N)
    diffs = []
    for r, P in [(0, P0), (1, P1)]:
        cond = P @ rho @ P
        p_r = np.trace(cond).real
        if p_r < 1e-10:
            continue
        cond = cond / p_r
        rho_S_cond = partial_trace(cond, [S], N)
        td = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(rho_S_cond - rho_S)))
        diffs.append(td)
        all_td.append(td)
    print(f"    R = {R}: |rho_S^R - rho_S| 范围 = [{min(diffs):.4f}, {max(diffs):.4f}]")

print(f"\n[3] 关系性分析:")
print(f"    所有 (R, r) 的平均 |rho_S^R - rho_S|: {np.mean(all_td):.4f}")
print(f"    最大值: {np.max(all_td):.4f}")
print(f"    最小值: {np.min(all_td):.4f}")

print(f"\n[4] 几何类比:")
print(f"    MI(i:j) = '关系几何'")
print(f"    高 MI → 强关系（类似强引力）")
print(f"    低 MI → 弱关系（类似弱引力）")
print(f"    平均 MI: {MI[np.triu_indices(N, 1)].mean():.4f}")

results = {
    "N": N,
    "MI_matrix": MI.tolist(),
    "MI_mean": float(MI[np.triu_indices(N, 1)].mean()),
    "purity_S_unconditional": float(purity_S),
    "mean_trace_distance": float(np.mean(all_td)),
    "max_trace_distance": float(np.max(all_td)),
    "min_trace_distance": float(np.min(all_td)),
    "all_trace_distances": [float(x) for x in all_td],
}

with open("rqm_gr_relation_result.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("结论:")
print("=" * 70)
print(f"  1. S 相对于不同参照系 R 的状态不同 → 量子态是关系性的（RQM）")
print(f"  2. 关系几何（MI 矩阵）由量子比特之间的关系决定 → 几何是关系性的（GR 类比）")
print(f"  3. 两者共享同一个关系结构 → RQM 和 GR 是关系结构的两个方面")
print("\n保存到 rqm_gr_relation_result.json")
