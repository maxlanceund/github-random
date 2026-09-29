import numpy as np
import json

def kron_op(op, k, N):
    out = np.array([[1.0]])
    for i in range(N):
        out = np.kron(out, op if i == k else np.eye(2))
    return out

def ptrace(rho, keep, n):
    keep = sorted(keep)
    tr = [i for i in range(n) if i not in keep]
    rt = rho.reshape([2]*(2*n))
    for i in sorted(tr, reverse=True):
        rt = np.trace(rt, axis1=i, axis2=i+n)
        n -= 1
    return rt.reshape(2**len(keep), -1)

def ent(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-12]
    return -np.sum(e * np.log2(e))

def mi(rho, i, j, N):
    return ent(ptrace(rho,[i],N)) + ent(ptrace(rho,[j],N)) - ent(ptrace(rho,[i,j],N))

N = 8
Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
J = [1.0, 1.0, 1.0, 0.5, 0.2, 0.2, 0.2]

H = np.zeros((2**N, 2**N), dtype=complex)
for i in range(N-1):
    H -= J[i] * kron_op(Z,i,N) @ kron_op(Z,i+1,N)
for i in range(N):
    H -= 1.0 * kron_op(X,i,N)

evals, evecs = np.linalg.eigh(H)
psi = evecs[:, 0]
rho = np.outer(psi, psi.conj())

MI = np.zeros((N,N))
for i in range(N):
    for j in range(i+1,N):
        MI[i,j] = MI[j,i] = mi(rho,i,j,N)

# 对数距离
d = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i != j:
            d[i,j] = np.log(1.0 / (MI[i,j] + 1e-6))

# 三角不等式
viol = 0
total = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            if i!=j and j!=k and i!=k:
                total += 1
                if d[i,k] > d[i,j] + d[j,k] + 1e-6:
                    viol += 1

print("=" * 60)
print("修正版：对数距离")
print("=" * 60)
print(f"\n三角不等式违反: {viol} / {total}")
print(f"违反率: {100*viol/total:.1f}%")

print(f"\n对数距离矩阵:")
print("      " + "  ".join(f"{i:>5}" for i in range(N)))
for i in range(N):
    print(f"  {i}: " + "  ".join(f"{d[i,j]:5.2f}" for j in range(N)))

results = {
    "N": N, "J": J,
    "MI_matrix": MI.tolist(),
    "log_distance": d.tolist(),
    "violations": int(viol),
    "total_triples": int(total),
    "violation_rate": float(viol/total),
    "mean_MI": float(MI[np.triu_indices(N,1)].mean()),
    "max_MI": float(MI[np.triu_indices(N,1)].max()),
    "min_MI": float(MI[np.triu_indices(N,1)].min()),
}
json.dump(results, open("entanglement_geometry2_result.json","w"), indent=2)
print("\n保存到 entanglement_geometry2_result.json")
