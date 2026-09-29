import numpy as np, json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def build_H_obc(N, h):
    dim = 2**N
    states = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim)
    for i in range(N-1):
        b_i = (states >> i) & 1
        b_j = (states >> (i+1)) & 1
        diag -= (1-2*b_i) * (1-2*b_j)
    rows = np.repeat(states, N)
    cols = np.zeros_like(rows)
    for i in range(N):
        cols[i::N] = states ^ (1 << i)
    vals = np.full(len(rows), -h, dtype=float)
    H = csr_matrix((diag, (states, states)), shape=(dim,dim))
    H += csr_matrix((vals, (rows, cols)), shape=(dim,dim))
    return H

def reduced_rho(psi, L, N):
    psi_t = psi.reshape([2]*N)
    perm = list(range(L)) + list(range(L, N))
    psi_t = np.transpose(psi_t, perm)
    psi_mat = psi_t.reshape(2**L, 2**(N-L))
    return psi_mat @ psi_mat.conj().T

results = {}
Ns = []
D1s = []
D2s = []

for N in [14, 16, 18, 20]:
    print(f"\n=== N = {N} ===")
    H = build_H_obc(N, 1.0)
    val, vec = eigsh(H, k=1, which='SA')
    psi = vec[:, 0]
    L = N // 2
    rho = reduced_rho(psi, L, N)
    evals = np.linalg.eigvalsh(rho)
    evals = np.sort(evals)[::-1]
    evals = evals[evals > 1e-14]
    eps = -np.log(evals)
    gaps = eps - eps[0]
    
    print(f"  前 6 个本征值: {evals[:6]}")
    print(f"  前 6 个间隙: {gaps[:6]}")
    
    # 正确归一化
    D1 = N * gaps[1] / (8 * np.pi)
    D2 = N * gaps[2] / (8 * np.pi)
    
    print(f"  Δ₁ = {D1:.6f}  (期望 1，能量场)")
    print(f"  Δ₂ = {D2:.6f}")
    
    Ns.append(N)
    D1s.append(D1)
    D2s.append(D2)
    
    results[str(N)] = {
        "N": N,
        "L": L,
        "gaps": [float(x) for x in gaps[:10]],
        "Delta1": float(D1),
        "Delta2": float(D2),
    }
    json.dump(results, open("ent_spectrum2_result.json","w"), indent=2)

# 一阶拟合: Δ₁(N) = Δ_∞ + a/N
Ns_arr = np.array(Ns, dtype=float)
D1_arr = np.array(D1s)
A1 = np.vstack([np.ones_like(Ns_arr), 1/Ns_arr]).T
coef1, _, _, _ = np.linalg.lstsq(A1, D1_arr, rcond=None)
D_inf_1, a1 = coef1
print(f"\n一阶拟合 Δ₁ = Δ_∞ + a/N:")
print(f"  Δ_∞ = {D_inf_1:.6f}")
print(f"  a = {a1:.6f}")

# 二阶拟合: Δ₁(N) = Δ_∞ + a/N + b/N²
A2 = np.vstack([np.ones_like(Ns_arr), 1/Ns_arr, 1/Ns_arr**2]).T
coef2, _, _, _ = np.linalg.lstsq(A2, D1_arr, rcond=None)
D_inf_2, a2, b2 = coef2
print(f"\n二阶拟合 Δ₁ = Δ_∞ + a/N + b/N²:")
print(f"  Δ_∞ = {D_inf_2:.6f}")
print(f"  a = {a2:.6f}")
print(f"  b = {b2:.6f}")

results["fit_1st"] = {"D_inf": float(D_inf_1), "a": float(a1)}
results["fit_2nd"] = {"D_inf": float(D_inf_2), "a": float(a2), "b": float(b2)}
json.dump(results, open("ent_spectrum2_result.json","w"), indent=2)

print(f"\n=== 收敛检查 ===")
for N, D1 in zip(Ns, D1s):
    print(f"  N={N}:  Δ₁ = {D1:.6f}")
print(f"\n外推 N→∞:")
print(f"  一阶: Δ₁ → {D_inf_1:.6f}")
print(f"  二阶: Δ₁ → {D_inf_2:.6f}")
print(f"\nCFT 预言: Δ_ε = 1")
