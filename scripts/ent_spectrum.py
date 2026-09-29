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
    dk = 2**L
    dt = 2**(N-L)
    psi_mat = psi_t.reshape(dk, dt)
    rho = psi_mat @ psi_mat.conj().T
    return rho

results = {}
for N in [14, 16, 18]:
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
    
    print(f"  半链大小 L = {L}")
    print(f"  前 15 个本征值:")
    for i in range(min(15, len(evals))):
        print(f"    i={i:>2}  λ={evals[i]:.6e}  ε={eps[i]:.6f}  Δε={gaps[i]:.6f}")
    
    # CFT 预言: N*(ε₁-ε₀) / (2π) → Δ_min
    N_times_gap1 = N * gaps[1] if len(gaps) > 1 else None
    N_times_gap2 = N * gaps[2] if len(gaps) > 2 else None
    Delta1 = N_times_gap1 / (2*np.pi) if N_times_gap1 else None
    Delta2 = N_times_gap2 / (2*np.pi) if N_times_gap2 else None
    
    print(f"\n  N*(ε₁-ε₀) = {N_times_gap1:.6f}")
    print(f"  N*(ε₂-ε₀) = {N_times_gap2:.6f}")
    print(f"  Δ₁ = N*(ε₁-ε₀)/(2π) = {Delta1:.6f}  (CFT 预言 1/8 = 0.125)")
    print(f"  Δ₂ = N*(ε₂-ε₀)/(2π) = {Delta2:.6f}")
    
    results[str(N)] = {
        "L": L,
        "evals_top15": [float(x) for x in evals[:15]],
        "eps_top15": [float(x) for x in eps[:15]],
        "gaps_top15": [float(x) for x in gaps[:15]],
        "Delta1": float(Delta1) if Delta1 else None,
        "Delta2": float(Delta2) if Delta2 else None,
        "N_times_gap1": float(N_times_gap1) if N_times_gap1 else None,
    }
    json.dump(results, open("ent_spectrum_result.json","w"), indent=2)

print(f"\n=== 收敛检查 ===")
for N in [14,16,18]:
    d1 = results[str(N)]["Delta1"]
    print(f"  N={N:>2}:  Δ₁ = {d1:.6f}")
print(f"\nCFT 预言: Δ₁ → 0.125 (Ising 自旋场 σ)")
print("\n保存到 ent_spectrum_result.json")
