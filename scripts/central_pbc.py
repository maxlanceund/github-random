import numpy as np, json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def build_H_pbc(N, h):
    dim = 2**N
    states = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim)
    for i in range(N):
        j = (i+1) % N
        b_i = (states >> i) & 1
        b_j = (states >> j) & 1
        diag -= (1-2*b_i) * (1-2*b_j)
    rows = np.repeat(states, N)
    cols = np.zeros_like(rows)
    for i in range(N):
        cols[i::N] = states ^ (1 << i)
    vals = np.full(len(rows), -h, dtype=float)
    H = csr_matrix((diag, (states, states)), shape=(dim,dim))
    H += csr_matrix((vals, (rows, cols)), shape=(dim,dim))
    return H

def ent_L(psi, L, N):
    psi_t = psi.reshape([2]*N)
    perm = list(range(L)) + list(range(L, N))
    psi_t = np.transpose(psi_t, perm)
    dk = 2**L
    dt = 2**(N-L)
    psi_mat = psi_t.reshape(dk, dt)
    rho = psi_mat @ psi_mat.conj().T
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-12]
    return float(-np.sum(e * np.log(e)))

def fit_pbc(Ls, Ss, N):
    Ls = np.array(Ls, dtype=float)
    Ss = np.array(Ss)
    x = np.log(np.sin(np.pi * Ls / N))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, Ss, rcond=None)
    a, b = coef
    c = 3 * a
    pred = A @ coef
    residual = float(np.sum((Ss - pred)**2))
    return float(c), float(a), float(b), residual

results = {}
for N in [14, 16, 18]:
    print(f"\n=== N = {N} ===")
    H = build_H_pbc(N, 1.0)
    val, vec = eigsh(H, k=1, which='SA')
    psi = vec[:, 0]
    Lmax = N // 2
    Ls = list(range(1, Lmax+1))
    Ss = [ent_L(psi, L, N) for L in Ls]
    for L, S in zip(Ls, Ss):
        print(f"  L={L:>2}  S={S:.6f}")
    c, a, b, resid = fit_pbc(Ls, Ss, N)
    print(f"  c = 3a = {c:.4f}, 残差 = {resid:.6f}")
    results[str(N)] = {"L": Ls, "S": Ss, "a": a, "b": b, "c": c, "residual": resid}
    # 每跑完一个 N 立刻写文件
    json.dump(results, open("central_pbc_result.json","w"), indent=2)

print(f"\n=== 中心荷 c ===")
for N in [14,16,18]:
    print(f"  N={N:>2}:  c = {results[str(N)]['c']:.4f}")
print(f"\nCFT 预言: c → 0.5")
print("\n保存到 central_pbc_result.json")
