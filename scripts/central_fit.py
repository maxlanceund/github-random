import numpy as np, json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def build_H(N, h):
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

def ent_L(psi, L, N):
    psi_t = psi.reshape([2]*N)
    keep = list(range(L))
    trace = list(range(L, N))
    perm = keep + trace
    psi_t = np.transpose(psi_t, perm)
    dk = 2**L
    dt = 2**(N-L)
    psi_mat = psi_t.reshape(dk, dt)
    rho = psi_mat @ psi_mat.conj().T
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-12]
    return float(-np.sum(e * np.log2(e)))

def fit_c(Ls, Ss, N):
    # 开边界 CFT: S(L) = a * ln(sin(pi L / N)) + b
    # c = 6a
    Ls = np.array(Ls, dtype=float)
    Ss = np.array(Ss)
    x = np.log(np.sin(np.pi * Ls / N))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, rank, sv = np.linalg.lstsq(A, Ss, rcond=None)
    a, b = coef
    c = 6 * a
    pred = A @ coef
    residual = float(np.sum((Ss - pred)**2))
    return float(c), float(a), float(b), residual

results = {}
for N in [14, 16, 18, 20]:
    print(f"\n=== N = {N} ===")
    H = build_H(N, 1.0)
    val, vec = eigsh(H, k=1, which='SA')
    psi = vec[:, 0]
    Lmax = N // 2
    Ls = list(range(1, Lmax+1))
    Ss = [ent_L(psi, L, N) for L in Ls]
    for L, S in zip(Ls, Ss):
        print(f"  L={L:>2}  S={S:.6f}")
    c, a, b, resid = fit_c(Ls, Ss, N)
    print(f"  拟合: S(L) = {a:.4f} * ln(sin(pi L/N)) + {b:.4f}")
    print(f"  a = {a:.4f}, c = 6a = {c:.4f}, 残差 = {resid:.6f}")
    results[str(N)] = {
        "L": Ls, "S": Ss,
        "a": a, "b": b, "c": c,
        "residual": resid
    }

# 收敛趋势
cs = [results[str(N)]["c"] for N in [14,16,18,20]]
print(f"\n=== 中心荷 c 随 N 的变化 ===")
for N, c in zip([14,16,18,20], cs):
    print(f"  N={N:>2}:  c = {c:.4f}")
print(f"\nCFT 预言: c → 0.5")
print(f"趋势: {'收敛' if abs(cs[-1] - cs[-2]) < abs(cs[1] - cs[0]) else '发散'}")

json.dump(results, open("central_fit_result.json","w"), indent=2)
print("\n保存到 central_fit_result.json")
