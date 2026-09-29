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
    logLs = np.log(Ls)
    Ss_arr = np.array(Ss)
    slopes = []
    for i in range(1, len(Ls)):
        dS = Ss_arr[i] - Ss_arr[i-1]
        dlog = logLs[i] - logLs[i-1]
        c_eff = 6 * np.log(2) * dS / dlog
        slopes.append(float(c_eff))
    print(f"  有效中心荷 c_eff (L→L+1):")
    for L, c in zip(Ls[1:], slopes):
        print(f"    L={L:>2}  c_eff={c:.4f}")
    results[str(N)] = {"L": Ls, "S": Ss, "c_eff": slopes}

json.dump(results, open("central_charge_result.json","w"), indent=2)
print("\n保存到 central_charge_result.json")
print(f"CFT 预言: c_eff → 0.5（中心荷 c=1/2）")
